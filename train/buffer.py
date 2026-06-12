"""RolloutBuffer — stockage GPU des trajectoires + GAE + fenêtres d'attention.

Pour économiser la mémoire, l'historique n'est PAS stocké par step (x16) :
on garde une seule séquence d'obs [T, B, D] précédée d'un préfixe [H-1, B, D]
(l'historique d'avant le rollout), et les fenêtres [M, H, D] sont
reconstruites à la volée par minibatch. Les ticks antérieurs au début de
l'épisode sont masqués à zéro grâce à `age` (ticks depuis le reset).
"""

import torch


class RolloutBuffer:
    def __init__(self, T: int, B: int, obs_dim: int, history: int,
                 device: torch.device):
        self.T, self.B, self.D, self.H = T, B, obs_dim, history
        self.device = device
        d = device
        self.obs = torch.zeros(T, B, obs_dim, device=d)
        self.age = torch.zeros(T, B, dtype=torch.long, device=d)
        self.pre = torch.zeros(T, B, 2, device=d)
        self.fwd = torch.zeros(T, B, dtype=torch.long, device=d)
        self.strafe = torch.zeros(T, B, dtype=torch.long, device=d)
        self.bins = torch.zeros(T, B, 3, device=d)
        self.logp = torch.zeros(T, B, device=d)
        self.value = torch.zeros(T, B, device=d)
        self.reward = torch.zeros(T, B, device=d)
        self.done = torch.zeros(T, B, device=d)
        self.prefix = torch.zeros(max(history - 1, 0), B, obs_dim, device=d)
        self.adv = torch.zeros(T, B, device=d)
        self.ret = torch.zeros(T, B, device=d)
        self.t = 0
        self._padded = None   # cache [H-1+T, B, D] pour windows()
        self._gae_done = False

    def set_prefix(self, hist_now: torch.Tensor) -> None:
        """hist_now [B, H, D] : l'historique courant AVANT le 1er step."""
        if self.H > 1:
            self.prefix.copy_(hist_now[:, :-1].permute(1, 0, 2))

    def add(self, obs, age, raw, logp, value, reward, done) -> None:
        t = self.t
        self.obs[t] = obs
        self.age[t] = age
        self.pre[t] = raw["pre"]
        self.fwd[t] = raw["fwd"]
        self.strafe[t] = raw["strafe"]
        self.bins[t] = raw["bins"]
        self.logp[t] = logp
        self.value[t] = value
        self.reward[t] = reward
        self.done[t] = done
        self.t += 1

    # ------------------------------------------------------------------- GAE
    @torch.no_grad()
    def compute_gae(self, last_value: torch.Tensor, gamma: float, lam: float) -> None:
        # idempotent par rollout : en mode population, chaque membre appelle
        # update() sur le même buffer — gamma/lam sont partagés (le PBT
        # n'explore PAS gamma/lam, le cache serait invalide sinon)
        if self._gae_done:
            return
        self._gae_done = True
        adv = torch.zeros(self.B, device=self.device)
        next_value = last_value
        for t in reversed(range(self.T)):
            not_done = 1.0 - self.done[t]
            delta = self.reward[t] + gamma * next_value * not_done - self.value[t]
            adv = delta + gamma * lam * not_done * adv
            self.adv[t] = adv
            next_value = self.value[t]
        self.ret = self.adv + self.value

    # ------------------------------------------------------- fenêtres window
    @torch.no_grad()
    def windows(self, t_idx: torch.Tensor, b_idx: torch.Tensor) -> torch.Tensor:
        """Reconstruit les historiques [M, H, D] des échantillons (t, b)."""
        if self._padded is None:
            self._padded = torch.cat([self.prefix, self.obs], dim=0)  # [H-1+T,B,D]
        padded = self._padded
        k = torch.arange(self.H, device=self.device)
        ti = t_idx.unsqueeze(1) + k                           # [M, H]
        win = padded[ti, b_idx.unsqueeze(1)]                  # [M, H, D]
        valid = self.age[t_idx, b_idx].unsqueeze(1) >= (self.H - 1 - k)
        return win * valid.unsqueeze(-1)

    def raw_at(self, t_idx, b_idx) -> dict:
        return {
            "pre": self.pre[t_idx, b_idx],
            "fwd": self.fwd[t_idx, b_idx],
            "strafe": self.strafe[t_idx, b_idx],
            "bins": self.bins[t_idx, b_idx],
        }

    def reset(self) -> None:
        self.t = 0
        self._padded = None
        self._gae_done = False
