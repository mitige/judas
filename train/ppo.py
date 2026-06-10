"""PPO clipped + GAE, mixed precision, sur les fenêtres d'attention."""

from dataclasses import dataclass

import torch

from .buffer import RolloutBuffer
from .model import JudasPolicy


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.995
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 3
    minibatch_size: int = 16384
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    amp: bool = True
    # stabilité / qualité d'apprentissage
    target_kl: float = 0.02      # early-stop des epochs si KL moyen dépasse 1.5x
    value_clip: float = 0.2      # clipping PPO2 du critic (0 = désactivé)
    anneal: bool = True          # lr et entropie -> 0 linéairement sur le run


class PPO:
    def __init__(self, policy: JudasPolicy, cfg: PPOConfig, device: torch.device):
        self.policy = policy
        self.cfg = cfg
        self.device = device
        self.opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr, eps=1e-5)
        self.use_amp = cfg.amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def update(self, buf: RolloutBuffer, learner_mask: torch.Tensor,
               last_value: torch.Tensor, progress: float = 0.0) -> dict:
        """progress dans [0,1] : avancement du run (pour l'annealing)."""
        cfg = self.cfg
        frac = max(1.0 - progress, 0.05) if cfg.anneal else 1.0
        for g in self.opt.param_groups:
            g["lr"] = cfg.lr * frac
        ent_coef = cfg.ent_coef * frac
        buf.compute_gae(last_value, cfg.gamma, cfg.lam)

        # échantillons (t, b) des agents contrôlés par le learner
        b_keep = torch.nonzero(learner_mask, as_tuple=False).squeeze(-1)
        T = buf.T
        t_all = torch.arange(T, device=self.device).repeat_interleave(b_keep.numel())
        b_all = b_keep.repeat(T)
        n = t_all.numel()

        adv_all = buf.adv[t_all, b_all]
        adv_mean, adv_std = adv_all.mean(), adv_all.std().clamp_min(1e-6)

        # stats accumulées sur GPU : une seule synchro à la fin de l'update
        acc = {k: torch.zeros((), device=self.device)
               for k in ("loss_pi", "loss_v", "entropy", "clip_frac", "approx_kl")}
        n_updates = 0

        stop = False
        for _ in range(cfg.epochs):
            if stop:
                break
            kl_epoch = torch.zeros((), device=self.device)
            n_mb = 0
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, cfg.minibatch_size):
                mb = perm[start:start + cfg.minibatch_size]
                ti, bi = t_all[mb], b_all[mb]
                hist = buf.windows(ti, bi)
                raw = buf.raw_at(ti, bi)
                old_logp = buf.logp[ti, bi]
                old_value = buf.value[ti, bi]
                ret = buf.ret[ti, bi]
                adv = (buf.adv[ti, bi] - adv_mean) / adv_std

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    logp, entropy, value = self.policy.evaluate(hist, raw)
                    ratio = (logp - old_logp).exp()
                    s1 = ratio * adv
                    s2 = ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv
                    loss_pi = -torch.min(s1, s2).mean()
                    if cfg.value_clip > 0:
                        v_clip = old_value + (value - old_value).clamp(
                            -cfg.value_clip, cfg.value_clip)
                        loss_v = 0.5 * torch.max((value - ret).pow(2),
                                                 (v_clip - ret).pow(2)).mean()
                    else:
                        loss_v = 0.5 * (value - ret).pow(2).mean()
                    loss_ent = -entropy.mean()
                    loss = (loss_pi + cfg.vf_coef * loss_v
                            + ent_coef * loss_ent)

                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(),
                                               cfg.max_grad_norm)
                self.scaler.step(self.opt)
                self.scaler.update()

                with torch.no_grad():
                    kl = (old_logp - logp).mean().detach()
                    acc["loss_pi"] += loss_pi.detach()
                    acc["loss_v"] += loss_v.detach()
                    acc["entropy"] += entropy.mean().detach()
                    acc["clip_frac"] += ((ratio - 1).abs() > cfg.clip).float().mean()
                    acc["approx_kl"] += kl
                    kl_epoch += kl
                n_updates += 1
                n_mb += 1

            # early-stop : une seule synchro GPU par epoch
            if cfg.target_kl > 0 and n_mb > 0:
                if float(kl_epoch.item()) / n_mb > 1.5 * cfg.target_kl:
                    stop = True

        d = max(n_updates, 1)
        out = {k: float(v.item()) / d for k, v in acc.items()}
        out["lr"] = cfg.lr * frac
        return out
