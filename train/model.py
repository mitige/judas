"""JudasPolicy — policy transformer à actions hybrides.

Entrée  : historique d'observations [B, H, OBS_DIM] (H=16 ticks)
Encodage: MLP par tick -> + embedding positionnel -> TransformerEncoder
          (attention multi-têtes sur le temps) -> dernier token
Sorties : - Gaussienne tanh-squashed (Δyaw, Δpitch) normalisée [-1, 1]
          - Catégorielles 3 classes : forward, strafe (-1/0/+1)
          - Bernoulli : jump, sprint, attack
          - Value (critic)

Les actions échantillonnées sont stockées sous forme brute (pré-tanh +
indices) pour le calcul exact des log-probs PPO, et converties au format
sim [.., 7] par to_sim_actions().
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from sim.obs import OBS_DIM

LOG_STD_MIN, LOG_STD_MAX = -4.0, 1.0
TANH_EPS = 1e-6


def _categorical_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits.float(), dim=-1)
    return -(logp.exp() * logp).sum(-1)


def _gumbel_sample(logits: torch.Tensor) -> torch.Tensor:
    """Échantillonnage catégoriel par gumbel-max — équivalent exact de
    Categorical(logits).sample(), mais en ops tenseur pures : capturable par
    CUDA graph et sans surcoût d'objets distribution. Le bruit est tiré en
    float32 (un clamp 1e-9 sous-déborderait en fp16 sous autocast)."""
    u = torch.rand(logits.shape, device=logits.device,
                   dtype=torch.float32).clamp_(1e-9, 1.0)
    return (logits.float() - torch.log(-torch.log(u))).argmax(-1)


def _bernoulli_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    return (F.softplus(logits) - logits * torch.sigmoid(logits)).sum(-1)


@dataclass
class PolicyConfig:
    obs_dim: int = OBS_DIM
    history: int = 8
    d_model: int = 96           # taille du cerveau
    n_heads: int = 4
    n_layers: int = 2
    ff_mult: int = 4
    dropout: float = 0.0
    attention: bool = True      # False -> MLP sur l'historique aplati (plus rapide)


class JudasPolicy(nn.Module):
    def __init__(self, cfg: PolicyConfig | None = None):
        super().__init__()
        self.cfg = cfg or PolicyConfig()
        c = self.cfg

        if c.attention:
            self.encoder = nn.Sequential(
                nn.Linear(c.obs_dim, c.d_model),
                nn.LayerNorm(c.d_model),
                nn.GELU(),
                nn.Linear(c.d_model, c.d_model),
            )
            self.pos_emb = nn.Parameter(torch.zeros(1, c.history, c.d_model))
            layer = nn.TransformerEncoderLayer(
                d_model=c.d_model, nhead=c.n_heads,
                dim_feedforward=c.d_model * c.ff_mult,
                dropout=c.dropout, activation="gelu",
                batch_first=True, norm_first=True)
            self.transformer = nn.TransformerEncoder(layer, num_layers=c.n_layers)
            self.norm = nn.LayerNorm(c.d_model)
            nn.init.normal_(self.pos_emb, std=0.02)
        else:
            # trunk MLP : historique aplati, n_layers couches cachées
            dims = [c.obs_dim * c.history] + [c.d_model] * max(c.n_layers, 1)
            mlp = []
            for a, b in zip(dims[:-1], dims[1:]):
                mlp += [nn.Linear(a, b), nn.LayerNorm(b), nn.GELU()]
            self.mlp = nn.Sequential(*mlp)

        self.mean_head = nn.Linear(c.d_model, 2)        # Δyaw, Δpitch
        self.log_std = nn.Parameter(torch.full((2,), -0.5))
        self.fwd_head = nn.Linear(c.d_model, 3)
        self.strafe_head = nn.Linear(c.d_model, 3)
        self.bin_head = nn.Linear(c.d_model, 3)         # jump, sprint, attack
        self.value_head = nn.Linear(c.d_model, 1)
        # tête auxiliaire : prédit l'état adverse au tick suivant (obs[0:7] =
        # position relative + distance + vélocité adverse) — force le trunk à
        # modéliser l'adversaire
        self.aux_head = nn.Linear(c.d_model, 7)

        for head in (self.mean_head, self.fwd_head, self.strafe_head, self.bin_head):
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    # ------------------------------------------------------------------ core
    def trunk(self, hist: torch.Tensor) -> torch.Tensor:
        """[B, H, obs] -> [B, d_model]."""
        if self.cfg.attention:
            x = self.encoder(hist) + self.pos_emb
            x = self.transformer(x)
            return self.norm(x[:, -1])
        return self.mlp(hist.flatten(1))

    def heads(self, z: torch.Tensor):
        mean = self.mean_head(z)
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).expand_as(mean)
        return (mean, log_std, self.fwd_head(z), self.strafe_head(z),
                self.bin_head(z), self.value_head(z).squeeze(-1))

    # ------------------------------------------------------------------- act
    @torch.no_grad()
    def act(self, hist: torch.Tensor, deterministic: bool = False) -> dict:
        """Échantillonne une action. Retourne un dict de tenseurs bruts +
        logp + value (tout sur le device de hist)."""
        mean, log_std, fwd_l, str_l, bin_l, value = self.heads(self.trunk(hist))
        if deterministic:
            pre = mean
            fwd = fwd_l.argmax(-1)
            strafe = str_l.argmax(-1)
            bins = (bin_l > 0).float()
        else:
            pre = mean + torch.randn_like(mean) * log_std.exp()
            fwd = _gumbel_sample(fwd_l)
            strafe = _gumbel_sample(str_l)
            bins = (torch.rand(bin_l.shape, device=bin_l.device,
                               dtype=torch.float32)
                    < torch.sigmoid(bin_l.float())).to(bin_l.dtype)
        raw = {"pre": pre, "fwd": fwd, "strafe": strafe, "bins": bins}
        logp = self.log_prob(mean, log_std, fwd_l, str_l, bin_l, raw)
        return {**raw, "logp": logp, "value": value}

    # ------------------------------------------------------------- log-probs
    @staticmethod
    def log_prob(mean, log_std, fwd_l, str_l, bin_l, raw) -> torch.Tensor:
        std = log_std.exp()
        pre = raw["pre"]
        normal_lp = (-0.5 * (((pre - mean) / std) ** 2)
                     - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        # correction tanh (changement de variable)
        squash = torch.log(1 - torch.tanh(pre) ** 2 + TANH_EPS).sum(-1)
        lp_cont = normal_lp - squash
        lp_fwd = -F.cross_entropy(fwd_l, raw["fwd"].long(), reduction="none")
        lp_str = -F.cross_entropy(str_l, raw["strafe"].long(), reduction="none")
        lp_bin = -F.binary_cross_entropy_with_logits(
            bin_l, raw["bins"], reduction="none").sum(-1)
        return lp_cont + lp_fwd + lp_str + lp_bin

    def evaluate(self, hist: torch.Tensor, raw: dict):
        """Log-probs / entropie / value / prédiction aux pour PPO."""
        z = self.trunk(hist)
        mean, log_std, fwd_l, str_l, bin_l, value = self.heads(z)
        logp = self.log_prob(mean, log_std, fwd_l, str_l, bin_l, raw)
        ent_cont = (0.5 * (1.0 + math.log(2 * math.pi)) + log_std.float()).sum(-1)
        ent_fwd = _categorical_entropy_from_logits(fwd_l)
        ent_str = _categorical_entropy_from_logits(str_l)
        ent_bin = _bernoulli_entropy_from_logits(bin_l)
        entropy = ent_cont + ent_fwd + ent_str + ent_bin
        return logp, entropy, value, self.aux_head(z)


def to_sim_actions(raw: dict) -> torch.Tensor:
    """dict d'actions brutes [B, ...] -> tenseur sim [B, 7] (convention sim/)."""
    pre = raw["pre"]
    out = torch.zeros(pre.shape[0], 7, dtype=torch.float32, device=pre.device)
    out[:, 0:2] = torch.tanh(pre)
    out[:, 2] = raw["fwd"].float() - 1.0      # 0,1,2 -> -1,0,1
    out[:, 3] = raw["strafe"].float() - 1.0
    out[:, 4:7] = raw["bins"]
    return out
