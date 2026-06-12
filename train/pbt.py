"""Population Based Training — population de policies co-entraînées.

Chaque membre possède sa policy, son optimiseur et ses hyperparamètres
(lr, ent_coef, clip). La population s'affronte (population-play), partage la
league de snapshots et le chase-bot, et est classée par ELO. Périodiquement
(exploit/explore) : les membres du bas copient poids + optimiseur + hypers
d'un membre du haut, avec hypers perturbés (×perturb_low / ×perturb_high,
bornés). Référence : Jaderberg et al., "Population Based Training of Neural
Networks" (2017), adapté au self-play league.

L'annealing lr/entropie du PPO est désactivé en mode population : c'est le
PBT qui pilote ces hyperparamètres, en continu et par sélection.
"""

import random
from dataclasses import dataclass, field

import torch

DEFAULT_PBT = {
    "population": 1,        # 1 = mode single-policy (comportement historique)
    "interval": 25,         # itérations entre deux exploit/explore
    "truncation": 0.25,     # part du bas qui copie la part du haut
    "perturb_low": 0.8,
    "perturb_high": 1.25,
    "cross_frac": 0.25,     # part des envs de chaque membre vs autres membres
    # bornes de l'espace d'exploration (clés de PPOConfig)
    "explore": {"lr": [6e-5, 6e-4],
                "ent_coef": [0.002, 0.02],
                "clip": [0.1, 0.3]},
}


@dataclass
class Member:
    """Un membre de la population : policy + optimiseur + hypers + fitness."""
    idx: int
    policy: object                  # JudasPolicy
    ppo: object                     # PPO (possède opt + scaler + cfg)
    hypers: dict
    elo: float = 1000.0
    env_lo: int = 0                 # slice d'envs [lo, hi)
    env_hi: int = 0
    games: int = 0

    @property
    def row_lo(self) -> int:        # lignes agent [B] correspondantes
        return self.env_lo * 2

    @property
    def row_hi(self) -> int:
        return self.env_hi * 2


def slice_envs(n_envs: int, k: int) -> list:
    """Découpe [0, n_envs) en k tranches contiguës quasi égales."""
    base, rem = divmod(n_envs, k)
    out, lo = [], 0
    for i in range(k):
        hi = lo + base + (1 if i < rem else 0)
        out.append((lo, hi))
        lo = hi
    return out


def perturb_hypers(hypers: dict, explore: dict, low: float, high: float,
                   rng: random.Random) -> dict:
    """Perturbe chaque hyperparamètre exploré (×low ou ×high), borné."""
    out = dict(hypers)
    for key, (lo, hi) in explore.items():
        factor = low if rng.random() < 0.5 else high
        out[key] = min(max(hypers[key] * factor, lo), hi)
    return out


def apply_hypers(ppo, hypers: dict) -> None:
    """Applique les hypers d'un membre dans son instance PPO (lr incluse)."""
    for key, value in hypers.items():
        setattr(ppo.cfg, key, value)
    for group in ppo.opt.param_groups:
        group["lr"] = hypers.get("lr", group["lr"])


def elo_delta(rating_a: float, rating_b: float, score_a: float,
              k: float = 16.0) -> float:
    """Delta ELO du joueur A pour un résultat score_a (1 / 0.5 / 0)."""
    expected = 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
    return k * (score_a - expected)


def exploit_explore(members: list, cfg: dict, rng: random.Random) -> list:
    """Truncation selection : le bas copie le haut (poids + optimiseur +
    hypers perturbés). Retourne la liste des événements [(loser, winner)]."""
    k = len(members)
    q = max(1, int(round(k * cfg["truncation"])))
    if k < 2 or q * 2 > k:
        return []
    ranked = sorted(members, key=lambda m: m.elo, reverse=True)
    top, bottom = ranked[:q], ranked[-q:]
    events = []
    for loser in bottom:
        winner = rng.choice(top)
        loser.policy.load_state_dict(winner.policy.state_dict())
        loser.ppo.opt.load_state_dict(winner.ppo.opt.state_dict())
        loser.ppo.scaler.load_state_dict(winner.ppo.scaler.state_dict())
        loser.hypers = perturb_hypers(winner.hypers, cfg["explore"],
                                      cfg["perturb_low"], cfg["perturb_high"],
                                      rng)
        apply_hypers(loser.ppo, loser.hypers)
        loser.elo = winner.elo          # repart de la fitness du parent
        events.append((loser.idx, winner.idx))
    return events
