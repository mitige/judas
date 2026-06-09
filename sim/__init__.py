"""sim — simulateur boxing 1.8.9 vectorisé (CUDA + backend de référence CPU)."""

from .config import ACTION_DIM, MAX_ACTION_DELAY, SimConfig
from .obs import OBS_DIM
from .ref_backend import JudasSimRef

__all__ = ["SimConfig", "ACTION_DIM", "MAX_ACTION_DELAY", "OBS_DIM",
           "JudasSimRef", "make_sim"]


def make_sim(n_envs: int, cfg: SimConfig | None = None, seed: int = 0,
             force_cpu: bool = False):
    """JudasSim (CUDA) si dispo, sinon JudasSimRef (CPU)."""
    import torch
    if not force_cpu and torch.cuda.is_available():
        from .judas_sim import JudasSim
        return JudasSim(n_envs, cfg, seed=seed)
    return JudasSimRef(n_envs, cfg, seed=seed)
