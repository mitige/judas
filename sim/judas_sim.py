"""JudasSim — simulateur boxing CUDA vectorisé.

API identique à sim.ref_backend.JudasSimRef mais sur GPU :
des dizaines de milliers de matchs simulés en parallèle, tenseurs torch
restant sur le device (zéro copie pendant l'entraînement).

Précision :
  - "float"  (défaut) : physique en float32 — vitesse maximale (le FP64 des
    GPU grand public est ~32x plus lent). C'est le mode entraînement.
  - "double" : physique en double exacte — utilisé par sim.verify et
    tests/test_equivalence.py pour la comparaison stricte avec sim_ref.
Variable d'env JUDAS_PRECISION=double pour forcer globalement.
"""

import os

import torch

from .config import ACTION_DIM, MAX_ACTION_DELAY, SimConfig
from .obs import OBS_DIM

_ext_cache: dict = {}


def _load_extension(precision: str):
    """Compile (JIT) et charge l'extension CUDA. Sous Windows, lancer depuis
    un 'x64 Native Tools Command Prompt' pour que MSVC soit dans le PATH."""
    if precision not in _ext_cache:
        from pathlib import Path

        from torch.utils.cpp_extension import load

        # RTX 3060 = compute 8.6 ; évite de compiler pour toutes les archs
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
        csrc = Path(__file__).parent / "csrc" / "boxing_kernel.cu"
        flags = ["-O3"]
        if precision == "double":
            flags.append("-DJUDAS_DOUBLE")
        _ext_cache[precision] = load(
            name=f"judas_boxing_{precision}",
            sources=[str(csrc)],
            extra_cuda_cflags=flags,
            extra_cflags=["-DJUDAS_DOUBLE"] if precision == "double" else [],
            verbose=False,
        )
    return _ext_cache[precision]


class JudasSim:
    def __init__(self, n_envs: int, cfg: SimConfig | None = None,
                 device: str = "cuda", seed: int = 0,
                 precision: str | None = None):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA indisponible. Utiliser sim.ref_backend.JudasSimRef sur CPU.")
        self.precision = precision or os.environ.get("JUDAS_PRECISION", "float")
        assert self.precision in ("float", "double")
        self.ext = _load_extension(self.precision)
        self.n_envs = n_envs
        self.cfg = cfg or SimConfig()
        self.device = torch.device(device)
        self.seed = seed
        self._params = [float(v) for v in self.cfg.as_floats()]

        N = n_envs
        dev = self.device
        real_dtype = torch.float64 if self.precision == "double" else torch.float32
        self._pos = torch.zeros((N, 2, 8), dtype=real_dtype, device=dev)
        self._ints = torch.zeros((N, 2, 10), dtype=torch.int32, device=dev)
        self._human = torch.zeros((N, 2, 2), dtype=torch.float32, device=dev)
        self._tick = torch.zeros((N,), dtype=torch.int32, device=dev)
        self._queue = torch.zeros((N, 2, MAX_ACTION_DELAY, ACTION_DIM),
                                  dtype=torch.float32, device=dev)
        self._last = torch.zeros((N, 2, ACTION_DIM), dtype=torch.float32, device=dev)
        self._rng = torch.zeros((N,), dtype=torch.int64, device=dev)

        self.obs = torch.zeros((N, 2, OBS_DIM), dtype=torch.float32, device=dev)
        self.reward = torch.zeros((N, 2), dtype=torch.float32, device=dev)
        self.done = torch.zeros((N,), dtype=torch.uint8, device=dev)
        self.winner = torch.zeros((N,), dtype=torch.int32, device=dev)

    def reset(self) -> torch.Tensor:
        self.ext.reset(self._pos, self._ints, self._human, self._tick,
                       self._queue, self._last, self._rng, self.obs,
                       self._params, self.seed)
        return self.obs

    def step(self, actions: torch.Tensor):
        """actions float32 [N, 2, 7] sur le device.
        -> (obs [N,2,48], reward [N,2], done [N] uint8, info)"""
        actions = actions.to(self.device, torch.float32).contiguous()
        self.ext.tick(self._pos, self._ints, self._human, self._tick,
                      self._queue, self._last, self._rng, actions, self.obs,
                      self.reward, self.done, self.winner, self._params)
        return self.obs, self.reward, self.done, {"winner": self.winner}

    def set_reward_dist(self, v: float) -> None:
        """Shaping de distance modifiable à chaud (decay automatique)."""
        self.cfg.reward_dist = float(v)
        self._params = [float(x) for x in self.cfg.as_floats()]

    def set_spawn_gap(self, v: float) -> None:
        """Curriculum : distance de spawn modifiable à chaud (0 = arène/3)."""
        self.cfg.spawn_gap = float(v)
        self._params = [float(x) for x in self.cfg.as_floats()]

    # ------------------------------------------------------------ inspection
    def raw_state(self) -> dict:
        """État brut (copie CPU) — debug / tests d'équivalence."""
        return {
            "pos": self._pos.cpu().numpy(),
            "ints": self._ints.cpu().numpy(),
            "tick": self._tick.cpu().numpy(),
        }
