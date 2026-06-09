"""JudasSim — simulateur boxing CUDA vectorisé.

API identique à sim.ref_backend.JudasSimRef mais sur GPU :
des dizaines de milliers de matchs simulés en parallèle, tenseurs torch
restant sur le device (zéro copie pendant l'entraînement).
"""

import torch

from .config import ACTION_DIM, MAX_ACTION_DELAY, SimConfig
from .obs import OBS_DIM

_ext = None


def _load_extension():
    """Compile (JIT) et charge l'extension CUDA. Sous Windows, lancer depuis
    un 'x64 Native Tools Command Prompt' pour que MSVC soit dans le PATH."""
    global _ext
    if _ext is None:
        from pathlib import Path

        from torch.utils.cpp_extension import load

        csrc = Path(__file__).parent / "csrc" / "boxing_kernel.cu"
        # NB: pas de --use_fast_math : la précision double exacte est requise
        # pour l'équivalence avec sim_ref.
        _ext = load(
            name="judas_boxing",
            sources=[str(csrc)],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    return _ext


class JudasSim:
    def __init__(self, n_envs: int, cfg: SimConfig | None = None,
                 device: str = "cuda", seed: int = 0):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA indisponible. Utiliser sim.ref_backend.JudasSimRef sur CPU.")
        self.ext = _load_extension()
        self.n_envs = n_envs
        self.cfg = cfg or SimConfig()
        self.device = torch.device(device)
        self.seed = seed
        self._params = [float(v) for v in self.cfg.as_floats()]

        N = n_envs
        dev = self.device
        self._pos = torch.zeros((N, 2, 8), dtype=torch.float64, device=dev)
        self._ints = torch.zeros((N, 2, 8), dtype=torch.int32, device=dev)
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

    # ------------------------------------------------------------ inspection
    def raw_state(self) -> dict:
        """État brut (copie CPU) — debug / tests d'équivalence."""
        return {
            "pos": self._pos.cpu().numpy(),
            "ints": self._ints.cpu().numpy(),
            "tick": self._tick.cpu().numpy(),
        }
