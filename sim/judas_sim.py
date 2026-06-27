"""JudasSim â€” simulateur boxing CUDA vectorisÃ©.

API identique Ã  sim.ref_backend.JudasSimRef mais sur GPU :
des dizaines de milliers de matchs simulÃ©s en parallÃ¨le, tenseurs torch
restant sur le device (zÃ©ro copie pendant l'entraÃ®nement).

PrÃ©cision :
  - "float"  (dÃ©faut) : physique en float32 â€” vitesse maximale (le FP64 des
    GPU grand public est ~32x plus lent). C'est le mode entraÃ®nement.
  - "double" : physique en double exacte â€” utilisÃ© par sim.verify et
    tests/test_equivalence.py pour la comparaison stricte avec sim_ref.
Variable d'env JUDAS_PRECISION=double pour forcer globalement.
"""

import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path

import torch

from .config import ACTION_DIM, MAX_ACTION_DELAY, SimConfig
from .obs import OBS_DIM

_ext_cache: dict = {}
_PARAM_ABI = "p32"


def _extension_source_mtime() -> float:
    csrc = Path(__file__).parent / "csrc"
    return max(
        (csrc / "boxing_kernel.cu").stat().st_mtime,
        (csrc / "boxing_binding.cpp").stat().st_mtime,
        (csrc / "boxing_core.h").stat().st_mtime,
    )


def _load_cached_extension(name: str):
    root = os.environ.get("TORCH_EXTENSIONS_DIR")
    if not root:
        return None
    build_dir = Path(root) / name
    source_mtime = _extension_source_mtime()
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        candidate = build_dir / f"{name}{suffix}"
        if not candidate.exists():
            continue
        if candidate.stat().st_mtime < source_mtime:
            return None
        try:
            spec = importlib.util.spec_from_file_location(name, str(candidate))
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module
        except Exception:  # noqa: BLE001 - fallback to the normal JIT path
            sys.modules.pop(name, None)
            return None
    return None


def _load_extension(precision: str):
    """Compile (JIT) et charge l'extension CUDA. Sous Windows, lancer depuis
    un 'x64 Native Tools Command Prompt' pour que MSVC soit dans le PATH."""
    if os.name == "nt":
        add_paths = []
        scripts_dir = os.path.dirname(sys.executable)
        if os.path.exists(os.path.join(scripts_dir, "ninja.exe")):
            add_paths.append(scripts_dir)
        vc_tools = os.environ.get("VCToolsInstallDir")
        if vc_tools:
            cl_dir = os.path.join(vc_tools, "bin", "Hostx64", "x64")
            if os.path.exists(os.path.join(cl_dir, "cl.exe")):
                add_paths.append(cl_dir)
        current = os.environ.get("PATH") or os.environ.get("Path", "")
        parts = current.split(os.pathsep) if current else []
        prefix = [p for p in add_paths if p and p not in parts]
        merged = os.pathsep.join(prefix + parts)
        os.environ["PATH"] = merged
        os.environ["Path"] = merged
    if precision not in _ext_cache:
        name = f"judas_boxing_{precision}_{_PARAM_ABI}"
        cached = _load_cached_extension(name)
        if cached is not None:
            _ext_cache[precision] = cached
            return cached

        from torch.utils.cpp_extension import load

        # compile pour l'arch du GPU prÃ©sent (Ã©vite toutes les archs) â€”
        # dÃ©tectÃ©e dynamiquement : portable au-delÃ  de la 3060 (8.6)
        if "TORCH_CUDA_ARCH_LIST" not in os.environ and torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        csrc = Path(__file__).parent / "csrc" / "boxing_kernel.cu"
        binding = Path(__file__).parent / "csrc" / "boxing_binding.cpp"
        flags = ["-O3"]
        if os.name == "nt":
            # CUDA 12.9 rejects Visual Studio 2026's MSVC version by default.
            flags.append("-allow-unsupported-compiler")
        if precision == "double":
            flags.append("-DJUDAS_DOUBLE")
        _ext_cache[precision] = load(
            name=name,
            sources=[str(csrc), str(binding)],
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
        self._pos = torch.zeros((N, 2, 10), dtype=real_dtype, device=dev)
        self._ints = torch.zeros((N, 2, 10), dtype=torch.int32, device=dev)
        self._human = torch.zeros((N, 2, 3), dtype=torch.float32, device=dev)
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
        -> (obs [N,2,48], reward [N,2], done [N] uint8, info)

        ATTENTION : les tenseurs retournÃ©s sont les buffers internes,
        rÃ©Ã©crits in-place au step suivant (zÃ©ro copie). Cloner pour
        conserver un tick (le Trainer le fait via _sim_step)."""
        actions = actions.to(self.device, torch.float32).contiguous()
        self.ext.tick(self._pos, self._ints, self._human, self._tick,
                      self._queue, self._last, self._rng, actions, self.obs,
                      self.reward, self.done, self.winner, self._params)
        return self.obs, self.reward, self.done, {"winner": self.winner, "combo": self._ints[:, :, 8]}

    def set_reward_dist(self, v: float) -> None:
        """Shaping de distance modifiable Ã  chaud (decay automatique)."""
        self.cfg.reward_dist = float(v)
        self._params = [float(x) for x in self.cfg.as_floats()]

    def set_spawn_gap(self, v: float) -> None:
        """Curriculum : distance de spawn modifiable Ã  chaud (0 = arÃ¨ne/3)."""
        self.cfg.spawn_gap = float(v)
        self._params = [float(x) for x in self.cfg.as_floats()]

    # ------------------------------------------------------------ inspection
    def raw_state(self) -> dict:
        """Ã‰tat brut (copie CPU) â€” debug / tests d'Ã©quivalence."""
        return {
            "pos": self._pos.cpu().numpy(),
            "ints": self._ints.cpu().numpy(),
            "tick": self._tick.cpu().numpy(),
        }
