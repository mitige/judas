"""Vérification d'équivalence sim_ref <-> kernel CUDA.

    python -m sim.verify [--envs 64] [--ticks 2000]

Joue des actions pseudo-aléatoires identiques dans les deux backends et
compare état + observations à chaque tick. À lancer sur le PC RTX 3060.
"""

import argparse
import sys

import numpy as np
import torch

from .config import SimConfig
from .ref_backend import JudasSimRef

ATOL = 1e-6


def random_actions(rng, n_envs):
    a = np.zeros((n_envs, 2, 7), dtype=np.float32)
    a[:, :, 0] = rng.uniform(-1, 1, (n_envs, 2))
    a[:, :, 1] = rng.uniform(-1, 1, (n_envs, 2))
    a[:, :, 2] = rng.choice([-1.0, 0.0, 1.0], (n_envs, 2), p=[0.1, 0.2, 0.7])
    a[:, :, 3] = rng.choice([-1.0, 0.0, 1.0], (n_envs, 2))
    a[:, :, 4] = (rng.random((n_envs, 2)) < 0.15).astype(np.float32)
    a[:, :, 5] = (rng.random((n_envs, 2)) < 0.8).astype(np.float32)
    a[:, :, 6] = (rng.random((n_envs, 2)) < 0.5).astype(np.float32)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--ticks", type=int, default=2000)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA indisponible — rien à vérifier.")
        sys.exit(1)

    from .judas_sim import JudasSim

    cfg = SimConfig(randomize=False, target_hits=20, max_ticks=600)
    gpu = JudasSim(args.envs, cfg, seed=0)
    cpu = JudasSimRef(args.envs, cfg, seed=0)

    obs_g = gpu.reset().cpu().numpy()
    obs_c = cpu.reset()
    _check("reset/obs", obs_g, obs_c)

    rng = np.random.default_rng(42)
    worst = 0.0
    for t in range(args.ticks):
        a = random_actions(rng, args.envs)
        og, rg, dg, _ = gpu.step(torch.from_numpy(a))
        oc, rc, dc, _ = cpu.step(a)
        og, rg, dg = og.cpu().numpy(), rg.cpu().numpy(), dg.cpu().numpy().astype(bool)
        worst = max(worst,
                    _check(f"tick {t}/obs", og, oc),
                    _check(f"tick {t}/reward", rg, rc))
        if not np.array_equal(dg, dc):
            print(f"ECHEC tick {t}: done divergent")
            sys.exit(1)
    print(f"OK — {args.ticks} ticks x {args.envs} envs, "
          f"écart max obs/reward = {worst:.2e} (tolérance {ATOL:.0e})")


def _check(label, a, b):
    diff = float(np.max(np.abs(a - b)))
    if diff > ATOL:
        idx = np.unravel_index(np.argmax(np.abs(a - b)), a.shape)
        print(f"ECHEC {label}: écart {diff:.3e} à l'index {idx} "
              f"(gpu={a[idx]:.9f} ref={b[idx]:.9f})")
        sys.exit(1)
    return diff


if __name__ == "__main__":
    main()
