"""Benchmark du simulateur CUDA.

    python -m sim.bench [--envs 16384] [--ticks 2000]

Cible RTX 3060 : > 1 000 000 agent-steps/s.
"""

import argparse
import time

import torch

from .config import SimConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=16384)
    ap.add_argument("--ticks", type=int, default=2000)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA indisponible.")
        return

    from .judas_sim import JudasSim

    cfg = SimConfig(randomize=True, spawn_jitter=2.0,
                    cps_min=8, cps_max=16, rot_speed_min=20, rot_speed_max=60)
    sim = JudasSim(args.envs, cfg, seed=0)
    sim.reset()
    actions = torch.rand((args.envs, 2, 7), device="cuda") * 2 - 1

    # chauffe
    for _ in range(50):
        sim.step(actions)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(args.ticks):
        sim.step(actions)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    steps = args.envs * 2 * args.ticks
    print(f"{args.envs} matchs x {args.ticks} ticks en {dt:.2f}s")
    print(f"-> {steps / dt / 1e6:.2f} M agent-steps/s "
          f"({args.envs * args.ticks / dt / 1e6:.2f} M match-ticks/s)")


if __name__ == "__main__":
    main()
