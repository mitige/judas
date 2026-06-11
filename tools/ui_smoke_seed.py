"""Create tiny local model artifacts for UI smoke tests.

The Electron apps need at least one checkpoint/exported model to exercise the
Models, Live, and Arena flows without requiring a long training run.
"""

import json
from pathlib import Path

import torch

from train.export import export
from train.model import JudasPolicy, PolicyConfig


def main() -> None:
    run_dir = Path("runs") / "ui_smoke"
    run_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    cfg = PolicyConfig(history=4, d_model=32, n_heads=2, n_layers=1)
    policy = JudasPolicy(cfg)
    ckpt = {
        "iter": 1,
        "total_steps": 32,
        "policy": policy.state_dict(),
        "policy_cfg": cfg.__dict__,
        "cfg": {
            "name": "ui_smoke",
            "policy": cfg.__dict__,
        },
    }

    ckpt_path = run_dir / "ckpt_000001.pt"
    latest_path = run_dir / "latest.pt"
    torch.save(ckpt, ckpt_path)
    torch.save(ckpt, latest_path)

    metrics = {
        "iter": 1,
        "total_steps": 32,
        "sps": 1024,
        "reward_mean": 0.125,
        "elo": 1000.0,
        "pool_size": 1,
        "league_winrate": 0.5,
        "matches": 1,
        "entropy": 1.0,
        "approx_kl": 0.0,
        "clip_frac": 0.0,
        "loss_v": 0.0,
        "time": 0.01,
    }
    (run_dir / "metrics.jsonl").write_text(json.dumps(metrics) + "\n")

    out = export(str(ckpt_path), str(models_dir / "ui-smoke.pts"))
    print(f"seeded {ckpt_path} and {out}")


if __name__ == "__main__":
    main()
