"""Export TorchScript d'un checkpoint pour l'inférence temps réel (serve/).

    python -m train.export runs/boxing/latest.pt --out models/judas_gen.pts
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from .model import JudasPolicy, PolicyConfig


class InferenceWrapper(nn.Module):
    """hist [B, H, OBS_DIM] -> action sim [B, 7] (déterministe)."""

    def __init__(self, policy: JudasPolicy):
        super().__init__()
        self.policy = policy

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        z = self.policy.trunk(hist)
        mean = self.policy.mean_head(z)
        fwd = self.policy.fwd_head(z).argmax(-1)
        strafe = self.policy.strafe_head(z).argmax(-1)
        bins = (self.policy.bin_head(z) > 0).float()
        out = torch.zeros(hist.shape[0], 7, dtype=torch.float32, device=hist.device)
        out[:, 0:2] = torch.tanh(mean)
        out[:, 2] = fwd.float() - 1.0
        out[:, 3] = strafe.float() - 1.0
        out[:, 4:7] = bins
        return out


def export(ckpt_path: str, out_path: str, device: str = "cpu") -> Path:
    ckpt, pol_cfg, policy = _load_policy(ckpt_path)

    wrapper = InferenceWrapper(policy).to(device).eval()
    example = torch.zeros(1, pol_cfg.history, pol_cfg.obs_dim, device=device)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out))

    # le contrat d'inférence (serve/live.py) : histoire, obs, et l'humanisation
    # MEDIANE du run — l'inférence doit reproduire le modèle moteur du sim
    sim = dict(ckpt.get("cfg", {}).get("sim", {}))
    meta = {"history": pol_cfg.history, "obs_dim": pol_cfg.obs_dim,
            "iter": ckpt.get("iter"), "source": str(ckpt_path),
            "max_ticks": int(_sim_value(sim, "max_ticks", 6000)),
            "target_hits": int(_sim_value(sim, "target_hits", 100)),
            "max_cps": _mid(sim, "cps_min", "cps_max", 12.0),
            "max_rot_speed": _mid(sim, "rot_speed_min", "rot_speed_max", 40.0),
            "aim_smooth": _mid(sim, "aim_smooth_min", "aim_smooth_max", 0.0)}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return out


def _load_policy(ckpt_path: str) -> tuple[dict, PolicyConfig, JudasPolicy]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pol_cfg = PolicyConfig(**ckpt.get("policy_cfg", {}))
    policy = JudasPolicy(pol_cfg)
    # strict=False : mêmes checkpoints acceptés que Trainer.load
    # (ex. pré-aux_head) — l'inférence n'utilise pas les têtes manquantes
    policy.load_state_dict(ckpt["policy"], strict=False)
    policy.eval()
    return ckpt, pol_cfg, policy


def _mid(raw: dict, lo_key: str, hi_key: str, default: float) -> float:
    try:
        lo = float(raw.get(lo_key, default))
        hi = float(raw.get(hi_key, default))
    except (TypeError, ValueError):
        return float(default)
    return (lo + hi) / 2.0


def _sim_value(raw: dict, key: str, default):
    return raw.get(key, default)


def export_orchid(ckpt_path: str, out_path: str, device: str = "cpu") -> Path:
    """Export a Judas checkpoint for Orchid's native LibTorch loader."""
    ckpt, pol_cfg, policy = _load_policy(ckpt_path)

    wrapper = InferenceWrapper(policy).to(device).eval()
    example = torch.zeros(1, pol_cfg.history, pol_cfg.obs_dim, device=device)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out))

    sim = dict(ckpt.get("cfg", {}).get("sim", {}))
    max_rot_speed = _mid(sim, "rot_speed_min", "rot_speed_max", 40.0)
    max_cps = _mid(sim, "cps_min", "cps_max", 12.0)
    # int(x + 0.5) et non round() : même arrondi que le kernel et ref_backend
    action_delay = int(_mid(sim, "delay_min", "delay_max", 0.0) + 0.5)
    meta = {
        "schema": "orchid-judas-export-v1",
        "policy_type": "judas-v1-torchscript",
        "history": pol_cfg.history,
        "obs_dim": pol_cfg.obs_dim,
        "action_dim": 7,
        "iter": ckpt.get("iter"),
        "total_steps": ckpt.get("total_steps", 0),
        "source_checkpoint": str(ckpt_path),
        "arena_size_x": float(_sim_value(sim, "arena_size_x", 18.0)),
        "arena_size_z": float(_sim_value(sim, "arena_size_z", 18.0)),
        "target_hits": int(_sim_value(sim, "target_hits", 100)),
        "max_ticks": int(_sim_value(sim, "max_ticks", 6000)),
        "max_cps": max_cps,
        "max_rot_speed": max_rot_speed,
        "action_delay": int(action_delay),
        "aim_smooth": _mid(sim, "aim_smooth_min", "aim_smooth_max", 0.0),
        "floor_y": float(_sim_value(sim, "floor_y", 0.0)),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--format", choices=("serve", "orchid"), default="serve")
    args = ap.parse_args()
    out_path = args.out
    if out_path is None:
        out_path = "models/judas.judas.orchid.pt" if args.format == "orchid" else "models/judas.pts"
    if args.format == "orchid":
        out = export_orchid(args.ckpt, out_path, args.device)
    else:
        out = export(args.ckpt, out_path, args.device)
    print(f"exporté -> {out}")


if __name__ == "__main__":
    main()
