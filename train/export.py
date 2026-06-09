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
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pol_cfg = PolicyConfig(**ckpt.get("policy_cfg", {}))
    policy = JudasPolicy(pol_cfg)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    wrapper = InferenceWrapper(policy).to(device).eval()
    example = torch.zeros(1, pol_cfg.history, pol_cfg.obs_dim, device=device)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out))

    meta = {"history": pol_cfg.history, "obs_dim": pol_cfg.obs_dim,
            "iter": ckpt.get("iter"), "source": str(ckpt_path)}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", default="models/judas.pts")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    out = export(args.ckpt, args.out, args.device)
    print(f"exporté -> {out}")


if __name__ == "__main__":
    main()
