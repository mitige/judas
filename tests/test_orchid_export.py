"""Orchid export contract for Judas checkpoints."""

import json

torch = __import__("pytest").importorskip("torch")


def test_export_orchid_torchscript_contract(tmp_path):
    from train.export import export_orchid
    from train.model import JudasPolicy, PolicyConfig

    cfg = PolicyConfig(history=4, d_model=32, n_heads=2, n_layers=1)
    policy = JudasPolicy(cfg)
    ckpt_path = tmp_path / "ckpt_000001.pt"
    torch.save(
        {
            "iter": 1,
            "total_steps": 128,
            "policy": policy.state_dict(),
            "policy_cfg": cfg.__dict__,
            "cfg": {
                "sim": {
                    "arena_size_x": 18.0,
                    "arena_size_z": 18.0,
                    "target_hits": 100,
                    "max_ticks": 6000,
                    "cps_min": 8.0,
                    "cps_max": 16.0,
                    "rot_speed_min": 20.0,
                    "rot_speed_max": 60.0,
                    "delay_min": 0,
                    "delay_max": 3,
                }
            },
        },
        ckpt_path,
    )

    out_path = tmp_path / "ckpt_000001.judas.orchid.pt"
    exported = export_orchid(str(ckpt_path), str(out_path))

    assert exported == out_path
    module = torch.jit.load(str(exported))
    hist = torch.zeros(2, cfg.history, cfg.obs_dim)
    actions = module(hist)
    assert actions.shape == (2, 7)
    assert (actions[:, 0:2].abs() <= 1.0).all()
    assert set(actions[:, 2].unique().tolist()) <= {-1.0, 0.0, 1.0}
    assert set(actions[:, 4:].unique().tolist()) <= {0.0, 1.0}

    metadata = json.loads(exported.with_suffix(".json").read_text())
    assert metadata["schema"] == "orchid-judas-export-v1"
    assert metadata["policy_type"] == "judas-v1-torchscript"
    assert metadata["history"] == cfg.history
    assert metadata["obs_dim"] == cfg.obs_dim == 48
    assert metadata["action_dim"] == 7
    assert metadata["max_rot_speed"] == 40.0
    assert metadata["max_cps"] == 12.0
    assert metadata["action_delay"] == 2
    assert metadata["source_checkpoint"] == str(ckpt_path)
