"""Tests de fumée de l'entraînement (CPU, configurations minuscules)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from train.model import JudasPolicy, PolicyConfig, to_sim_actions  # noqa: E402
from train.run import Trainer                                       # noqa: E402

TINY = {
    "name": "_smoke",
    "n_envs": 2,
    "rollout_ticks": 8,
    "league_frac": 0.5,
    "pool_every": 1,
    "save_every": 1000,
    "eval_every": 0,
    "eval_envs": 2,
    "eval_target_hits": 2,
    "eval_max_ticks": 30,
    "sim": {"target_hits": 3, "max_ticks": 60, "randomize": False},
    "policy": {"history": 4, "d_model": 32, "n_heads": 2, "n_layers": 1},
    "ppo": {"epochs": 1, "minibatch_size": 16, "amp": False},
}


@pytest.fixture()
def tiny_trainer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return Trainer(TINY, device="cpu")


def test_policy_forward_shapes():
    pol = JudasPolicy(PolicyConfig(history=4, d_model=32, n_heads=2, n_layers=1))
    hist = torch.randn(5, 4, pol.cfg.obs_dim)
    out = pol.act(hist)
    assert out["pre"].shape == (5, 2)
    assert out["fwd"].shape == (5,)
    assert out["bins"].shape == (5, 3)
    assert torch.isfinite(out["logp"]).all()
    assert torch.isfinite(out["value"]).all()


def test_evaluate_matches_act_logp():
    """Le logp d'evaluate() doit être identique à celui d'act() (même action)."""
    pol = JudasPolicy(PolicyConfig(history=4, d_model=32, n_heads=2, n_layers=1))
    pol.eval()
    hist = torch.randn(7, 4, pol.cfg.obs_dim)
    out = pol.act(hist)
    raw = {k: out[k] for k in ("pre", "fwd", "strafe", "bins")}
    logp, entropy, value = pol.evaluate(hist, raw)
    assert torch.allclose(logp, out["logp"], atol=1e-5)
    assert torch.allclose(value, out["value"], atol=1e-5)
    assert torch.isfinite(entropy).all()


def test_to_sim_actions_ranges():
    raw = {
        "pre": torch.randn(10, 2) * 3,
        "fwd": torch.randint(0, 3, (10,)),
        "strafe": torch.randint(0, 3, (10,)),
        "bins": torch.randint(0, 2, (10, 3)).float(),
    }
    a = to_sim_actions(raw)
    assert a.shape == (10, 7)
    assert (a[:, 0:2].abs() <= 1.0).all()
    assert set(a[:, 2].unique().tolist()) <= {-1.0, 0.0, 1.0}
    assert set(a[:, 4].unique().tolist()) <= {0.0, 1.0}


def test_trainer_two_iters(tiny_trainer):
    m1 = tiny_trainer.train_iter()
    m2 = tiny_trainer.train_iter()   # itération 2 : league active (pool_every=1)
    for m in (m1, m2):
        assert np.isfinite(m["reward_mean"])
        assert np.isfinite(m["approx_kl"])
    assert m2["pool_size"] >= 1
    assert (tiny_trainer.run_dir / "metrics.jsonl").exists()


def test_save_load_roundtrip(tiny_trainer, tmp_path):
    tiny_trainer.train_iter()
    path = tiny_trainer.save()
    t2 = Trainer(TINY, device="cpu")
    t2.load(str(path))
    assert t2.iter == tiny_trainer.iter
    assert abs(t2.league.learner_elo - tiny_trainer.league.learner_elo) < 1e-9


def test_policy_mlp_mode():
    """attention=False -> trunk MLP, mêmes interfaces."""
    pol = JudasPolicy(PolicyConfig(history=4, d_model=32, n_layers=2,
                                   attention=False))
    hist = torch.randn(5, 4, pol.cfg.obs_dim)
    out = pol.act(hist)
    assert out["pre"].shape == (5, 2)
    assert torch.isfinite(out["logp"]).all()
    raw = {k: out[k] for k in ("pre", "fwd", "strafe", "bins")}
    logp, entropy, value = pol.evaluate(hist, raw)
    assert torch.isfinite(logp).all() and torch.isfinite(value).all()


def test_auto_eval_logged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    t = Trainer({**TINY, "eval_every": 1}, device="cpu")
    t.train_iter()                       # pool créé à l'itération 1
    m = t.train_iter()
    assert "eval_first" in m
    assert 0.0 <= m["eval_first"] <= 1.0


def test_metrics_have_automation_fields(tiny_trainer):
    m = tiny_trainer.train_iter()
    for k in ("hit_rate", "shaping", "warn_entropy", "total_steps"):
        assert k in m


def test_checkpoint_pruning(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    t = Trainer({**TINY, "keep_ckpts": 3}, device="cpu")
    for i in range(1, 7):
        t.iter = i
        t.save()
    remaining = sorted(t.run_dir.glob("ckpt_*.pt"))
    assert len(remaining) == 3
    assert remaining[-1].name == "ckpt_000006.pt"


def test_export_torchscript(tiny_trainer, tmp_path):
    from train.export import export
    tiny_trainer.train_iter()
    ckpt = tiny_trainer.save()
    out = export(str(ckpt), str(tmp_path / "m.pts"))
    mod = torch.jit.load(str(out))
    hist = torch.zeros(1, 4, 48)
    a = mod(hist)
    assert a.shape == (1, 7)
    assert (a[:, 0:2].abs() <= 1.0).all()
