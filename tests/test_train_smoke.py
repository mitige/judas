"""Tests de fumée de l'entraînement (CPU, configurations minuscules)."""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from train.model import (                                         # noqa: E402
    JudasPolicy,
    PolicyConfig,
    _bernoulli_entropy_from_logits,
    _categorical_entropy_from_logits,
    to_sim_actions,
)
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


def test_boxing_short_run_profile_defaults():
    """Profil court RTX 3060 : ~315M steps, modèle 96x8, matchs courts."""
    from train.run import DEFAULT_CFG

    boxing = json.loads(Path("train/configs/boxing.json").read_text())
    policy = PolicyConfig()

    for cfg in (DEFAULT_CFG, boxing):
        assert cfg["total_iters"] == 300
        assert cfg["pool_every"] == 25
        assert cfg["save_every"] == 25
        assert cfg["eval_every"] == 25
        assert cfg["league_bot_frac"] == 0.25
        assert cfg["sim"]["target_hits"] == 50
        assert cfg["policy"]["history"] == 8
        assert cfg["policy"]["d_model"] == 96
        assert cfg["policy"]["n_layers"] == 2
        assert cfg["policy"]["n_heads"] == 4

    assert policy.history == 8
    assert policy.d_model == 96
    assert policy.n_layers == 2
    assert policy.n_heads == 4


def test_ppo_value_clip_disabled_by_default():
    from train.ppo import PPOConfig

    assert PPOConfig().value_clip == 0.0


def test_evaluate_matches_act_logp():
    """Le logp d'evaluate() doit être identique à celui d'act() (même action)."""
    pol = JudasPolicy(PolicyConfig(history=4, d_model=32, n_heads=2, n_layers=1))
    pol.eval()
    hist = torch.randn(7, 4, pol.cfg.obs_dim)
    out = pol.act(hist)
    raw = {k: out[k] for k in ("pre", "fwd", "strafe", "bins")}
    logp, entropy, value, aux = pol.evaluate(hist, raw)
    assert torch.allclose(logp, out["logp"], atol=1e-5)
    assert torch.allclose(value, out["value"], atol=1e-5)
    assert torch.isfinite(entropy).all()
    assert aux.shape == (7, 7)


def test_entropy_helpers_handle_saturated_half_logits():
    logits = torch.tensor([[100.0, -100.0, 0.0]], dtype=torch.float16)

    ent_cat = _categorical_entropy_from_logits(logits)
    ent_bin = _bernoulli_entropy_from_logits(logits)

    assert torch.isfinite(ent_cat).all()
    assert torch.isfinite(ent_bin).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA requis")
def test_evaluate_entropy_finite_under_amp_with_saturated_binary_logits():
    pol = JudasPolicy(PolicyConfig(history=4, d_model=32, n_heads=2,
                                   n_layers=1)).cuda()
    with torch.no_grad():
        pol.bin_head.weight.zero_()
        pol.bin_head.bias.fill_(100.0)

    hist = torch.zeros(8, 4, pol.cfg.obs_dim, device="cuda")
    out = pol.act(hist)
    raw = {k: out[k] for k in ("pre", "fwd", "strafe", "bins")}

    with torch.amp.autocast("cuda", enabled=True):
        _, entropy, _, _ = pol.evaluate(hist, raw)

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
        assert "combo_hits" in m
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
    logp, entropy, value, _aux = pol.evaluate(hist, raw)
    assert torch.isfinite(logp).all() and torch.isfinite(value).all()


def test_auto_eval_logged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    t = Trainer({**TINY, "eval_every": 1}, device="cpu")
    t.train_iter()                       # pool créé à l'itération 1
    m = t.train_iter()
    assert "eval_first" in m and "eval_bot" in m
    assert 0.0 <= m["eval_first"] <= 1.0
    assert 0.0 <= m["eval_bot"] <= 1.0


def test_spawn_gap_curriculum():
    """spawn_gap configuré -> les joueurs spawnent à 2x gap l'un de l'autre."""
    from sim import JudasSimRef, SimConfig
    env = JudasSimRef(1, SimConfig(spawn_gap=2.0, target_hits=5, max_ticks=50))
    env.reset()
    p0, p1 = env._matches[0].players
    assert abs(abs(p1.z - p0.z) - 4.0) < 1e-9
    env.set_spawn_gap(0.0)               # retour au standard (arène/3)
    env._matches[0] = env._new_match()
    p0, p1 = env._matches[0].players
    assert abs(abs(p1.z - p0.z) - 12.0) < 1e-9


def test_bot_opponents_in_league(tmp_path, monkeypatch):
    """league_bot_frac=1 -> tous les agents 1 sont contrôlés par le chase-bot."""
    monkeypatch.chdir(tmp_path)
    t = Trainer({**TINY, "league_bot_frac": 1.0}, device="cpu")
    mask = t._assign_opponents()
    assert int((~mask).sum()) == t.N        # agents 1 exclus de l'apprentissage
    m = t.train_iter()                       # tourne sans erreur avec les bots
    assert "reward_mean" in m


def test_ramp_staggered_and_adaptive(tiny_trainer):
    """Phase 1 : le spawn s'élargit, le shaping reste plein.
    Phase 2 : le shaping décroît. Effondrement du hit rate -> la rampe recule."""
    t = tiny_trainer
    t.cfg["shaping_decay_iters"] = 10
    t._ramp_on = True
    t._shaping_base = 0.002      # TINY ne configure pas le shaping (défaut 0)

    for _ in range(5):                     # combat sain -> pos 0.5
        t._update_ramp(10.0)
    assert abs(t._ramp_pos - 0.5) < 1e-9
    assert abs(t._auto_shaping() - t._shaping_base) < 1e-12   # shaping intact
    assert abs(t._auto_curriculum() - t._full_gap) < 1e-9     # spawn standard

    for _ in range(3):                     # combat sain -> shaping décroît
        t._update_ramp(10.0)
    assert t._auto_shaping() < t._shaping_base

    pos_before = t._ramp_pos
    for _ in range(4):                     # effondrement -> recul (x2 plus vite)
        t._update_ramp(0.1)
    assert t._ramp_pos < pos_before
    # le shaping est restauré en reculant sous 0.5
    while t._ramp_pos > 0.4:
        t._update_ramp(0.1)
    assert abs(t._auto_shaping() - t._shaping_base) < 1e-12


def test_chase_bot_actions():
    from train.scripted import ChaseBot
    hist = torch.zeros(3, 4, 48)
    hist[:, -1, 36] = 40.0 / 180.0       # rot speed
    hist[:, -1, 11] = 1.0                # sin(yaw_err) = 1 -> tourner a fond
    hist[:, -1, 12] = 0.0
    a = ChaseBot().act7(hist)
    assert a.shape == (3, 7)
    assert (a[:, 0] == 1.0).all()        # dyaw saturé
    assert (a[:, 2] == 1.0).all() and (a[:, 6] == 1.0).all()


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


def _zero_raw(b):
    return {"pre": torch.zeros(b, 2), "fwd": torch.zeros(b, dtype=torch.long),
            "strafe": torch.zeros(b, dtype=torch.long), "bins": torch.zeros(b, 3)}


def test_gae_respects_done_boundaries():
    """Le done au milieu du buffer coupe le bootstrap ET la propagation GAE."""
    from train.buffer import RolloutBuffer
    T, B = 5, 1
    buf = RolloutBuffer(T, B, obs_dim=2, history=1, device=torch.device("cpu"))
    for t in range(T):
        buf.add(torch.zeros(B, 2), torch.zeros(B, dtype=torch.long),
                _zero_raw(B), torch.zeros(B), torch.zeros(B),   # logp, value=0
                torch.ones(B),                                  # reward = 1
                torch.ones(B) if t == 2 else torch.zeros(B))    # done au tick 2
    buf.compute_gae(torch.zeros(B), gamma=0.9, lam=1.0)
    # values nulles -> adv = somme discountée des rewards jusqu'au done
    assert abs(buf.adv[2, 0].item() - 1.0) < 1e-6      # done : pas de suite
    assert abs(buf.adv[3, 0].item() - 1.9) < 1e-6      # 1 + 0.9 (fin de buffer)
    assert abs(buf.adv[0, 0].item() - 2.71) < 1e-6     # 1 + 0.9 * adv[1]


def test_windows_mask_pre_episode_history():
    """windows() reconstruit l'historique et masque les ticks d'AVANT le
    reset d'épisode (age) — exactement ce que voit la policy au rollout."""
    from train.buffer import RolloutBuffer
    T, B, H = 4, 1, 3
    buf = RolloutBuffer(T, B, obs_dim=1, history=H, device=torch.device("cpu"))
    hist0 = torch.tensor([[[-2.0], [-1.0], [1.0]]])    # historique avant le rollout
    buf.set_prefix(hist0)
    ages = [5, 6, 0, 1]                                 # reset au tick 2
    for t in range(T):
        buf.add(torch.full((B, 1), float(t + 1)),
                torch.tensor([ages[t]]), _zero_raw(B),
                torch.zeros(B), torch.zeros(B), torch.zeros(B), torch.zeros(B))
    # t=3 (age 1) : fenêtre brute [2, 3, 4], le 1er tick précède le reset
    win = buf.windows(torch.tensor([3]), torch.tensor([0]))
    assert win.shape == (1, H, 1)
    np.testing.assert_allclose(win[0, :, 0].numpy(), [0.0, 3.0, 4.0])
    # t=1 (age 6 >= H) : fenêtre complète [prefix[-1], obs[0], obs[1]]
    win = buf.windows(torch.tensor([1]), torch.tensor([0]))
    np.testing.assert_allclose(win[0, :, 0].numpy(), [-1.0, 1.0, 2.0])


def test_export_parity_with_policy(tiny_trainer, tmp_path):
    """Le .pts exporté doit produire EXACTEMENT l'action déterministe de la
    policy d'entraînement sur les mêmes obs — parité train <-> inférence."""
    from train.export import export
    tiny_trainer.train_iter()
    ckpt = tiny_trainer.save()
    out = export(str(ckpt), str(tmp_path / "m.pts"))
    mod = torch.jit.load(str(out)).eval()

    policy = tiny_trainer.policy.eval()
    hist = torch.randn(16, 4, 48)
    with torch.no_grad():
        a_export = mod(hist)
        act = policy.act(hist, deterministic=True)
        a_policy = to_sim_actions(
            {k: act[k] for k in ("pre", "fwd", "strafe", "bins")})
    assert torch.allclose(a_export, a_policy, atol=1e-6), \
        "export TorchScript != policy déterministe sur les mêmes obs"
