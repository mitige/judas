"""Tests de fumÃƒÆ’Ã‚Â©e de l'entraÃƒÆ’Ã‚Â®nement (CPU, configurations minuscules)."""

import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch.utils.cpp_extension import CUDA_HOME  # noqa: E402


def _has_cuda_build_toolchain() -> bool:
    if not torch.cuda.is_available() or CUDA_HOME is None:
        return False
    if os.name == "nt":
        return shutil.which("cl") is not None
    return shutil.which("c++") is not None or shutil.which("g++") is not None

from train.model import (                                         # noqa: E402
    COUNTER_CLOSE_COUNTER_REACH,
    COUNTER_CLOSE_RECOVERY_CLICK_HURT,
    COUNTER_FAR_TRADE_REACH,
    COUNTER_HIT_REACH,
    COUNTER_HIT_SELECT_CLEAN_HURT,
    COUNTER_HIT_SELECT_CLEAN_MAX_REACH,
    COUNTER_HIT_SELECT_CLEAN_MIN_REACH,
    COUNTER_HIT_SELECT_MIN_OWN_HURT,
    COUNTER_HIT_SELECT_MIN_REACH,
    COUNTER_HIT_SELECT_OPP_COOLDOWN,
    COUNTER_RECOVERY_CLICK_HURT,
    JudasPolicy,
    PolicyConfig,
    _bernoulli_entropy_from_logits,
    _categorical_entropy_from_logits,
    to_sim_actions,
)
from train.run import (                                           # noqa: E402
    Trainer,
    _behavior_reward_bonus,
    _chain_followup_stats,
    _chase_transfer_reward_bonus,
    _hit_event_reward_bonus,
    _opener_boxing_reward_bonus,
    _post_hit_wtap_reward_bonus,
    _round_train_stat,
)
from train.ppo import _coach_loss                                  # noqa: E402

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


PBT_TINY = {**TINY, "n_envs": 8,
            "pbt": {"population": 2, "interval": 1, "cross_frac": 0.25}}


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



def test_direct_movement_lock_blocks_back_strafe_and_jump():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True,
    ))
    with torch.no_grad():
        pol.bin_head.bias[0] = 10.0
    hist = torch.zeros(32, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 45] = 4.0 / 8.0
    out = pol.act(hist, deterministic=True)
    raw = {k: out[k] for k in ("pre", "fwd", "strafe", "bins")}
    sim_actions = to_sim_actions(raw)

    assert int(out["fwd"].min()) >= 1
    assert torch.equal(out["strafe"], torch.ones_like(out["strafe"]))
    assert float(sim_actions[:, 2].min()) >= 0.0
    assert torch.equal(sim_actions[:, 3], torch.zeros_like(sim_actions[:, 3]))
    assert torch.equal(sim_actions[:, 4], torch.zeros_like(sim_actions[:, 4]))
    assert torch.isfinite(out["logp"]).all()
    logp, entropy, value, _aux = pol.evaluate(hist, raw)
    assert torch.isfinite(logp).all()
    assert torch.isfinite(entropy).all()
    assert torch.isfinite(value).all()



def test_direct_movement_lock_forces_forward_when_far_even_if_idle_bias_wins():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True,
    ))
    with torch.no_grad():
        pol.fwd_head.bias[1] = 10.0
        pol.fwd_head.bias[2] = 0.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 45] = 8.0 / 8.0

    out = pol.act(hist, deterministic=True)
    sim_actions = to_sim_actions({k: out[k] for k in ("pre", "fwd", "strafe", "bins")})

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 2))
    assert torch.equal(sim_actions[:, 2], torch.ones_like(sim_actions[:, 2]))


def test_direct_movement_lock_does_not_force_far_under_combo_trade_click():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True, leaderboard_boxing=True,
    ))
    with torch.no_grad():
        pol.bin_head.bias[2] = 10.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = COUNTER_HIT_SELECT_CLEAN_HURT - 0.03
    hist[:, -1, 22] = 0.0
    hist[:, -1, 45] = ((COUNTER_HIT_REACH + COUNTER_FAR_TRADE_REACH) * 0.5) / 8.0

    out = pol.act(hist, deterministic=True)
    sim_actions = to_sim_actions({k: out[k] for k in ("pre", "fwd", "strafe", "bins")})

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 2))
    assert torch.equal(sim_actions[:, 2], torch.ones_like(sim_actions[:, 2]))
    assert torch.equal(out["bins"][:, 2], torch.zeros_like(out["bins"][:, 2]))
    assert torch.equal(sim_actions[:, 6], torch.zeros_like(sim_actions[:, 6]))


def test_direct_movement_lock_forces_reachable_under_combo_counter_click():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True,
    ))
    with torch.no_grad():
        pol.bin_head.bias[2] = -10.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = COUNTER_HIT_SELECT_CLEAN_HURT - 0.03
    hist[:, -1, 22] = 0.0
    hist[:, -1, 21] = COUNTER_RECOVERY_CLICK_HURT - 0.05
    hist[:, -1, 45] = (COUNTER_CLOSE_COUNTER_REACH - 0.10) / 8.0

    out = pol.act(hist, deterministic=True)
    sim_actions = to_sim_actions({k: out[k] for k in ("pre", "fwd", "strafe", "bins")})

    assert torch.equal(out["bins"][:, 2], torch.ones_like(out["bins"][:, 2]))
    assert torch.equal(sim_actions[:, 6], torch.ones_like(sim_actions[:, 6]))


def test_direct_movement_lock_waits_for_under_combo_recovery_timing():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True,
    ))
    with torch.no_grad():
        pol.fwd_head.bias[1] = 10.0
        pol.fwd_head.bias[2] = 0.0
        pol.bin_head.bias[2] = 10.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = COUNTER_RECOVERY_CLICK_HURT + 0.10
    hist[:, -1, 22] = 0.0
    hist[:, -1, 45] = (COUNTER_HIT_REACH - 0.05) / 8.0

    out = pol.act(hist, deterministic=True)
    sim_actions = to_sim_actions({k: out[k] for k in ("pre", "fwd", "strafe", "bins")})

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 2))
    assert torch.equal(out["bins"][:, 2], torch.zeros_like(out["bins"][:, 2]))
    assert torch.equal(sim_actions[:, 6], torch.zeros_like(sim_actions[:, 6]))


def test_direct_movement_lock_blocks_dirty_midrange_under_combo_counter_click():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True,
    ))
    with torch.no_grad():
        pol.fwd_head.bias[1] = 10.0
        pol.fwd_head.bias[2] = 0.0
        pol.bin_head.bias[2] = 10.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = COUNTER_HIT_SELECT_CLEAN_HURT - 0.03
    hist[:, -1, 22] = 0.0
    hist[:, -1, 45] = ((2.10 + COUNTER_HIT_SELECT_MIN_REACH) * 0.5) / 8.0

    out = pol.act(hist, deterministic=True)
    sim_actions = to_sim_actions({k: out[k] for k in ("pre", "fwd", "strafe", "bins")})

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 2))
    assert torch.equal(out["bins"][:, 2], torch.zeros_like(out["bins"][:, 2]))
    assert torch.equal(sim_actions[:, 6], torch.zeros_like(sim_actions[:, 6]))


def test_direct_movement_lock_lines_up_under_combo_counter_without_strafe():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True,
        leaderboard_boxing=True,
    ))
    with torch.no_grad():
        pol.bin_head.bias[2] = -10.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -1, 1] = 1.0
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = COUNTER_HIT_SELECT_CLEAN_HURT - 0.03
    hist[:, -1, 22] = 0.0
    hist[:, -1, 45] = (COUNTER_CLOSE_COUNTER_REACH - 0.10) / 8.0

    out = pol.act(hist, deterministic=True)
    sim_actions = to_sim_actions({k: out[k] for k in ("pre", "fwd", "strafe", "bins")})

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 2))
    assert torch.equal(out["strafe"], torch.full_like(out["strafe"], 1))
    assert torch.equal(sim_actions[:, 2], torch.ones_like(sim_actions[:, 2]))
    assert torch.equal(sim_actions[:, 3], torch.zeros_like(sim_actions[:, 3]))
    assert torch.equal(sim_actions[:, 6], torch.ones_like(sim_actions[:, 6]))


def test_live_direct_pad_lines_up_under_combo_counter_before_strafe_breaks():
    live = Path("serve/live.py").read_text(encoding="utf-8")

    assert "counter_lineup = (" in live
    assert "elif counter_lineup:" in live
    assert "action[3] = 0.0" in live
    assert live.index("elif counter_lineup:") < live.index("elif counter_break_strafe:")


def test_direct_movement_lock_can_leave_reachable_counter_click_to_policy():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True,
        leaderboard_boxing=True,
        direct_counter_attack_lock=False,
    ))
    with torch.no_grad():
        pol.bin_head.bias[2] = -10.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = COUNTER_RECOVERY_CLICK_HURT - 0.05
    hist[:, -1, 22] = 0.0
    hist[:, -1, 45] = (COUNTER_HIT_REACH - 0.05) / 8.0

    out = pol.act(hist, deterministic=True)
    sim_actions = to_sim_actions({k: out[k] for k in ("pre", "fwd", "strafe", "bins")})

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 2))
    assert torch.equal(out["bins"][:, 2], torch.zeros_like(out["bins"][:, 2]))
    assert torch.equal(sim_actions[:, 6], torch.zeros_like(sim_actions[:, 6]))


def test_direct_movement_lock_learns_hit_select_without_blocking_close_counter():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True,
        leaderboard_boxing=True,
        direct_counter_attack_lock=True,
        direct_hit_select_attack_lock=False,
    ))
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = COUNTER_HIT_SELECT_CLEAN_HURT - 0.03
    hist[:, -1, 22] = 0.0
    hist[:, -1, 37] = 0.10
    hist[:, -1, 45] = (COUNTER_HIT_SELECT_CLEAN_MIN_REACH + 0.10) / 8.0

    with torch.no_grad():
        pol.bin_head.bias[2] = -10.0
    out = pol.act(hist, deterministic=True)
    assert torch.equal(out["fwd"], torch.ones_like(out["fwd"]))
    assert torch.equal(out["bins"][:, 1], torch.zeros_like(out["bins"][:, 1]))
    assert torch.equal(out["bins"][:, 2], torch.zeros_like(out["bins"][:, 2]))

    with torch.no_grad():
        pol.bin_head.bias[2] = 10.0
    out = pol.act(hist, deterministic=True)
    assert torch.equal(out["bins"][:, 2], torch.ones_like(out["bins"][:, 2]))

    with torch.no_grad():
        pol.bin_head.bias[2] = -10.0
    hist[:, -1, 21] = COUNTER_RECOVERY_CLICK_HURT - 0.05
    hist[:, -1, 45] = (COUNTER_CLOSE_COUNTER_REACH - 0.10) / 8.0
    out = pol.act(hist, deterministic=True)
    assert torch.equal(out["bins"][:, 2], torch.ones_like(out["bins"][:, 2]))


def test_direct_movement_lock_soft_bias_clicks_legal_hit_select_only():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True,
        leaderboard_boxing=True,
        direct_counter_attack_lock=True,
        direct_hit_select_attack_lock=False,
        direct_hit_select_attack_bias=12.0,
    ))
    with torch.no_grad():
        pol.fwd_head.bias[1] = 10.0
        pol.fwd_head.bias[2] = 0.0
        pol.bin_head.bias[2] = -10.0

    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = COUNTER_HIT_SELECT_CLEAN_HURT - 0.03
    hist[:, -1, 22] = 0.0
    hist[:, -1, 23] = 0.0
    hist[:, -1, 37] = 0.10
    hist[:, -1, 45] = (COUNTER_HIT_SELECT_CLEAN_MIN_REACH + 0.10) / 8.0

    out = pol.act(hist, deterministic=True)
    assert torch.equal(out["fwd"], torch.ones_like(out["fwd"]))
    assert torch.equal(out["bins"][:, 1], torch.zeros_like(out["bins"][:, 1]))
    assert torch.equal(out["bins"][:, 2], torch.ones_like(out["bins"][:, 2]))

    hist[:, -1, 45] = ((COUNTER_CLOSE_COUNTER_REACH + COUNTER_HIT_SELECT_MIN_REACH) * 0.5) / 8.0
    out = pol.act(hist, deterministic=True)
    assert torch.equal(out["bins"][:, 2], torch.zeros_like(out["bins"][:, 2]))

    hist[:, -1, 45] = (COUNTER_HIT_REACH + 0.20) / 8.0
    out = pol.act(hist, deterministic=True)
    assert torch.equal(out["bins"][:, 2], torch.zeros_like(out["bins"][:, 2]))

    hist[:, -1, 21] = COUNTER_RECOVERY_CLICK_HURT - 0.05
    hist[:, -1, 45] = (COUNTER_CLOSE_COUNTER_REACH - 0.10) / 8.0
    out = pol.act(hist, deterministic=True)
    assert torch.equal(out["bins"][:, 2], torch.ones_like(out["bins"][:, 2]))


def test_direct_movement_lock_waits_for_counter_click_cooldown():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True,
    ))
    with torch.no_grad():
        pol.bin_head.bias[2] = 10.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = COUNTER_RECOVERY_CLICK_HURT - 0.05
    hist[:, -1, 22] = 0.0
    hist[:, -1, 23] = 0.10
    hist[:, -1, 45] = (COUNTER_HIT_REACH - 0.05) / 8.0

    out = pol.act(hist, deterministic=True)
    sim_actions = to_sim_actions({k: out[k] for k in ("pre", "fwd", "strafe", "bins")})

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 2))
    assert torch.equal(out["bins"][:, 2], torch.zeros_like(out["bins"][:, 2]))
    assert torch.equal(sim_actions[:, 6], torch.zeros_like(sim_actions[:, 6]))


def test_leaderboard_boxing_forces_opener_strafe_at_spawn_gap():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True, leaderboard_boxing=True,
    ))
    with torch.no_grad():
        pol.fwd_head.bias[1] = 10.0
        pol.strafe_head.bias[1] = 10.0
        pol.bin_head.bias[0] = 10.0
        pol.bin_head.bias[1] = -10.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 45] = 7.0 / 8.0
    hist[0:2, -1, 41] = -1.0
    hist[2:4, -1, 41] = 1.0

    out = pol.act(hist, deterministic=True)
    sim_actions = to_sim_actions({k: out[k] for k in ("pre", "fwd", "strafe", "bins")})

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 2))
    assert not torch.any(out["strafe"] == 1)
    assert torch.equal(sim_actions[0:2, 3], -torch.ones_like(sim_actions[0:2, 3]))
    assert torch.equal(sim_actions[2:, 3], torch.ones_like(sim_actions[2:, 3]))
    assert torch.equal(sim_actions[:, 2], torch.ones_like(sim_actions[:, 2]))
    assert torch.equal(sim_actions[:, 5], torch.ones_like(sim_actions[:, 5]))
    assert torch.equal(sim_actions[:, 3].abs(), torch.ones_like(sim_actions[:, 3]))
    assert torch.equal(sim_actions[:, 4], torch.zeros_like(sim_actions[:, 4]))


def test_leaderboard_boxing_forces_opener_strafe_from_arena_spawn_gap():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True, leaderboard_boxing=True,
    ))
    with torch.no_grad():
        pol.fwd_head.bias[1] = 10.0
        pol.strafe_head.bias[1] = 10.0
        pol.bin_head.bias[0] = 10.0
        pol.bin_head.bias[1] = -10.0
    hist = torch.zeros(4, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 45] = 8.0 / 8.0
    hist[0:2, -1, 41] = -1.0
    hist[2:4, -1, 41] = 1.0

    out = pol.act(hist, deterministic=True)
    sim_actions = to_sim_actions({k: out[k] for k in ("pre", "fwd", "strafe", "bins")})

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 2))
    assert not torch.any(out["strafe"] == 1)
    assert torch.equal(sim_actions[:, 2], torch.ones_like(sim_actions[:, 2]))
    assert torch.equal(sim_actions[:, 5], torch.ones_like(sim_actions[:, 5]))
    assert torch.equal(sim_actions[:, 3].abs(), torch.ones_like(sim_actions[:, 3]))


def test_leaderboard_boxing_holds_opener_strafe_side_near_centerline():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True, leaderboard_boxing=True,
    ))
    with torch.no_grad():
        pol.fwd_head.bias[1] = 10.0
        pol.strafe_head.bias[1] = 10.0
    hist = torch.zeros(2, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 45] = 8.0 / 8.0
    hist[:, -1, 1] = 0.20
    hist[:, -1, 41] = -1.0

    out = pol.act(hist, deterministic=True)
    sim_actions = to_sim_actions({k: out[k] for k in ("pre", "fwd", "strafe", "bins")})

    assert torch.equal(sim_actions[:, 2], torch.ones_like(sim_actions[:, 2]))
    assert torch.equal(sim_actions[:, 3], -torch.ones_like(sim_actions[:, 3]))


def test_leaderboard_coach_prefers_opener_strafe_over_straightline():
    hist = torch.zeros(2, 4, 48)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 45] = 7.0 / 8.0
    hist[0, -1, 41] = -1.0
    hist[1, -1, 41] = 1.0
    mean = torch.zeros(2, 2)
    fwd_l = torch.tensor([[-8.0, -8.0, 8.0], [-8.0, -8.0, 8.0]])
    bin_l = torch.zeros(2, 3)
    bin_l[:, 1] = 8.0
    good_strafe = torch.tensor([[8.0, -8.0, -8.0], [-8.0, -8.0, 8.0]])
    straightline = torch.tensor([[-8.0, 8.0, -8.0], [-8.0, 8.0, -8.0]])

    good = _coach_loss(mean, fwd_l, good_strafe, bin_l, hist, True)
    bad = _coach_loss(mean, fwd_l, straightline, bin_l, hist, True)

    assert good < bad


def test_leaderboard_coach_uses_age_to_force_opener_pressure():
    hist = torch.zeros(2, 4, 48)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 45] = 3.0 / 8.0
    mean = torch.zeros(2, 2)
    str_l = torch.tensor([[8.0, -8.0, -8.0], [-8.0, -8.0, 8.0]])
    bin_l = torch.zeros(2, 3)
    no_forward = torch.tensor([[-8.0, 8.0, -8.0], [-8.0, 8.0, -8.0]])
    forward = torch.tensor([[-8.0, -8.0, 8.0], [-8.0, -8.0, 8.0]])
    age = torch.tensor([4, 40])

    early_good = _coach_loss(mean, forward, str_l, bin_l, hist, True, True, age=age)
    early_bad = _coach_loss(mean, no_forward, str_l, bin_l, hist, True, True, age=age)

    assert early_good < early_bad


def test_leaderboard_coach_selective_counter_respects_click_cooldown():
    hist = torch.zeros(2, 4, 48)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = COUNTER_HIT_SELECT_CLEAN_HURT - 0.03
    hist[:, -1, 22] = 0.0
    hist[:, -1, 21] = COUNTER_RECOVERY_CLICK_HURT - 0.05
    hist[:, -1, 45] = (COUNTER_CLOSE_COUNTER_REACH - 0.10) / 8.0
    hist[1, -1, 23] = 0.10
    mean = torch.zeros(2, 2)
    fwd_l = torch.tensor([[-8.0, -8.0, 8.0], [-8.0, -8.0, 8.0]])
    str_l = torch.tensor([[-8.0, 8.0, -8.0], [-8.0, 8.0, -8.0]])
    selective = torch.tensor([[0.0, 8.0, 8.0], [0.0, 8.0, -8.0]])
    always_click = torch.tensor([[0.0, 8.0, 8.0], [0.0, 8.0, 8.0]])

    good = _coach_loss(mean, fwd_l, str_l, selective, hist, True, True)
    bad = _coach_loss(mean, fwd_l, str_l, always_click, hist, True, True)

    assert good < bad


def test_leaderboard_coach_leaves_hit_select_attack_to_policy_without_hard_lock():
    hist = torch.zeros(1, 4, 48)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = COUNTER_HIT_SELECT_CLEAN_HURT - 0.03
    hist[:, -1, 22] = 0.0
    hist[:, -1, 23] = 0.0
    hist[:, -1, 37] = 0.10
    hist[:, -1, 45] = (COUNTER_HIT_SELECT_CLEAN_MIN_REACH + 0.10) / 8.0
    mean = torch.zeros(1, 2)
    fwd_l = torch.tensor([[-8.0, -8.0, 8.0]])
    str_l = torch.tensor([[-8.0, 8.0, -8.0]])
    click = torch.tensor([[0.0, 8.0, 8.0]])
    no_click = torch.tensor([[0.0, 8.0, -8.0]])

    good = _coach_loss(mean, fwd_l, str_l, no_click, hist, True, True)
    bad = _coach_loss(mean, fwd_l, str_l, click, hist, True, True)

    assert good < bad
    assert bad > good + 0.25


def test_leaderboard_coach_keeps_counter_lineup_strafe_neutral():
    hist = torch.zeros(1, 4, 48)
    hist[:, -1, 1] = 1.0
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = COUNTER_HIT_SELECT_CLEAN_HURT - 0.03
    hist[:, -1, 22] = 0.0
    hist[:, -1, 37] = 0.10
    hist[:, -1, 45] = (COUNTER_HIT_SELECT_CLEAN_MIN_REACH + 0.10) / 8.0
    mean = torch.zeros(1, 2)
    fwd_l = torch.tensor([[-8.0, -8.0, 8.0]])
    bin_l = torch.tensor([[0.0, 8.0, 8.0]])
    neutral_strafe = torch.tensor([[-8.0, 8.0, -8.0]])
    side_strafe = torch.tensor([[-8.0, -8.0, 8.0]])

    good = _coach_loss(mean, fwd_l, neutral_strafe, bin_l, hist, True, True)
    bad = _coach_loss(mean, fwd_l, side_strafe, bin_l, hist, True, True)

    assert good < bad


def test_leaderboard_coach_penalizes_dirty_midrange_counter_click():
    hist = torch.zeros(1, 4, 48)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = COUNTER_RECOVERY_CLICK_HURT - 0.05
    hist[:, -1, 22] = 0.0
    hist[:, -1, 23] = 0.0
    hist[:, -1, 45] = ((COUNTER_CLOSE_COUNTER_REACH + COUNTER_HIT_SELECT_MIN_REACH) * 0.5) / 8.0
    mean = torch.zeros(1, 2)
    fwd_l = torch.tensor([[-8.0, -8.0, 8.0]])
    str_l = torch.tensor([[-8.0, 8.0, -8.0]])
    click = torch.tensor([[0.0, 8.0, 8.0]])
    no_click = torch.tensor([[0.0, 8.0, -8.0]])

    good = _coach_loss(mean, fwd_l, str_l, no_click, hist, True, True)
    bad = _coach_loss(mean, fwd_l, str_l, click, hist, True, True)

    assert good < bad


def test_direct_movement_lock_allows_close_combo_s_tap():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True,
    ))
    with torch.no_grad():
        pol.fwd_head.bias[0] = 10.0
        pol.fwd_head.bias[1] = 20.0
        pol.fwd_head.bias[2] = 0.0
        pol.bin_head.bias[1] = 10.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = 0.0
    hist[:, -1, 22] = 0.50
    hist[:, -1, 45] = 2.35 / 8.0
    hist[:, -1, 40] = 1.0
    hist[:, -1, 43] = 1.0

    out = pol.act(hist, deterministic=True)
    sim_actions = to_sim_actions({k: out[k] for k in ("pre", "fwd", "strafe", "bins")})

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 1))
    assert torch.equal(out["bins"][:, 1], torch.zeros_like(out["bins"][:, 1]))
    assert torch.equal(sim_actions[:, 2], torch.zeros_like(sim_actions[:, 2]))
    assert torch.equal(sim_actions[:, 5], torch.zeros_like(sim_actions[:, 5]))


def test_direct_movement_lock_forces_too_close_combo_s_tap():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True,
    ))
    with torch.no_grad():
        pol.fwd_head.bias[0] = 0.0
        pol.fwd_head.bias[2] = 10.0
        pol.bin_head.bias[1] = 10.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = 0.0
    hist[:, -1, 22] = 0.50
    hist[:, -1, 45] = 2.10 / 8.0
    hist[:, -1, 40] = 1.0
    hist[:, -1, 43] = 1.0

    out = pol.act(hist, deterministic=True)
    sim_actions = to_sim_actions({k: out[k] for k in ("pre", "fwd", "strafe", "bins")})

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 1))
    assert torch.equal(sim_actions[:, 2], torch.zeros_like(sim_actions[:, 2]))
    assert torch.equal(sim_actions[:, 5], torch.zeros_like(sim_actions[:, 5]))


def test_direct_movement_lock_presses_combo_rehit_instead_of_overtapping():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True,
    ))
    with torch.no_grad():
        pol.fwd_head.bias[0] = 10.0
        pol.fwd_head.bias[1] = 20.0
        pol.fwd_head.bias[2] = 0.0
        pol.bin_head.bias[1] = -10.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = 0.0
    hist[:, -1, 22] = 0.50
    hist[:, -1, 45] = 2.80 / 8.0
    hist[:, -1, 40] = 1.0
    hist[:, -1, 43] = 1.0

    out = pol.act(hist, deterministic=True)
    sim_actions = to_sim_actions({k: out[k] for k in ("pre", "fwd", "strafe", "bins")})

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 2))
    assert torch.equal(out["bins"][:, 1], torch.ones_like(out["bins"][:, 1]))
    assert torch.equal(sim_actions[:, 2], torch.ones_like(sim_actions[:, 2]))
    assert torch.equal(sim_actions[:, 5], torch.ones_like(sim_actions[:, 5]))


def test_direct_movement_lock_releases_s_tap_before_losing_combo_range():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True,
    ))
    with torch.no_grad():
        pol.fwd_head.bias[1] = 20.0
        pol.fwd_head.bias[2] = 0.0
        pol.bin_head.bias[1] = 10.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = 0.0
    hist[:, -1, 22] = 0.50
    hist[:, -1, 45] = 2.45 / 8.0
    hist[:, -1, 40] = -1.0
    hist[:, -1, 43] = 0.0

    out = pol.act(hist, deterministic=True)
    sim_actions = to_sim_actions({k: out[k] for k in ("pre", "fwd", "strafe", "bins")})

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 2))
    assert torch.equal(out["bins"][:, 1], torch.ones_like(out["bins"][:, 1]))
    assert torch.equal(sim_actions[:, 2], torch.ones_like(sim_actions[:, 2]))
    assert torch.equal(sim_actions[:, 5], torch.ones_like(sim_actions[:, 5]))


def test_direct_movement_lock_s_taps_combo_wait_at_reach_edge():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True,
    ))
    with torch.no_grad():
        pol.fwd_head.bias[1] = 20.0
        pol.fwd_head.bias[2] = 0.0
        pol.bin_head.bias[1] = 10.0
        pol.bin_head.bias[2] = 10.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = 0.0
    hist[:, -1, 22] = 0.70
    hist[:, -1, 45] = 3.25 / 8.0
    hist[:, -1, 40] = 1.0
    hist[:, -1, 43] = 1.0

    out = pol.act(hist, deterministic=True)
    sim_actions = to_sim_actions({k: out[k] for k in ("pre", "fwd", "strafe", "bins")})

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 1))
    assert torch.equal(out["bins"][:, 1], torch.zeros_like(out["bins"][:, 1]))
    assert torch.equal(out["bins"][:, 2], torch.zeros_like(out["bins"][:, 2]))
    assert torch.equal(sim_actions[:, 2], torch.zeros_like(sim_actions[:, 2]))
    assert torch.equal(sim_actions[:, 5], torch.zeros_like(sim_actions[:, 5]))


def test_direct_movement_lock_s_taps_after_z_release_when_combo_cooldown_is_close():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True,
    ))
    with torch.no_grad():
        pol.fwd_head.bias[1] = 20.0
        pol.fwd_head.bias[2] = 0.0
        pol.bin_head.bias[1] = 10.0
        pol.bin_head.bias[2] = 10.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = 0.0
    hist[:, -1, 22] = 0.70       # target still cannot receive re-hit
    hist[:, -1, 45] = 2.85 / 8.0 # close enough for an active chaser to steal
    hist[:, -1, 40] = 0.0        # already released W last tick
    hist[:, -1, 43] = 0.0

    out = pol.act(hist, deterministic=True)
    sim_actions = to_sim_actions({k: out[k] for k in ("pre", "fwd", "strafe", "bins")})

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 1))
    assert torch.equal(out["bins"][:, 1], torch.zeros_like(out["bins"][:, 1]))
    assert torch.equal(out["bins"][:, 2], torch.zeros_like(out["bins"][:, 2]))
    assert torch.equal(sim_actions[:, 2], torch.zeros_like(sim_actions[:, 2]))
    assert torch.equal(sim_actions[:, 5], torch.zeros_like(sim_actions[:, 5]))


def test_direct_movement_lock_uses_z_release_when_combo_cooldown_has_space():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True,
    ))
    with torch.no_grad():
        pol.fwd_head.bias[0] = 20.0
        pol.fwd_head.bias[1] = 10.0
        pol.fwd_head.bias[2] = 0.0
        pol.bin_head.bias[1] = 10.0
        pol.bin_head.bias[2] = 10.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = 0.0
    hist[:, -1, 22] = 0.70
    hist[:, -1, 45] = 3.25 / 8.0
    hist[:, -1, 40] = 0.0
    hist[:, -1, 43] = 0.0

    out = pol.act(hist, deterministic=True)
    sim_actions = to_sim_actions({k: out[k] for k in ("pre", "fwd", "strafe", "bins")})

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 1))
    assert torch.equal(out["bins"][:, 1], torch.zeros_like(out["bins"][:, 1]))
    assert torch.equal(out["bins"][:, 2], torch.zeros_like(out["bins"][:, 2]))
    assert torch.equal(sim_actions[:, 2], torch.zeros_like(sim_actions[:, 2]))
    assert torch.equal(sim_actions[:, 5], torch.zeros_like(sim_actions[:, 5]))


def test_direct_movement_lock_releases_s_tap_while_combo_cools_down():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True,
    ))
    with torch.no_grad():
        pol.fwd_head.bias[0] = 20.0
        pol.fwd_head.bias[1] = 10.0
        pol.fwd_head.bias[2] = 0.0
        pol.bin_head.bias[1] = 10.0
        pol.bin_head.bias[2] = 10.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = 0.0
    hist[:, -1, 22] = 0.70
    hist[:, -1, 45] = 3.25 / 8.0
    hist[:, -1, 40] = -1.0       # previous tick was already an S-tap
    hist[:, -1, 43] = 0.0

    out = pol.act(hist, deterministic=True)
    sim_actions = to_sim_actions({k: out[k] for k in ("pre", "fwd", "strafe", "bins")})

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 1))
    assert torch.equal(out["bins"][:, 1], torch.zeros_like(out["bins"][:, 1]))
    assert torch.equal(out["bins"][:, 2], torch.zeros_like(out["bins"][:, 2]))
    assert torch.equal(sim_actions[:, 2], torch.zeros_like(sim_actions[:, 2]))
    assert torch.equal(sim_actions[:, 5], torch.zeros_like(sim_actions[:, 5]))


def test_direct_movement_lock_holds_s_tap_while_combo_cooldown_is_too_close():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True,
    ))
    with torch.no_grad():
        pol.fwd_head.bias[0] = 0.0
        pol.fwd_head.bias[1] = 20.0
        pol.fwd_head.bias[2] = 10.0
        pol.bin_head.bias[1] = 10.0
        pol.bin_head.bias[2] = 10.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = 0.0
    hist[:, -1, 22] = 0.70
    hist[:, -1, 45] = 2.85 / 8.0
    hist[:, -1, 40] = -1.0
    hist[:, -1, 43] = 0.0

    out = pol.act(hist, deterministic=True)
    sim_actions = to_sim_actions({k: out[k] for k in ("pre", "fwd", "strafe", "bins")})

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 1))
    assert torch.equal(out["bins"][:, 1], torch.zeros_like(out["bins"][:, 1]))
    assert torch.equal(out["bins"][:, 2], torch.zeros_like(out["bins"][:, 2]))
    assert torch.equal(sim_actions[:, 2], torch.zeros_like(sim_actions[:, 2]))
    assert torch.equal(sim_actions[:, 5], torch.zeros_like(sim_actions[:, 5]))


def test_direct_movement_lock_presses_after_landed_hit_at_combo_gap():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True,
    ))
    with torch.no_grad():
        pol.fwd_head.bias[2] = 10.0
        pol.bin_head.bias[1] = 10.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -2, 31] = 0.00
    hist[:, -1, 31] = 0.01
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = 0.0
    hist[:, -1, 22] = 1.0
    hist[:, -1, 45] = 4.30 / 8.0
    hist[:, -1, 40] = 1.0
    hist[:, -1, 43] = 1.0

    out = pol.act(hist, deterministic=True)
    sim_actions = to_sim_actions({k: out[k] for k in ("pre", "fwd", "strafe", "bins")})

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 2))
    assert torch.equal(out["bins"][:, 1], torch.ones_like(out["bins"][:, 1]))
    assert torch.equal(sim_actions[:, 2], torch.ones_like(sim_actions[:, 2]))
    assert torch.equal(sim_actions[:, 5], torch.ones_like(sim_actions[:, 5]))


def test_direct_movement_lock_coasts_after_hit_while_rehit_cools_down():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True,
    ))
    with torch.no_grad():
        pol.fwd_head.bias[0] = 0.0
        pol.fwd_head.bias[1] = 0.0
        pol.fwd_head.bias[2] = 20.0
        pol.bin_head.bias[1] = 10.0
        pol.bin_head.bias[2] = 10.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -2, 31] = 0.00
    hist[:, -1, 31] = 0.01
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = 0.0
    hist[:, -1, 22] = 0.95
    hist[:, -1, 45] = 4.03 / 8.0
    hist[:, -1, 40] = 1.0
    hist[:, -1, 43] = 1.0

    out = pol.act(hist, deterministic=True)
    sim_actions = to_sim_actions({k: out[k] for k in ("pre", "fwd", "strafe", "bins")})

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 1))
    assert torch.equal(out["bins"][:, 1], torch.zeros_like(out["bins"][:, 1]))
    assert torch.equal(out["bins"][:, 2], torch.zeros_like(out["bins"][:, 2]))
    assert torch.equal(sim_actions[:, 2], torch.zeros_like(sim_actions[:, 2]))
    assert torch.equal(sim_actions[:, 5], torch.zeros_like(sim_actions[:, 5]))


def test_direct_movement_lock_represses_forward_after_combo_tap():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True,
    ))
    with torch.no_grad():
        pol.fwd_head.bias[1] = 10.0
        pol.fwd_head.bias[2] = 0.0
        pol.bin_head.bias[1] = -10.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = 0.0
    hist[:, -1, 22] = 0.50
    hist[:, -1, 45] = 2.10 / 8.0
    hist[:, -1, 40] = 0.0
    hist[:, -1, 43] = 0.0

    out = pol.act(hist, deterministic=True)
    sim_actions = to_sim_actions({k: out[k] for k in ("pre", "fwd", "strafe", "bins")})

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 2))
    assert torch.equal(out["bins"][:, 1], torch.ones_like(out["bins"][:, 1]))
    assert torch.equal(sim_actions[:, 2], torch.ones_like(sim_actions[:, 2]))
    assert torch.equal(sim_actions[:, 5], torch.ones_like(sim_actions[:, 5]))


def test_direct_movement_lock_holds_z_tap_through_ready_rehit():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True, leaderboard_boxing=True,
    ))
    with torch.no_grad():
        pol.fwd_head.bias[1] = 10.0
        pol.fwd_head.bias[2] = 0.0
        pol.bin_head.bias[1] = -10.0
        pol.bin_head.bias[2] = -10.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -1, 1] = 0.20
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = 0.0
    hist[:, -1, 22] = 0.50
    hist[:, -1, 45] = 2.85 / 8.0
    hist[:, -1, 40] = 0.0
    hist[:, -1, 41] = 1.0
    hist[:, -1, 43] = 0.0

    out = pol.act(hist, deterministic=True)
    sim_actions = to_sim_actions({k: out[k] for k in ("pre", "fwd", "strafe", "bins")})

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 1))
    assert torch.equal(out["bins"][:, 1], torch.zeros_like(out["bins"][:, 1]))
    assert torch.equal(out["bins"][:, 2], torch.ones_like(out["bins"][:, 2]))
    assert torch.equal(sim_actions[:, 2], torch.zeros_like(sim_actions[:, 2]))
    assert torch.equal(sim_actions[:, 3].abs(), torch.ones_like(sim_actions[:, 3]))
    assert torch.equal(sim_actions[:, 5], torch.zeros_like(sim_actions[:, 5]))
    assert torch.equal(sim_actions[:, 6], torch.ones_like(sim_actions[:, 6]))


def test_direct_movement_lock_forces_point_blank_s_tap_after_forward_sprint():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True,
    ))
    with torch.no_grad():
        pol.fwd_head.bias[2] = 10.0
        pol.bin_head.bias[1] = 10.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 45] = 1.20 / 8.0
    hist[:, -1, 40] = 1.0
    hist[:, -1, 43] = 1.0

    out = pol.act(hist, deterministic=True)
    sim_actions = to_sim_actions({k: out[k] for k in ("pre", "fwd", "strafe", "bins")})

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 1))
    assert torch.equal(out["bins"][:, 1], torch.zeros_like(out["bins"][:, 1]))
    assert torch.equal(out["bins"][:, 2], torch.ones_like(out["bins"][:, 2]))
    assert torch.equal(sim_actions[:, 2], torch.zeros_like(sim_actions[:, 2]))
    assert torch.equal(sim_actions[:, 5], torch.zeros_like(sim_actions[:, 5]))


def test_direct_movement_lock_represses_after_point_blank_s_tap():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True,
    ))
    with torch.no_grad():
        pol.fwd_head.bias[1] = 10.0
        pol.fwd_head.bias[2] = 0.0
        pol.bin_head.bias[1] = -10.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 45] = 1.20 / 8.0
    hist[:, -1, 40] = -1.0
    hist[:, -1, 43] = 0.0

    out = pol.act(hist, deterministic=True)
    sim_actions = to_sim_actions({k: out[k] for k in ("pre", "fwd", "strafe", "bins")})

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 2))
    assert torch.equal(out["bins"][:, 1], torch.ones_like(out["bins"][:, 1]))
    assert torch.equal(sim_actions[:, 2], torch.ones_like(sim_actions[:, 2]))
    assert torch.equal(sim_actions[:, 5], torch.ones_like(sim_actions[:, 5]))


def test_direct_movement_lock_uses_s_tap_when_combo_adv_point_blank():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True,
    ))
    with torch.no_grad():
        pol.fwd_head.bias[0] = 10.0
        pol.fwd_head.bias[2] = 0.0
        pol.bin_head.bias[1] = 10.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = 0.0
    hist[:, -1, 22] = 1.0
    hist[:, -1, 45] = 1.20 / 8.0
    hist[:, -1, 40] = 1.0
    hist[:, -1, 43] = 1.0

    out = pol.act(hist, deterministic=True)
    sim_actions = to_sim_actions({k: out[k] for k in ("pre", "fwd", "strafe", "bins")})

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 1))
    assert torch.equal(out["bins"][:, 1], torch.zeros_like(out["bins"][:, 1]))
    assert torch.equal(sim_actions[:, 2], torch.zeros_like(sim_actions[:, 2]))
    assert torch.equal(sim_actions[:, 5], torch.zeros_like(sim_actions[:, 5]))


def test_direct_movement_lock_uses_s_tap_after_landed_hit_at_combo_range():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True,
    ))
    with torch.no_grad():
        pol.fwd_head.bias[0] = 10.0
        pol.fwd_head.bias[2] = 10.0
        pol.bin_head.bias[1] = 10.0
        pol.bin_head.bias[2] = 10.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -2, 12] = 1.0
    hist[:, -1, 12] = 1.0
    hist[:, -2, 31] = 0.0
    hist[:, -1, 31] = 0.01
    hist[:, -1, 21] = 0.0
    hist[:, -1, 22] = 1.0
    hist[:, -1, 45] = 2.80 / 8.0
    hist[:, -1, 40] = 1.0
    hist[:, -1, 43] = 1.0

    out = pol.act(hist, deterministic=True)
    sim_actions = to_sim_actions({k: out[k] for k in ("pre", "fwd", "strafe", "bins")})

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 1))
    assert torch.equal(out["bins"][:, 1], torch.zeros_like(out["bins"][:, 1]))
    assert torch.equal(out["bins"][:, 2], torch.zeros_like(out["bins"][:, 2]))
    assert torch.equal(sim_actions[:, 2], torch.zeros_like(sim_actions[:, 2]))
    assert torch.equal(sim_actions[:, 5], torch.zeros_like(sim_actions[:, 5]))


def test_direct_movement_lock_s_taps_fresh_combo_cooldown_without_landed_delta():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True,
    ))
    with torch.no_grad():
        pol.fwd_head.bias[1] = 10.0
        pol.fwd_head.bias[2] = 10.0
        pol.bin_head.bias[1] = 10.0
        pol.bin_head.bias[2] = 10.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = 0.0
    hist[:, -1, 22] = 0.80
    hist[:, -1, 45] = 2.85 / 8.0
    hist[:, -1, 40] = 1.0
    hist[:, -1, 43] = 1.0

    out = pol.act(hist, deterministic=True)
    sim_actions = to_sim_actions({k: out[k] for k in ("pre", "fwd", "strafe", "bins")})

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 1))
    assert torch.equal(out["bins"][:, 1], torch.zeros_like(out["bins"][:, 1]))
    assert torch.equal(out["bins"][:, 2], torch.zeros_like(out["bins"][:, 2]))
    assert torch.equal(sim_actions[:, 2], torch.zeros_like(sim_actions[:, 2]))
    assert torch.equal(sim_actions[:, 5], torch.zeros_like(sim_actions[:, 5]))


def test_direct_movement_lock_holds_release_when_combo_rehit_is_ready():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True, leaderboard_boxing=True,
    ))
    with torch.no_grad():
        pol.fwd_head.bias[1] = 10.0
        pol.fwd_head.bias[2] = 0.0
        pol.bin_head.bias[1] = 0.0
        pol.bin_head.bias[2] = -10.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = 0.0
    hist[:, -1, 22] = 0.50
    hist[:, -1, 45] = 2.80 / 8.0
    hist[:, -1, 40] = 0.0
    hist[:, -1, 43] = 0.0

    out = pol.act(hist, deterministic=True)
    sim_actions = to_sim_actions({k: out[k] for k in ("pre", "fwd", "strafe", "bins")})

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 1))
    assert torch.equal(out["bins"][:, 1], torch.zeros_like(out["bins"][:, 1]))
    assert torch.equal(out["bins"][:, 2], torch.ones_like(out["bins"][:, 2]))
    assert torch.equal(sim_actions[:, 2], torch.zeros_like(sim_actions[:, 2]))
    assert torch.equal(sim_actions[:, 5], torch.zeros_like(sim_actions[:, 5]))


def test_direct_movement_lock_coasts_until_rehit_edge_is_in_range():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True,
    ))
    with torch.no_grad():
        pol.fwd_head.bias[1] = 0.0
        pol.fwd_head.bias[2] = 20.0
        pol.bin_head.bias[1] = 10.0
        pol.bin_head.bias[2] = 10.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = 0.0
    hist[:, -1, 22] = 0.50
    hist[:, -1, 45] = 3.80 / 8.0
    hist[:, -1, 40] = 0.0
    hist[:, -1, 43] = 0.0

    out = pol.act(hist, deterministic=True)
    sim_actions = to_sim_actions({k: out[k] for k in ("pre", "fwd", "strafe", "bins")})

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 1))
    assert torch.equal(out["bins"][:, 1], torch.zeros_like(out["bins"][:, 1]))
    assert torch.equal(out["bins"][:, 2], torch.zeros_like(out["bins"][:, 2]))
    assert torch.equal(sim_actions[:, 2], torch.zeros_like(sim_actions[:, 2]))
    assert torch.equal(sim_actions[:, 5], torch.zeros_like(sim_actions[:, 5]))


def test_direct_movement_lock_brakes_at_rehit_edge_before_chaser_can_trade():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True,
    ))
    with torch.no_grad():
        pol.fwd_head.bias[1] = 20.0
        pol.fwd_head.bias[2] = 10.0
        pol.bin_head.bias[1] = 10.0
        pol.bin_head.bias[2] = 10.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = 0.0
    hist[:, -1, 22] = 0.45
    hist[:, -1, 45] = 3.50 / 8.0
    hist[:, -1, 40] = 0.0
    hist[:, -1, 43] = 0.0

    out = pol.act(hist, deterministic=True)
    sim_actions = to_sim_actions({k: out[k] for k in ("pre", "fwd", "strafe", "bins")})

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 1))
    assert torch.equal(out["bins"][:, 1], torch.zeros_like(out["bins"][:, 1]))
    assert torch.equal(out["bins"][:, 2], torch.zeros_like(out["bins"][:, 2]))
    assert torch.equal(sim_actions[:, 2], torch.zeros_like(sim_actions[:, 2]))
    assert torch.equal(sim_actions[:, 5], torch.zeros_like(sim_actions[:, 5]))


def test_direct_movement_lock_starts_rehit_brake_before_edge_band():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True,
    ))
    with torch.no_grad():
        pol.fwd_head.bias[1] = 20.0
        pol.fwd_head.bias[2] = 10.0
        pol.bin_head.bias[1] = 10.0
        pol.bin_head.bias[2] = 10.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = 0.0
    hist[:, -1, 22] = 0.50
    hist[:, -1, 45] = 3.75 / 8.0
    hist[:, -1, 40] = 0.0
    hist[:, -1, 43] = 0.0

    out = pol.act(hist, deterministic=True)
    sim_actions = to_sim_actions({k: out[k] for k in ("pre", "fwd", "strafe", "bins")})

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 1))
    assert torch.equal(out["bins"][:, 1], torch.zeros_like(out["bins"][:, 1]))
    assert torch.equal(out["bins"][:, 2], torch.zeros_like(out["bins"][:, 2]))
    assert torch.equal(sim_actions[:, 2], torch.zeros_like(sim_actions[:, 2]))
    assert torch.equal(sim_actions[:, 5], torch.zeros_like(sim_actions[:, 5]))


def test_direct_movement_lock_brakes_just_outside_reliable_rehit_range():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True,
    ))
    with torch.no_grad():
        pol.fwd_head.bias[1] = 20.0
        pol.fwd_head.bias[2] = 10.0
        pol.bin_head.bias[1] = 10.0
        pol.bin_head.bias[2] = 10.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = 0.0
    hist[:, -1, 22] = 0.40
    hist[:, -1, 45] = 3.38 / 8.0
    hist[:, -1, 40] = -1.0
    hist[:, -1, 43] = 0.0

    out = pol.act(hist, deterministic=True)
    sim_actions = to_sim_actions({k: out[k] for k in ("pre", "fwd", "strafe", "bins")})

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 1))
    assert torch.equal(out["bins"][:, 1], torch.zeros_like(out["bins"][:, 1]))
    assert torch.equal(out["bins"][:, 2], torch.zeros_like(out["bins"][:, 2]))
    assert torch.equal(sim_actions[:, 2], torch.zeros_like(sim_actions[:, 2]))
    assert torch.equal(sim_actions[:, 5], torch.zeros_like(sim_actions[:, 5]))


def test_direct_movement_lock_edge_pokes_with_s_brake_at_rehit_limit():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True,
    ))
    with torch.no_grad():
        pol.fwd_head.bias[1] = 20.0
        pol.fwd_head.bias[2] = 10.0
        pol.bin_head.bias[1] = 10.0
        pol.bin_head.bias[2] = -10.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = 0.0
    hist[:, -1, 22] = 0.40
    hist[:, -1, 45] = 3.36 / 8.0
    hist[:, -1, 40] = -1.0
    hist[:, -1, 43] = 0.0

    out = pol.act(hist, deterministic=True)
    sim_actions = to_sim_actions({k: out[k] for k in ("pre", "fwd", "strafe", "bins")})

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 1))
    assert torch.equal(out["bins"][:, 1], torch.zeros_like(out["bins"][:, 1]))
    assert torch.equal(out["bins"][:, 2], torch.ones_like(out["bins"][:, 2]))
    assert torch.equal(sim_actions[:, 2], torch.zeros_like(sim_actions[:, 2]))
    assert torch.equal(sim_actions[:, 5], torch.zeros_like(sim_actions[:, 5]))
    assert torch.equal(sim_actions[:, 6], torch.ones_like(sim_actions[:, 6]))


def test_direct_movement_lock_attacks_neutral_hit_window_first():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True,
    ))
    with torch.no_grad():
        pol.fwd_head.bias[1] = 10.0
        pol.fwd_head.bias[2] = 0.0
        pol.bin_head.bias[2] = -10.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 45] = 3.00 / 8.0

    out = pol.act(hist, deterministic=True)
    sim_actions = to_sim_actions({k: out[k] for k in ("pre", "fwd", "strafe", "bins")})

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 2))
    assert torch.equal(out["bins"][:, 1], torch.ones_like(out["bins"][:, 1]))
    assert torch.equal(out["bins"][:, 2], torch.ones_like(out["bins"][:, 2]))
    assert torch.equal(sim_actions[:, 6], torch.ones_like(sim_actions[:, 6]))


def test_direct_movement_lock_attacks_combo_rehit_at_aabb_reach_edge():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True,
    ))
    with torch.no_grad():
        pol.fwd_head.bias[1] = 10.0
        pol.fwd_head.bias[2] = 0.0
        pol.bin_head.bias[2] = -10.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = 0.0
    hist[:, -1, 22] = 0.50
    hist[:, -1, 45] = 3.34 / 8.0

    out = pol.act(hist, deterministic=True)

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 2))
    assert torch.equal(out["bins"][:, 1], torch.ones_like(out["bins"][:, 1]))
    assert torch.equal(out["bins"][:, 2], torch.ones_like(out["bins"][:, 2]))


def test_direct_movement_lock_attacks_one_tick_before_hurt_threshold():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True,
    ))
    with torch.no_grad():
        pol.fwd_head.bias[1] = 10.0
        pol.fwd_head.bias[2] = 0.0
        pol.bin_head.bias[1] = 0.0
        pol.bin_head.bias[2] = -10.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = 0.0
    hist[:, -1, 22] = 11.0 / 20.0
    hist[:, -1, 45] = 2.80 / 8.0
    hist[:, -1, 40] = 0.0
    hist[:, -1, 43] = 0.0

    out = pol.act(hist, deterministic=True)
    sim_actions = to_sim_actions({k: out[k] for k in ("pre", "fwd", "strafe", "bins")})

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 2))
    assert torch.equal(out["bins"][:, 1], torch.ones_like(out["bins"][:, 1]))
    assert torch.equal(out["bins"][:, 2], torch.ones_like(out["bins"][:, 2]))
    assert torch.equal(sim_actions[:, 2], torch.ones_like(sim_actions[:, 2]))


def test_direct_movement_lock_coasts_after_combo_knockback_gap():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True,
    ))
    with torch.no_grad():
        pol.fwd_head.bias[1] = 10.0
        pol.fwd_head.bias[2] = 0.0
        pol.bin_head.bias[1] = 0.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = 0.0
    hist[:, -1, 22] = 1.0
    hist[:, -1, 45] = 3.60 / 8.0
    hist[:, -1, 40] = 1.0
    hist[:, -1, 43] = 1.0

    out = pol.act(hist, deterministic=True)
    sim_actions = to_sim_actions({k: out[k] for k in ("pre", "fwd", "strafe", "bins")})

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 1))
    assert torch.equal(out["bins"][:, 1], torch.zeros_like(out["bins"][:, 1]))
    assert torch.equal(out["bins"][:, 2], torch.zeros_like(out["bins"][:, 2]))
    assert torch.equal(sim_actions[:, 2], torch.zeros_like(sim_actions[:, 2]))
    assert torch.equal(sim_actions[:, 5], torch.zeros_like(sim_actions[:, 5]))


def test_direct_movement_lock_forces_close_under_combo_counter_pressure():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True,
    ))
    with torch.no_grad():
        pol.fwd_head.bias[1] = 10.0
        pol.fwd_head.bias[2] = 0.0
        pol.bin_head.bias[2] = -10.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = COUNTER_RECOVERY_CLICK_HURT - 0.05
    hist[:, -1, 22] = 0.0
    hist[:, -1, 45] = 2.70 / 8.0

    out = pol.act(hist, deterministic=True)
    sim_actions = to_sim_actions({k: out[k] for k in ("pre", "fwd", "strafe", "bins")})

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 2))
    assert torch.equal(out["bins"][:, 1], torch.ones_like(out["bins"][:, 1]))
    assert torch.equal(out["bins"][:, 2], torch.ones_like(out["bins"][:, 2]))
    assert torch.equal(sim_actions[:, 2], torch.ones_like(sim_actions[:, 2]))
    assert torch.equal(sim_actions[:, 5], torch.ones_like(sim_actions[:, 5]))


def test_direct_movement_lock_forces_early_reach_under_combo_counter():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True,
    ))
    with torch.no_grad():
        pol.fwd_head.bias[1] = 10.0
        pol.fwd_head.bias[2] = 0.0
        pol.bin_head.bias[2] = -10.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = COUNTER_RECOVERY_CLICK_HURT - 0.05
    hist[:, -1, 22] = 0.0
    hist[:, -1, 45] = (COUNTER_HIT_SELECT_MIN_REACH + 0.05) / 8.0

    out = pol.act(hist, deterministic=True)
    sim_actions = to_sim_actions({k: out[k] for k in ("pre", "fwd", "strafe", "bins")})

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 2))
    assert torch.equal(out["bins"][:, 1], torch.ones_like(out["bins"][:, 1]))
    assert torch.equal(out["bins"][:, 2], torch.ones_like(out["bins"][:, 2]))
    assert torch.equal(sim_actions[:, 2], torch.ones_like(sim_actions[:, 2]))
    assert torch.equal(sim_actions[:, 5], torch.ones_like(sim_actions[:, 5]))


def test_direct_movement_lock_forces_reach_under_combo_counter_pressure():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True,
    ))
    with torch.no_grad():
        pol.fwd_head.bias[1] = 10.0
        pol.fwd_head.bias[2] = 0.0
        pol.bin_head.bias[2] = -10.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = COUNTER_RECOVERY_CLICK_HURT - 0.05
    hist[:, -1, 22] = 0.0
    hist[:, -1, 45] = 3.10 / 8.0

    out = pol.act(hist, deterministic=True)

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 2))
    assert torch.equal(out["bins"][:, 1], torch.ones_like(out["bins"][:, 1]))
    assert torch.equal(out["bins"][:, 2], torch.ones_like(out["bins"][:, 2]))


def test_direct_movement_lock_counters_under_combo_at_reach_edge():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True,
    ))
    with torch.no_grad():
        pol.fwd_head.bias[1] = 10.0
        pol.fwd_head.bias[2] = 0.0
        pol.bin_head.bias[2] = -10.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = COUNTER_RECOVERY_CLICK_HURT - 0.05
    hist[:, -1, 22] = 0.0
    hist[:, -1, 45] = 3.22 / 8.0

    out = pol.act(hist, deterministic=True)

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 2))
    assert torch.equal(out["bins"][:, 1], torch.ones_like(out["bins"][:, 1]))
    assert torch.equal(out["bins"][:, 2], torch.ones_like(out["bins"][:, 2]))


def test_direct_movement_lock_counters_under_combo_at_aabb_reach_edge():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True,
    ))
    with torch.no_grad():
        pol.fwd_head.bias[1] = 10.0
        pol.fwd_head.bias[2] = 0.0
        pol.bin_head.bias[2] = -10.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = COUNTER_RECOVERY_CLICK_HURT - 0.05
    hist[:, -1, 22] = 0.0
    hist[:, -1, 45] = 3.38 / 8.0

    out = pol.act(hist, deterministic=True)

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 2))
    assert torch.equal(out["bins"][:, 1], torch.ones_like(out["bins"][:, 1]))
    assert torch.equal(out["bins"][:, 2], torch.ones_like(out["bins"][:, 2]))


def test_direct_movement_lock_reenters_without_far_under_combo_trade_click():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True,
    ))
    with torch.no_grad():
        pol.fwd_head.bias[1] = 10.0
        pol.fwd_head.bias[2] = 0.0
        pol.bin_head.bias[2] = -10.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = COUNTER_RECOVERY_CLICK_HURT - 0.05
    hist[:, -1, 22] = 0.0
    hist[:, -1, 45] = 3.52 / 8.0

    out = pol.act(hist, deterministic=True)

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 2))
    assert torch.equal(out["bins"][:, 1], torch.ones_like(out["bins"][:, 1]))
    assert torch.equal(out["bins"][:, 2], torch.zeros_like(out["bins"][:, 2]))


def test_direct_movement_lock_z_tap_attacks_ready_combo_rehit():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True,
    ))
    with torch.no_grad():
        pol.fwd_head.bias[1] = 0.0
        pol.fwd_head.bias[2] = 10.0
        pol.bin_head.bias[2] = -10.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = 0.0
    hist[:, -1, 22] = 0.40
    hist[:, -1, 40] = 1.0
    hist[:, -1, 43] = 1.0
    hist[:, -1, 45] = 3.25 / 8.0

    out = pol.act(hist, deterministic=True)
    sim_actions = to_sim_actions({k: out[k] for k in ("pre", "fwd", "strafe", "bins")})

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 1))
    assert torch.equal(out["bins"][:, 1], torch.zeros_like(out["bins"][:, 1]))
    assert torch.equal(out["bins"][:, 2], torch.ones_like(out["bins"][:, 2]))
    assert torch.equal(sim_actions[:, 2], torch.zeros_like(sim_actions[:, 2]))
    assert torch.equal(sim_actions[:, 5], torch.zeros_like(sim_actions[:, 5]))
    assert torch.equal(sim_actions[:, 6], torch.ones_like(sim_actions[:, 6]))


def test_direct_movement_lock_blocks_far_under_combo_trade_click():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        direct_movement_lock=True,
    ))
    with torch.no_grad():
        pol.fwd_head.bias[1] = 10.0
        pol.fwd_head.bias[2] = 0.0
        pol.bin_head.bias[2] = 10.0
    hist = torch.zeros(8, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = COUNTER_RECOVERY_CLICK_HURT - 0.05
    hist[:, -1, 22] = 0.0
    hist[:, -1, 45] = 3.60 / 8.0

    out = pol.act(hist, deterministic=True)
    sim_actions = to_sim_actions({k: out[k] for k in ("pre", "fwd", "strafe", "bins")})

    assert torch.equal(out["fwd"], torch.full_like(out["fwd"], 2))
    assert torch.equal(out["bins"][:, 1], torch.ones_like(out["bins"][:, 1]))
    assert torch.equal(out["bins"][:, 2], torch.zeros_like(out["bins"][:, 2]))
    assert torch.equal(sim_actions[:, 2], torch.ones_like(sim_actions[:, 2]))
    assert torch.equal(sim_actions[:, 5], torch.ones_like(sim_actions[:, 5]))


def test_under_combo_attack_lock_masks_attack_only_when_behind():
    pol = JudasPolicy(PolicyConfig(
        history=4, d_model=32, n_heads=2, n_layers=1,
        under_combo_attack_lock=True,
    ))
    with torch.no_grad():
        pol.bin_head.bias[2] = 10.0
    hist = torch.zeros(2, 4, pol.cfg.obs_dim)
    hist[:, -1, 12] = 1.0
    hist[0, -1, 21] = 1.0
    hist[0, -1, 22] = 0.0
    hist[1, -1, 21] = 0.0
    hist[1, -1, 22] = 1.0

    out = pol.act(hist, deterministic=True)

    assert float(out["bins"][0, 2]) == 0.0
    assert float(out["bins"][1, 2]) == 1.0
    assert torch.isfinite(out["logp"]).all()

def test_boxing_short_run_profile_defaults():
    """Profil god RTX 3060 : transformer d96 (SEULE architecture ayant
    prouvÃƒÆ’Ã‚Â© le niveau god ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â eval_bot 0.76, sprint_hits 0.77 ; le MLP plafonne
    ÃƒÆ’Ã‚Â  0.00 vs bot malgrÃƒÆ’Ã‚Â© 6.8B steps), 8192 envs, PBT-4, epochs 2."""
    from train.run import DEFAULT_CFG

    boxing = json.loads(Path("train/configs/boxing.json").read_text())
    policy = PolicyConfig()

    for cfg in (DEFAULT_CFG, boxing):
        assert cfg["pool_every"] == 25
        assert cfg["save_every"] == 25
        assert cfg["eval_every"] == 25
        assert cfg["league_bot_frac"] == 0.25
        assert cfg["sim"]["target_hits"] == 50
        assert cfg["policy"]["history"] == 8
        assert cfg["policy"]["n_layers"] == 2

    # profil de production (boxing.json) : transformer god + ÃƒÆ’Ã‚Â©chelle 3060
    assert boxing["n_envs"] == 8192
    assert boxing["policy"]["d_model"] == 96
    assert boxing["policy"]["attention"] is True
    assert boxing["ppo"]["epochs"] == 2
    assert boxing["ppo"]["minibatch_size"] == 32768
    assert boxing["pbt"]["population"] == 4
    assert boxing["sim"]["arena_size_x"] == 18.0
    assert boxing["sim"]["reward_sprint_hit"] == 0.35
    assert boxing["sim"]["reward_trade_penalty"] == 0.4
    assert DEFAULT_CFG["safety_min_strafe_frac"] >= 0.50
    assert DEFAULT_CFG["safety_min_opener_strafe_frac"] >= 0.75
    assert DEFAULT_CFG["safety_min_opener_strafe_hold_frac"] < 0.0
    assert DEFAULT_CFG["safety_opener_ticks"] == 20
    assert DEFAULT_CFG["safety_min_opener_pressure_frac"] < 0.0
    assert DEFAULT_CFG["safety_min_combo_tap_frac"] < 0.0
    assert DEFAULT_CFG["safety_min_combo_z_tap_frac"] < 0.0
    assert DEFAULT_CFG["safety_max_combo_s_tap_frac"] < 0.0
    assert DEFAULT_CFG["safety_min_hit_wtap_frac"] < 0.0
    assert DEFAULT_CFG["safety_min_under_combo_counter_hit_frac"] < 0.0

    # dÃƒÆ’Ã‚Â©fauts du code (compat checkpoints transformer)
    assert policy.history == 8
    assert policy.d_model == 96
    assert policy.n_layers == 2
    assert policy.n_heads == 4
    assert policy.attention is True


def test_combo_god_attn96_combo12_profile_defaults():
    cfg = json.loads(Path("train/configs/combo_god_attn96_combo12.json").read_text())

    assert cfg["name"] == "combo_god_countertap96_combo12"
    assert cfg["resume_as_seed"] is True
    assert cfg["policy"] == {
        "history": 8,
        "d_model": 96,
        "n_heads": 4,
        "n_layers": 2,
        "attention": True,
        "aim_residual": 1.15,
        "direct_movement_lock": True,
        "under_combo_attack_lock": False,
    }
    assert cfg["ppo"]["epochs"] == 1
    assert cfg["ppo"]["minibatch_size"] == 16384
    assert cfg["ppo"]["ent_coef"] >= 0.008
    assert cfg["ppo"]["coach_coef"] >= 0.22
    assert cfg["ppo"]["coach_until"] >= 0.75
    assert cfg["sim"]["arena_size_x"] == 40.0
    assert cfg["sim"]["arena_size_z"] == 40.0
    assert cfg["sim"]["post_sprint_hit_stop"] is True
    assert cfg["sim"]["cps_min"] == 10.0
    assert cfg["sim"]["cps_max"] == 10.0
    assert cfg["league_bot_frac"] == 0.0
    assert cfg["league_spar_bot_frac"] == 0.15
    assert cfg["league_rehit_bot_frac"] == 0.15
    assert cfg["league_pressure_bot_frac"] == 0.30
    assert cfg["league_combo_chase_bot_frac"] == 0.35
    assert cfg["league_pad_bot_frac"] == 0.05
    assert cfg["keep_ckpts"] >= 80
    assert abs((
        cfg["league_bot_frac"]
        + cfg["league_spar_bot_frac"]
        + cfg["league_rehit_bot_frac"]
        + cfg["league_pressure_bot_frac"]
        + cfg["league_combo_chase_bot_frac"]
        + cfg["league_pad_bot_frac"]
    ) - 1.0) < 1e-9
    assert cfg["safety_stop_on_regression"] is True
    assert cfg["safety_restore_on_low_combo"] is False
    assert cfg["fresh_optimizer_on_resume"] is True
    assert cfg["combo_eval_every"] == 8
    assert cfg["combo_eval_envs"] == 128
    assert cfg["combo_eval_ticks"] == 1200
    assert cfg["combo_eval_chase"] is True
    assert cfg["combo_eval_chase_envs"] == 32
    assert cfg["combo_eval_chase_ticks"] == 450
    assert cfg["combo_eval_spar"] is True
    assert cfg["combo_eval_spar_envs"] == 64
    assert cfg["combo_eval_spar_ticks"] == 900
    assert cfg["combo_eval_rehit"] is True
    assert cfg["combo_eval_rehit_envs"] == 64
    assert cfg["combo_eval_rehit_ticks"] == 900
    assert cfg["combo_eval_pressure"] is True
    assert cfg["combo_eval_pressure_envs"] == 64
    assert cfg["combo_eval_pressure_ticks"] == 900
    assert cfg["safety_require_chase_combo"] is True
    assert cfg["safety_under_combo_escape"] <= 0.02
    assert cfg["safety_back_frac"] <= 0.002
    assert cfg["safety_min_strafe_frac"] >= 0.50
    assert cfg["safety_min_opener_strafe_frac"] >= 0.75
    assert cfg["safety_min_opener_strafe_hold_frac"] >= 0.65
    assert cfg["safety_opener_ticks"] == 20
    assert cfg["safety_min_opener_pressure_frac"] >= 0.40
    assert cfg["safety_min_combo_tap_frac"] >= 0.12
    assert cfg["safety_min_combo_z_tap_frac"] >= 0.10
    assert cfg["safety_max_combo_s_tap_frac"] <= 0.02
    assert cfg["safety_min_hit_wtap_frac"] >= 0.40
    assert cfg["safety_min_under_combo_counter_hit_frac"] >= 0.05
    assert cfg["safety_strafe_frac"] >= 0.9
    assert cfg["safety_min_hit_rate"] >= 40.0
    assert cfg["safety_fresh_min_hit_rate"] == 10.0
    assert cfg["safety_promote_min_combo_max"] == 8.0
    assert cfg["sim"]["reward_trade_penalty"] >= 3.4
    assert cfg["sim"]["reward_combo"] >= 3.0
    assert cfg["sim"]["combo_cap"] >= 28
    assert cfg["sim"]["reward_aim"] > 0.0
    assert cfg["sim"]["reward_bad_pitch"] > 0.0
    assert cfg["sim"]["reward_chase"] > 0.0
    assert cfg["sim"]["reward_turn_aim"] > 0.0
    assert cfg["sim"]["reward_aggression"] > 0.0
    assert cfg["sim"]["reward_no_escape"] > 0.0
    assert cfg["sim"]["reward_combo_focus"] >= 1.6
    assert cfg["sim"]["reward_combo_tap"] >= 4.6
    assert cfg["sim"]["reward_opener_strafe"] >= 1.0
    assert cfg["sim"]["reward_hit_wtap"] >= 3.4
    assert cfg["sim"]["reward_counter_hit"] >= 5.0
    assert cfg["sim"]["reward_chase_rechain"] >= 1.3
    assert cfg["sim"]["reward_chase_counter"] >= 1.15
    assert cfg["shaping_sky_frac"] <= 0.3
    assert cfg["shaping_combo12_state"] >= 0.45
    assert cfg["pbt"]["population"] == 1


def test_combo_god_leaderboard10_combo12_profile_defaults():
    cfg = json.loads(Path("train/configs/combo_god_leaderboard10_combo12.json").read_text())

    assert cfg["name"] == "combo_god_leaderboard10_combo12"
    assert cfg["resume_as_seed"] is True
    assert cfg["resume_same_run_as_seed"] is True
    assert cfg["policy"]["history"] == 8
    assert cfg["policy"]["d_model"] == 96
    assert cfg["policy"]["n_heads"] == 4
    assert cfg["policy"]["n_layers"] == 2
    assert cfg["policy"]["attention"] is True
    assert cfg["policy"]["direct_movement_lock"] is True
    assert cfg["policy"]["leaderboard_boxing"] is True
    assert cfg["policy"]["direct_counter_attack_lock"] is True
    assert cfg["policy"]["direct_hit_select_attack_lock"] is True
    assert cfg["policy"]["direct_hit_select_attack_bias"] == pytest.approx(18.0)
    assert cfg["policy"]["under_combo_attack_lock"] is False
    assert cfg["sim"]["arena_size_x"] == 40.0
    assert cfg["sim"]["arena_size_z"] == 40.0
    assert cfg["sim"]["post_sprint_hit_stop"] is True
    assert cfg["sim"]["cps_min"] == 10.0
    assert cfg["sim"]["cps_max"] == 10.0
    assert cfg["league_spar_bot_frac"] == pytest.approx(0.22)
    assert cfg["league_rehit_bot_frac"] == pytest.approx(0.10)
    assert cfg["league_pressure_bot_frac"] == pytest.approx(0.06)
    assert cfg["league_combo_chase_bot_frac"] == pytest.approx(0.32)
    assert cfg["league_counter_bot_frac"] == pytest.approx(0.30)
    assert cfg["league_pad_bot_frac"] == pytest.approx(0.0)
    scripted_frac = sum(cfg[k] for k in (
        "league_spar_bot_frac",
        "league_rehit_bot_frac",
        "league_pressure_bot_frac",
        "league_combo_chase_bot_frac",
        "league_counter_bot_frac",
        "league_pad_bot_frac",
    ))
    assert scripted_frac == pytest.approx(1.0)
    assert cfg["ppo"]["lr"] == pytest.approx(0.0000015)
    assert cfg["ppo"]["coach_coef"] == pytest.approx(0.36)
    assert cfg["ppo"]["clip"] == pytest.approx(0.04)
    assert cfg["ppo"]["sample_frac"] == pytest.approx(0.28)
    assert cfg["safety_stop_on_regression"] is True
    assert cfg["safety_min_strafe_frac"] >= 0.50
    assert cfg["safety_min_opener_strafe_frac"] == pytest.approx(0.72)
    assert cfg["safety_min_opener_strafe_hold_frac"] >= 0.65
    assert cfg["safety_opener_ticks"] == 20
    assert cfg["safety_min_opener_pressure_frac"] >= 0.40
    assert cfg["safety_min_combo_tap_frac"] >= 0.12
    assert cfg["safety_min_combo_z_tap_frac"] >= 0.10
    assert cfg["safety_max_combo_s_tap_frac"] <= 0.02
    assert cfg["safety_min_hit_wtap_frac"] == pytest.approx(0.035)
    assert cfg["safety_min_chase_hit_wtap_frac"] == pytest.approx(0.035)
    assert cfg["safety_rollout_hit_wtap_slack"] == pytest.approx(0.32)
    assert cfg["safety_hit_wtap_blocks_promotion"] is True
    assert cfg["safety_min_under_combo_counter_hit_frac"] == pytest.approx(0.085)
    assert cfg["safety_under_combo_avoid_frac"] == pytest.approx(0.055)
    assert cfg["safety_under_combo_avoid_min_combo12"] == pytest.approx(0.08)
    assert cfg["safety_under_combo_avoid_min_hit_rate"] == pytest.approx(80)
    assert cfg["score_under_combo_avoid_target"] == pytest.approx(0.14)
    assert cfg["score_under_combo_avoid_weight"] == pytest.approx(0.30)
    assert cfg["score_under_combo_avoid_cap"] == pytest.approx(0.015)
    assert cfg["safety_min_under_combo_hit_select_clean_frac"] == pytest.approx(0.20)
    assert cfg["safety_max_under_combo_hit_select_trade_frac"] == pytest.approx(0.12)
    assert cfg["safety_require_counter_recovery"] is True
    assert cfg["sim"]["reward_opener_strafe"] >= 2.8
    assert cfg["sim"]["reward_trade_penalty"] >= 6.2
    assert cfg["sim"]["reward_counter_hit"] >= 26.0
    assert cfg["sim"]["reward_combo"] >= 12.2
    assert cfg["sim"]["reward_combo_focus"] >= 3.8
    assert cfg["sim"]["reward_combo_tap"] >= 6.4
    assert cfg["sim"]["reward_hit_wtap"] >= 15.6
    assert cfg["sim"]["reward_hit_select"] >= 28.0
    assert cfg["sim"]["reward_combo_pressure"] >= 3.6
    assert cfg["sim"]["reward_chase_rechain"] >= 9.2
    assert cfg["sim"]["reward_chase_hit_select"] == pytest.approx(22.0)
    assert cfg["sim"]["reward_chase_close_counter"] == pytest.approx(14.0)
    assert cfg["sim"]["reward_chase_counter"] == pytest.approx(30.0)
    assert cfg["sim"]["reward_spar_counter"] == pytest.approx(26.0)
    assert cfg["combo_eval_chase_envs"] >= 64
    assert cfg["combo_eval_chase_ticks"] >= 900
    assert cfg["combo_eval_counter"] is True
    assert cfg["combo_eval_counter_envs"] >= 64
    assert cfg["combo_eval_counter_ticks"] >= 900
    assert cfg["combo_eval_every"] == 24
    assert cfg["fresh_optimizer_on_resume"] is True
    assert cfg["pbt"]["population"] == 1
    assert "sim.reward_chase_hit_select" in cfg["pbt"]["mutate"]
    assert "sim.reward_chase_close_counter" in cfg["pbt"]["mutate"]
    assert "sim.reward_spar_counter" in cfg["pbt"]["mutate"]


def test_combo_god_recovery_kb092_combo12_profile_defaults():
    cfg = json.loads(Path("train/configs/combo_god_recovery_kb092_combo12.json").read_text())
    leaderboard = json.loads(Path("train/configs/combo_god_leaderboard10_combo12.json").read_text())

    assert cfg["name"] == "combo_god_recovery_kb092_combo12"
    assert cfg["resume_as_seed"] is True
    assert cfg["resume_same_run_as_seed"] is False
    assert cfg["fresh_optimizer_on_resume"] is True
    assert cfg["policy"]["history"] == 8
    assert cfg["policy"]["d_model"] == 96
    assert cfg["policy"]["n_heads"] == 4
    assert cfg["policy"]["n_layers"] == 2
    assert cfg["policy"]["attention"] is True
    assert cfg["policy"]["direct_movement_lock"] is True
    assert cfg["policy"]["leaderboard_boxing"] is True
    assert cfg["policy"]["direct_counter_attack_lock"] is True
    assert cfg["policy"]["direct_hit_select_attack_lock"] is True
    assert cfg["policy"]["direct_hit_select_attack_bias"] >= 20.0
    assert cfg["policy"]["under_combo_attack_lock"] is False
    assert cfg["sim"]["kb_h_mult"] == pytest.approx(0.92)
    assert cfg["sim"]["kb_v_mult"] == pytest.approx(0.90)
    assert cfg["sim"]["kb_idle_mult"] == pytest.approx(0.6)
    assert cfg["sim"]["arena_size_x"] == 40.0
    assert cfg["sim"]["arena_size_z"] == 40.0
    assert cfg["sim"]["post_sprint_hit_stop"] is True
    assert cfg["league_counter_bot_frac"] > leaderboard["league_counter_bot_frac"]
    assert cfg["league_rehit_bot_frac"] > leaderboard["league_rehit_bot_frac"]
    assert cfg["league_combo_chase_bot_frac"] >= 0.24
    scripted_frac = sum(cfg[k] for k in (
        "league_bot_frac",
        "league_spar_bot_frac",
        "league_rehit_bot_frac",
        "league_pressure_bot_frac",
        "league_combo_chase_bot_frac",
        "league_counter_bot_frac",
        "league_pad_bot_frac",
    ))
    assert scripted_frac == pytest.approx(1.0)
    assert cfg["safety_back_frac"] <= 0.001
    assert cfg["safety_under_combo_escape"] <= 0.015
    assert cfg["safety_under_combo_avoid_frac"] == pytest.approx(0.070)
    assert cfg["safety_under_combo_avoid_frac"] < cfg["score_under_combo_avoid_target"]
    assert cfg["score_under_combo_avoid_target"] < leaderboard["score_under_combo_avoid_target"]
    assert cfg["score_under_combo_avoid_weight"] > leaderboard["score_under_combo_avoid_weight"]
    assert cfg["safety_min_under_combo_counter_hit_frac"] >= 0.115
    assert cfg["safety_min_under_combo_hit_select_clean_frac"] >= 0.30
    assert cfg["safety_max_under_combo_hit_select_trade_frac"] <= 0.08
    assert cfg["safety_min_combo_tap_frac"] >= 0.18
    assert cfg["safety_min_combo_z_tap_frac"] >= 0.16
    assert cfg["safety_max_combo_s_tap_frac"] <= 0.015
    assert cfg["safety_min_hit_wtap_frac"] >= 0.055
    assert cfg["safety_min_chase_hit_wtap_frac"] >= 0.055
    assert cfg["safety_require_counter_recovery"] is True
    assert cfg["safety_require_chase_combo"] is True
    assert cfg["safety_promote_min_combo_max"] >= 10
    assert cfg["combo_eval_counter"] is True
    assert cfg["combo_eval_counter_envs"] >= 96
    assert cfg["sim"]["reward_hurt"] <= -2.6
    assert cfg["sim"]["reward_trade_penalty"] > leaderboard["sim"]["reward_trade_penalty"]
    assert cfg["sim"]["reward_counter_hit"] > leaderboard["sim"]["reward_counter_hit"]
    assert cfg["sim"]["reward_hit_select"] > leaderboard["sim"]["reward_hit_select"]
    assert cfg["sim"]["reward_chase_hit_select"] > leaderboard["sim"]["reward_chase_hit_select"]
    assert cfg["sim"]["reward_chase_close_counter"] > leaderboard["sim"]["reward_chase_close_counter"]
    assert cfg["sim"]["reward_chase_counter"] > leaderboard["sim"]["reward_chase_counter"]
    assert cfg["sim"]["reward_spar_counter"] > leaderboard["sim"]["reward_spar_counter"]
    assert cfg["sim"]["reward_combo_tap"] > leaderboard["sim"]["reward_combo_tap"]
    assert cfg["sim"]["reward_hit_wtap"] > leaderboard["sim"]["reward_hit_wtap"]
    assert "sim.reward_hit_select" in cfg["pbt"]["mutate"]
    assert "sim.reward_chase_counter" in cfg["pbt"]["mutate"]


def test_behavior_reward_bonus_rewards_body_aim_and_punishes_sky_or_backoff():
    obs = torch.zeros(4, 48)
    obs[:, 12] = 1.0      # yaw error = 0 by default
    obs[:, 45] = 4.0 / 8.0
    obs[:, 0] = 4.0 / 8.0

    obs[1, 10] = 1.0      # pitch = 90 deg -> sky/floor saturation penalty
    obs[1, 13] = 1.0      # pitch error too high -> no aim bonus

    obs[2, 11] = 1.0      # yaw error = 90 deg -> no aim bonus
    obs[2, 12] = 0.0
    obs[2, 45] = 6.0 / 8.0
    obs[2, 0] = 6.0 / 8.0

    obs[3, 45] = 2.8 / 8.0  # close circular strafe while not attacking

    actions = torch.zeros(4, 7)
    actions[0, 2] = 1.0     # forward
    actions[0, 5] = 1.0     # sprint
    actions[0, 6] = 1.0     # attack
    actions[1, 2] = 1.0
    actions[2, 2] = -1.0    # backward
    actions[3, 3] = 1.0     # circular strafe

    bonus = _behavior_reward_bonus(
        obs, actions,
        {
            "reward_aim": 0.05,
            "reward_bad_pitch": 0.10,
            "reward_chase": 0.05,
            "reward_aggression": 0.10,
            "reward_no_escape": 0.10,
        },
    )

    assert bonus[0] > 0.05
    assert bonus[1] < 0.0
    assert bonus[2] < 0.0
    assert bonus[3] < 0.0



def test_behavior_reward_bonus_focuses_active_combo_on_body_aim():
    obs = torch.zeros(3, 48)
    obs[:, 12] = 1.0
    obs[:, 45] = 3.0 / 8.0
    obs[:, 0] = 3.0 / 8.0
    obs[1, 10] = 1.0
    obs[1, 13] = 1.0

    actions = torch.zeros(3, 7)
    actions[:, 2] = 1.0
    actions[:, 5] = 1.0
    actions[:, 6] = 1.0
    combo_lengths = torch.tensor([12.0, 12.0, 0.0])

    bonus = _behavior_reward_bonus(
        obs, actions, {"reward_combo_focus": 1.0}, combo_lengths)

    assert bonus[0] > 0.0
    assert bonus[1] < 0.0
    assert float(bonus[2]) == pytest.approx(0.0)


def test_behavior_reward_bonus_prefers_combo_tap_over_hold_sprint():
    obs = torch.zeros(3, 48)
    obs[:, 12] = 1.0
    obs[:, 15] = -1.0
    obs[:, 21] = 0.0
    obs[:, 22] = 0.50
    obs[:, 45] = 2.45 / 8.0
    obs[:, 0] = 2.45 / 8.0
    obs[:, 40] = 1.0
    obs[:, 43] = 1.0

    actions = torch.zeros(3, 7)
    actions[0, 2] = -1.0  # real back tap, forbidden by no-back contract
    actions[1, 2] = 1.0
    actions[1, 5] = 1.0   # hold sprint into close range
    actions[2, 2] = 0.0   # Z/W release reset

    bonus = _behavior_reward_bonus(obs, actions, {"reward_combo_tap": 1.0})

    assert bonus[2] > bonus[1]
    assert bonus[2] > bonus[0] + 0.25


def test_behavior_reward_bonus_prefers_z_release_during_fresh_combo_cooldown():
    obs = torch.zeros(2, 48)
    obs[:, 12] = 1.0
    obs[:, 15] = -1.0
    obs[:, 21] = 0.0
    obs[:, 22] = 0.80
    obs[:, 45] = 2.85 / 8.0
    obs[:, 0] = 2.85 / 8.0
    obs[:, 40] = 1.0
    obs[:, 43] = 1.0

    actions = torch.zeros(2, 7)
    actions[0, 2] = -1.0
    actions[1, 2] = 0.0

    bonus = _behavior_reward_bonus(obs, actions, {"reward_combo_tap": 1.0})

    assert bonus[1] > bonus[0] + 1.0


def test_behavior_reward_bonus_rewards_repress_after_combo_tap():
    obs = torch.zeros(2, 48)
    obs[:, 12] = 1.0
    obs[:, 15] = -1.0
    obs[:, 21] = 0.0
    obs[:, 22] = 0.50
    obs[:, 45] = 2.35 / 8.0
    obs[:, 0] = 2.35 / 8.0
    obs[:, 40] = 0.0
    obs[:, 43] = 0.0

    actions = torch.zeros(2, 7)
    actions[0, 2] = 1.0
    actions[0, 5] = 1.0   # re-press after tap
    actions[1, 2] = 0.0   # stuck in repeated Z release

    bonus = _behavior_reward_bonus(obs, actions, {"reward_combo_tap": 1.0})

    assert bonus[0] > 0.0
    assert bonus[0] > bonus[1] + 0.5


def test_behavior_reward_bonus_rewards_under_combo_counter_and_punishes_escape():
    obs = torch.zeros(5, 48)
    obs[:, 12] = 1.0
    obs[:, 45] = 2.50 / 8.0
    obs[:, 0] = 2.50 / 8.0
    obs[:, 21] = 1.0      # own hurt resistance: we are being comboed
    obs[:, 22] = 0.0
    obs[4, 45] = 3.75 / 8.0
    obs[4, 0] = 3.75 / 8.0

    actions = torch.zeros(5, 7)
    actions[0, 2] = 1.0   # hold the line without breaking the combo
    actions[1, 2] = 1.0
    actions[1, 6] = 1.0   # counter-hit while comboed
    actions[2, 2] = -1.0  # back escape while comboed
    actions[3, 3] = 1.0   # lateral escape while comboed
    actions[4, 2] = 1.0
    actions[4, 6] = 1.0   # too far: this is a trade click, not an opportunity

    bonus = _behavior_reward_bonus(
        obs, actions, {"reward_no_escape": 1.0, "reward_counter_hit": 1.0})

    assert bonus[1] > bonus[0]
    assert bonus[0] > bonus[4]
    assert bonus[1] > bonus[2] + 1.0
    assert bonus[1] > bonus[3] + 1.0
    assert bonus[2] < 0.0
    assert bonus[3] < 0.0


def test_behavior_reward_bonus_strongly_prefers_timed_close_counter_click():
    obs = torch.zeros(5, 48)
    obs[:, 12] = 1.0
    obs[:, 21] = COUNTER_CLOSE_RECOVERY_CLICK_HURT - 0.04
    obs[:, 22] = 0.0
    obs[:, 45] = 1.90 / 8.0
    obs[:, 0] = 1.90 / 8.0

    actions = torch.zeros(5, 7)
    actions[0, 6] = 1.0   # timed close-counter click
    actions[1, 6] = 0.0   # freezes during the recovery window
    actions[2, 2] = -1.0  # backs out instead of countering
    actions[2, 6] = 1.0
    actions[3, 3] = 1.0   # circles instead of holding the line
    actions[3, 6] = 1.0
    actions[4, 2] = 1.0   # weaker untimed hold-forward click
    actions[4, 6] = 1.0

    bonus = _behavior_reward_bonus(obs, actions, {"reward_counter_hit": 1.0})

    assert bonus[0] > bonus[1] + 2.0
    assert bonus[0] > bonus[2] + 2.0
    assert bonus[0] > bonus[3] + 2.0
    assert bonus[0] > bonus[4]
    assert bonus[1] < 0.0
    assert bonus[2] < 0.0
    assert bonus[3] < 0.0


def test_behavior_reward_bonus_prefers_early_reach_counter_click():
    obs = torch.zeros(4, 48)
    obs[:, 12] = 1.0
    obs[:, 21] = COUNTER_RECOVERY_CLICK_HURT - 0.05
    obs[:, 22] = 0.0
    obs[:, 45] = (COUNTER_HIT_SELECT_MIN_REACH + 0.05) / 8.0
    obs[:, 0] = (COUNTER_HIT_SELECT_MIN_REACH + 0.05) / 8.0

    actions = torch.zeros(4, 7)
    actions[0, 2] = 1.0
    actions[0, 5] = 1.0
    actions[0, 6] = 1.0   # timed early-reach counter click
    actions[1, 2] = 1.0
    actions[1, 5] = 1.0   # misses the ready click
    actions[2, 2] = -1.0  # backs out
    actions[2, 6] = 1.0
    actions[3, 2] = 1.0
    actions[3, 5] = 1.0
    actions[3, 6] = 1.0
    obs[3, 45] = (COUNTER_FAR_TRADE_REACH + 0.20) / 8.0

    bonus = _behavior_reward_bonus(obs, actions, {"reward_counter_hit": 1.0})

    assert bonus[0] > bonus[1] + 2.0
    assert bonus[0] > bonus[2] + 2.0
    assert bonus[0] > bonus[3] + 2.0
    assert bonus[1] < 0.0
    assert bonus[2] < 0.0
    assert bonus[3] < 0.0


def test_behavior_reward_bonus_hit_select_prefers_timed_counter_click():
    obs = torch.zeros(8, 48)
    obs[:, 12] = 1.0
    obs[:, 21] = COUNTER_HIT_SELECT_CLEAN_HURT - 0.05
    obs[:, 22] = 0.0
    obs[:, 37] = 0.10
    obs[:, 45] = 2.90 / 8.0
    obs[:, 0] = 2.90 / 8.0
    obs[2, 45] = 4.25 / 8.0
    obs[2, 0] = 4.25 / 8.0
    obs[4, 11] = 1.0     # yaw error = 90 deg, bad click even in range
    obs[4, 12] = 0.0
    obs[5, 21] = 0.0     # not under combo -> hit-select should not fire

    actions = torch.zeros(8, 7)
    actions[0, 6] = 1.0  # valid select click
    actions[1, 2] = 1.0
    actions[1, 5] = 1.0  # sprinting through the select window trades
    actions[2, 6] = 1.0  # far spam
    actions[3, 2] = -1.0 # back escape
    actions[4, 6] = 1.0  # bad aim spam
    actions[5, 6] = 1.0
    actions[6, 3] = 1.0  # circling while clicking loses the line
    actions[6, 6] = 1.0
    actions[7, 6] = 0.0  # correct release, but misses the timed click

    bonus = _behavior_reward_bonus(obs, actions, {"reward_hit_select": 1.0})

    assert bonus[0] > bonus[1] + 2.0
    assert bonus[0] > bonus[6] + 1.0
    assert bonus[0] > bonus[7] + 2.0
    assert bonus[0] > 0.0
    assert bonus[1] < 0.0
    assert bonus[2] < 0.0
    assert bonus[3] < 0.0
    assert bonus[4] < 0.0
    assert float(bonus[5]) == pytest.approx(0.0)


def test_hit_event_reward_bonus_rewards_rehit_and_counter_break():
    obs = torch.zeros(6, 48)
    obs[:, 12] = 1.0
    obs[:, 45] = 2.85 / 8.0
    obs[0, 22] = 1.0   # combo advantage, clean re-hit
    obs[1, 22] = 1.0   # combo advantage, got hit back
    obs[4, 22] = 1.0   # combo advantage, trade still breaks chain
    obs[2, 21] = 1.0   # under combo, clean counter
    obs[3, 21] = 1.0   # under combo, ate next hit
    obs[5, 21] = 1.0   # under combo, trade still means we ate the combo

    actions = torch.zeros(6, 7)
    actions[:, 2] = 1.0
    actions[:, 5] = 1.0
    actions[:, 6] = 1.0
    dealt = torch.tensor([True, False, True, False, True, True])
    taken = torch.tensor([False, True, False, True, True, True])

    bonus = _hit_event_reward_bonus(
        obs, actions, dealt, taken,
        {"reward_combo_tap": 1.0, "reward_counter_hit": 1.0},
    )

    assert bonus[0] > 0.0
    assert bonus[0] > bonus[1] + 1.0
    assert bonus[4] < 0.0
    assert bonus[0] > bonus[4] + 1.0
    assert bonus[2] > 0.0
    assert bonus[2] > bonus[3] + 1.0
    assert bonus[5] < 0.0
    assert bonus[2] > bonus[5] + 1.0


def test_hit_event_reward_bonus_hit_select_rewards_clean_counter_not_trade():
    obs = torch.zeros(5, 48)
    obs[:, 12] = 1.0
    obs[:, 21] = COUNTER_HIT_SELECT_CLEAN_HURT - 0.05
    obs[:, 22] = 0.0
    obs[:, 37] = 0.10
    obs[:, 45] = 2.90 / 8.0
    obs[3, 45] = 4.25 / 8.0
    obs[4, 21] = 0.0
    obs[4, 22] = 0.80

    actions = torch.zeros(5, 7)
    actions[:, 2] = 1.0
    actions[:, 5] = 1.0
    actions[:, 6] = 1.0
    dealt = torch.tensor([True, False, True, False, True])
    taken = torch.tensor([False, True, True, False, False])

    bonus = _hit_event_reward_bonus(
        obs, actions, dealt, taken, {"reward_hit_select": 1.0})

    assert bonus[0] > 0.0
    assert bonus[1] < 0.0
    assert bonus[2] < 0.0
    assert bonus[3] < 0.0
    assert float(bonus[4]) == pytest.approx(0.0)
    assert bonus[0] > bonus[1] + 2.0


def test_hit_event_reward_bonus_hit_select_allows_controlled_strafe_counter():
    obs = torch.zeros(3, 48)
    obs[:, 12] = 1.0
    obs[:, 21] = COUNTER_HIT_SELECT_CLEAN_HURT - 0.05
    obs[:, 22] = 0.0
    obs[:, 37] = 0.10
    obs[:, 45] = 2.90 / 8.0

    actions = torch.zeros(3, 7)
    actions[:, 2] = 1.0
    actions[:, 5] = 1.0
    actions[:, 6] = 1.0
    actions[1, 3] = 1.0
    actions[2, 3] = 1.0
    dealt = torch.tensor([True, True, True])
    taken = torch.tensor([False, False, True])

    bonus = _hit_event_reward_bonus(
        obs, actions, dealt, taken, {"reward_hit_select": 1.0})

    assert bonus[1] > 0.0
    assert bonus[0] > bonus[1]
    assert bonus[1] > bonus[2] + 2.0
    assert bonus[2] < 0.0


def test_under_combo_escape_metric_tracks_back_not_active_strafe():
    source = Path("train/run.py").read_text(encoding="utf-8")

    assert "((back_mask & under_combo).float().sum() / under_combo_count)" in source
    assert "strafe_dir_metric.abs() > 0.5))\n              & under_combo" not in source


def test_hit_event_reward_bonus_rewards_wtap_strafed_hits():
    obs = torch.zeros(3, 48)
    obs[:, 12] = 1.0
    obs[:, 45] = 2.85 / 8.0
    actions = torch.zeros(3, 7)
    actions[0, 2] = 0.0
    actions[0, 3] = 1.0
    actions[1, 2] = 1.0
    actions[1, 3] = 1.0
    actions[1, 5] = 1.0
    actions[2, 2] = 0.0
    dealt = torch.tensor([True, True, True])
    taken = torch.tensor([False, False, False])

    bonus = _hit_event_reward_bonus(
        obs, actions, dealt, taken, {"reward_hit_wtap": 1.0})

    assert bonus[0] > 0.0
    assert bonus[1] < 0.0
    assert bonus[0] > bonus[2]


def test_post_hit_wtap_reward_bonus_targets_live_hit_counter_timing():
    hist = torch.zeros(3, 4, 48)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 45] = 2.85 / 8.0
    hist[:, -2, 31] = 0.00
    hist[:, -1, 31] = 0.01
    actions = torch.zeros(3, 7)
    actions[0, 2] = 0.0
    actions[0, 3] = 1.0
    actions[1, 2] = 1.0
    actions[1, 3] = 1.0
    actions[1, 5] = 1.0
    actions[2, 2] = 0.0

    bonus = _post_hit_wtap_reward_bonus(
        hist, actions, {"reward_hit_wtap": 1.0})

    assert bonus[0] > 0.0
    assert bonus[1] < 0.0
    assert bonus[0] > bonus[2]


def test_opener_boxing_reward_prefers_forward_strafe_pressure():
    obs = torch.zeros(6, 48)
    obs[:, 12] = 1.0
    obs[:, 45] = 5.5 / 8.0
    age = torch.tensor([0, 3, 3, 3, 3, 24])
    actions = torch.zeros(6, 7)
    actions[0, 2] = 1.0
    actions[0, 3] = 1.0
    actions[0, 5] = 1.0
    actions[1, 2] = 1.0
    actions[1, 5] = 1.0
    actions[2, 2] = -1.0
    actions[2, 3] = 1.0
    actions[3, 2] = 1.0
    actions[3, 3] = 1.0
    actions[3, 4] = 1.0
    actions[3, 5] = 1.0
    actions[4, 3] = 1.0
    actions[5, 2] = 1.0
    actions[5, 3] = 1.0
    actions[5, 5] = 1.0

    bonus = _opener_boxing_reward_bonus(
        age, obs, actions, {"reward_opener_strafe": 1.0}, opener_ticks=20)

    assert bonus[0] > 0.0
    assert bonus[0] > bonus[1]
    assert bonus[2] < 0.0
    assert bonus[3] < bonus[0]
    assert bonus[4] < 0.0
    assert bonus[0] > bonus[4]
    assert float(bonus[5]) == pytest.approx(0.0)


def test_chase_transfer_reward_targets_active_chaser_rechains_and_counters():
    obs = torch.zeros(7, 48)
    obs[:, 12] = 1.0
    obs[:, 45] = 2.85 / 8.0
    obs[:, 37] = 0.10
    obs[0, 22] = 0.30   # combo advantage, clean re-hit
    obs[1, 22] = 0.30   # combo advantage, got stolen
    obs[5, 22] = 0.30   # same as row 0, but not a chase learner row
    obs[2, 21] = COUNTER_HIT_SELECT_CLEAN_HURT - 0.05   # legal recovery counter
    obs[3, 21] = COUNTER_HIT_SELECT_CLEAN_HURT - 0.05   # had a counter window but froze
    obs[4, 21] = COUNTER_HIT_SELECT_CLEAN_HURT - 0.05   # trade is still bad
    obs[6, 21] = COUNTER_HIT_SELECT_CLEAN_HURT - 0.05   # clean but circling

    actions = torch.zeros(7, 7)
    actions[:, 2] = 1.0
    actions[:, 5] = 1.0
    actions[:, 6] = 1.0
    actions[3, 6] = 0.0
    actions[6, 3] = 1.0

    dealt = torch.tensor([True, False, True, False, True, True, True])
    taken = torch.tensor([False, True, False, False, True, False, False])
    chase_mask = torch.tensor([True, True, True, True, True, False, True])

    bonus = _chase_transfer_reward_bonus(
        obs, actions, dealt, taken, chase_mask, chase_mask,
        torch.zeros_like(chase_mask),
        {"reward_chase_rechain": 1.0, "reward_chase_counter": 1.0},
    )

    assert bonus[0] > 0.0
    assert bonus[1] < 0.0
    assert bonus[2] > 0.0
    assert bonus[2] > bonus[6] + 1.0
    assert bonus[3] < 0.0
    assert bonus[4] < 0.0
    assert bonus[5] == 0.0


def test_chase_transfer_reward_can_scope_spar_counter_without_rechain():
    obs = torch.zeros(4, 48)
    obs[:, 12] = 1.0
    obs[:, 45] = 2.85 / 8.0
    obs[:, 37] = 0.10
    obs[0, 22] = 0.30   # combo advantage, clean re-hit
    obs[1, 22] = 0.30   # combo advantage, but not in rechain mask
    obs[2, 21] = COUNTER_HIT_SELECT_CLEAN_HURT - 0.05   # spar counter recovery mask
    obs[3, 21] = COUNTER_HIT_SELECT_CLEAN_HURT - 0.05   # not masked

    actions = torch.zeros(4, 7)
    actions[:, 2] = 1.0
    actions[:, 5] = 1.0
    actions[:, 6] = 1.0

    dealt = torch.tensor([True, True, True, True])
    taken = torch.zeros(4, dtype=torch.bool)
    rechain_mask = torch.tensor([True, False, False, False])
    counter_mask = torch.tensor([False, True, False, False])
    spar_counter_mask = torch.tensor([False, False, True, False])

    bonus = _chase_transfer_reward_bonus(
        obs, actions, dealt, taken, rechain_mask, counter_mask,
        spar_counter_mask,
        {"reward_chase_rechain": 1.0, "reward_chase_counter": 1.0},
    )

    assert bonus[0] > 0.0
    assert bonus[1] == 0.0
    assert bonus[2] > 0.0
    assert bonus[3] == 0.0


def test_chase_transfer_reward_uses_spar_counter_coeff_without_boosting_chase():
    obs = torch.zeros(2, 48)
    obs[:, 12] = 1.0
    obs[:, 45] = 2.85 / 8.0
    obs[:, 21] = COUNTER_HIT_SELECT_CLEAN_HURT - 0.05
    obs[:, 37] = 0.10

    actions = torch.zeros(2, 7)
    actions[:, 2] = 1.0
    actions[:, 5] = 1.0
    actions[:, 6] = 1.0

    dealt = torch.ones(2, dtype=torch.bool)
    taken = torch.zeros(2, dtype=torch.bool)
    rechain_mask = torch.zeros(2, dtype=torch.bool)
    chase_counter_mask = torch.tensor([True, False])
    spar_counter_mask = torch.tensor([False, True])

    bonus = _chase_transfer_reward_bonus(
        obs, actions, dealt, taken, rechain_mask, chase_counter_mask,
        spar_counter_mask,
        {
            "reward_chase_rechain": 0.0,
            "reward_chase_counter": 1.0,
            "reward_spar_counter": 3.0,
        },
    )

    assert bonus[0] > 0.0
    assert bonus[1] == pytest.approx(bonus[0] * 3.0)


def test_chase_transfer_reward_chase_hit_select_prefers_clean_counter():
    source = Path("train/run.py").read_text(encoding="utf-8")
    assert "- 18.80 * hit_select_trade" in source
    assert "- 12.40 * hit_select_stolen" in source

    obs = torch.zeros(5, 48)
    obs[:, 12] = 1.0
    obs[:, 21] = COUNTER_HIT_SELECT_CLEAN_HURT - 0.05
    obs[:, 37] = 0.10
    obs[:, 45] = 2.85 / 8.0

    actions = torch.zeros(5, 7)
    actions[:, 2] = 1.0
    actions[:, 5] = 1.0
    actions[:, 6] = 1.0
    actions[0, 2] = 0.0
    actions[0, 5] = 0.0
    actions[3, 6] = 0.0

    dealt = torch.tensor([True, True, False, False, True])
    taken = torch.tensor([False, True, True, False, False])
    rechain_mask = torch.zeros(5, dtype=torch.bool)
    chase_counter_mask = torch.tensor([True, True, True, True, False])
    spar_counter_mask = torch.zeros(5, dtype=torch.bool)

    bonus = _chase_transfer_reward_bonus(
        obs, actions, dealt, taken, rechain_mask, chase_counter_mask,
        spar_counter_mask,
        {
            "reward_chase_rechain": 0.0,
            "reward_chase_counter": 0.0,
            "reward_spar_counter": 0.0,
            "reward_chase_hit_select": 1.0,
        },
    )

    assert bonus[0] > 0.0
    assert bonus[1] < 0.0
    assert bonus[2] < 0.0
    assert bonus[3] < 0.0
    assert bonus[4] == 0.0


def test_chase_transfer_reward_close_counter_prefers_clean_point_blank_counter():
    source = Path("train/run.py").read_text(encoding="utf-8")
    assert "- 13.20 * close_counter_trade" in source
    assert "- 8.40 * close_counter_stolen" in source

    obs = torch.zeros(5, 48)
    obs[:, 12] = 1.0
    obs[:, 21] = COUNTER_RECOVERY_CLICK_HURT - 0.05
    obs[:, 45] = 1.80 / 8.0

    actions = torch.zeros(5, 7)
    actions[:, 2] = 1.0
    actions[:, 5] = 1.0
    actions[:, 6] = 1.0
    actions[3, 6] = 0.0

    dealt = torch.tensor([True, True, False, False, True])
    taken = torch.tensor([False, True, True, False, False])
    rechain_mask = torch.zeros(5, dtype=torch.bool)
    chase_counter_mask = torch.tensor([True, True, True, True, False])
    spar_counter_mask = torch.zeros(5, dtype=torch.bool)

    bonus = _chase_transfer_reward_bonus(
        obs, actions, dealt, taken, rechain_mask, chase_counter_mask,
        spar_counter_mask,
        {
            "reward_chase_rechain": 0.0,
            "reward_chase_counter": 0.0,
            "reward_spar_counter": 0.0,
            "reward_chase_hit_select": 0.0,
            "reward_chase_close_counter": 1.0,
        },
    )

    assert bonus[0] > 0.0
    assert bonus[1] < 0.0
    assert bonus[2] < 0.0
    assert bonus[3] < 0.0
    assert bonus[4] == 0.0


def test_chase_transfer_reward_rewards_early_reach_counter():
    obs = torch.zeros(5, 48)
    obs[:, 12] = 1.0
    obs[:, 21] = COUNTER_RECOVERY_CLICK_HURT - 0.05
    obs[:, 45] = (COUNTER_HIT_SELECT_MIN_REACH + 0.05) / 8.0

    actions = torch.zeros(5, 7)
    actions[:, 2] = 1.0
    actions[:, 5] = 1.0
    actions[:, 6] = 1.0
    actions[3, 6] = 0.0

    dealt = torch.tensor([True, True, False, False, True])
    taken = torch.tensor([False, True, True, False, False])
    rechain_mask = torch.zeros(5, dtype=torch.bool)
    chase_counter_mask = torch.tensor([True, True, True, True, False])
    spar_counter_mask = torch.zeros(5, dtype=torch.bool)

    bonus = _chase_transfer_reward_bonus(
        obs, actions, dealt, taken, rechain_mask, chase_counter_mask,
        spar_counter_mask,
        {
            "reward_chase_rechain": 0.0,
            "reward_chase_counter": 1.0,
            "reward_spar_counter": 0.0,
            "reward_chase_hit_select": 0.0,
            "reward_chase_close_counter": 0.0,
        },
    )

    assert bonus[0] > 0.0
    assert bonus[1] < 0.0
    assert bonus[2] < 0.0
    assert bonus[3] < 0.0
    assert bonus[4] == 0.0


def test_collect_applies_recovery_counter_transfer_to_spar_lane():
    source = Path("train/run.py").read_text(encoding="utf-8")

    assert "combo_spar_counter_learner_mask" in source
    assert "combo_recovery_counter_learner_mask" in source
    assert "combo_spar_counter_learner_mask = learner_mask & (self._env_opp[row_env] == -4)" in source
    assert "combo_rechain_learner_mask" in source


def test_chase_transfer_reward_teaches_drive_release_and_hit_select_timing():
    obs = torch.zeros(6, 48)
    obs[:, 12] = 1.0
    obs[:, 22] = 0.30
    obs[2:4, 22] = 0.50
    obs[4:6, 22] = 0.80
    obs[0:2, 45] = 2.85 / 8.0
    obs[2:4, 45] = 3.85 / 8.0
    obs[4:6, 45] = 3.10 / 8.0

    actions = torch.zeros(6, 7)
    actions[0, 6] = 1.0          # ready re-hit: click even during z-release
    actions[2, 2] = 1.0          # chase drive: sprint forward while safely out of edge-brake
    actions[2, 5] = 1.0
    actions[5, 2] = 1.0          # bad wait: run/click too early instead of z-release
    actions[5, 5] = 1.0
    actions[5, 6] = 1.0

    dealt = torch.zeros(6, dtype=torch.bool)
    taken = torch.zeros(6, dtype=torch.bool)
    chase_mask = torch.ones(6, dtype=torch.bool)

    bonus = _chase_transfer_reward_bonus(
        obs, actions, dealt, taken, chase_mask, chase_mask,
        torch.zeros_like(chase_mask),
        {"reward_chase_rechain": 1.0, "reward_chase_counter": 0.0},
    )

    assert bonus[0] > 0.0
    assert bonus[0] > bonus[1] + 2.0
    assert bonus[2] > 0.0
    assert bonus[2] > bonus[3] + 1.0
    assert bonus[4] > 0.0
    assert bonus[4] > bonus[5] + 1.0
    assert bonus[5] < 0.0


def test_chase_transfer_reward_edge_brakes_before_chaser_steals_rehit():
    obs = torch.zeros(4, 48)
    obs[:, 12] = 1.0
    obs[:, 22] = 0.50
    obs[:, 45] = 3.55 / 8.0

    actions = torch.zeros(4, 7)
    actions[1, 2] = 1.0          # overrun: still sprinting into the chaser
    actions[1, 5] = 1.0
    actions[2, 6] = 1.0          # prefire: click before reliable re-hit reach
    actions[3, 2] = 1.0
    actions[3, 5] = 1.0
    actions[3, 6] = 1.0

    dealt = torch.zeros(4, dtype=torch.bool)
    taken = torch.zeros(4, dtype=torch.bool)
    chase_mask = torch.ones(4, dtype=torch.bool)

    bonus = _chase_transfer_reward_bonus(
        obs, actions, dealt, taken, chase_mask, chase_mask,
        torch.zeros_like(chase_mask),
        {"reward_chase_rechain": 1.0, "reward_chase_counter": 0.0},
    )

    assert bonus[0] > 0.0
    assert bonus[0] > bonus[1] + 1.0
    assert bonus[0] > bonus[2] + 1.0
    assert bonus[1] < 0.0
    assert bonus[2] < 0.0
    assert bonus[3] < bonus[1]


def test_combo_god_entrypoints_use_attn96_profile_and_arena_defaults():
    script = Path("scripts/train_combo_god.bat").read_text(encoding="utf-8")
    start = Path("scripts/start_combo_god.ps1").read_text(encoding="utf-8")
    assert "start_combo_god.bat" in script
    assert "-Force -Iters 8 -TimeoutMinutes 20" in script
    assert "python -m train.run" not in script
    assert "combo_god_recovery_kb092_combo12.json" in start
    assert "combo_god_leaderboard10_combo12.json" in start
    assert 'combo_god_recovery_kb092_combo12' in start
    assert 'combo_god_leaderboard10_combo12' in start
    assert 'combo_god_countertap96_combo12' in start
    assert start.index('"runs/$Run/safe_latest.pt"') < start.index('"runs/$Run/latest.pt"')
    assert start.index('"runs/combo_god_leaderboard10_combo12/safe_latest.pt"') < start.index('"runs/combo_god_countertap96_combo12/safe_latest.pt"')
    assert '"runs/combo_god_directpad_lock_combo12/safe_latest.pt"' in start
    assert "runs\\combo_god_aggro\\best.pt" not in script
    assert "runs\\combo_god_ft\\best.pt" not in script
    assert "runs/god/best.pt" in start
    run_menu = Path("run.bat").read_text(encoding="utf-8")
    assert "scripts\\start_combo_god.bat" in run_menu
    assert "scripts\\start_combo_god.bat -Force -Iters 8 -TimeoutMinutes 20" in run_menu
    assert "scripts\\stop_combo_god.bat" in run_menu
    assert "[9] field preflight (no start)" in run_menu
    assert "[p] combo proof local (no training)" in run_menu
    assert "[f] field proof quick (deployed mod, no rebuild)" in run_menu
    assert "call scripts\\check_field_preflight.bat" in run_menu
    assert "call scripts\\prove_combo_god.bat" in run_menu
    assert "call scripts\\field_test_aim_os_quick.bat" in run_menu
    assert "cmd /k scripts\\train_combo_god.bat" not in run_menu

    app = Path("app/src/pages/Training.jsx").read_text(encoding="utf-8")
    dashboard = Path("app/src/pages/Dashboard.jsx").read_text(encoding="utf-8")
    health = Path("app/src/health.js").read_text(encoding="utf-8")
    assert "judas:app:training:v92" in app
    assert "judas:app:training:v91" not in app
    assert "judas:app:training:v90" not in app
    assert "judas:app:training:v89" not in app
    assert "judas:app:training:v88" not in app
    assert "judas:app:training:v82" not in app
    assert "judas:app:training:v81" not in app
    assert "judas:app:training:v80" not in app
    assert "judas:app:training:v79" not in app
    assert "judas:app:training:v73" not in app
    assert "judas:app:training:v72" not in app
    assert "judas:app:training:v71" not in app
    assert "judas:app:training:v70" not in app
    assert "judas:app:training:v69" not in app
    assert "judas:app:training:v68" not in app
    assert "judas:app:training:v67" not in app
    assert "judas:app:training:v66" not in app
    assert "judas:app:training:v65" not in app
    assert "judas:app:training:v64" not in app
    assert "judas:app:training:v63" not in app
    assert "judas:app:training:v61" not in app
    assert "judas:app:training:v60" not in app
    assert "judas:app:training:v59" not in app
    assert "judas:app:training:v58" not in app
    assert "judas:app:training:v57" not in app
    assert "judas:app:training:v56" not in app
    assert "judas:app:training:v55" not in app
    assert "judas:app:training:v51" not in app
    assert 'name: "combo_god_recovery_kb092_combo12"' in app
    assert "lr: 0.0000015" in app
    assert "clip: 0.04" in app
    assert "sample_frac: 0.28" in app
    assert "leaderboard_boxing" in app
    assert "direct_counter_attack_lock: true" in app
    assert "direct_hit_select_attack_lock: true" in app
    assert "direct_hit_select_attack_bias: 22.0" in app
    assert "cps_min: 10, cps_max: 10" in app
    assert '? `runs/${name}/latest.pt`' in app
    assert "resume_as_seed" in app
    assert "resume_same_run_as_seed" in app
    assert "same run seed" in app
    assert "keep_ckpts" in app
    assert "safety_restore_on_low_combo" in app
    assert "league_spar_bot_frac" in app
    assert "league_rehit_bot_frac" in app
    assert "league_pressure_bot_frac" in app
    assert "league_combo_chase_bot_frac" in app
    assert "league_counter_bot_frac" in app
    assert "counter recovery" in app
    assert "attention: true, d_model: 96" in app
    assert "coach_coef" in app
    assert "reward_aim" in app
    assert "reward_bad_pitch" in app
    assert "reward_chase" in app
    assert "reward_turn_aim" in app
    assert "reward_aggression" in app
    assert "reward_no_escape" in app
    assert "reward_combo_focus" in app
    assert "reward_combo_tap" in app
    assert "reward_opener_strafe" in app
    assert "reward_hit_wtap" in app
    assert "reward_hit_wtap: 18.00" in app
    assert "reward_trade_penalty: 7.0" in app
    assert "reward_counter_hit" in app
    assert "reward_hit_select" in app
    assert "reward_chase_rechain" in app
    assert "reward_chase_hit_select" in app
    assert "reward_chase_close_counter" in app
    assert "reward_chase_counter" in app
    assert "reward_spar_counter" in app
    assert "reward_opener_strafe: 3.00" in app
    assert "reward_counter_hit: 34.00" in app
    assert "reward_hit_select: 36.00" in app
    assert "reward_chase_rechain: 10.80" in app
    assert "reward_chase_hit_select: 30.00" in app
    assert "reward_chase_close_counter: 22.00" in app
    assert "reward_chase_counter: 38.00" in app
    assert "reward_spar_counter: 34.00" in app
    assert "safety_under_combo_avoid_frac: 0.070" in app
    assert "safety_under_combo_avoid_min_combo12: 0.10" in app
    assert "score_under_combo_avoid_cap: 0.025" in app
    assert "safety_min_under_combo_hit_select_clean_frac: 0.30" in app
    assert "safety_max_under_combo_hit_select_trade_frac: 0.08" in app
    assert "chase hit select" in app
    assert "chase close counter" in app
    assert "combo_eval_every" in app
    assert "combo_eval_envs" in app
    assert "combo_eval_ticks" in app
    assert "combo_eval_chase" in app
    assert "combo_eval_chase_envs" in app
    assert "combo_eval_chase_ticks" in app
    assert "combo_eval_spar" in app
    assert "combo_eval_spar_envs" in app
    assert "combo_eval_spar_ticks" in app
    assert "combo_eval_rehit" in app
    assert "combo_eval_rehit_envs" in app
    assert "combo_eval_rehit_ticks" in app
    assert "combo_eval_pressure" in app
    assert "combo_eval_pressure_envs" in app
    assert "combo_eval_pressure_ticks" in app
    assert "safety_require_chase_combo" in app
    assert "shaping_sky_frac" in app
    assert "shaping_combo12_state" in app
    assert "direct_movement_lock" in app
    assert "direct_counter_attack_lock" in app
    assert "direct_hit_select_attack_lock" in app
    assert "direct_hit_select_attack_bias" in app
    assert "hit-select bias" in app
    assert "under_combo_attack_lock" in app
    assert "safety_stop_on_regression" in app
    assert "fresh_optimizer_on_resume" in app
    assert "safety_fresh_min_hit_rate" in app
    assert "safety_promote_min_combo_max" in app
    assert "safety_min_strafe_frac: 0.50" in app
    assert "safety_min_opener_strafe_frac: 0.75" in app
    assert "safety_min_opener_strafe_hold_frac: 0.70" in app
    assert "safety_opener_ticks: 20" in app
    assert "safety_min_opener_pressure_frac: 0.45" in app
    assert "safety_min_combo_tap_frac: 0.18" in app
    assert "safety_min_combo_z_tap_frac: 0.16" in app
    assert "safety_max_combo_s_tap_frac: 0.015" in app
    assert "safety_min_hit_wtap_frac: 0.055" in app
    assert "safety_min_chase_hit_wtap_frac: 0.055" in app
    assert "chase wtap min" in app
    assert "safety_rollout_hit_wtap_slack: 0.30" in app
    assert "rollout hit wtap slack" in app
    assert "safety_hit_wtap_blocks_promotion: true" in app
    assert "wtap waits safe" in app
    assert "safety_min_under_combo_counter_hit_frac: 0.115" in app
    assert "kb_h: 0.92, kb_v: 0.90, kb_idle: 0.6" in app
    assert "judas:app:training:v92" in app
    assert "judas:app:training:v91" not in app
    assert "judas:app:training:v90" not in app
    assert "judas:app:training:v89" not in app
    assert "judas:app:training:v88" not in app
    assert "judas:app:training:v87" not in app
    assert "judas:app:training:v86" not in app
    assert "judas:app:training:v83" not in app
    assert "judas:app:training:v80" not in app
    assert "judas:app:training:v79" not in app
    assert "judas:app:training:v78" not in app
    assert "judas:app:training:v73" not in app
    assert "judas:app:training:v72" not in app
    assert "judas:app:training:v71" not in app
    assert "judas:app:training:v70" not in app
    assert "judas:app:training:v69" not in app
    assert "judas:app:training:v68" not in app
    assert "judas:app:training:v67" not in app
    assert "judas:app:training:v66" not in app
    assert "judas:app:training:v65" not in app
    assert "judas:app:training:v64" not in app
    assert "judas:app:training:v63" not in app
    assert "judas:app:training:v61" not in app
    assert "judas:app:training:v60" not in app
    assert "judas:app:training:v59" not in app
    assert "judas:app:training:v58" not in app
    assert "judas:app:training:v57" not in app
    assert "judas:app:training:v56" not in app
    assert "judas:app:training:v55" not in app
    assert "judas:app:training:v51" not in app
    assert "resumePathFor" in app
    assert "runs/${name}/latest.pt" in app
    assert "safe_latest.pt" in app
    assert "resume latest" in app
    assert 'label="safe state"' in dashboard
    assert 'stat("win rate", "league_winrate"' not in dashboard
    assert 'stat("league WR", "league_winrate"' in dashboard
    assert 'k="fresh combo12"' in dashboard
    assert 'k="spar combo"' in dashboard
    assert 'k="spar combo12"' in dashboard
    assert 'k="rehit combo"' in dashboard
    assert 'k="rehit rechain"' in dashboard
    assert 'k="pressure combo"' in dashboard
    assert 'k="pressure combo12"' in dashboard
    assert 'k="pressure rechain"' in dashboard
    assert 'k="pressure taken"' in dashboard
    assert 'k="spar hit/min"' in dashboard
    assert 'k="spar trade"' in dashboard
    assert 'k="spar rechain"' in dashboard
    assert 'k="spar counter"' in dashboard
    assert 'k="chase combo"' in dashboard
    assert 'k="chase combo12"' in dashboard
    assert 'k="chase hit/min"' in dashboard
    assert 'k="chase trade"' in dashboard
    assert 'k="chase rechain"' in dashboard
    assert 'k="chase counter"' in dashboard
    assert 'k="recover hit/min"' in dashboard
    assert 'k="recover rechain"' in dashboard
    assert 'k="recover counter"' in dashboard
    assert 'k="recover combo"' in dashboard
    assert 'k="s-tap"' in dashboard
    assert 'k="rechain"' in dashboard
    assert 'k="counter break"' in dashboard
    assert 'k="counter hit/min"' in dashboard
    assert 'k="counter hit share"' in dashboard
    assert 'k="fresh sky"' in dashboard
    assert 'k="safe ckpt"' in dashboard
    assert 'k="safety"' in dashboard
    assert "safetyHealth" in dashboard
    assert 'case "fresh_sky_frac"' in health
    assert 'case "combo_s_tap_frac"' in health
    assert 'case "combo_z_tap_frac"' in health
    assert 'health={kvHealth("combo_z_tap_frac")}' in dashboard
    assert 'case "rechain_hit_frac"' in health
    assert 'case "counter_break_hit_frac"' in health
    assert 'case "counter_lane_break_hit_frac"' in health
    assert 'case "counter_lane_rechain_hit_frac"' in health
    assert 'case "counter_lane_hit_rate"' in health
    assert 'case "counter_lane_combo_max"' in health
    assert 'case "under_combo_counter_hit_rate"' in health
    assert 'case "under_combo_counter_hit_frac"' in health
    assert 'case "fresh_combo12_state"' in health
    assert 'case "fresh_chase_combo_max"' in health
    assert 'case "fresh_chase_combo12_state"' in health
    assert 'case "fresh_chase_hit_rate"' in health
    assert 'case "fresh_chase_trade_hit_frac"' in health
    assert 'case "fresh_spar_combo_max"' in health
    assert 'case "fresh_spar_combo12_state"' in health
    assert 'case "fresh_rehit_combo_max"' in health
    assert 'case "fresh_rehit_combo12_state"' in health
    assert 'case "fresh_pressure_combo_max"' in health
    assert 'case "fresh_pressure_combo12_state"' in health
    assert 'case "fresh_spar_hit_rate"' in health
    assert 'case "fresh_spar_trade_hit_frac"' in health

    live_page = Path("app/src/pages/Live.jsx").read_text(encoding="utf-8")
    assert "judas:app:live:v47" in live_page
    assert "judas:app:live:v46" not in live_page
    assert "judas:app:live:v43" not in live_page
    assert "judas:app:live:v41" not in live_page
    assert "judas:app:live:v38" not in live_page
    assert "judas:app:live:v37" not in live_page
    assert "judas:app:live:v31" not in live_page
    assert "judas:app:live:v30" not in live_page
    assert "judas:app:live:v29" not in live_page
    assert "judas:app:live:v28" not in live_page
    assert "judas:app:live:v27" not in live_page
    assert "judas:app:live:v24" not in live_page
    assert "judas:app:live:v22" not in live_page
    assert "judas:app:live:v21" not in live_page
    assert "judas:app:live:v20" not in live_page
    assert "counterAssist: false" in live_page
    assert "counter assist" in live_page
    assert "autoGapple: true" in live_page
    assert "auto gapple" in live_page
    assert "autoGappleCriticalHealth: 8" in live_page
    assert "autoGappleSafeDistance: 11.50" in live_page
    assert "autoGappleRetreat: true" in live_page
    assert "autoGappleRetreatDistance: 18" in live_page
    assert "autoGappleFastRetreat: true" in live_page
    assert "autoGappleRetreatHops: true" in live_page
    assert "autoGappleSprintHopHold: true" in live_page
    assert "autoGappleAvoidObstacles: true" in live_page
    assert "autoGappleRetreatStrafe: true" in live_page
    assert "autoGappleWallSlide: true" in live_page
    assert "autoGappleSpeedLock: true" in live_page
    assert "autoGappleVelocityAssist: true" in live_page
    assert "autoGappleSpeedFirst: true" in live_page
    assert "autoGappleFullSpeed: true" in live_page
    assert "autoGappleSpeedFloor: 4.50" in live_page
    assert "autoGappleMaxSpeed: 4.80" in live_page
    assert "autoGappleAccel: 5.50" in live_page
    assert "autoGappleStepAssist: true" in live_page
    assert "autoGappleStepHeight: 1.20" in live_page
    assert "autoGappleFallbackRetreat: true" in live_page
    assert "autoGappleRetreatInputLock: true" in live_page
    assert "autoGappleForceSprintRetreat: true" in live_page
    assert "autoGappleReleaseRetreatOnHit: true" in live_page
    assert "autoGappleCriticalRearmOnly: true" in live_page
    assert "autoGappleCriticalTrappedEat: true" in live_page
    assert "autoGappleRetreatTurnDeg: 360" in live_page
    assert "autoGappleEatingRetreatTurnDeg: 360" in live_page
    assert "autoGappleRetreatPathHoldTicks: 2" in live_page
    assert "autoGappleRetreatStuckAbortTicks: 4" in live_page
    assert "autoGappleRetreatMinTicks: 0" in live_page
    assert "autoGappleRetreatMaxTicks: 64" in live_page
    assert "autoGappleCriticalRetreatMaxTicks: 6" in live_page
    assert "autoGappleCriticalEatCommitTicks: 12" in live_page
    assert "autoGappleCombatRecoveryTicks: 6" in live_page
    assert "autoGappleRetreatStrafeHoldTicks: 5" in live_page
    assert "autoGappleRetreatObstacleJumpHoldTicks: 60" in live_page
    assert "autoGappleRetreatObstacleEscapeTicks: 120" in live_page
    assert "autoGappleRetreatPanicSpeed: true" in live_page
    assert "autoGappleRetreatObstacleLookahead: 24.00" in live_page
    assert "autoGappleSprintRetap: true" in live_page
    assert "autoGappleSprintRetapTicks: 2" in live_page
    assert "autoGappleAirControl: true" in live_page
    assert "autoGappleCriticalTrappedStuckTicks: 2" in live_page
    assert "critical hp" in live_page
    assert "safe dist" in live_page
    assert "retreat before gapple" in live_page
    assert "retreat dist" in live_page
    assert "fast retreat" in live_page
    assert "retreat hops" in live_page
    assert "hold sprint-hop" in live_page
    assert "avoid obstacles" in live_page
    assert "retreat strafe" in live_page
    assert "wall slide" in live_page
    assert "retreat speed lock" in live_page
    assert "retreat velocity assist" in live_page
    assert "speed-first retreat" in live_page
    assert "full-speed retreat" in live_page
    assert "speed floor" in live_page
    assert "max retreat speed" in live_page
    assert "retreat accel" in live_page
    assert "sprint retap" in live_page
    assert "air control" in live_page
    assert "step assist" in live_page
    assert "step height" in live_page
    assert "fallback retreat" in live_page
    assert "retreat input lock" in live_page
    assert "force sprint retreat" in live_page
    assert "release retreat on hit" in live_page
    assert "critical rearm only" in live_page
    assert "critical trapped eat" in live_page
    assert "retreat turn deg" in live_page
    assert "eat turn deg" in live_page
    assert "path hold ticks" in live_page
    assert "stuck abort ticks" in live_page
    assert "min retreat ticks" in live_page
    assert "max retreat ticks" in live_page
    assert "critical retreat ticks" in live_page
    assert "critical eat commit ticks" in live_page
    assert "combat recovery ticks" in live_page
    assert "strafe hold ticks" in live_page
    assert "obstacle jump hold ticks" in live_page
    assert "obstacle escape ticks" in live_page
    assert "trapped stuck ticks" in live_page
    assert "autoJump: false" in live_page
    assert "auto jump" in live_page
    assert "knockbackDump: false" in live_page
    assert "kb dump" in live_page
    assert "friendMode: false" in live_page
    assert "friend mode" in live_page
    assert "combo_god_recovery_kb092_combo12-safe_latest" in live_page
    assert "combo_god_leaderboard10_combo12-safe_latest" in live_page
    assert "combo_god_countertap96_combo12-safe_latest" in live_page
    assert "combo_god_directpad_lock_combo12-safe_latest" in live_page
    assert "export_fresh !== false" in live_page
    assert "usable.some((m) => m.path === path)" in live_page
    assert 'model: hasModel(cur.model) ? cur.model : (preferred?.path || ""),' in live_page
    assert "export_status" in live_page
    assert "blockedModels" in live_page
    assert "no schema 8 combo-safe export available" in live_page

    models_page = Path("app/src/pages/Models.jsx").read_text(encoding="utf-8")
    assert "safe_latest.pt" in models_page
    assert "isComboSafeRun" in models_page
    assert "canExportSafe" in models_page
    assert "safe_status" in models_page
    assert "export_status" in models_page
    assert "export_error" in models_page
    assert 'r.name === "combo_god_recovery_kb092_combo12"' in models_page
    assert 'r.name === "combo_god_leaderboard10_combo12"' in models_page
    assert 'r.name === "combo_god_countertap96_combo12"' in models_page
    assert 'r.name === "combo_god_directpad_lock_combo12")' in models_page
    assert "(r.safe || r.combo_safe)" in models_page
    assert "!isComboSafeRun(r) && r.best" in models_page
    assert "!isComboSafeRun(r) && r.checkpoints" in models_page

    viz = Path("viz/src/App.jsx").read_text(encoding="utf-8")
    assert "judas:viz:fighters:v4" in viz
    assert "judas:viz:params:v3" in viz
    assert "cps: 10, rot: 190, arena: 40" in viz
    assert "spawn_gap: 8, target: 50, sample: true" in viz
    assert "kb_h: 0.92, kb_v: 0.90, kb_idle: 0.6" in viz
    assert "aim_smooth: 0.02" in viz
    assert "isComboSafeRun" in viz
    assert "safe latest" in viz
    assert "combo_god_recovery_kb092_combo12-safe_latest" in viz
    assert "combo_god_leaderboard10_combo12-safe_latest" in viz
    assert "combo_god_countertap96_combo12-safe_latest" in viz
    assert "combo_god_directpad_lock_combo12-safe_latest" in viz
    assert "export_fresh === false" in viz
    assert "runs/combo_god_recovery_kb092_combo12/safe_latest.pt" in viz
    assert "runs/combo_god_leaderboard10_combo12/safe_latest.pt" in viz
    assert "runs/combo_god_countertap96_combo12/safe_latest.pt" in viz
    assert "runs/combo_god_directpad_lock_combo12/safe_latest.pt" in viz
    assert 'r.name === "combo_god_recovery_kb092_combo12"' in viz
    assert 'r.name === "combo_god_leaderboard10_combo12"' in viz
    assert 'r.name === "combo_god_countertap96_combo12"' in viz
    assert 'r.name === "combo_god_directpad_lock_combo12") && r.safe' in viz
    assert "continue;" in viz
    assert "combo_god_aggro/latest.pt" not in viz
    assert "combo_god_consistent/latest.pt" not in viz
    assert "__combo_pad__" in viz
    assert "__combo_spar__" in viz
    assert 'list.find((m) => m.path === "__combo_spar__")' in viz


def test_combo_god_process_scripts_are_single_run_guarded():
    start = Path("scripts/start_combo_god.ps1").read_text(encoding="utf-8")
    stop = Path("scripts/stop_combo_god.ps1").read_text(encoding="utf-8")
    status = Path("scripts/status_combo_god.ps1").read_text(encoding="utf-8")
    ui_stop = Path("scripts/stop_judas_ui.ps1").read_text(encoding="utf-8")
    train_stop = Path("scripts/stop_judas_train.ps1").read_text(encoding="utf-8")
    train_bat = Path("scripts/train.bat").read_text(encoding="utf-8")
    app_bat = Path("scripts/app.bat").read_text(encoding="utf-8")
    viz_bat = Path("scripts/viz.bat").read_text(encoding="utf-8")
    run_menu = Path("run.bat").read_text(encoding="utf-8")

    assert "Read-PidFile" in start
    assert "REMOVED_INVALID_PID_FILE" in start
    assert "[int]$Iters" in start
    assert "[int]$TimeoutMinutes = 20" in start
    assert "[int]$Seed = -1" in start
    assert "Seed must be >= 0, or -1 to use the config default." in start
    assert "config_seed_$stamp.json" in start
    assert "$cfgObj.seed = $Seed" in start
    assert "ConvertTo-Json -Depth 64" in start
    assert "seed=$Seed config=$cfg" in start
    assert "TimeoutMinutes must be >= 0" in start
    assert "--iters" in start
    assert "taskkill.exe /PID $ProcId /T /F" in start
    assert "TORCH_EXTENSIONS_DIR" in start
    assert "timeout_seconds=" in start
    assert "timeout_killed_pid=" in start
    assert "child_pid=" in start
    assert "env_bat=loaded" in start
    assert "cmd.exe /d /c" in start
    assert "[void]$child.WaitForExit()" in start
    assert "$child.Refresh()" in start
    assert "exit 124" in start
    assert "Start-Process -FilePath $pythonExe" in start
    assert "('    $child = Start-Process -FilePath $pythonExe" in start
    assert "Get-OwnedTrainingPids" in start
    assert "STOPPED_ORPHAN_TRAINING" in start
    assert "Get-AncestorPids" in start
    assert "scripts\\train_combo_god.bat" in start
    assert "train/configs/combo_god_recovery_kb092_combo12.json" in start
    assert "train/configs/combo_god_leaderboard10_combo12.json" in start
    assert 'combo_god_recovery_kb092_combo12' in stop
    assert 'combo_god_recovery_kb092_combo12' in status
    assert 'combo_god_leaderboard10_combo12' in stop
    assert 'combo_god_leaderboard10_combo12' in status
    assert 'combo_god_countertap96_combo12' in stop
    assert 'combo_god_countertap96_combo12' in status

    assert "Read-PidFile" in stop
    assert "process=invalid pid_file" in stop
    assert "taskkill.exe /PID $ProcId /T /F" in stop
    assert "Stop-OrphanTraining" in stop
    assert "orphan_training=stopped" in stop
    assert "orphan_training=none" in stop

    assert "Read-PidFile" in status
    assert "process=invalid pid_file" in status
    assert "Remove-Item -LiteralPath $pidFile" in status
    assert "orphan_training=running" in status
    assert "orphan_training=none" in status

    assert "Get-AncestorPids" in ui_stop
    assert "taskkill.exe /PID $ProcId /T /F" in ui_stop
    assert "scripts\\app.bat" in ui_stop
    assert "scripts\\viz.bat" in ui_stop
    assert "ui_process=stopped" in ui_stop
    assert "Get-AncestorPids" in train_stop
    assert "taskkill.exe /PID $ProcId /T /F" in train_stop
    assert "-m train.run" in train_stop
    assert "scripts\\train.bat" in train_stop
    assert "train_process=stopped" in train_stop
    assert "JUDAS_SKIP_TRAIN_STOP" in train_bat
    assert "stop_judas_train.bat" in train_bat
    assert train_bat.index("stop_judas_train.bat") < train_bat.index("python -m train.run")
    assert "JUDAS_SKIP_UI_STOP" in app_bat
    assert "stop_judas_ui.bat -Surface app" in app_bat
    assert app_bat.index("stop_judas_ui.bat -Surface app") < app_bat.index("npm run dev")
    assert "JUDAS_SKIP_UI_STOP" in viz_bat
    assert "stop_judas_ui.bat -Surface viz" in viz_bat
    assert viz_bat.index("stop_judas_ui.bat -Surface viz") < viz_bat.index("npm run dev")
    assert "[0] stop app/viz/live/train" in run_menu
    assert "stop_judas_ui.bat -Surface all" in run_menu
    assert run_menu.index('if "%c%"=="2"') < run_menu.index('start "judas-train"')
    assert run_menu.index("call scripts\\stop_judas_train.bat") < run_menu.index('start "judas-train"')
    assert "call scripts\\stop_combo_god.bat" in run_menu


def test_viz_reload_does_not_keep_stale_running_hud():
    viz = Path("viz/src/App.jsx").read_text(encoding="utf-8")
    arena3d = Path("viz/src/components/Arena3D.jsx").read_text(encoding="utf-8")

    assert "const [loading, setLoading] = useState(false);" in viz
    assert "const loadingRef = useRef(false);" in viz
    assert "const clearArenaFrame = () => {" in viz
    assert "resetSeq: 0" in viz
    assert "const resetSeq = (stateRef.current.resetSeq || 0) + 1;" in viz
    assert "stateRef.current = { cur: null, prev: null, tCur: 0, tPrev: 0, frame: 0, resetSeq };" in viz
    assert "setHud(null);" in viz
    assert "clearArenaFrame();" in viz
    assert "if (loadingRef.current) return;" in viz
    assert "loadingRef.current = true;" in viz
    assert "setStatus((cur) => (cur ? { ...cur, running: false, ready: false } : cur));" in viz
    assert "await api.arenaControl({ running: false });" in viz
    assert "loadingRef.current = false;" in viz
    assert "if (loading) return;" in viz
    assert "}, [speed, loading]);" in viz
    assert "const running = loading ? false : (status?.running ?? hud?.running);" in viz
    assert "disabled={!sel.a || !sel.b || loading}" in viz
    assert "disabled={loading || !(status?.ready)}" in viz
    assert "function clearDynamicVisuals()" in arena3d
    assert "let lastResetSeq = stateRef.current?.resetSeq ?? 0;" in arena3d
    assert "if (resetSeq !== lastResetSeq)" in arena3d
    assert "clearDynamicVisuals();" in arena3d
    assert "clearGroup(arenaGroup);" in arena3d

def test_ppo_value_clip_disabled_by_default():
    from train.ppo import PPOConfig

    assert PPOConfig().value_clip == 0.0


def test_evaluate_matches_act_logp():
    """Le logp d'evaluate() doit ÃƒÆ’Ã‚Âªtre identique ÃƒÆ’Ã‚Â  celui d'act() (mÃƒÆ’Ã‚Âªme action)."""
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
    m2 = tiny_trainer.train_iter()   # itÃƒÆ’Ã‚Â©ration 2 : league active (pool_every=1)
    for m in (m1, m2):
        assert np.isfinite(m["reward_mean"])
        assert np.isfinite(m["approx_kl"])
        assert "combo_hits" in m
        assert "combo_max" in m
        assert "combo_mean" in m
        assert "combo5_hits" in m
        assert "combo8_hits" in m
        assert "combo12_hits" in m
        assert "s_tap_frac" in m
        assert "z_tap_frac" in m
        assert "combo_tap_frac" in m
        assert "combo_s_tap_frac" in m
        assert "combo_z_tap_frac" in m
        assert "hit_wtap_frac" in m
        assert "rechain_hit_frac" in m
        assert "rechain_taken_frac" in m
        assert "counter_break_hit_frac" in m
        assert "counter_break_taken_frac" in m
        assert "escape_back_frac" in m
        assert "under_combo_counter_hit_rate" in m
        assert "under_combo_counter_hit_frac" in m
        assert "under_combo_hit_select_attack_frac" in m
        assert "under_combo_hit_select_clean_frac" in m
        assert "under_combo_hit_select_trade_frac" in m
        assert 0.0 <= m["s_tap_frac"] <= 1.0
        assert 0.0 <= m["z_tap_frac"] <= 1.0
        assert 0.0 <= m["combo_tap_frac"] <= 1.0
        assert 0.0 <= m["combo_s_tap_frac"] <= 1.0
        assert 0.0 <= m["combo_z_tap_frac"] <= 1.0
        assert 0.0 <= m["hit_wtap_frac"] <= 1.0
        assert 0.0 <= m["rechain_hit_frac"] <= 1.0
        assert 0.0 <= m["rechain_taken_frac"] <= 1.0
        assert 0.0 <= m["counter_break_hit_frac"] <= 1.0
        assert 0.0 <= m["counter_break_taken_frac"] <= 1.0
        assert 0.0 <= m["escape_back_frac"] <= 1.0
        assert m["under_combo_counter_hit_rate"] >= 0.0
        assert 0.0 <= m["under_combo_counter_hit_frac"] <= 1.0
        assert 0.0 <= m["under_combo_hit_select_attack_frac"] <= 1.0
        assert 0.0 <= m["under_combo_hit_select_clean_frac"] <= 1.0
        assert 0.0 <= m["under_combo_hit_select_trade_frac"] <= 1.0
        assert "mirror_combo12_hits" in m
        assert "all_combo12_hits" in m
        assert "bot_hit_rate" in m
        assert "bot_combo_max" in m
        assert "bot_sim_combo12_state" in m
        assert "spar_hit_rate" in m
        assert "spar_combo_max" in m
        assert "spar_sim_combo12_state" in m
        assert "rehit_hit_rate" in m
        assert "rehit_combo_max" in m
        assert "rehit_rechain_hit_frac" in m
        assert "rehit_sim_combo12_state" in m
        assert "pressure_hit_rate" in m
        assert "pressure_combo_max" in m
        assert "pressure_rechain_hit_frac" in m
        assert "pressure_sim_combo12_state" in m
        assert "combo_chase_hit_rate" in m
        assert "combo_chase_combo_max" in m
        assert "combo_chase_rechain_hit_frac" in m
        assert "combo_chase_counter_break_hit_frac" in m
        assert "combo_chase_sim_combo12_state" in m
        assert "counter_lane_hit_rate" in m
        assert "counter_lane_combo_max" in m
        assert "counter_lane_rechain_hit_frac" in m
        assert "counter_lane_break_hit_frac" in m
        assert "counter_lane_sim_combo12_state" in m
        assert m["counter_lane_hit_rate"] >= 0.0
        assert 0.0 <= m["counter_lane_rechain_hit_frac"] <= 1.0
        assert 0.0 <= m["counter_lane_break_hit_frac"] <= 1.0
    assert m2["pool_size"] >= 1
    assert (tiny_trainer.run_dir / "metrics.jsonl").exists()

def test_fresh_combo_eval_logs_reset_metrics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = {
        **TINY,
        "combo_eval_every": 1,
        "combo_eval_envs": 2,
        "combo_eval_ticks": 8,
        "combo_eval_counter": True,
        "combo_eval_counter_envs": 2,
        "combo_eval_counter_ticks": 8,
        "n_envs": 2,
        "rollout_ticks": 4,
    }
    t = Trainer(cfg, device="cpu")

    metrics = t.train_iter()

    assert metrics["fresh_combo_max"] >= 0
    assert 0.0 <= metrics["fresh_combo12_state"] <= 1.0
    assert 0.0 <= metrics["fresh_combo8_state"] <= 1.0
    assert metrics["fresh_hit_rate"] >= 0.0
    assert 0.0 <= metrics["fresh_rechain_hit_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_rechain_taken_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_counter_break_hit_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_counter_break_taken_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_sky_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_aim_body"] <= 1.0
    assert 0.0 <= metrics["fresh_s_tap_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_z_tap_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_combo_tap_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_combo_s_tap_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_combo_z_tap_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_hit_wtap_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_under_combo_frac"] <= 1.0
    assert metrics["fresh_under_combo_counter_hit_rate"] >= 0.0
    assert 0.0 <= metrics["fresh_under_combo_counter_hit_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_under_combo_hit_select_attack_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_under_combo_hit_select_clean_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_under_combo_hit_select_trade_frac"] <= 1.0
    assert metrics["fresh_chase_combo_max"] >= 0
    assert 0.0 <= metrics["fresh_chase_combo12_state"] <= 1.0
    assert 0.0 <= metrics["fresh_chase_combo8_state"] <= 1.0
    assert metrics["fresh_chase_hit_rate"] >= 0.0
    assert metrics["fresh_chase_taken_rate"] >= 0.0
    assert 0.0 <= metrics["fresh_chase_trade_hit_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_chase_rechain_hit_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_chase_rechain_taken_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_chase_counter_break_hit_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_chase_counter_break_taken_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_chase_close_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_chase_sky_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_chase_combo_tap_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_chase_combo_s_tap_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_chase_combo_z_tap_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_chase_hit_wtap_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_chase_under_combo_frac"] <= 1.0
    assert metrics["fresh_chase_under_combo_counter_hit_rate"] >= 0.0
    assert 0.0 <= metrics["fresh_chase_under_combo_counter_hit_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_chase_under_combo_hit_select_attack_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_chase_under_combo_hit_select_clean_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_chase_under_combo_hit_select_trade_frac"] <= 1.0
    assert metrics["fresh_spar_combo_max"] >= 0
    assert 0.0 <= metrics["fresh_spar_combo12_state"] <= 1.0
    assert 0.0 <= metrics["fresh_spar_combo8_state"] <= 1.0
    assert metrics["fresh_spar_hit_rate"] >= 0.0
    assert metrics["fresh_spar_taken_rate"] >= 0.0
    assert 0.0 <= metrics["fresh_spar_trade_hit_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_spar_rechain_hit_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_spar_rechain_taken_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_spar_counter_break_hit_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_spar_counter_break_taken_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_spar_close_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_spar_sky_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_spar_combo_tap_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_spar_combo_s_tap_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_spar_combo_z_tap_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_spar_hit_wtap_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_spar_under_combo_frac"] <= 1.0
    assert metrics["fresh_spar_under_combo_counter_hit_rate"] >= 0.0
    assert 0.0 <= metrics["fresh_spar_under_combo_counter_hit_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_spar_under_combo_hit_select_attack_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_spar_under_combo_hit_select_clean_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_spar_under_combo_hit_select_trade_frac"] <= 1.0
    assert metrics["fresh_counter_combo_max"] >= 0
    assert 0.0 <= metrics["fresh_counter_combo12_state"] <= 1.0
    assert 0.0 <= metrics["fresh_counter_combo8_state"] <= 1.0
    assert metrics["fresh_counter_hit_rate"] >= 0.0
    assert metrics["fresh_counter_taken_rate"] >= 0.0
    assert 0.0 <= metrics["fresh_counter_trade_hit_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_counter_rechain_hit_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_counter_rechain_taken_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_counter_counter_break_hit_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_counter_counter_break_taken_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_counter_close_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_counter_sky_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_counter_combo_tap_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_counter_combo_s_tap_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_counter_combo_z_tap_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_counter_hit_wtap_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_counter_under_combo_frac"] <= 1.0
    assert metrics["fresh_counter_under_combo_counter_hit_rate"] >= 0.0
    assert 0.0 <= metrics["fresh_counter_under_combo_counter_hit_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_counter_under_combo_hit_select_attack_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_counter_under_combo_hit_select_clean_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_counter_under_combo_hit_select_trade_frac"] <= 1.0
    assert metrics["fresh_rehit_combo_max"] >= 0
    assert 0.0 <= metrics["fresh_rehit_combo12_state"] <= 1.0
    assert 0.0 <= metrics["fresh_rehit_combo8_state"] <= 1.0
    assert metrics["fresh_rehit_hit_rate"] >= 0.0
    assert metrics["fresh_rehit_taken_rate"] >= 0.0
    assert 0.0 <= metrics["fresh_rehit_trade_hit_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_rehit_rechain_hit_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_rehit_rechain_taken_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_rehit_counter_break_hit_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_rehit_counter_break_taken_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_rehit_close_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_rehit_sky_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_rehit_combo_tap_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_rehit_combo_s_tap_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_rehit_combo_z_tap_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_rehit_hit_wtap_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_rehit_under_combo_frac"] <= 1.0
    assert metrics["fresh_rehit_under_combo_counter_hit_rate"] >= 0.0
    assert 0.0 <= metrics["fresh_rehit_under_combo_counter_hit_frac"] <= 1.0
    assert metrics["fresh_pressure_combo_max"] >= 0
    assert 0.0 <= metrics["fresh_pressure_combo12_state"] <= 1.0
    assert 0.0 <= metrics["fresh_pressure_combo8_state"] <= 1.0
    assert metrics["fresh_pressure_hit_rate"] >= 0.0
    assert metrics["fresh_pressure_taken_rate"] >= 0.0
    assert 0.0 <= metrics["fresh_pressure_trade_hit_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_pressure_rechain_hit_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_pressure_rechain_taken_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_pressure_counter_break_hit_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_pressure_counter_break_taken_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_pressure_close_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_pressure_sky_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_pressure_combo_tap_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_pressure_combo_s_tap_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_pressure_combo_z_tap_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_pressure_hit_wtap_frac"] <= 1.0
    assert 0.0 <= metrics["fresh_pressure_under_combo_frac"] <= 1.0
    assert metrics["fresh_pressure_under_combo_counter_hit_rate"] >= 0.0
    assert 0.0 <= metrics["fresh_pressure_under_combo_counter_hit_frac"] <= 1.0


def test_hit_wtap_metrics_exclude_under_combo_recovery_hits():
    src = Path("train/run.py").read_text(encoding="utf-8")

    assert "hit_wtap_denom = hits_mask & (~under_combo)" in src
    assert "hit_wtap = hit_wtap_denom & z_tap" in src
    assert "/ hit_wtap_denom.float().sum().clamp(min=1.0)" in src
    assert "non_counter_dealt = dealt_obs & (~under_combo)" in src
    assert "non_counter_dealt = dealt & (~under_combo)" in src
    assert "hit_wtap_sum / max(hit_wtap_denom_sum, 1.0)" in src
    assert "hit_wtap_sum / max(dealt_sum, 1.0)" not in src
    assert "hit_wtap_sum / hit_denom" not in src


def test_combo_rollout_stats_counts_clean_12_chain():
    from train.run import _combo_rollout_stats

    dealt = torch.zeros(16, 2, dtype=torch.bool)
    taken = torch.zeros(16, 2, dtype=torch.bool)
    done = torch.zeros(16, 2, dtype=torch.bool)
    learner = torch.tensor([True, False])

    dealt[:, 0] = True
    dealt[4, 1] = True
    taken[4, 0] = True              # trade/hurt breaks player 0's chain

    stats = _combo_rollout_stats(dealt, taken, done, learner, window=25,
                                 threshold=12)
    assert stats["combo_max"] == 11
    assert stats["combo5_hits"] > 0.0
    assert stats["combo8_hits"] > 0.0
    assert stats["combo_mean"] > 1.0
    assert stats["combo12_hits"] == 0.0

    taken.zero_()
    stats = _combo_rollout_stats(dealt, taken, done, learner, window=25,
                                 threshold=12)
    assert stats["combo_max"] == 16
    assert stats["combo12_hits"] > 0.0


def test_chain_followup_stats_counts_rehit_and_counter_breaks():
    start = torch.zeros(8, 2, dtype=torch.bool)
    response = torch.zeros(8, 2, dtype=torch.bool)
    blocker = torch.zeros(8, 2, dtype=torch.bool)
    done = torch.zeros(8, 2, dtype=torch.bool)
    learner = torch.tensor([True, False])

    start[0, 0] = True
    response[2, 0] = True
    start[2, 0] = True
    blocker[4, 0] = True

    stats = _chain_followup_stats(start, response, blocker, done, learner,
                                  window=3)

    assert stats["opps"] == 2
    assert stats["hit_frac"] == 0.5
    assert stats["taken_frac"] == 0.5
    assert stats["miss_frac"] == 0.0

    counter = _chain_followup_stats(
        blocker, response, blocker, done, learner, window=3)

    assert counter["opps"] == 1
    assert counter["hit_frac"] == 0.0
    assert counter["taken_frac"] == 0.0




def test_safety_waits_for_fresh_combo_eval_before_marking_safe(tiny_trainer):
    tiny_trainer.cfg["safety_stop_on_regression"] = True
    tiny_trainer.cfg["combo_eval_every"] = 8
    tiny_trainer.iter = 1
    (tiny_trainer.run_dir / "ckpt_000001.pt").write_bytes(b"no-fresh")

    stopped = tiny_trainer._checkpoint_safety_guard({
        "iter": 1,
        "hit_rate": 80.0,
        "sky_frac": 0.0,
        "under_combo_escape_frac": 0.0,
        "back_frac": 0.0,
        "escape_back_frac": 0.0,
        "strafe_frac": 0.6,
        "opener_strafe_frac": 0.8,
        "sim_combo12_state": 0.9,
        "sim_combo_max": 200,
    })

    assert stopped is False
    assert not (tiny_trainer.run_dir / "safe_latest.pt").exists()


def test_combo_eval_schedule_counts_from_resume_origin(tiny_trainer):
    tiny_trainer.cfg["combo_eval_every"] = 16
    tiny_trainer.iter = 568
    tiny_trainer._combo_eval_origin = tiny_trainer.iter

    tiny_trainer.iter = 576
    assert tiny_trainer._should_combo_eval_now() is False

    tiny_trainer.iter = 584
    assert tiny_trainer._should_combo_eval_now() is True

    tiny_trainer.cfg["combo_eval_every"] = 0
    assert tiny_trainer._should_combo_eval_now() is False


def test_bounded_train_forces_final_fresh_combo_eval_if_unaligned(tiny_trainer):
    tiny_trainer.cfg["combo_eval_every"] = 16
    calls = []

    def fake_train_iter(force_combo_eval=False):
        calls.append(force_combo_eval)
        tiny_trainer.iter += 1
        return {
            "iter": tiny_trainer.iter,
            "sps": 1,
            "reward_mean": 0.0,
            "elo": 1000.0,
            "league_winrate": 0.0,
            "approx_kl": 0.0,
        }

    tiny_trainer.train_iter = fake_train_iter
    tiny_trainer.train(iters=3)

    assert calls == [False, False, True]


def test_bounded_train_does_not_duplicate_final_fresh_eval(tiny_trainer):
    tiny_trainer.cfg["combo_eval_every"] = 16
    calls = []

    def fake_train_iter(force_combo_eval=False):
        calls.append(force_combo_eval)
        tiny_trainer.iter += 1
        if len(calls) == 2:
            tiny_trainer._fresh_combo_eval_count += 1
        return {
            "iter": tiny_trainer.iter,
            "sps": 1,
            "reward_mean": 0.0,
            "elo": 1000.0,
            "league_winrate": 0.0,
            "approx_kl": 0.0,
        }

    tiny_trainer.train_iter = fake_train_iter
    tiny_trainer.train(iters=3)

    assert calls == [False, False, False]


def test_safety_uses_separate_fresh_hit_threshold(tiny_trainer):
    tiny_trainer.cfg["safety_stop_on_regression"] = True
    tiny_trainer.cfg["combo_eval_every"] = 1
    tiny_trainer.cfg["safety_min_hit_rate"] = 45.0
    tiny_trainer.cfg["safety_fresh_min_hit_rate"] = 10.0
    tiny_trainer.iter = 1
    (tiny_trainer.run_dir / "ckpt_000001.pt").write_bytes(b"fresh-ok")

    ok_metrics = {
        "iter": 1,
        "hit_rate": 80.0,
        "fresh_hit_rate": 15.0,
        "fresh_sky_frac": 0.0,
        "fresh_combo12_state": 0.1,
        "fresh_combo_max": 28,
        "fresh_combo8_state": 0.13,
        "under_combo_escape_frac": 0.0,
        "back_frac": 0.0,
        "escape_back_frac": 0.0,
        "strafe_frac": 0.6,
        "opener_strafe_frac": 0.8,
    }
    assert tiny_trainer._checkpoint_safety_guard(ok_metrics) is False
    assert ok_metrics["safety_state"] == "safe"

    tiny_trainer.iter = 2
    (tiny_trainer.run_dir / "ckpt_000002.pt").write_bytes(b"fresh-bad")
    bad_metrics = {**ok_metrics, "iter": 2, "fresh_hit_rate": 5.0}
    assert tiny_trainer._checkpoint_safety_guard(bad_metrics) is True
    assert "fresh_hit_rate" in bad_metrics["safety_reason"]


def test_safety_uses_active_fresh_hit_when_available(tiny_trainer):
    tiny_trainer.cfg["safety_stop_on_regression"] = True
    tiny_trainer.cfg["combo_eval_every"] = 1
    tiny_trainer.cfg["safety_min_hit_rate"] = 45.0
    tiny_trainer.cfg["safety_fresh_min_hit_rate"] = 10.0
    tiny_trainer.iter = 1
    (tiny_trainer.run_dir / "ckpt_000001.pt").write_bytes(b"active-ok")

    metrics = {
        "iter": 1,
        "hit_rate": 80.0,
        "fresh_hit_rate": 0.0,
        "fresh_spar_hit_rate": 15.0,
        "fresh_spar_combo12_state": 0.1,
        "fresh_spar_combo_max": 28,
        "fresh_spar_combo8_state": 0.13,
        "fresh_spar_sky_frac": 0.0,
        "fresh_spar_combo_s_tap_frac": 1.0,
        "fresh_spar_under_combo_counter_hit_frac": 0.5,
        "fresh_sky_frac": 0.0,
        "fresh_combo12_state": 0.0,
        "fresh_combo_max": 0,
        "fresh_combo8_state": 0.0,
        "under_combo_escape_frac": 0.0,
        "back_frac": 0.0,
        "escape_back_frac": 0.0,
        "strafe_frac": 0.6,
        "opener_strafe_frac": 0.8,
    }

    assert tiny_trainer._checkpoint_safety_guard(metrics) is False
    assert metrics["safety_state"] == "safe"


def test_safety_back_limit_blocks_any_back_input(tiny_trainer):
    tiny_trainer.cfg["safety_stop_on_regression"] = True
    tiny_trainer.cfg["combo_eval_every"] = 1
    tiny_trainer.cfg["safety_back_frac"] = 0.002
    tiny_trainer.iter = 1
    (tiny_trainer.run_dir / "ckpt_000001.pt").write_bytes(b"no-back-ok")

    ok_metrics = {
        "iter": 1,
        "hit_rate": 80.0,
        "fresh_hit_rate": 15.0,
        "fresh_sky_frac": 0.0,
        "fresh_combo12_state": 0.1,
        "fresh_combo_max": 28,
        "fresh_combo8_state": 0.13,
        "under_combo_escape_frac": 0.0,
        "back_frac": 0.0,
        "escape_back_frac": 0.0,
        "strafe_frac": 0.6,
        "opener_strafe_frac": 0.8,
    }
    assert tiny_trainer._checkpoint_safety_guard(ok_metrics) is False
    assert ok_metrics["safety_state"] == "safe"

    tiny_trainer.iter = 2
    (tiny_trainer.run_dir / "ckpt_000002.pt").write_bytes(b"back-bad")
    bad_metrics = {**ok_metrics, "iter": 2, "back_frac": 0.01}
    assert tiny_trainer._checkpoint_safety_guard(bad_metrics) is True
    assert "back_frac" in bad_metrics["safety_reason"]


def test_safety_strafe_gate_blocks_straightline_policy(tiny_trainer):
    tiny_trainer.cfg["safety_stop_on_regression"] = True
    tiny_trainer.cfg["combo_eval_every"] = 1
    tiny_trainer.cfg["safety_min_strafe_frac"] = 0.50
    tiny_trainer.iter = 1
    (tiny_trainer.run_dir / "ckpt_000001.pt").write_bytes(b"straightline")

    metrics = {
        "iter": 1,
        "hit_rate": 80.0,
        "fresh_hit_rate": 15.0,
        "fresh_sky_frac": 0.0,
        "fresh_combo12_state": 0.1,
        "fresh_combo_max": 28,
        "fresh_combo8_state": 0.13,
        "under_combo_escape_frac": 0.0,
        "back_frac": 0.0,
        "escape_back_frac": 0.0,
        "strafe_frac": 0.1,
        "opener_strafe_frac": 0.8,
    }

    assert tiny_trainer._checkpoint_safety_guard(metrics) is True
    assert metrics["safety_state"] == "stop"
    assert "strafe_frac=0.1<0.5" in metrics["safety_reason"]


def test_safety_opener_strafe_gate_blocks_late_only_strafe(tiny_trainer):
    tiny_trainer.cfg["safety_stop_on_regression"] = True
    tiny_trainer.cfg["combo_eval_every"] = 1
    tiny_trainer.cfg["safety_min_strafe_frac"] = 0.50
    tiny_trainer.cfg["safety_min_opener_strafe_frac"] = 0.75
    tiny_trainer.iter = 1
    (tiny_trainer.run_dir / "ckpt_000001.pt").write_bytes(b"late-strafe")

    metrics = {
        "iter": 1,
        "hit_rate": 80.0,
        "fresh_hit_rate": 15.0,
        "fresh_sky_frac": 0.0,
        "fresh_combo12_state": 0.1,
        "fresh_combo_max": 28,
        "fresh_combo8_state": 0.13,
        "under_combo_escape_frac": 0.0,
        "back_frac": 0.0,
        "escape_back_frac": 0.0,
        "strafe_frac": 0.8,
        "opener_strafe_frac": 0.2,
    }

    assert tiny_trainer._checkpoint_safety_guard(metrics) is True
    assert metrics["safety_state"] == "stop"
    assert "opener_strafe_frac=0.2<0.75" in metrics["safety_reason"]


def test_safety_opener_pressure_gate_blocks_passive_lateral_openers(tiny_trainer):
    tiny_trainer.cfg["safety_stop_on_regression"] = True
    tiny_trainer.cfg["combo_eval_every"] = 1
    tiny_trainer.cfg["safety_min_opener_strafe_frac"] = 0.75
    tiny_trainer.cfg["safety_min_opener_pressure_frac"] = 0.60
    tiny_trainer.iter = 1
    (tiny_trainer.run_dir / "ckpt_000001.pt").write_bytes(b"passive-opener")

    metrics = {
        "iter": 1,
        "hit_rate": 80.0,
        "fresh_hit_rate": 15.0,
        "fresh_sky_frac": 0.0,
        "fresh_combo12_state": 0.1,
        "fresh_combo_max": 28,
        "fresh_combo8_state": 0.13,
        "under_combo_escape_frac": 0.0,
        "back_frac": 0.0,
        "escape_back_frac": 0.0,
        "strafe_frac": 0.8,
        "opener_strafe_frac": 0.9,
        "opener_pressure_frac": 0.25,
    }

    assert tiny_trainer._checkpoint_safety_guard(metrics) is True
    assert metrics["safety_state"] == "stop"
    assert "opener_pressure_frac=0.25<0.6" in metrics["safety_reason"]


def test_safety_opener_hold_gate_blocks_jitter_strafe_openers(tiny_trainer):
    tiny_trainer.cfg["safety_stop_on_regression"] = True
    tiny_trainer.cfg["combo_eval_every"] = 1
    tiny_trainer.cfg["safety_min_opener_strafe_frac"] = 0.75
    tiny_trainer.cfg["safety_min_opener_strafe_hold_frac"] = 0.70
    tiny_trainer.iter = 1
    (tiny_trainer.run_dir / "ckpt_000001.pt").write_bytes(b"jitter-opener")

    metrics = {
        "iter": 1,
        "hit_rate": 80.0,
        "fresh_hit_rate": 15.0,
        "fresh_sky_frac": 0.0,
        "fresh_combo12_state": 0.1,
        "fresh_combo_max": 28,
        "fresh_combo8_state": 0.13,
        "under_combo_escape_frac": 0.0,
        "back_frac": 0.0,
        "escape_back_frac": 0.0,
        "strafe_frac": 0.8,
        "opener_strafe_frac": 0.9,
        "opener_strafe_hold_frac": 0.45,
        "opener_pressure_frac": 0.8,
    }

    assert tiny_trainer._checkpoint_safety_guard(metrics) is True
    assert metrics["safety_state"] == "stop"
    assert "opener_strafe_hold_frac=0.45<0.7" in metrics["safety_reason"]


def test_safety_combo_tap_gate_blocks_strafe_only_combos(tiny_trainer):
    tiny_trainer.cfg["safety_stop_on_regression"] = True
    tiny_trainer.cfg["combo_eval_every"] = 1
    tiny_trainer.cfg["safety_min_combo_tap_frac"] = 0.12
    tiny_trainer.iter = 1
    (tiny_trainer.run_dir / "ckpt_000001.pt").write_bytes(b"no-tap-combo")

    metrics = {
        "iter": 1,
        "hit_rate": 80.0,
        "fresh_hit_rate": 15.0,
        "fresh_sky_frac": 0.0,
        "fresh_combo12_state": 0.1,
        "fresh_combo_max": 28,
        "fresh_combo8_state": 0.13,
        "fresh_combo_tap_frac": 0.02,
        "under_combo_escape_frac": 0.0,
        "back_frac": 0.0,
        "escape_back_frac": 0.0,
        "strafe_frac": 0.8,
        "opener_strafe_frac": 0.8,
    }

    assert tiny_trainer._checkpoint_safety_guard(metrics) is True
    assert metrics["safety_state"] == "stop"
    assert "fresh_combo_tap_frac=0.02<0.12" in metrics["safety_reason"]


def test_safety_combo_tap_gate_requires_z_tap_not_s_tap(tiny_trainer):
    tiny_trainer.cfg["safety_stop_on_regression"] = True
    tiny_trainer.cfg["combo_eval_every"] = 1
    tiny_trainer.cfg["safety_min_combo_tap_frac"] = 0.12
    tiny_trainer.cfg["safety_min_combo_z_tap_frac"] = 0.10
    tiny_trainer.cfg["safety_max_combo_s_tap_frac"] = 0.02
    tiny_trainer.iter = 1
    (tiny_trainer.run_dir / "ckpt_000001.pt").write_bytes(b"s-tap-combo")

    metrics = {
        "iter": 1,
        "hit_rate": 80.0,
        "fresh_hit_rate": 15.0,
        "fresh_sky_frac": 0.0,
        "fresh_combo12_state": 0.1,
        "fresh_combo_max": 28,
        "fresh_combo8_state": 0.13,
        "fresh_combo_tap_frac": 0.18,
        "fresh_combo_s_tap_frac": 0.16,
        "fresh_combo_z_tap_frac": 0.02,
        "under_combo_escape_frac": 0.0,
        "back_frac": 0.0,
        "escape_back_frac": 0.0,
        "strafe_frac": 0.8,
        "opener_strafe_frac": 0.8,
    }

    assert tiny_trainer._checkpoint_safety_guard(metrics) is True
    assert metrics["safety_state"] == "stop"
    assert "fresh_combo_z_tap_frac=0.02<0.1" in metrics["safety_reason"]
    assert "fresh_combo_s_tap_frac=0.16>0.02" in metrics["safety_reason"]


def test_safety_hit_wtap_gate_blocks_hits_without_release(tiny_trainer):
    tiny_trainer.cfg["safety_stop_on_regression"] = True
    tiny_trainer.cfg["combo_eval_every"] = 1
    tiny_trainer.cfg["safety_min_hit_wtap_frac"] = 0.75
    tiny_trainer.iter = 1
    (tiny_trainer.run_dir / "ckpt_000001.pt").write_bytes(b"no-hit-wtap")

    metrics = {
        "iter": 1,
        "hit_rate": 80.0,
        "fresh_hit_rate": 15.0,
        "fresh_sky_frac": 0.0,
        "fresh_combo12_state": 0.1,
        "fresh_combo_max": 28,
        "fresh_combo8_state": 0.13,
        "fresh_hit_wtap_frac": 0.20,
        "under_combo_escape_frac": 0.0,
        "back_frac": 0.0,
        "escape_back_frac": 0.0,
        "strafe_frac": 0.8,
        "opener_strafe_frac": 0.8,
        "opener_pressure_frac": 0.8,
    }

    assert tiny_trainer._checkpoint_safety_guard(metrics) is True
    assert metrics["safety_state"] == "stop"
    assert "fresh_hit_wtap_frac=0.2<0.75" in metrics["safety_reason"]


def test_safety_hit_wtap_can_block_promotion_without_stopping_training(tiny_trainer):
    tiny_trainer.cfg["safety_stop_on_regression"] = True
    tiny_trainer.cfg["combo_eval_every"] = 1
    tiny_trainer.cfg["safety_min_hit_wtap_frac"] = 0.40
    tiny_trainer.cfg["safety_min_chase_hit_wtap_frac"] = 0.33
    tiny_trainer.cfg["safety_hit_wtap_blocks_promotion"] = True
    tiny_trainer.iter = 1
    (tiny_trainer.run_dir / "ckpt_000001.pt").write_bytes(b"style-pending")

    metrics = {
        "iter": 1,
        "hit_rate": 80.0,
        "fresh_spar_hit_rate": 83.0,
        "fresh_spar_sky_frac": 0.0,
        "fresh_spar_combo12_state": 0.40,
        "fresh_spar_combo_max": 34,
        "fresh_spar_combo8_state": 0.45,
        "fresh_spar_hit_wtap_frac": 0.0293,
        "fresh_chase_hit_rate": 79.0,
        "fresh_chase_sky_frac": 0.0,
        "fresh_chase_combo12_state": 0.10,
        "fresh_chase_combo_max": 34,
        "fresh_chase_combo8_state": 0.45,
        "fresh_chase_hit_wtap_frac": 0.0227,
        "under_combo_escape_frac": 0.0,
        "back_frac": 0.0,
        "escape_back_frac": 0.0,
        "strafe_frac": 0.8,
        "opener_samples": 0.0,
    }

    assert tiny_trainer._checkpoint_safety_guard(metrics) is False
    assert metrics["safety_state"] == "training"
    assert metrics["safety_checkpoint"] == "await_hit_wtap"
    assert "fresh_spar_hit_wtap_frac=0.0293<0.4" in metrics["safety_promotion_reason"]
    assert "fresh_chase_hit_wtap_frac=0.0227<0.33" in metrics["safety_promotion_reason"]
    assert "safety_reason" not in metrics


def test_safety_hit_wtap_slack_only_applies_to_opener_rollouts(tiny_trainer):
    tiny_trainer.cfg["safety_stop_on_regression"] = True
    tiny_trainer.cfg["combo_eval_every"] = 1
    tiny_trainer.cfg["safety_min_hit_wtap_frac"] = 0.40
    tiny_trainer.cfg["safety_rollout_hit_wtap_slack"] = 0.025
    tiny_trainer.iter = 1

    rollout_metrics = {
        "iter": 1,
        "hit_rate": 80.0,
        "sky_frac": 0.0,
        "hit_wtap_frac": 0.3935,
        "opener_samples": 100.0,
        "under_combo_escape_frac": 0.0,
        "back_frac": 0.0,
        "escape_back_frac": 0.0,
        "strafe_frac": 0.8,
        "opener_strafe_frac": 0.8,
        "opener_strafe_hold_frac": 0.8,
        "opener_pressure_frac": 0.8,
    }

    assert tiny_trainer._checkpoint_safety_guard(rollout_metrics) is False
    assert rollout_metrics["safety_state"] == "safe"
    assert "safety_reason" not in rollout_metrics

    no_opener_metrics = {
        **rollout_metrics,
        "hit_wtap_frac": 0.01,
        "opener_samples": 0.0,
    }
    assert tiny_trainer._checkpoint_safety_guard(no_opener_metrics) is False
    assert no_opener_metrics["safety_state"] == "safe"
    assert "safety_reason" not in no_opener_metrics

    fresh_metrics = {
        "iter": 1,
        "hit_rate": 80.0,
        "fresh_hit_rate": 80.0,
        "fresh_sky_frac": 0.0,
        "fresh_combo12_state": 0.1,
        "fresh_combo_max": 28,
        "fresh_combo8_state": 0.13,
        "fresh_hit_wtap_frac": 0.3935,
        "opener_samples": 100.0,
        "under_combo_escape_frac": 0.0,
        "back_frac": 0.0,
        "escape_back_frac": 0.0,
        "strafe_frac": 0.8,
        "opener_strafe_frac": 0.8,
        "opener_pressure_frac": 0.8,
    }

    assert tiny_trainer._checkpoint_safety_guard(fresh_metrics) is True
    assert fresh_metrics["safety_state"] == "stop"
    assert "fresh_hit_wtap_frac=0.3935<0.4" in fresh_metrics["safety_reason"]


def test_safety_chase_hit_wtap_has_own_floor(tiny_trainer):
    tiny_trainer.cfg["safety_stop_on_regression"] = True
    tiny_trainer.cfg["combo_eval_every"] = 1
    tiny_trainer.cfg["safety_min_hit_wtap_frac"] = 0.40
    tiny_trainer.cfg["safety_min_chase_hit_wtap_frac"] = 0.33
    tiny_trainer.iter = 1

    metrics = {
        "iter": 1,
        "hit_rate": 80.0,
        "fresh_chase_hit_rate": 80.0,
        "fresh_chase_sky_frac": 0.0,
        "fresh_chase_combo12_state": 0.1,
        "fresh_chase_combo_max": 28,
        "fresh_chase_combo8_state": 0.13,
        "fresh_chase_hit_wtap_frac": 0.3585,
        "under_combo_escape_frac": 0.0,
        "back_frac": 0.0,
        "escape_back_frac": 0.0,
        "strafe_frac": 0.8,
        "opener_samples": 0.0,
    }

    assert tiny_trainer._checkpoint_safety_guard(metrics) is False
    assert metrics["safety_state"] == "safe"

    metrics["fresh_chase_hit_wtap_frac"] = 0.30
    assert tiny_trainer._checkpoint_safety_guard(metrics) is True
    assert "fresh_chase_hit_wtap_frac=0.3<0.33" in metrics["safety_reason"]


def test_safety_under_combo_counter_gate_blocks_no_counter_hits(tiny_trainer):
    tiny_trainer.cfg["safety_stop_on_regression"] = True
    tiny_trainer.cfg["combo_eval_every"] = 1
    tiny_trainer.cfg["safety_min_under_combo_counter_hit_frac"] = 0.05
    tiny_trainer.iter = 1
    (tiny_trainer.run_dir / "ckpt_000001.pt").write_bytes(b"no-counter")

    metrics = {
        "iter": 1,
        "hit_rate": 80.0,
        "fresh_hit_rate": 15.0,
        "fresh_sky_frac": 0.0,
        "fresh_combo12_state": 0.1,
        "fresh_combo_max": 28,
        "fresh_combo8_state": 0.13,
        "fresh_under_combo_counter_hit_frac": 0.0,
        "under_combo_escape_frac": 0.0,
        "back_frac": 0.0,
        "escape_back_frac": 0.0,
        "strafe_frac": 0.8,
        "opener_strafe_frac": 0.8,
        "opener_pressure_frac": 0.8,
    }

    assert tiny_trainer._checkpoint_safety_guard(metrics) is True
    assert metrics["safety_state"] == "stop"
    assert "fresh_under_combo_counter_hit_frac=0<0.05" in metrics["safety_reason"]


def test_safety_under_combo_counter_gate_prefers_active_fresh_over_rollout(tiny_trainer):
    metrics = {
        "under_combo_counter_hit_frac": 0.25,
        "fresh_spar_under_combo_counter_hit_frac": 0.08,
        "fresh_chase_under_combo_counter_hit_frac": 0.12,
    }

    assert tiny_trainer._safety_under_combo_counter_values(metrics) == [
        ("fresh_spar_under_combo_counter_hit_frac", 0.08),
        ("fresh_chase_under_combo_counter_hit_frac", 0.12),
    ]


def test_safety_under_combo_counter_gate_includes_counter_recovery(tiny_trainer):
    tiny_trainer.cfg["safety_require_counter_recovery"] = True
    metrics = {
        "under_combo_counter_hit_frac": 0.25,
        "fresh_spar_under_combo_counter_hit_frac": 0.08,
        "fresh_chase_under_combo_counter_hit_frac": 0.12,
        "fresh_counter_under_combo_counter_hit_frac": 0.04,
    }

    assert tiny_trainer._safety_under_combo_counter_values(metrics) == [
        ("fresh_spar_under_combo_counter_hit_frac", 0.08),
        ("fresh_chase_under_combo_counter_hit_frac", 0.12),
        ("fresh_counter_under_combo_counter_hit_frac", 0.04),
    ]


def test_safety_under_combo_counter_gate_uses_rollout_without_fresh(tiny_trainer):
    metrics = {
        "under_combo_counter_hit_frac": 0.25,
    }

    assert tiny_trainer._safety_under_combo_counter_values(metrics) == [
        ("under_combo_counter_hit_frac", 0.25),
    ]


def test_safety_under_combo_hit_select_gate_blocks_dirty_recovery(tiny_trainer):
    tiny_trainer.cfg["safety_stop_on_regression"] = True
    tiny_trainer.cfg["safety_under_combo_escape"] = 1.0
    tiny_trainer.cfg["safety_back_frac"] = 1.0
    tiny_trainer.cfg["safety_min_strafe_frac"] = -1.0
    tiny_trainer.cfg["safety_min_combo_tap_frac"] = -1.0
    tiny_trainer.cfg["safety_min_combo_z_tap_frac"] = -1.0
    tiny_trainer.cfg["safety_max_combo_s_tap_frac"] = -1.0
    tiny_trainer.cfg["safety_min_hit_wtap_frac"] = -1.0
    tiny_trainer.cfg["safety_min_under_combo_counter_hit_frac"] = -1.0
    tiny_trainer.cfg["safety_min_under_combo_hit_select_clean_frac"] = 0.20
    tiny_trainer.cfg["safety_max_under_combo_hit_select_trade_frac"] = 0.12
    metrics = {
        "under_combo_escape_frac": 0.0,
        "back_frac": 0.0,
        "opener_samples": 0.0,
        "fresh_spar_under_combo_hit_select_clean_frac": 0.19,
        "fresh_spar_under_combo_hit_select_trade_frac": 0.05,
        "fresh_chase_under_combo_hit_select_clean_frac": 0.25,
        "fresh_chase_under_combo_hit_select_trade_frac": 0.13,
    }

    assert tiny_trainer._checkpoint_safety_guard(metrics) is True
    assert metrics["safety_state"] == "stop"
    assert "fresh_spar_under_combo_hit_select_clean_frac=0.19<0.2" in metrics["safety_reason"]
    assert "fresh_chase_under_combo_hit_select_trade_frac=0.13>0.12" in metrics["safety_reason"]


def test_safety_under_combo_hit_select_gate_ignores_rollout_noise(tiny_trainer):
    tiny_trainer.cfg["safety_min_under_combo_hit_select_clean_frac"] = 0.20
    tiny_trainer.cfg["safety_max_under_combo_hit_select_trade_frac"] = 0.12
    metrics = {
        "under_combo_hit_select_clean_frac": 0.01,
        "under_combo_hit_select_trade_frac": 0.90,
    }

    assert tiny_trainer._safety_under_combo_hit_select_values(metrics, "clean") == []
    assert tiny_trainer._safety_under_combo_hit_select_values(metrics, "trade") == []


def test_safety_under_combo_hit_select_gate_accepts_low_exposure_avoidance(tiny_trainer):
    tiny_trainer.cfg["safety_stop_on_regression"] = True
    tiny_trainer.cfg["safety_under_combo_escape"] = 1.0
    tiny_trainer.cfg["safety_back_frac"] = 1.0
    tiny_trainer.cfg["safety_min_strafe_frac"] = -1.0
    tiny_trainer.cfg["safety_min_combo_tap_frac"] = -1.0
    tiny_trainer.cfg["safety_min_combo_z_tap_frac"] = -1.0
    tiny_trainer.cfg["safety_max_combo_s_tap_frac"] = -1.0
    tiny_trainer.cfg["safety_min_hit_wtap_frac"] = -1.0
    tiny_trainer.cfg["safety_min_under_combo_counter_hit_frac"] = -1.0
    tiny_trainer.cfg["safety_min_under_combo_hit_select_clean_frac"] = 0.20
    tiny_trainer.cfg["safety_max_under_combo_hit_select_trade_frac"] = 0.12
    tiny_trainer.cfg["safety_under_combo_avoid_frac"] = 0.070
    tiny_trainer.cfg["safety_under_combo_avoid_min_combo12"] = 0.35
    tiny_trainer.cfg["safety_under_combo_avoid_min_hit_rate"] = 80.0
    metrics = {
        "under_combo_escape_frac": 0.0,
        "back_frac": 0.0,
        "opener_samples": 0.0,
        "hit_rate": 82.0,
        "fresh_spar_under_combo_hit_select_clean_frac": 0.0,
        "fresh_spar_under_combo_hit_select_trade_frac": 0.0,
        "fresh_spar_under_combo_frac": 0.0,
        "fresh_spar_combo12_state": 0.4281,
        "fresh_spar_hit_rate": 104.33,
        "fresh_chase_under_combo_hit_select_clean_frac": 0.2542,
        "fresh_chase_under_combo_hit_select_trade_frac": 0.1186,
        "fresh_chase_under_combo_frac": 0.0661,
        "fresh_chase_combo12_state": 0.4281,
        "fresh_chase_hit_rate": 99.0,
    }

    assert tiny_trainer._checkpoint_safety_guard(metrics) is False
    assert metrics["safety_state"] == "safe"
    assert metrics["fresh_spar_under_combo_hit_select_exposure_gate"] == 1.0


def test_safety_under_combo_avoidance_accepts_moderate_combo_progress(tiny_trainer):
    tiny_trainer.cfg["safety_under_combo_avoid_frac"] = 0.070
    tiny_trainer.cfg["safety_under_combo_avoid_min_combo12"] = 0.08
    tiny_trainer.cfg["safety_under_combo_avoid_min_hit_rate"] = 70.0
    metrics = {
        "fresh_spar_under_combo_frac": 0.0339,
        "fresh_spar_combo12_state": 0.1022,
        "fresh_spar_hit_rate": 83.0,
    }

    assert tiny_trainer._under_combo_avoidance_gate_passes(
        metrics, "fresh_spar_under_combo_counter_hit_frac")
    assert metrics["fresh_spar_under_combo_counter_avoidance_gate"] == 1.0


def test_safety_counter_hit_select_accepts_low_exposure_counter_lane(tiny_trainer):
    tiny_trainer.cfg["safety_under_combo_avoid_frac"] = 0.070
    tiny_trainer.cfg["safety_under_combo_avoid_min_combo12"] = 0.08
    tiny_trainer.cfg["safety_under_combo_avoid_min_hit_rate"] = 80.0
    metrics = {
        "fresh_counter_under_combo_frac": 0.0383,
        "fresh_counter_combo12_state": 0.2628,
        "fresh_counter_hit_rate": 84.0,
    }

    assert tiny_trainer._under_combo_hit_select_exposure_gate_passes(
        metrics, "fresh_counter_under_combo_hit_select_clean_frac")
    assert metrics["fresh_counter_under_combo_hit_select_exposure_gate"] == 1.0


def test_safety_counter_recovery_blocks_promotion_without_stopping_training(tiny_trainer):
    tiny_trainer.cfg["safety_stop_on_regression"] = True
    tiny_trainer.cfg["safety_under_combo_escape"] = 1.0
    tiny_trainer.cfg["safety_back_frac"] = 1.0
    tiny_trainer.cfg["safety_min_strafe_frac"] = -1.0
    tiny_trainer.cfg["safety_min_combo_tap_frac"] = -1.0
    tiny_trainer.cfg["safety_min_combo_z_tap_frac"] = -1.0
    tiny_trainer.cfg["safety_max_combo_s_tap_frac"] = -1.0
    tiny_trainer.cfg["safety_min_hit_wtap_frac"] = -1.0
    tiny_trainer.cfg["safety_min_under_combo_counter_hit_frac"] = -1.0
    tiny_trainer.cfg["safety_min_under_combo_hit_select_clean_frac"] = 0.20
    tiny_trainer.cfg["safety_max_under_combo_hit_select_trade_frac"] = 0.12
    tiny_trainer.cfg["safety_under_combo_avoid_frac"] = 0.070
    tiny_trainer.cfg["safety_under_combo_avoid_min_combo12"] = 0.35
    tiny_trainer.cfg["safety_under_combo_avoid_min_hit_rate"] = 80.0
    tiny_trainer.cfg["safety_require_counter_recovery"] = True
    metrics = {
        "under_combo_escape_frac": 0.0,
        "back_frac": 0.0,
        "opener_samples": 0.0,
        "hit_rate": 82.0,
        "fresh_spar_under_combo_hit_select_clean_frac": 0.0,
        "fresh_spar_under_combo_hit_select_trade_frac": 0.0,
        "fresh_spar_under_combo_frac": 0.0,
        "fresh_spar_combo12_state": 0.4281,
        "fresh_spar_hit_rate": 104.33,
        "fresh_chase_under_combo_hit_select_clean_frac": 0.2542,
        "fresh_chase_under_combo_hit_select_trade_frac": 0.1186,
        "fresh_chase_under_combo_frac": 0.0661,
        "fresh_chase_combo12_state": 0.4281,
        "fresh_chase_hit_rate": 99.0,
        "fresh_counter_under_combo_hit_select_clean_frac": 0.0,
        "fresh_counter_under_combo_hit_select_trade_frac": 1.0,
        "fresh_counter_under_combo_frac": 0.22,
        "fresh_counter_combo12_state": 0.20,
        "fresh_counter_hit_rate": 80.0,
    }

    assert tiny_trainer._checkpoint_safety_guard(metrics) is False
    assert metrics["safety_state"] == "training"
    assert metrics["safety_checkpoint"] == "await_counter_recovery"
    assert "fresh_counter_under_combo_hit_select_clean_frac=0<0.2" in metrics["safety_promotion_reason"]
    assert "fresh_counter_under_combo_hit_select_trade_frac=1>0.12" in metrics["safety_promotion_reason"]


def test_safety_chase_counter_recovery_keeps_training_alive(tiny_trainer):
    tiny_trainer.cfg["safety_stop_on_regression"] = True
    tiny_trainer.cfg["safety_under_combo_escape"] = 1.0
    tiny_trainer.cfg["safety_back_frac"] = 1.0
    tiny_trainer.cfg["safety_min_strafe_frac"] = -1.0
    tiny_trainer.cfg["safety_min_combo_tap_frac"] = -1.0
    tiny_trainer.cfg["safety_min_combo_z_tap_frac"] = -1.0
    tiny_trainer.cfg["safety_max_combo_s_tap_frac"] = -1.0
    tiny_trainer.cfg["safety_min_hit_wtap_frac"] = -1.0
    tiny_trainer.cfg["safety_min_under_combo_counter_hit_frac"] = 0.095
    tiny_trainer.cfg["safety_min_under_combo_hit_select_clean_frac"] = 0.20
    tiny_trainer.cfg["safety_max_under_combo_hit_select_trade_frac"] = 0.12
    tiny_trainer.cfg["safety_under_combo_avoid_frac"] = 0.070
    tiny_trainer.cfg["safety_under_combo_avoid_min_combo12"] = 0.35
    tiny_trainer.cfg["safety_under_combo_avoid_min_hit_rate"] = 80.0
    tiny_trainer.cfg["safety_require_counter_recovery"] = True
    metrics = {
        "under_combo_escape_frac": 0.0,
        "back_frac": 0.0,
        "opener_samples": 0.0,
        "hit_rate": 82.0,
        "fresh_chase_under_combo_counter_hit_frac": 0.0469,
        "fresh_chase_under_combo_hit_select_clean_frac": 0.0,
        "fresh_chase_under_combo_hit_select_trade_frac": 0.0,
        "fresh_chase_under_combo_frac": 0.0769,
        "fresh_chase_combo12_state": 0.42,
        "fresh_chase_hit_rate": 100.33,
    }

    assert tiny_trainer._checkpoint_safety_guard(metrics) is False
    assert metrics["safety_state"] == "training"
    assert metrics["safety_checkpoint"] == "await_counter_recovery"
    assert "fresh_chase_under_combo_counter_hit_frac=0.0469<0.095" in metrics["safety_promotion_reason"]
    assert "fresh_chase_under_combo_hit_select_clean_frac=0<0.2" in metrics["safety_promotion_reason"]
    assert "safety_reason" not in metrics


def test_safety_under_combo_counter_gate_accepts_low_exposure_avoidance(tiny_trainer):
    tiny_trainer.cfg["safety_under_combo_avoid_frac"] = 0.070
    tiny_trainer.cfg["safety_under_combo_avoid_min_combo12"] = 0.35
    tiny_trainer.cfg["safety_under_combo_avoid_min_hit_rate"] = 80.0
    metrics = {
        "fresh_spar_under_combo_counter_hit_frac": 0.0821,
        "fresh_spar_under_combo_frac": 0.0575,
        "fresh_spar_combo12_state": 0.4000,
        "fresh_spar_hit_rate": 100.0,
    }

    assert tiny_trainer._under_combo_avoidance_gate_passes(
        metrics, "fresh_spar_under_combo_counter_hit_frac") is True
    assert metrics["fresh_spar_under_combo_counter_avoidance_gate"] == 1.0


def test_safety_under_combo_counter_gate_accepts_low_exposure_counter_lane(tiny_trainer):
    tiny_trainer.cfg["safety_under_combo_avoid_frac"] = 0.070
    tiny_trainer.cfg["safety_under_combo_avoid_min_combo12"] = 0.08
    tiny_trainer.cfg["safety_under_combo_avoid_min_hit_rate"] = 80.0
    metrics = {
        "fresh_counter_under_combo_counter_hit_frac": 0.0509,
        "fresh_counter_under_combo_frac": 0.0600,
        "fresh_counter_combo12_state": 0.4156,
        "fresh_counter_hit_rate": 97.0,
    }

    assert tiny_trainer._under_combo_avoidance_gate_passes(
        metrics, "fresh_counter_under_combo_counter_hit_frac") is True
    assert metrics["fresh_counter_under_combo_counter_avoidance_gate"] == 1.0


def test_safe_record_counter_contract_accepts_avoidance_bonus(tiny_trainer):
    tiny_trainer.cfg["safety_require_counter_recovery"] = True
    tiny_trainer.cfg["safety_min_under_combo_counter_hit_frac"] = 0.085
    data = {
        "under_combo_counter_hit_frac": 0.0509,
        "under_combo_avoidance_score_bonus": 0.015,
        "under_combo_hit_select_clean_frac": 1.0,
        "under_combo_hit_select_trade_frac": 0.0,
        "hit_wtap_frac": 0.8,
    }

    assert tiny_trainer._safe_record_counter_contract_violated(
        data, None) is False


def test_safe_meta_under_combo_counter_uses_gate_metric_when_rollout_is_present(tiny_trainer):
    run_dir = tiny_trainer.run_dir
    tiny_trainer.cfg["safety_min_under_combo_counter_hit_frac"] = 0.10
    tiny_trainer.cfg["safety_require_chase_combo"] = True
    tiny_trainer.iter = 1
    (run_dir / "ckpt_000001.pt").write_bytes(b"rollout-counter-safe")

    tiny_trainer._mark_safe_checkpoint({
        "iter": 1,
        "under_combo_counter_hit_frac": 0.25,
        "fresh_spar_under_combo_counter_hit_frac": 0.0,
        "fresh_chase_under_combo_counter_hit_frac": 0.0,
        "fresh_spar_combo12_state": 0.4,
        "fresh_spar_combo8_state": 0.4,
        "fresh_spar_combo_max": 80,
        "fresh_spar_hit_rate": 90.0,
        "fresh_spar_sky_frac": 0.0,
        "fresh_spar_combo_tap_frac": 1.0,
        "fresh_spar_combo_z_tap_frac": 1.0,
        "fresh_spar_combo_s_tap_frac": 0.0,
        "fresh_spar_hit_wtap_frac": 0.7,
        "fresh_chase_combo12_state": 0.35,
        "fresh_chase_combo8_state": 0.38,
        "fresh_chase_combo_max": 70,
        "fresh_chase_hit_rate": 88.0,
        "fresh_chase_sky_frac": 0.0,
        "fresh_chase_combo_tap_frac": 1.0,
        "fresh_chase_combo_z_tap_frac": 1.0,
        "fresh_chase_combo_s_tap_frac": 0.0,
        "fresh_chase_hit_wtap_frac": 0.6,
        "back_frac": 0.0,
        "strafe_frac": 0.8,
    })

    meta = json.loads((run_dir / "safe_latest.meta.json").read_text())
    assert meta["under_combo_counter_hit_frac"] == 0.0
    assert meta["under_combo_counter_score_frac"] == 0.0
    assert meta["under_combo_counter_source"] == "fresh_spar_under_combo_counter_hit_frac"


def test_safe_meta_under_combo_counter_uses_worst_counter_lane(tiny_trainer):
    run_dir = tiny_trainer.run_dir
    tiny_trainer.cfg["safety_require_counter_recovery"] = True
    tiny_trainer.iter = 1
    (run_dir / "ckpt_000001.pt").write_bytes(b"counter-safe")

    tiny_trainer._mark_safe_checkpoint({
        "iter": 1,
        "fresh_spar_combo12_state": 0.2,
        "fresh_spar_combo8_state": 0.2,
        "fresh_spar_combo_max": 12,
        "fresh_spar_hit_rate": 85.0,
        "fresh_spar_sky_frac": 0.0,
        "fresh_spar_under_combo_counter_hit_frac": 0.18,
        "fresh_chase_combo12_state": 0.2,
        "fresh_chase_combo8_state": 0.2,
        "fresh_chase_combo_max": 12,
        "fresh_chase_hit_rate": 84.0,
        "fresh_chase_sky_frac": 0.0,
        "fresh_chase_under_combo_counter_hit_frac": 0.13,
        "fresh_counter_under_combo_counter_hit_frac": 0.09,
    })

    meta = json.loads((run_dir / "safe_latest.meta.json").read_text())
    assert meta["under_combo_counter_hit_frac"] == pytest.approx(0.09)
    assert meta["under_combo_counter_score_frac"] == pytest.approx(0.09)
    assert meta["under_combo_counter_source"] == "fresh_counter_under_combo_counter_hit_frac"


def test_safety_opener_gates_skip_rollouts_without_opener_samples(tiny_trainer):
    tiny_trainer.cfg["safety_stop_on_regression"] = True
    tiny_trainer.cfg["combo_eval_every"] = 1
    tiny_trainer.cfg["safety_min_opener_strafe_frac"] = 0.75
    tiny_trainer.cfg["safety_min_opener_strafe_hold_frac"] = 0.70
    tiny_trainer.cfg["safety_min_opener_pressure_frac"] = 0.40
    metrics = {
        "iter": 1,
        "hit_rate": 80.0,
        "sky_frac": 0.0,
        "under_combo_escape_frac": 0.0,
        "back_frac": 0.0,
        "escape_back_frac": 0.0,
        "strafe_frac": 0.8,
        "opener_samples": 0.0,
        "opener_strafe_frac": 0.0,
        "opener_strafe_hold_frac": 0.0,
        "opener_pressure_frac": 0.0,
    }

    assert tiny_trainer._checkpoint_safety_guard(metrics) is False
    assert metrics["safety_state"] == "safe"
    assert "safety_reason" not in metrics


def test_safe_checkpoint_keeps_best_combo_score(tiny_trainer):
    run_dir = tiny_trainer.run_dir

    tiny_trainer.iter = 1
    best = run_dir / "ckpt_000001.pt"
    best.write_bytes(b"best-combo")
    tiny_trainer._mark_safe_checkpoint({
        "iter": 1, "sim_combo12_state": 0.5, "sim_combo_max": 29,
        "combo8_hits": 0.2, "hit_rate": 55.0, "sky_frac": 0.03,
    })
    assert (run_dir / "safe_latest.pt").read_bytes() == b"best-combo"

    tiny_trainer.iter = 2
    worse = run_dir / "ckpt_000002.pt"
    worse.write_bytes(b"worse-combo")
    metrics = {
        "iter": 2, "sim_combo12_state": 0.0, "sim_combo_max": 10,
        "combo8_hits": 0.05, "hit_rate": 70.0, "sky_frac": 0.01,
        "pad_sim_combo12_state": 1.0, "pad_sim_combo_max": 200,
    }
    (run_dir / "latest.pt").write_bytes(b"worse-combo")
    tiny_trainer._mark_safe_checkpoint(metrics)
    assert metrics["safety_checkpoint"] == "kept_best"
    assert metrics["safety_restored"] == "safe_latest.pt"
    assert (run_dir / "safe_latest.pt").read_bytes() == b"best-combo"
    assert (run_dir / "latest.pt").read_bytes() == b"best-combo"

    tiny_trainer.iter = 3
    inflated = run_dir / "ckpt_000003.pt"
    inflated.write_bytes(b"inflated-combo")
    metrics = {
        "iter": 3, "fresh_combo12_state": 0.1, "fresh_combo_max": 12,
        "fresh_combo8_state": 0.02, "fresh_hit_rate": 80.0,
        "fresh_sky_frac": 0.01, "sim_combo12_state": 0.9,
        "sim_combo_max": 200, "combo8_hits": 0.9,
    }
    (run_dir / "latest.pt").write_bytes(b"inflated-combo")
    tiny_trainer._mark_safe_checkpoint(metrics)
    assert metrics["safety_checkpoint"] == "kept_best"
    assert (run_dir / "safe_latest.pt").read_bytes() == b"best-combo"
    assert (run_dir / "latest.pt").read_bytes() == b"best-combo"

    tiny_trainer.iter = 4
    fresh_only = run_dir / "ckpt_000004.pt"
    fresh_only.write_bytes(b"fresh-only-combo")
    metrics = {
        "iter": 4, "fresh_combo12_state": 0.95, "fresh_combo_max": 120,
        "fresh_combo8_state": 0.90, "fresh_hit_rate": 90.0,
        "fresh_sky_frac": 0.01, "mirror_combo12_hits": 0.0,
        "mirror_sim_combo12_state": 0.0, "mirror_combo_max": 2,
        "mirror_sim_combo_max": 2, "mirror_combo8_hits": 0.0,
    }
    (run_dir / "latest.pt").write_bytes(b"fresh-only-combo")
    tiny_trainer._mark_safe_checkpoint(metrics)
    assert metrics["safety_checkpoint"] == "kept_best"
    assert (run_dir / "safe_latest.pt").read_bytes() == b"best-combo"
    assert (run_dir / "latest.pt").read_bytes() == b"best-combo"

    tiny_trainer.iter = 5
    better = run_dir / "ckpt_000005.pt"
    better.write_bytes(b"better-combo")
    tiny_trainer._mark_safe_checkpoint({
        "iter": 5, "sim_combo12_state": 0.6, "sim_combo_max": 31,
        "combo8_hits": 0.3, "hit_rate": 50.0, "sky_frac": 0.04,
    })
    assert (run_dir / "safe_latest.pt").read_bytes() == b"better-combo"


def test_safe_checkpoint_prefers_style_when_combo_is_tied(tiny_trainer):
    run_dir = tiny_trainer.run_dir

    tiny_trainer.iter = 1
    old = run_dir / "ckpt_000001.pt"
    old.write_bytes(b"old-style")
    tiny_trainer._mark_safe_checkpoint({
        "iter": 1, "sim_combo12_state": 0.5, "sim_combo_max": 30,
        "combo8_hits": 0.3, "hit_rate": 60.0, "sky_frac": 0.01,
        "combo_s_tap_frac": 0.0, "under_combo_counter_hit_frac": 0.0,
    })

    tiny_trainer.iter = 2
    better = run_dir / "ckpt_000002.pt"
    better.write_bytes(b"better-style")
    tiny_trainer._mark_safe_checkpoint({
        "iter": 2, "sim_combo12_state": 0.5, "sim_combo_max": 30,
        "combo8_hits": 0.3, "hit_rate": 60.0, "sky_frac": 0.01,
        "combo_s_tap_frac": 0.35, "under_combo_counter_hit_frac": 0.4,
    })

    assert (run_dir / "safe_latest.pt").read_bytes() == b"better-style"


def test_safe_checkpoint_prefers_no_back_z_tap_when_combo_is_tied(tiny_trainer):
    run_dir = tiny_trainer.run_dir

    tiny_trainer.iter = 1
    old = run_dir / "ckpt_000001.pt"
    old.write_bytes(b"old-no-tap")
    tiny_trainer._mark_safe_checkpoint({
        "iter": 1, "sim_combo12_state": 0.5, "sim_combo_max": 30,
        "combo8_hits": 0.3, "hit_rate": 60.0, "sky_frac": 0.01,
        "combo_s_tap_frac": 0.0, "combo_z_tap_frac": 0.0,
        "under_combo_counter_hit_frac": 0.0,
    })

    tiny_trainer.iter = 2
    better = run_dir / "ckpt_000002.pt"
    better.write_bytes(b"better-z-tap")
    tiny_trainer._mark_safe_checkpoint({
        "iter": 2, "sim_combo12_state": 0.5, "sim_combo_max": 30,
        "combo8_hits": 0.3, "hit_rate": 60.0, "sky_frac": 0.01,
        "combo_s_tap_frac": 0.0, "combo_z_tap_frac": 0.35,
        "hit_wtap_frac": 0.90,
        "under_combo_counter_hit_frac": 0.4,
    })

    assert (run_dir / "safe_latest.pt").read_bytes() == b"better-z-tap"
    meta = json.loads((run_dir / "safe_latest.meta.json").read_text())
    assert meta["score_schema"] == 9
    assert meta["combo_tap_frac"] == 0.35
    assert meta["combo_z_tap_frac"] == 0.35
    assert meta["combo_s_tap_frac"] == 0.0
    assert meta["hit_wtap_frac"] == 0.90
    assert meta["under_combo_counter_hit_frac"] == 0.4
    assert "opener_strafe_frac" in meta
    assert "safety_min_opener_strafe_frac" in meta
    assert "opener_strafe_hold_frac" in meta
    assert "safety_min_opener_strafe_hold_frac" in meta
    assert "opener_pressure_frac" in meta
    assert "safety_min_opener_pressure_frac" in meta
    assert "safety_min_combo_tap_frac" in meta
    assert "safety_min_combo_z_tap_frac" in meta
    assert "safety_max_combo_s_tap_frac" in meta
    assert "safety_min_hit_wtap_frac" in meta
    assert "safety_min_under_combo_counter_hit_frac" in meta
    assert "under_combo_hit_select_clean_frac" in meta
    assert "under_combo_hit_select_trade_frac" in meta
    assert "safety_min_under_combo_hit_select_clean_frac" in meta
    assert "safety_max_under_combo_hit_select_trade_frac" in meta


def test_safe_combo_score_requires_active_spar_combo(tiny_trainer):
    score = tiny_trainer._safe_combo_score({
        "fresh_combo12_state": 0.8,
        "fresh_combo_max": 30,
        "fresh_combo8_state": 0.7,
        "fresh_hit_rate": 80.0,
        "fresh_sky_frac": 0.0,
        "fresh_combo_s_tap_frac": 0.6,
        "fresh_under_combo_counter_hit_frac": 0.4,
        "fresh_spar_combo12_state": 0.0,
        "fresh_spar_combo_max": 2,
        "fresh_spar_combo8_state": 0.0,
        "fresh_spar_hit_rate": 70.0,
        "fresh_spar_sky_frac": 0.0,
        "fresh_spar_combo_s_tap_frac": 0.0,
        "fresh_spar_under_combo_counter_hit_frac": 0.0,
    })

    assert score[1] == 0.0
    assert score[3] == 2.0


def test_safe_combo_score_uses_active_spar_when_mirror_is_absent(tiny_trainer):
    score = tiny_trainer._safe_combo_score({
        "fresh_combo12_state": 0.0,
        "fresh_combo_max": 0,
        "fresh_combo8_state": 0.0,
        "fresh_hit_rate": 0.0,
        "fresh_sky_frac": 0.0,
        "fresh_spar_combo12_state": 0.4454,
        "fresh_spar_combo_max": 120,
        "fresh_spar_combo8_state": 0.4621,
        "fresh_spar_hit_rate": 103.0,
        "fresh_spar_sky_frac": 0.0,
        "fresh_spar_combo_s_tap_frac": 0.1688,
        "fresh_spar_under_combo_counter_hit_frac": 0.1531,
        "mirror_hit_rate": 0.0,
        "mirror_engage_rate": 0.0,
        "mirror_combo12_hits": 0.0,
        "mirror_sim_combo12_state": 0.0,
        "mirror_combo_max": 0,
        "mirror_sim_combo_max": 0,
    })

    assert score[1] == 0.4454
    assert score[3] == 120.0
    assert score[6] == 103.0


def test_safe_combo_score_requires_chase_when_enabled(tiny_trainer):
    tiny_trainer.cfg["safety_require_chase_combo"] = True
    score = tiny_trainer._safe_combo_score({
        "fresh_spar_combo12_state": 0.45,
        "fresh_spar_combo_max": 120,
        "fresh_spar_combo8_state": 0.50,
        "fresh_spar_hit_rate": 100.0,
        "fresh_spar_sky_frac": 0.0,
        "fresh_spar_combo_s_tap_frac": 0.20,
        "fresh_spar_under_combo_counter_hit_frac": 0.25,
        "fresh_chase_combo12_state": 0.0,
        "fresh_chase_combo_max": 1,
        "fresh_chase_combo8_state": 0.0,
        "fresh_chase_hit_rate": 70.0,
        "fresh_chase_sky_frac": 0.0,
        "fresh_chase_combo_s_tap_frac": 0.0,
        "fresh_chase_under_combo_counter_hit_frac": 0.05,
    })

    assert score[1] == 0.0
    assert score[3] == 1.0
    assert score[5] == 0.05


def test_safe_combo_score_rewards_low_under_combo_exposure(tiny_trainer):
    tiny_trainer.cfg["safety_require_chase_combo"] = True
    tiny_trainer.cfg["score_under_combo_avoid_target"] = 0.14
    tiny_trainer.cfg["score_under_combo_avoid_weight"] = 0.30
    tiny_trainer.cfg["score_under_combo_avoid_cap"] = 0.015

    score = tiny_trainer._safe_combo_score({
        "fresh_spar_combo12_state": 0.4000,
        "fresh_spar_combo8_state": 0.4222,
        "fresh_spar_combo_max": 90,
        "fresh_spar_hit_rate": 100.0,
        "fresh_spar_sky_frac": 0.0,
        "fresh_spar_combo_tap_frac": 1.0,
        "fresh_spar_under_combo_counter_hit_frac": 0.0821,
        "fresh_spar_under_combo_frac": 0.0575,
        "fresh_chase_combo12_state": 0.3889,
        "fresh_chase_combo8_state": 0.4111,
        "fresh_chase_combo_max": 90,
        "fresh_chase_hit_rate": 94.67,
        "fresh_chase_sky_frac": 0.0,
        "fresh_chase_combo_tap_frac": 1.0,
        "fresh_chase_under_combo_counter_hit_frac": 0.104,
        "fresh_chase_under_combo_frac": 0.0908,
    })

    assert score[0] == pytest.approx(0.727256)
    assert score[1] == pytest.approx(0.3889)
    assert score[5] == pytest.approx(0.0821)


def test_safe_checkpoint_refuses_low_combo_promotion(tiny_trainer):
    run_dir = tiny_trainer.run_dir
    tiny_trainer.cfg["safety_promote_min_combo_max"] = 8.0

    tiny_trainer.iter = 1
    existing = run_dir / "ckpt_000001.pt"
    existing.write_bytes(b"existing-safe")
    tiny_trainer._mark_safe_checkpoint({
        "iter": 1, "fresh_combo12_state": 0.0, "fresh_combo_max": 9,
        "fresh_combo8_state": 0.1, "fresh_hit_rate": 50.0,
        "fresh_sky_frac": 0.0,
    })
    assert (run_dir / "safe_latest.pt").read_bytes() == b"existing-safe"

    tiny_trainer.iter = 2
    weak = run_dir / "ckpt_000002.pt"
    weak.write_bytes(b"weak-rush")
    (run_dir / "latest.pt").write_bytes(b"weak-rush")
    metrics = {
        "iter": 2, "fresh_combo12_state": 0.0, "fresh_combo_max": 1,
        "fresh_combo8_state": 0.0, "fresh_hit_rate": 70.0,
        "fresh_sky_frac": 0.0, "fresh_combo_s_tap_frac": 1.0,
        "fresh_under_combo_counter_hit_frac": 0.2,
    }

    tiny_trainer._mark_safe_checkpoint(metrics)

    assert metrics["safety_checkpoint"] == "kept_best"
    assert "combo_max=1<8" in metrics["safety_promotion_reason"]
    assert "safety_restored" not in metrics
    assert "safety_memory_restored" not in metrics
    assert (run_dir / "safe_latest.pt").read_bytes() == b"existing-safe"
    assert (run_dir / "latest.pt").read_bytes() == b"weak-rush"


def test_safe_checkpoint_can_restore_low_combo_when_enabled(tiny_trainer):
    run_dir = tiny_trainer.run_dir
    tiny_trainer.cfg["safety_promote_min_combo_max"] = 8.0
    tiny_trainer.cfg["safety_restore_on_low_combo"] = True

    tiny_trainer.iter = 1
    existing = run_dir / "ckpt_000001.pt"
    existing.write_bytes(b"existing-safe")
    tiny_trainer._mark_safe_checkpoint({
        "iter": 1, "fresh_combo12_state": 0.0, "fresh_combo_max": 9,
        "fresh_combo8_state": 0.1, "fresh_hit_rate": 50.0,
        "fresh_sky_frac": 0.0,
    })

    tiny_trainer.iter = 2
    weak = run_dir / "ckpt_000002.pt"
    weak.write_bytes(b"weak-rush")
    (run_dir / "latest.pt").write_bytes(b"weak-rush")
    metrics = {
        "iter": 2, "fresh_combo12_state": 0.0, "fresh_combo_max": 1,
        "fresh_combo8_state": 0.0, "fresh_hit_rate": 70.0,
        "fresh_sky_frac": 0.0,
    }

    tiny_trainer._mark_safe_checkpoint(metrics)

    assert metrics["safety_checkpoint"] == "kept_best"
    assert metrics["safety_restored"] == "safe_latest.pt"
    assert metrics["safety_memory_restored"] == "missing"
    assert (run_dir / "safe_latest.pt").read_bytes() == b"existing-safe"
    assert (run_dir / "latest.pt").read_bytes() == b"existing-safe"


def test_safe_checkpoint_ignores_legacy_non_chase_best_when_chase_required(tiny_trainer):
    run_dir = tiny_trainer.run_dir
    tiny_trainer.cfg["safety_require_chase_combo"] = True
    tiny_trainer.cfg["safety_promote_min_combo_max"] = 8.0
    (run_dir / "safe_latest.pt").write_bytes(b"legacy-safe")
    (run_dir / "recover_noescape.pt").write_bytes(b"legacy-safe")
    (run_dir / "safe_latest.meta.json").write_text(json.dumps({
        "score_schema": 3,
        "requires_chase_combo": False,
        "score": [0.99, 0.8, 0.7, 30.0, 0.5, 0.5, 80.0, 0.0],
    }))

    tiny_trainer.iter = 2
    candidate = run_dir / "ckpt_000002.pt"
    candidate.write_bytes(b"chase-gated-candidate")
    (run_dir / "latest.pt").write_bytes(b"chase-gated-candidate")
    metrics = {
        "iter": 2,
        "fresh_spar_combo12_state": 0.0,
        "fresh_spar_combo_max": 10,
        "fresh_spar_combo8_state": 0.2,
        "fresh_spar_hit_rate": 80.0,
        "fresh_spar_sky_frac": 0.0,
        "fresh_spar_combo_s_tap_frac": 0.3,
        "fresh_spar_under_combo_counter_hit_frac": 0.2,
        "fresh_chase_combo12_state": 0.0,
        "fresh_chase_combo_max": 8,
        "fresh_chase_combo8_state": 0.1,
        "fresh_chase_hit_rate": 60.0,
        "fresh_chase_sky_frac": 0.0,
        "fresh_chase_combo_s_tap_frac": 0.2,
        "fresh_chase_under_combo_counter_hit_frac": 0.1,
    }

    tiny_trainer._mark_safe_checkpoint(metrics)

    assert metrics["safety_checkpoint"] == "ckpt_000002.pt"
    assert (run_dir / "safe_latest.pt").read_bytes() == b"chase-gated-candidate"
    meta = json.loads((run_dir / "safe_latest.meta.json").read_text())
    assert meta["requires_chase_combo"] is True


def test_safe_checkpoint_downgrades_counter_blind_current_safe(tiny_trainer):
    run_dir = tiny_trainer.run_dir
    tiny_trainer.cfg["safety_require_chase_combo"] = True
    tiny_trainer.cfg["safety_require_counter_recovery"] = True
    tiny_trainer.cfg["safety_promote_min_combo_max"] = 8.0
    tiny_trainer.cfg["safety_min_hit_wtap_frac"] = 0.035
    tiny_trainer.cfg["safety_min_under_combo_counter_hit_frac"] = 0.085
    tiny_trainer.cfg["safety_min_under_combo_hit_select_clean_frac"] = 0.20
    tiny_trainer.cfg["safety_max_under_combo_hit_select_trade_frac"] = 0.12
    tiny_trainer.cfg["safety_under_combo_avoid_frac"] = 0.06
    tiny_trainer.cfg["safety_under_combo_avoid_min_combo12"] = 0.08
    tiny_trainer.cfg["safety_under_combo_avoid_min_hit_rate"] = 70.0
    (run_dir / "safe_latest.pt").write_bytes(b"old-combo-only-safe")
    (run_dir / "recover_noescape.pt").write_bytes(b"old-combo-only-safe")
    (run_dir / "safe_latest.meta.json").write_text(json.dumps({
        "score_schema": 9,
        "requires_chase_combo": True,
        "requires_counter_recovery": True,
        "score": [0.99, 0.44, 0.45, 90.0, 1.0, 0.0465, 99.0, 0.0],
        "hit_wtap_frac": 0.80,
        "under_combo_counter_hit_frac": 0.0465,
        "under_combo_hit_select_clean_frac": 0.0,
        "under_combo_hit_select_trade_frac": 1.0,
    }))

    tiny_trainer.iter = 2
    candidate = run_dir / "ckpt_000002.pt"
    candidate.write_bytes(b"recovery-safe")
    (run_dir / "latest.pt").write_bytes(b"recovery-safe")
    metrics = {
        "iter": 2,
        "fresh_spar_combo12_state": 0.0736,
        "fresh_spar_combo8_state": 0.10,
        "fresh_spar_combo_max": 22,
        "fresh_spar_hit_rate": 85.33,
        "fresh_spar_sky_frac": 0.0,
        "fresh_spar_combo_z_tap_frac": 0.4164,
        "fresh_spar_hit_wtap_frac": 0.0372,
        "fresh_spar_under_combo_counter_hit_frac": 0.1186,
        "fresh_spar_under_combo_hit_select_clean_frac": 0.3333,
        "fresh_spar_under_combo_hit_select_trade_frac": 0.0,
        "fresh_chase_combo12_state": 0.0736,
        "fresh_chase_combo8_state": 0.10,
        "fresh_chase_combo_max": 22,
        "fresh_chase_hit_rate": 84.67,
        "fresh_chase_sky_frac": 0.0,
        "fresh_chase_combo_z_tap_frac": 0.3725,
        "fresh_chase_hit_wtap_frac": 0.0466,
        "fresh_chase_under_combo_counter_hit_frac": 0.0973,
        "fresh_chase_under_combo_hit_select_clean_frac": 0.4286,
        "fresh_chase_under_combo_hit_select_trade_frac": 0.0,
        "fresh_counter_combo12_state": 0.2628,
        "fresh_counter_hit_rate": 84.0,
        "fresh_counter_under_combo_frac": 0.0383,
        "fresh_counter_under_combo_counter_hit_frac": 0.087,
        "fresh_counter_under_combo_hit_select_clean_frac": 0.0,
        "fresh_counter_under_combo_hit_select_trade_frac": 0.0,
    }

    tiny_trainer._mark_safe_checkpoint(metrics)

    assert metrics["safety_checkpoint"] == "ckpt_000002.pt"
    assert (run_dir / "safe_latest.pt").read_bytes() == b"recovery-safe"
    meta = json.loads((run_dir / "safe_latest.meta.json").read_text())
    assert meta["requires_counter_recovery"] is True
    assert meta["under_combo_counter_score_frac"] == pytest.approx(0.087)
    assert meta["under_combo_hit_select_clean_frac"] == pytest.approx(0.3333)


def test_safe_checkpoint_does_not_restore_invalid_low_combo_best(tiny_trainer):
    run_dir = tiny_trainer.run_dir
    tiny_trainer.cfg["safety_promote_min_combo_max"] = 8.0
    (run_dir / "safe_latest.pt").write_bytes(b"old-invalid-safe")
    (run_dir / "recover_noescape.pt").write_bytes(b"old-invalid-safe")
    (run_dir / "safe_latest.meta.json").write_text(json.dumps({
        "score_schema": 2,
        "score": [0.01, 0.0, 0.0, 1.0, 1.0, 0.1, 50.0, 0.0],
    }))
    tiny_trainer.iter = 2
    candidate = run_dir / "ckpt_000002.pt"
    candidate.write_bytes(b"keep-training")
    (run_dir / "latest.pt").write_bytes(b"keep-training")

    metrics = {
        "iter": 2, "fresh_combo12_state": 0.0, "fresh_combo_max": 2,
        "fresh_combo8_state": 0.0, "fresh_hit_rate": 70.0,
        "fresh_sky_frac": 0.0,
    }

    tiny_trainer._mark_safe_checkpoint(metrics)

    assert metrics["safety_checkpoint"] == "kept_best"
    assert "safety_restored" not in metrics
    assert (run_dir / "safe_latest.pt").read_bytes() == b"old-invalid-safe"
    assert (run_dir / "latest.pt").read_bytes() == b"keep-training"


def test_safety_stop_does_not_restore_invalid_safe(tiny_trainer):
    run_dir = tiny_trainer.run_dir
    tiny_trainer.cfg["safety_stop_on_regression"] = True
    tiny_trainer.cfg["safety_promote_min_combo_max"] = 8.0
    tiny_trainer.cfg["safety_min_hit_rate"] = 40.0
    (run_dir / "safe_latest.pt").write_bytes(b"old-invalid-safe")
    (run_dir / "recover_noescape.pt").write_bytes(b"old-invalid-safe")
    (run_dir / "safe_latest.meta.json").write_text(json.dumps({
        "score_schema": 2,
        "score": [0.01, 0.0, 0.0, 1.0, 1.0, 0.1, 50.0, 0.0],
    }))
    tiny_trainer.iter = 2
    (run_dir / "ckpt_000002.pt").write_bytes(b"candidate")
    (run_dir / "latest.pt").write_bytes(b"candidate")

    metrics = {
        "iter": 2,
        "hit_rate": 10.0,
        "under_combo_escape_frac": 0.0,
        "back_frac": 0.0,
        "escape_back_frac": 0.0,
        "strafe_frac": 0.1,
        "sky_frac": 0.0,
    }

    assert tiny_trainer._checkpoint_safety_guard(metrics) is True
    assert metrics["safety_restored"] == "skipped_invalid_safe"
    assert (run_dir / "latest.pt").read_bytes() == b"candidate"
    assert (run_dir / "ckpt_000002.pt").exists()


def test_safe_checkpoint_rollback_restores_policy_memory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    t = Trainer(TINY, device="cpu")
    t.iter = 1
    safe_ckpt = t.save()
    original = {k: v.detach().clone() for k, v in t.policy.state_dict().items()}
    for name in ("safe_latest.pt", "recover_noescape.pt"):
        shutil.copyfile(safe_ckpt, t.run_dir / name)
    t._write_safe_meta(
        {"iter": 1}, safe_ckpt,
        (0.9, 0.9, 0.8, 100.0, 0.0, 0.0, 60.0, -0.01),
    )

    with torch.no_grad():
        for param in t.policy.parameters():
            param.add_(0.5)
    t.iter = 2
    t.save()
    degraded = {
        "iter": 2,
        "fresh_combo12_state": 0.1,
        "fresh_combo_max": 10,
        "fresh_combo8_state": 0.1,
        "fresh_hit_rate": 80.0,
        "fresh_sky_frac": 0.01,
    }

    t._mark_safe_checkpoint(degraded)

    assert degraded["safety_checkpoint"] == "kept_best"
    assert degraded["safety_memory_restored"] == "safe_latest.pt"
    restored = t.policy.state_dict()
    for key, value in original.items():
        assert torch.equal(restored[key], value)


def test_safe_checkpoint_migrates_legacy_fresh_only_score(tiny_trainer):
    run_dir = tiny_trainer.run_dir
    safe = run_dir / "safe_latest.pt"
    safe.write_bytes(b"legacy-fresh-only")
    (run_dir / "recover_noescape.pt").write_bytes(b"legacy-fresh-only")
    (run_dir / "safe_latest.meta.json").write_text(json.dumps({
        "checkpoint": "safe_latest.pt",
        "score": [0.95, 120.0, 0.90, 90.0, -0.01],
    }))

    tiny_trainer.iter = 2
    candidate = run_dir / "ckpt_000002.pt"
    candidate.write_bytes(b"mirror-progress")
    (run_dir / "latest.pt").write_bytes(b"mirror-progress")
    metrics = {
        "iter": 2, "fresh_combo12_state": 0.0, "fresh_combo_max": 4,
        "fresh_combo8_state": 0.0, "fresh_hit_rate": 70.0,
        "fresh_sky_frac": 0.0, "mirror_combo12_hits": 0.0,
        "mirror_sim_combo12_state": 0.0, "mirror_combo_max": 4,
        "mirror_sim_combo_max": 4, "mirror_combo8_hits": 0.0,
    }

    tiny_trainer._mark_safe_checkpoint(metrics)

    assert metrics["safety_checkpoint"] == "ckpt_000002.pt"
    assert safe.read_bytes() == b"mirror-progress"


def test_save_load_roundtrip(tiny_trainer, tmp_path):
    tiny_trainer.train_iter()
    path = tiny_trainer.save()
    t2 = Trainer(TINY, device="cpu")
    t2.load(str(path))
    assert t2.iter == tiny_trainer.iter
    assert abs(t2.league.learner_elo - tiny_trainer.league.learner_elo) < 1e-9



def test_same_run_resume_keeps_optimizer_state_even_when_seed_resume_wants_fresh(tiny_trainer):
    tiny_trainer.train_iter()
    path = tiny_trainer.save()
    saved = torch.load(path, map_location="cpu", weights_only=False)
    for group in saved["optimizer"]["param_groups"]:
        group["lr"] = 0.123
    torch.save(saved, path)

    cfg = {**TINY, "fresh_optimizer_on_resume": True,
           "ppo": {**TINY["ppo"], "lr": 0.0007, "amp": False}}
    t2 = Trainer(cfg, device="cpu")
    t2.load(str(path))

    assert t2.iter == tiny_trainer.iter
    assert t2.ppo.opt.param_groups[0]["lr"] == pytest.approx(0.123)


def test_seed_resume_uses_fresh_optimizer_for_other_run(tiny_trainer):
    tiny_trainer.train_iter()
    path = tiny_trainer.save()
    saved = torch.load(path, map_location="cpu", weights_only=False)
    for group in saved["optimizer"]["param_groups"]:
        group["lr"] = 0.123
    torch.save(saved, path)

    cfg = {**TINY, "name": "_seeded", "resume_as_seed": True,
           "fresh_optimizer_on_resume": True,
           "ppo": {**TINY["ppo"], "lr": 0.0007, "amp": False}}
    t2 = Trainer(cfg, device="cpu")
    t2.load(str(path))

    assert t2.iter == 0
    assert t2.ppo.opt.param_groups[0]["lr"] == pytest.approx(0.0007)


def test_load_policy_only_checkpoint_seeds_new_run(tiny_trainer, tmp_path):
    path = tmp_path / "best.pt"
    torch.save({
        "iter": 123,
        "eval_bot": 0.75,
        "policy": tiny_trainer.policy.state_dict(),
        "policy_cfg": tiny_trainer.pol_cfg.__dict__,
    }, path)

    t2 = Trainer(TINY, device="cpu")
    t2.load(str(path))

    assert t2.iter == 0
    assert t2.total_steps == 0
    assert t2._best_bot == 0.75
    saved = torch.load(t2.run_dir / "best.pt", map_location="cpu",
                       weights_only=False)
    assert saved["eval_bot"] == 0.75



def test_load_incompatible_policy_only_checkpoint_ignores_old_best(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    old_policy = JudasPolicy(PolicyConfig(history=4, d_model=96, n_heads=4,
                                          n_layers=2, attention=True))
    path = tmp_path / "old_best.pt"
    torch.save({
        "iter": 999,
        "eval_bot": 0.99,
        "policy": old_policy.state_dict(),
        "policy_cfg": old_policy.cfg.__dict__,
    }, path)

    t = Trainer(TINY, device="cpu")
    t.load(str(path))

    assert t.iter == 0
    assert t.total_steps == 0
    assert t._best_bot == -1.0
    assert not (t.run_dir / "best.pt").exists()

def test_policy_mlp_mode():
    """attention=False -> trunk MLP, mÃƒÆ’Ã‚Âªmes interfaces."""
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
    t.train_iter()                       # pool crÃƒÆ’Ã‚Â©ÃƒÆ’Ã‚Â© ÃƒÆ’Ã‚Â  l'itÃƒÆ’Ã‚Â©ration 1
    m = t.train_iter()
    assert "eval_first" in m and "eval_bot" in m
    assert 0.0 <= m["eval_first"] <= 1.0
    assert 0.0 <= m["eval_bot"] <= 1.0


def test_best_pt_never_regresses(tmp_path, monkeypatch):
    """Un best.pt plus fort sur disque ne peut pas ÃƒÆ’Ã‚Âªtre ÃƒÆ’Ã‚Â©crasÃƒÆ’Ã‚Â© par un run
    dont la barre est repartie de -1 (start sans resume) ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â la protection de
    lignÃƒÆ’Ã‚Â©e. Et la barre se rÃƒÆ’Ã‚Â©initialise depuis le disque au chargement."""
    monkeypatch.chdir(tmp_path)
    t = Trainer({**TINY, "eval_every": 1}, device="cpu")
    t.run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"eval_bot": 0.76, "policy": t.policy.state_dict(),
                "policy_cfg": t.pol_cfg.__dict__}, t.run_dir / "best.pt")
    t.train_iter()                          # crÃƒÆ’Ã‚Â©e le pool
    t.train_iter()                          # ÃƒÆ’Ã‚Â©val (policy alÃƒÆ’Ã‚Â©atoire, < 0.76)
    saved = torch.load(t.run_dir / "best.pt", map_location="cpu",
                       weights_only=False)
    assert abs(saved["eval_bot"] - 0.76) < 1e-9, \
        "best.pt a rÃƒÆ’Ã‚Â©gressÃƒÆ’Ã‚Â© ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â la protection de lignÃƒÆ’Ã‚Â©e est cassÃƒÆ’Ã‚Â©e"


def test_full_gap_capped_for_huge_arenas(tmp_path, monkeypatch):
    """Sur une arÃƒÆ’Ã‚Â¨ne gÃƒÆ’Ã‚Â©ante, le spawn standard reste bornÃƒÆ’Ã‚Â© ÃƒÆ’Ã‚Â  8 blocs ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â
    arÃƒÆ’Ã‚Â¨ne 45 ne doit plus produire des spawns ÃƒÆ’Ã‚Â  15 blocs (dÃƒÆ’Ã‚Â©sert)."""
    monkeypatch.chdir(tmp_path)
    cfg = {**TINY, "sim": {**TINY["sim"],
                           "arena_size_x": 45.0, "arena_size_z": 45.0}}
    t = Trainer(cfg, device="cpu")
    assert t._full_gap == 8.0
    t2 = Trainer(TINY, device="cpu")        # arÃƒÆ’Ã‚Â¨ne 18 : comportement inchangÃƒÆ’Ã‚Â©
    assert abs(t2._full_gap - 6.0) < 1e-9


def test_spawn_gap_curriculum():
    """spawn_gap configurÃƒÆ’Ã‚Â© -> les joueurs spawnent ÃƒÆ’Ã‚Â  2x gap l'un de l'autre."""
    from sim import JudasSimRef, SimConfig
    env = JudasSimRef(1, SimConfig(spawn_gap=2.0, target_hits=5, max_ticks=50))
    env.reset()
    p0, p1 = env._matches[0].players
    assert abs(abs(p1.z - p0.z) - 4.0) < 1e-9
    env.set_spawn_gap(0.0)               # retour au standard (arÃƒÆ’Ã‚Â¨ne/3)
    env._matches[0] = env._new_match()
    p0, p1 = env._matches[0].players
    assert abs(abs(p1.z - p0.z) - 12.0) < 1e-9


def test_bot_opponents_in_league(tmp_path, monkeypatch):
    """league_bot_frac=1 -> tous les agents 1 sont contrÃƒÆ’Ã‚Â´lÃƒÆ’Ã‚Â©s par le chase-bot."""
    monkeypatch.chdir(tmp_path)
    t = Trainer({**TINY, "league_bot_frac": 1.0}, device="cpu")
    mask = t._assign_opponents()
    assert int((~mask).sum()) == t.N        # agents 1 exclus de l'apprentissage
    m = t.train_iter()                       # tourne sans erreur avec les bots
    assert "reward_mean" in m


def test_ramp_staggered_and_adaptive(tiny_trainer):
    """Phase 1 : le spawn s'ÃƒÆ’Ã‚Â©largit, le shaping reste plein.
    Phase 2 : le shaping dÃƒÆ’Ã‚Â©croÃƒÆ’Ã‚Â®t. Effondrement du hit rate -> la rampe recule."""
    t = tiny_trainer
    t.cfg["shaping_decay_iters"] = 10
    t._ramp_on = True
    t._shaping_base = 0.002      # TINY ne configure pas le shaping (dÃƒÆ’Ã‚Â©faut 0)

    for _ in range(5):                     # combat sain -> pos 0.5
        t._update_ramp(10.0)
    assert abs(t._ramp_pos - 0.5) < 1e-9
    assert abs(t._auto_shaping() - t._shaping_base) < 1e-12   # shaping intact
    assert abs(t._auto_curriculum() - t._full_gap) < 1e-9     # spawn standard

    for _ in range(3):                     # combat sain -> shaping dÃƒÆ’Ã‚Â©croÃƒÆ’Ã‚Â®t
        t._update_ramp(10.0)
    assert t._auto_shaping() < t._shaping_base

    pos_before = t._ramp_pos
    for _ in range(4):                     # effondrement -> recul (x2 plus vite)
        t._update_ramp(0.1)
    assert t._ramp_pos < pos_before
    # le shaping est restaurÃƒÆ’Ã‚Â© en reculant sous 0.5
    while t._ramp_pos > 0.4:
        t._update_ramp(0.1)
    assert abs(t._auto_shaping() - t._shaping_base) < 1e-12


def test_pad_bot_opponents_in_league(tmp_path, monkeypatch):
    """league_pad_bot_frac=1 -> agent 1 poursuit sans attaquer et sans gradient."""
    monkeypatch.chdir(tmp_path)
    t = Trainer({**TINY, "n_envs": 8, "league_bot_frac": 0.0,
                 "league_pad_bot_frac": 1.0}, device="cpu")
    mask = t._assign_opponents()
    assert int((~mask).sum()) == t.N
    assert t._pad_rows.numel() == t.N
    assert t._bot_rows.numel() == 0


def test_spar_bot_opponents_in_league(tmp_path, monkeypatch):
    """league_spar_bot_frac=1 -> agent 1 uses active combo sparring bot."""
    monkeypatch.chdir(tmp_path)
    t = Trainer({**TINY, "n_envs": 8, "league_bot_frac": 0.0,
                 "league_pad_bot_frac": 0.0,
                 "league_spar_bot_frac": 1.0,
                 "rollout_ticks": 6,
                 "sim": {"target_hits": 99, "max_ticks": 4,
                         "randomize": False}}, device="cpu")
    mask = t._assign_opponents()
    assert int((~mask).sum()) == t.N
    assert t._spar_rows.numel() == t.N
    assert t._bot_rows.numel() == 0
    assert t._pad_rows.numel() == 0
    m = t.train_iter()
    assert "reward_mean" in m


def test_rehit_bot_opponents_in_league(tmp_path, monkeypatch):
    """league_rehit_bot_frac=1 -> agent 1 uses controlled re-hit drill bot."""
    monkeypatch.chdir(tmp_path)
    t = Trainer({**TINY, "n_envs": 8, "league_bot_frac": 0.0,
                 "league_pad_bot_frac": 0.0,
                 "league_spar_bot_frac": 0.0,
                 "league_rehit_bot_frac": 1.0,
                 "rollout_ticks": 6,
                 "sim": {"target_hits": 99, "max_ticks": 4,
                         "randomize": False}}, device="cpu")
    mask = t._assign_opponents()
    assert int((~mask).sum()) == t.N
    assert t._rehit_rows.numel() == t.N
    assert t._bot_rows.numel() == 0
    assert t._pad_rows.numel() == 0
    assert t._spar_rows.numel() == 0
    m = t.train_iter()
    assert "rehit_combo_max" in m


def test_pressure_bot_opponents_in_league(tmp_path, monkeypatch):
    """league_pressure_bot_frac=1 -> agent 1 uses active re-hit transfer bot."""
    monkeypatch.chdir(tmp_path)
    t = Trainer({**TINY, "n_envs": 8, "league_bot_frac": 0.0,
                 "league_pad_bot_frac": 0.0,
                 "league_spar_bot_frac": 0.0,
                 "league_rehit_bot_frac": 0.0,
                 "league_pressure_bot_frac": 1.0,
                 "rollout_ticks": 6,
                 "sim": {"target_hits": 99, "max_ticks": 4,
                         "randomize": False}}, device="cpu")
    mask = t._assign_opponents()
    assert int((~mask).sum()) == t.N
    assert t._pressure_rows.numel() == t.N
    assert t._bot_rows.numel() == 0
    assert t._pad_rows.numel() == 0
    assert t._spar_rows.numel() == 0
    assert t._rehit_rows.numel() == 0
    m = t.train_iter()
    assert "pressure_combo_max" in m


def test_combo_chase_bot_opponents_in_league(tmp_path, monkeypatch):
    """league_combo_chase_bot_frac=1 -> agent 1 uses live-like combo chaser."""
    monkeypatch.chdir(tmp_path)
    t = Trainer({**TINY, "n_envs": 8, "league_bot_frac": 0.0,
                 "league_pad_bot_frac": 0.0,
                 "league_spar_bot_frac": 0.0,
                 "league_rehit_bot_frac": 0.0,
                 "league_pressure_bot_frac": 0.0,
                 "league_combo_chase_bot_frac": 1.0,
                 "rollout_ticks": 6,
                 "sim": {"target_hits": 99, "max_ticks": 4,
                         "randomize": False}}, device="cpu")
    mask = t._assign_opponents()
    assert int((~mask).sum()) == t.N
    assert t._combo_chase_rows.numel() == t.N
    assert t._bot_rows.numel() == 0
    assert t._pad_rows.numel() == 0
    assert t._spar_rows.numel() == 0
    assert t._rehit_rows.numel() == 0
    assert t._pressure_rows.numel() == 0
    m = t.train_iter()
    assert "combo_chase_combo_max" in m


def test_counter_bot_opponents_in_league(tmp_path, monkeypatch):
    """league_counter_bot_frac=1 -> agent 1 gets recovery hit-select drills."""
    monkeypatch.chdir(tmp_path)
    t = Trainer({**TINY, "n_envs": 8, "league_bot_frac": 0.0,
                 "league_pad_bot_frac": 0.0,
                 "league_spar_bot_frac": 0.0,
                 "league_rehit_bot_frac": 0.0,
                 "league_pressure_bot_frac": 0.0,
                 "league_combo_chase_bot_frac": 0.0,
                 "league_counter_bot_frac": 1.0,
                 "rollout_ticks": 6,
                 "sim": {"target_hits": 99, "max_ticks": 4,
                         "randomize": False}}, device="cpu")
    mask = t._assign_opponents()
    assert int((~mask).sum()) == t.N
    assert t._counter_rows.numel() == t.N
    assert t._bot_rows.numel() == 0
    assert t._pad_rows.numel() == 0
    assert t._spar_rows.numel() == 0
    assert t._rehit_rows.numel() == 0
    assert t._pressure_rows.numel() == 0
    assert t._combo_chase_rows.numel() == 0
    m = t.train_iter()
    assert "counter_lane_combo_max" in m
    assert "counter_lane_rechain_hit_frac" in m


def test_full_scripted_curriculum_absorbs_remainder(tmp_path, monkeypatch):
    """A 100% scripted mix must not leave a rounding/ramp mirror env."""
    monkeypatch.chdir(tmp_path)
    cfg = {
        **TINY,
        "n_envs": 16,
        "league_frac": 0.0,
        "league_bot_frac": 0.0,
        "league_pad_bot_frac": 0.10,
        "league_spar_bot_frac": 0.20,
        "league_rehit_bot_frac": 0.20,
        "league_pressure_bot_frac": 0.30,
        "league_combo_chase_bot_frac": 0.20,
    }
    t = Trainer(cfg, device="cpu")
    mask = t._assign_opponents()
    assert int((~mask).sum()) == t.N
    assert int((t._env_opp == -1).sum()) == 0
    assert t._pressure_rows.numel() == 6
    assert t._combo_chase_rows.numel() == 3

    t._ramp_pos = 0.5
    mask = t._assign_opponents()
    assert int((~mask).sum()) == t.N
    assert int((t._env_opp == -1).sum()) == 0
    assert t._pressure_rows.numel() > 3


def test_chase_bot_actions():
    from train.scripted import ChaseBot
    hist = torch.zeros(3, 4, 48)
    hist[:, -1, 36] = 40.0 / 180.0       # rot speed
    hist[:, -1, 11] = 1.0                # sin(yaw_err) = 1 -> tourner a fond
    hist[:, -1, 12] = 0.0
    a = ChaseBot().act7(hist)
    assert a.shape == (3, 7)
    assert (a[:, 0] == 1.0).all()        # dyaw saturÃƒÆ’Ã‚Â©
    assert (a[:, 2] == 1.0).all() and (a[:, 6] == 1.0).all()


def test_combo_pad_bot_actions():
    from train.scripted import ComboPadBot
    hist = torch.zeros(3, 4, 48)
    hist[:, -1, 36] = 40.0 / 180.0
    hist[:, -1, 11] = 1.0
    hist[:, -1, 12] = 0.0
    a = ComboPadBot().act7(hist)
    assert (a[:, 0] == 1.0).all()
    assert (a[:, 2] == 1.0).all()
    assert (a[:, 5] == 1.0).all()
    assert (a[:, 6] == 0.0).all()


def test_combo_spar_bot_actions():
    from train.scripted import ComboSparBot
    hist = torch.zeros(6, 4, 48)
    hist[:, -1, 36] = 40.0 / 180.0
    hist[:, -1, 12] = 1.0
    hist[:, -1, 45] = 3.0 / 8.0
    hist[0, -1, 22] = 0.5              # ahead in combo -> tap reset
    hist[0, -1, 45] = 2.10 / 8.0       # too close -> single W/Z release reset
    hist[0, -1, 40] = 1.0
    hist[0, -1, 43] = 1.0
    hist[1, -1, 21] = 0.5              # medium lane -> holds re-hit window
    hist[2, -1, 21] = 0.5              # hard lane -> counter attack
    hist[3, -1, 21] = 0.5              # contest lane -> counter attack
    hist[4, -1, 45] = 4.0 / 8.0        # too far -> no trade click
    hist[5, -1, 45] = 3.0 / 8.0        # neutral opener -> no prefire trade

    a = ComboSparBot().act7(hist)

    assert a[0, 2] == 0.0 and a[0, 5] == 0.0
    assert a[1, 6] == 0.0
    assert a[2, 6] == 0.0
    assert a[3, 6] == 1.0
    assert a[4, 2] == 1.0 and a[4, 5] == 1.0 and a[4, 6] == 0.0
    assert a[5, 2] == 1.0 and a[5, 5] == 1.0 and a[5, 6] == 0.0


def test_combo_rehit_bot_gives_rechain_window_then_punishes():
    from train.scripted import ComboRehitBot
    hist = torch.zeros(4, 4, 48)
    hist[:, -1, 36] = 40.0 / 180.0
    hist[:, -1, 12] = 1.0
    hist[:, -1, 45] = 2.80 / 8.0
    hist[0, -1, 21] = 0.60             # freshly hurt -> hold fire
    hist[1, -1, 21] = 0.20             # missed window -> counter
    hist[2, -1, 22] = 12.0 / 20.0      # ahead but re-hit not legal this tick
    hist[3, -1, 22] = 11.0 / 20.0      # ahead, click becomes legal after decrement

    a = ComboRehitBot().act7(hist)

    assert a[0, 6] == 0.0
    assert a[0, 2] == 1.0 and a[0, 5] == 1.0
    assert a[1, 6] == 1.0
    assert a[2, 6] == 0.0
    assert a[2, 2] == 1.0 and a[2, 5] == 1.0
    assert a[3, 6] == 1.0


def test_combo_pressure_bot_shortens_rechain_window():
    from train.scripted import ComboPressureBot
    hist = torch.zeros(4, 4, 48)
    hist[:, -1, 36] = 40.0 / 180.0
    hist[:, -1, 12] = 1.0
    hist[:, -1, 45] = 2.80 / 8.0
    hist[0, -1, 21] = 0.50             # easy lane still leaves a legal re-hit
    hist[1, -1, 21] = 0.50             # hard lane contests the same timing
    hist[2, -1, 21] = 0.44             # easy lane missed window -> contest
    hist[3, -1, 21] = 0.60             # hard lane fresh hurt -> still holds fire

    a = ComboPressureBot().act7(hist)

    assert a[0, 6] == 0.0
    assert a[1, 6] == 1.0
    assert a[2, 6] == 0.0
    assert a[3, 6] == 0.0


def test_combo_chase_bot_pressures_without_permanent_prefire():
    from train.scripted import ComboChaseBot
    hist = torch.zeros(5, 4, 48)
    hist[:, -1, 36] = 40.0 / 180.0
    hist[:, -1, 12] = 1.0
    hist[:, -1, 45] = 2.80 / 8.0
    hist[0, -1, 45] = 2.80 / 8.0       # neutral easy lane -> no prefire
    hist[1, -1, 45] = 2.80 / 8.0       # neutral probe lane -> close prefire
    hist[2, -1, 21] = 0.60             # behind, still leaves legal re-hit
    hist[3, -1, 21] = 0.85             # contest lane, very short window
    hist[4, -1, 22] = 0.50             # ahead in combo -> tap reset like spar
    hist[4, -1, 45] = 2.10 / 8.0
    hist[4, -1, 40] = 1.0
    hist[4, -1, 43] = 1.0

    a = ComboChaseBot().act7(hist)

    assert a[0, 6] == 0.0
    assert a[1, 6] == 1.0
    assert a[2, 6] == 0.0
    assert a[3, 6] == 1.0
    assert a[4, 2] == 0.0 and a[4, 5] == 0.0


def test_combo_chase_bot_contests_edge_overruns_only_on_hard_lanes():
    from train.scripted import ComboChaseBot
    hist = torch.zeros(4, 4, 48)
    hist[:, -1, 36] = 40.0 / 180.0
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = 0.45
    hist[:, -1, 45] = 3.50 / 8.0

    a = ComboChaseBot().act7(hist)

    assert a[0, 6] == 0.0
    assert a[1, 6] == 0.0
    assert a[2, 6] == 0.0
    assert a[3, 6] == 1.0


def test_combo_counter_bot_creates_recovery_windows():
    from train.scripted import ComboCounterBot
    hist = torch.zeros(6, 4, 48)
    hist[:, -1, 36] = 40.0 / 180.0
    hist[:, -1, 12] = 1.0
    hist[:, -1, 45] = 2.80 / 8.0
    hist[0, -1, 45] = 3.00 / 8.0       # neutral opener -> starts pressure
    hist[1, -1, 22] = 0.40             # easy lane -> recovery window
    hist[2, -1, 22] = 0.40             # hard lane -> contest
    hist[3, -1, 22] = 0.40             # contest lane -> contest
    hist[4, -1, 22] = 0.60             # fresh combo -> bot keeps pressure
    hist[5, -1, 22] = 0.50
    hist[5, -1, 45] = 2.10 / 8.0       # over-close -> release sprint/W

    a = ComboCounterBot().act7(hist)

    assert a[0, 6] == 1.0
    assert a[1, 6] == 0.0
    assert a[2, 6] == 1.0
    assert a[3, 6] == 1.0
    assert a[4, 2] == 1.0 and a[4, 5] == 1.0 and a[4, 6] == 1.0
    assert a[5, 2] == 0.0 and a[5, 5] == 0.0


def test_combo_counter_bot_exposes_clean_hit_select_recovery_window():
    from train.scripted import ComboCounterBot
    hist = torch.zeros(4, 4, 48)
    hist[:, -1, 36] = 40.0 / 180.0
    hist[:, -1, 12] = 1.0
    hist[:, -1, 22] = COUNTER_HIT_SELECT_MIN_OWN_HURT + 0.04
    hist[:, -1, 45] = (COUNTER_HIT_SELECT_CLEAN_MIN_REACH + 0.10) / 8.0

    a = ComboCounterBot().act7(hist)

    assert a[0, 2] == 0.0 and a[0, 5] == 0.0 and a[0, 6] == 0.0
    assert a[1, 2] == 0.0 and a[1, 5] == 0.0 and a[1, 6] == 0.0
    assert a[2, 2] == 0.0 and a[2, 5] == 0.0 and a[2, 6] == 0.0
    assert a[3, 6] == 1.0


def test_combo_counter_bot_keeps_pressure_outside_hit_select_recovery_window():
    from train.scripted import ComboCounterBot
    hist = torch.zeros(2, 4, 48)
    hist[:, -1, 36] = 40.0 / 180.0
    hist[:, -1, 12] = 1.0
    hist[:, -1, 22] = COUNTER_HIT_SELECT_MIN_OWN_HURT + 0.04
    hist[0, -1, 45] = (COUNTER_HIT_SELECT_CLEAN_MAX_REACH + 0.20) / 8.0
    hist[1, -1, 45] = (COUNTER_HIT_SELECT_CLEAN_MIN_REACH - 0.20) / 8.0

    a = ComboCounterBot().act7(hist)

    assert a[0, 2] == 1.0 and a[0, 5] == 1.0
    assert a[1, 6] == 1.0


def test_combo_counter_bot_exposes_close_counter_recovery_window():
    from train.scripted import ComboCounterBot
    hist = torch.zeros(4, 4, 48)
    hist[:, -1, 36] = 40.0 / 180.0
    hist[:, -1, 12] = 1.0
    hist[:, -1, 22] = COUNTER_CLOSE_RECOVERY_CLICK_HURT - 0.04
    hist[:, -1, 45] = (COUNTER_CLOSE_COUNTER_REACH - 0.05) / 8.0

    a = ComboCounterBot().act7(hist)

    assert a[0, 2] == 0.0 and a[0, 5] == 0.0 and a[0, 6] == 0.0
    assert a[1, 2] == 0.0 and a[1, 5] == 0.0 and a[1, 6] == 0.0
    assert a[2, 2] == 0.0 and a[2, 5] == 0.0 and a[2, 6] == 0.0
    assert a[3, 6] == 1.0


def test_metrics_have_automation_fields(tiny_trainer):
    m = tiny_trainer.train_iter()
    for k in ("hit_rate", "shaping", "warn_entropy", "total_steps",
              "engage_rate", "opener_strafe_frac", "opener_strafe_hold_frac",
              "opener_pressure_frac"):
        assert k in m
    assert 0.0 <= m["engage_rate"] <= 1.0
    assert 0.0 <= m["opener_strafe_frac"] <= 1.0
    assert 0.0 <= m["opener_strafe_hold_frac"] <= 1.0
    assert 0.0 <= m["opener_pressure_frac"] <= 1.0


def test_train_iter_accepts_hit_wtap_behavior_reward(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = {
        **TINY,
        "n_envs": 2,
        "rollout_ticks": 4,
        "sim": {
            **TINY["sim"],
            "reward_hit_wtap": 1.0,
            "reward_opener_strafe": 1.0,
            "reward_spar_counter": 3.0,
        },
    }
    t = Trainer(cfg, device="cpu")

    m = t.train_iter()

    assert t.cfg["behavior_reward"]["reward_hit_wtap"] == 1.0
    assert t.cfg["behavior_reward"]["reward_opener_strafe"] == 1.0
    assert t.cfg["behavior_reward"]["reward_spar_counter"] == 3.0
    assert np.isfinite(m["reward_mean"])


def test_shaping_floor_keeps_pressure(tiny_trainer):
    """shaping_floor_frac > 0 : le shaping distance ne s'ÃƒÆ’Ã‚Â©teint plus en fin
    de rampe (pression de rapprochement permanente, anti-passivitÃƒÆ’Ã‚Â©)."""
    t = tiny_trainer
    t._ramp_on = True
    t._shaping_base = 0.002
    t._ramp_pos = 1.0
    t.cfg["shaping_floor_frac"] = 0.0
    assert t._auto_shaping() == 0.0          # comportement historique
    t.cfg["shaping_floor_frac"] = 0.25
    assert abs(t._auto_shaping() - 0.002 * 0.25) < 1e-12


def test_engage_gate_rolls_back_passive_combat(tiny_trainer):
    """Un hit_rate correct ne suffit pas si le modÃƒÆ’Ã‚Â¨le reste trop loin."""
    t = tiny_trainer
    t.cfg["shaping_hit_rate"] = 20.0
    t.cfg["shaping_engage_rate"] = 0.1
    t.cfg["shaping_decay_iters"] = 10
    t._ramp_on = True
    t._ramp_pos = 0.8

    t._update_ramp(25.0, 0.03)
    assert t._ramp_pos < 0.8

    pos = t._ramp_pos
    t._update_ramp(25.0, 0.15)
    assert t._ramp_pos > pos

def test_combo_gate_rolls_back_short_chains(tiny_trainer):
    """La rampe ne doit pas s'elargir si les hits ne deviennent pas des combos."""
    t = tiny_trainer
    t.cfg["shaping_hit_rate"] = 20.0
    t.cfg["shaping_engage_rate"] = 0.1
    t.cfg["shaping_combo5_rate"] = 0.04
    t.cfg["shaping_decay_iters"] = 10
    t._ramp_on = True
    t._ramp_pos = 0.8

    t._update_ramp(30.0, 0.2, 0.005)
    assert t._ramp_pos < 0.8

    pos = t._ramp_pos
    t._update_ramp(30.0, 0.2, 0.06)
    assert t._ramp_pos > pos



def test_combo12_gate_rolls_back_until_long_chains(tiny_trainer):
    """The combo12 profile must not widen spawns on combo5 alone."""
    t = tiny_trainer
    t.cfg["shaping_hit_rate"] = 20.0
    t.cfg["shaping_engage_rate"] = 0.1
    t.cfg["shaping_combo5_rate"] = 0.04
    t.cfg["shaping_combo12_state"] = 0.45
    t.cfg["shaping_decay_iters"] = 10
    t._ramp_on = True
    t._ramp_pos = 0.8

    t._update_ramp(30.0, 0.2, 0.50, 0.01, 0.42)
    assert t._ramp_pos < 0.8

    pos = t._ramp_pos
    t._update_ramp(30.0, 0.2, 0.50, 0.01, 0.50)
    assert t._ramp_pos > pos
def test_resume_truncates_future_metrics(tiny_trainer, tmp_path):
    """Un resume coupe les lignes de mÃƒÆ’Ã‚Â©triques d'itÃƒÆ’Ã‚Â©rations > checkpoint
    (progrÃƒÆ’Ã‚Â¨s perdu d'une session tuÃƒÆ’Ã‚Â©e) : pas de doublons dans les courbes."""
    tiny_trainer.train_iter()
    path = tiny_trainer.save()                  # checkpoint ÃƒÆ’Ã‚Â  iter 1
    with open(tiny_trainer.run_dir / "metrics.jsonl", "a") as f:
        f.write('{"iter": 2}\n{"iter": 3}\n')   # progrÃƒÆ’Ã‚Â¨s non sauvegardÃƒÆ’Ã‚Â©
    t2 = Trainer(TINY, device="cpu")
    t2.load(str(path))
    lines = (t2.run_dir / "metrics.jsonl").read_text().strip().splitlines()
    iters = [json.loads(ln)["iter"] for ln in lines]
    assert max(iters) == 1


def test_resume_as_seed_resets_run_state(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    t1 = Trainer(TINY, device="cpu")
    t1.train_iter()
    path = t1.save()
    assert (t1.run_dir / "metrics.jsonl").exists()

    t2 = Trainer({**TINY, "name": "_seeded", "resume_as_seed": True,
                  "fresh_optimizer_on_resume": True}, device="cpu")
    t2.load(str(path))

    assert t2.iter == 0
    assert t2.total_steps == 0
    assert not (t2.run_dir / "metrics.jsonl").exists()


def test_resume_as_seed_keeps_same_run_state(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    t1 = Trainer(TINY, device="cpu")
    t1.train_iter()
    path = t1.save()

    t2 = Trainer({**TINY, "resume_as_seed": True,
                  "fresh_optimizer_on_resume": True}, device="cpu")
    t2.load(str(path))

    assert t2.iter == t1.iter
    assert t2.total_steps == t1.total_steps
    assert (t2.run_dir / "metrics.jsonl").exists()


def test_resume_same_run_as_seed_resets_same_run_state(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    t1 = Trainer(TINY, device="cpu")
    t1.train_iter()
    path = t1.save()
    assert (t1.run_dir / "metrics.jsonl").exists()

    t2 = Trainer({**TINY, "resume_as_seed": True,
                  "resume_same_run_as_seed": True,
                  "fresh_optimizer_on_resume": True}, device="cpu")
    t2.load(str(path))

    assert t2.iter == 0
    assert t2.total_steps == 0
    assert not (t2.run_dir / "metrics.jsonl").exists()
    assert (t2.run_dir / "metrics-001.jsonl").exists()


def test_same_run_resume_reapplies_current_ppo_lr(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    old_cfg = {
        **TINY,
        "ppo": {**TINY["ppo"], "lr": 1.0e-5},
    }
    t1 = Trainer(old_cfg, device="cpu")
    t1.train_iter()
    path = t1.save()

    new_cfg = {
        **TINY,
        "ppo": {**TINY["ppo"], "lr": 8.0e-6},
    }
    t2 = Trainer(new_cfg, device="cpu")
    t2.load(str(path))

    assert t2.ppo.cfg.lr == pytest.approx(8.0e-6)
    assert t2.ppo.opt.param_groups[0]["lr"] == pytest.approx(8.0e-6)


def test_train_metric_lr_keeps_small_learning_rate_precision():
    assert _round_train_stat("lr", 7.1808e-6) == pytest.approx(7.18e-6)
    assert _round_train_stat("loss_pi", 0.1234567) == pytest.approx(0.12346)


@pytest.mark.skipif(not _has_cuda_build_toolchain(),
                    reason="CUDA + toolchain C++ requis")
def test_resume_old_checkpoint_into_fused_adam(tmp_path, monkeypatch):
    """Un checkpoint sauvÃƒÆ’Ã‚Â© par l'Adam non-fused (steps CPU) doit se charger
    dans l'Adam fused (CUDA) sans dÃƒÆ’Ã‚Â©clencher l'assertion
    Ãƒâ€šÃ‚Â« Expected grad_scale and found_inf to be None Ãƒâ€šÃ‚Â» au premier update."""
    monkeypatch.chdir(tmp_path)
    t1 = Trainer(TINY, device="cpu")
    t1.train_iter()
    path = t1.save()

    t2 = Trainer({**TINY, "ppo": {**TINY["ppo"], "amp": True}}, device="cuda")
    t2.load(str(path))
    m = t2.train_iter()
    assert np.isfinite(m["reward_mean"])


# ----------------------------------------------------------------------- PBT
def test_pbt_perturb_within_bounds():
    from train.pbt import perturb_hypers
    rng = random.Random(0)
    base = {"lr": 3e-4, "ent_coef": 0.005, "clip": 0.2}
    explore = {"lr": [6e-5, 6e-4], "ent_coef": [0.002, 0.02], "clip": [0.1, 0.3]}
    for _ in range(50):
        h = perturb_hypers(base, explore, 0.8, 1.25, rng)
        for key, (lo, hi) in explore.items():
            assert lo <= h[key] <= hi
            # x0.8 ou x1.25 (bornÃƒÆ’Ã‚Â©) : jamais identique ÃƒÆ’Ã‚Â  la base ici
            assert h[key] != base[key]


def test_pbt_exploit_copies_top():
    """Le membre du bas copie poids + hypers (perturbÃƒÆ’Ã‚Â©s) du membre du haut."""
    from train.pbt import DEFAULT_PBT, Member, exploit_explore
    from train.ppo import PPO, PPOConfig
    dev = torch.device("cpu")
    cfg_pol = PolicyConfig(history=4, d_model=32, n_heads=2, n_layers=1)
    top_pol, low_pol = JudasPolicy(cfg_pol), JudasPolicy(cfg_pol)
    top = Member(0, top_pol, PPO(top_pol, PPOConfig(amp=False), dev),
                 {"lr": 3e-4, "ent_coef": 0.005, "clip": 0.2}, elo=1500.0)
    low = Member(1, low_pol, PPO(low_pol, PPOConfig(amp=False), dev),
                 {"lr": 1e-4, "ent_coef": 0.01, "clip": 0.25}, elo=900.0)
    cfg = {**DEFAULT_PBT, "truncation": 0.5}

    events = exploit_explore([top, low], cfg, random.Random(0))

    assert events == [(1, 0)]
    for k, v in low_pol.state_dict().items():
        assert torch.equal(v, top_pol.state_dict()[k])
    assert low.elo == top.elo
    for key, (lo, hi) in cfg["explore"].items():
        assert lo <= low.hypers[key] <= hi
        assert low.hypers[key] != top.hypers[key]
    # les hypers perturbÃƒÆ’Ã‚Â©s sont APPLIQUÃƒÆ’Ã¢â‚¬Â°S dans l'optimiseur
    assert low.ppo.opt.param_groups[0]["lr"] == low.hypers["lr"]


def test_pbt_smoke_two_iters(tmp_path, monkeypatch):
    """Population 2 : rollout multi-membres, ELO cross-play, exploit/explore
    et mÃƒÆ’Ã‚Â©triques ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â deux itÃƒÆ’Ã‚Â©rations complÃƒÆ’Ã‚Â¨tes sans erreur."""
    monkeypatch.chdir(tmp_path)
    t = Trainer(PBT_TINY, device="cpu")
    assert len(t.members) == 2
    assert t.members[0].policy is not t.members[1].policy
    m1 = t.train_iter()
    m2 = t.train_iter()
    for m in (m1, m2):
        assert np.isfinite(m["reward_mean"])
        assert np.isfinite(m["approx_kl"])
    assert len(m2["pbt_elo"]) == 2
    assert m2["pbt_best"] in (0, 1)
    assert len(m2["pbt_lr"]) == 2


def test_pbt_seed_from_single_checkpoint(tmp_path, monkeypatch):
    """Un checkpoint single-policy seed TOUTE la population (lignÃƒÆ’Ã‚Â©e
    conservÃƒÆ’Ã‚Â©e) ; les hypers restent diversifiÃƒÆ’Ã‚Â©s par l'init."""
    monkeypatch.chdir(tmp_path)
    t1 = Trainer(TINY, device="cpu")
    t1.train_iter()
    path = t1.save()

    t2 = Trainer(PBT_TINY, device="cpu")
    t2.load(str(path))
    ref = torch.load(path, map_location="cpu", weights_only=False)["policy"]
    for mb in t2.members:
        sd = mb.policy.state_dict()
        for k in ref:
            assert torch.equal(sd[k], ref[k]), f"membre {mb.idx}: {k} diverge"


def test_pbt_checkpoint_roundtrip(tmp_path, monkeypatch):
    """Sauvegarde/restauration complÃƒÆ’Ã‚Â¨te de la population (poids par membre,
    hypers, elo) + tÃƒÆ’Ã‚Âªte de checkpoint = meilleur membre (compat export)."""
    monkeypatch.chdir(tmp_path)
    t1 = Trainer(PBT_TINY, device="cpu")
    t1.train_iter()
    t1.members[0].elo = 1234.5
    t1.members[1].hypers["lr"] = 1.1e-4
    path = t1.save()

    t2 = Trainer(PBT_TINY, device="cpu")
    t2.load(str(path))
    assert abs(t2.members[0].elo - 1234.5) < 1e-9
    assert abs(t2.members[1].hypers["lr"] - 1.1e-4) < 1e-12
    for m1, m2 in zip(t1.members, t2.members):
        sd1, sd2 = m1.policy.state_dict(), m2.policy.state_dict()
        for k in sd1:
            assert torch.equal(sd1[k], sd2[k])


@pytest.mark.skipif(not _has_cuda_build_toolchain(),
                    reason="CUDA + toolchain C++ requis")
def test_pbt_cuda_graphs_population(tmp_path, monkeypatch):
    """Mode population sur GPU : capture des graphs par membre puis replay
    sur plusieurs itÃƒÆ’Ã‚Â©rations (le 2e tour exerce le chemin replay + cat)."""
    monkeypatch.chdir(tmp_path)
    cfg = {**PBT_TINY, "n_envs": 16, "ppo": {**TINY["ppo"], "amp": True}}
    t = Trainer(cfg, device="cuda")
    m1 = t.train_iter()
    m2 = t.train_iter()
    assert np.isfinite(m1["reward_mean"])
    assert np.isfinite(m2["reward_mean"])
    assert len(m2["pbt_elo"]) == 2


def test_metrics_rotation_on_fresh_start(tmp_path, monkeypatch):
    """Un run frais archive le metrics.jsonl du run prÃƒÆ’Ã‚Â©cÃƒÆ’Ã‚Â©dent du mÃƒÆ’Ã‚Âªme nom
    (les courbes de l'app ne doivent pas concatÃƒÆ’Ã‚Â©ner les runs)."""
    monkeypatch.chdir(tmp_path)
    t = Trainer(TINY, device="cpu")
    (t.run_dir / "metrics.jsonl").write_text('{"iter": 1}\n')
    t.rotate_metrics()
    assert not (t.run_dir / "metrics.jsonl").exists()
    assert (t.run_dir / "metrics-001.jsonl").read_text() == '{"iter": 1}\n'
    # une 2e rotation n'ÃƒÆ’Ã‚Â©crase pas l'archive existante
    (t.run_dir / "metrics.jsonl").write_text('{"iter": 2}\n')
    t.rotate_metrics()
    assert (t.run_dir / "metrics-002.jsonl").exists()


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
    meta = json.loads(out.with_suffix(".json").read_text())
    hist = torch.zeros(1, 4, 48)
    a = mod(hist)
    assert a.shape == (1, 7)
    assert (a[:, 0:2].abs() <= 1.0).all()
    assert meta["source"] == str(ckpt)
    assert len(meta["source_sha256"]) == 64
    assert meta["source_size"] == ckpt.stat().st_size


def test_export_direct_lock_does_not_click_during_combo_wait_s_tap(tmp_path, monkeypatch):
    from train.export import export
    monkeypatch.chdir(tmp_path)
    cfg = {
        **TINY,
        "policy": {
            **TINY["policy"],
            "direct_movement_lock": True,
        },
    }
    trainer = Trainer(cfg, device="cpu")
    with torch.no_grad():
        trainer.policy.fwd_head.bias[1] = 20.0
        trainer.policy.fwd_head.bias[2] = 0.0
        trainer.policy.bin_head.bias[1] = 10.0
        trainer.policy.bin_head.bias[2] = 10.0
    ckpt = trainer.save()
    out = export(str(ckpt), str(tmp_path / "direct.pts"))
    mod = torch.jit.load(str(out)).eval()

    hist = torch.zeros(1, 4, 48)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 22] = 0.70
    hist[:, -1, 45] = 3.25 / 8.0
    hist[:, -1, 40] = 1.0
    hist[:, -1, 43] = 1.0

    a = mod(hist)

    assert a[0, 2] == 0.0
    assert a[0, 5] == 0.0
    assert a[0, 6] == 0.0


def test_export_direct_lock_does_not_reenable_unreliable_edge_rehit(tmp_path, monkeypatch):
    from train.export import export
    monkeypatch.chdir(tmp_path)
    cfg = {
        **TINY,
        "policy": {
            **TINY["policy"],
            "direct_movement_lock": True,
        },
    }
    trainer = Trainer(cfg, device="cpu")
    with torch.no_grad():
        trainer.policy.fwd_head.bias[1] = 20.0
        trainer.policy.fwd_head.bias[2] = 10.0
        trainer.policy.bin_head.bias[1] = 10.0
        trainer.policy.bin_head.bias[2] = 10.0
    ckpt = trainer.save()
    out = export(str(ckpt), str(tmp_path / "direct_edge.pts"))
    mod = torch.jit.load(str(out)).eval()

    hist = torch.zeros(1, 4, 48)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = 0.0
    hist[:, -1, 22] = 0.40
    hist[:, -1, 45] = 3.38 / 8.0
    hist[:, -1, 40] = -1.0
    hist[:, -1, 43] = 0.0

    a = mod(hist)

    assert a[0, 2] == 0.0
    assert a[0, 5] == 0.0
    assert a[0, 6] == 0.0


def test_export_direct_lock_reenters_without_far_under_combo_trade_click(tmp_path, monkeypatch):
    from train.export import export
    monkeypatch.chdir(tmp_path)
    cfg = {
        **TINY,
        "policy": {
            **TINY["policy"],
            "direct_movement_lock": True,
        },
    }
    trainer = Trainer(cfg, device="cpu")
    with torch.no_grad():
        trainer.policy.fwd_head.bias[1] = 20.0
        trainer.policy.fwd_head.bias[2] = 0.0
        trainer.policy.bin_head.bias[2] = -10.0
    ckpt = trainer.save()
    out = export(str(ckpt), str(tmp_path / "direct_counter_edge.pts"))
    mod = torch.jit.load(str(out)).eval()

    hist = torch.zeros(1, 4, 48)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = COUNTER_RECOVERY_CLICK_HURT - 0.05
    hist[:, -1, 22] = 0.0
    hist[:, -1, 45] = 3.52 / 8.0

    a = mod(hist)

    assert a[0, 2] == 1.0
    assert a[0, 5] == 1.0
    assert a[0, 6] == 0.0


def test_export_direct_lock_blocks_dirty_midrange_under_combo_trade_click(tmp_path, monkeypatch):
    from train.export import export
    monkeypatch.chdir(tmp_path)
    cfg = {
        **TINY,
        "policy": {
            **TINY["policy"],
            "direct_movement_lock": True,
        },
    }
    trainer = Trainer(cfg, device="cpu")
    with torch.no_grad():
        trainer.policy.fwd_head.bias[1] = 20.0
        trainer.policy.fwd_head.bias[2] = 0.0
        trainer.policy.bin_head.bias[2] = 10.0
    ckpt = trainer.save()
    out = export(str(ckpt), str(tmp_path / "direct_midrange_counter.pts"))
    mod = torch.jit.load(str(out)).eval()

    hist = torch.zeros(1, 4, 48)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = COUNTER_RECOVERY_CLICK_HURT - 0.05
    hist[:, -1, 22] = 0.0
    hist[:, -1, 45] = ((2.10 + COUNTER_HIT_SELECT_MIN_REACH) * 0.5) / 8.0

    a = mod(hist)

    assert a[0, 2] == 1.0
    assert a[0, 5] == 1.0
    assert a[0, 6] == 0.0


def test_export_direct_lock_leaves_hit_select_click_to_policy(tmp_path, monkeypatch):
    from train.export import export
    monkeypatch.chdir(tmp_path)
    cfg = {
        **TINY,
        "policy": {
            **TINY["policy"],
            "direct_movement_lock": True,
            "leaderboard_boxing": True,
            "direct_counter_attack_lock": True,
            "direct_hit_select_attack_lock": False,
        },
    }
    trainer = Trainer(cfg, device="cpu")
    with torch.no_grad():
        trainer.policy.fwd_head.bias[1] = 20.0
        trainer.policy.fwd_head.bias[2] = 0.0
        trainer.policy.bin_head.bias[2] = -10.0
    ckpt = trainer.save()
    out = export(str(ckpt), str(tmp_path / "direct_hit_select_policy.pts"))
    mod = torch.jit.load(str(out)).eval()

    hist = torch.zeros(1, 4, 48)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = COUNTER_HIT_SELECT_CLEAN_HURT - 0.03
    hist[:, -1, 22] = 0.0
    hist[:, -1, 37] = 0.10
    hist[:, -1, 45] = (COUNTER_HIT_SELECT_CLEAN_MIN_REACH + 0.10) / 8.0

    a = mod(hist)
    assert a[0, 2] == 0.0
    assert a[0, 5] == 0.0
    assert a[0, 6] == 0.0

    hist[:, -1, 21] = COUNTER_RECOVERY_CLICK_HURT - 0.05
    hist[:, -1, 45] = (COUNTER_CLOSE_COUNTER_REACH - 0.10) / 8.0
    a = mod(hist)
    assert a[0, 6] == 1.0


def test_export_direct_lock_soft_bias_clicks_legal_hit_select(tmp_path, monkeypatch):
    from train.export import export
    monkeypatch.chdir(tmp_path)
    cfg = {
        **TINY,
        "policy": {
            **TINY["policy"],
            "direct_movement_lock": True,
            "leaderboard_boxing": True,
            "direct_counter_attack_lock": True,
            "direct_hit_select_attack_lock": False,
            "direct_hit_select_attack_bias": 12.0,
        },
    }
    trainer = Trainer(cfg, device="cpu")
    with torch.no_grad():
        trainer.policy.fwd_head.bias[1] = 20.0
        trainer.policy.fwd_head.bias[2] = 0.0
        trainer.policy.bin_head.bias[2] = -10.0
    ckpt = trainer.save()
    out = export(str(ckpt), str(tmp_path / "direct_hit_select_bias.pts"))
    mod = torch.jit.load(str(out)).eval()

    hist = torch.zeros(1, 4, 48)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = COUNTER_HIT_SELECT_CLEAN_HURT - 0.03
    hist[:, -1, 22] = 0.0
    hist[:, -1, 23] = 0.0
    hist[:, -1, 37] = 0.10
    hist[:, -1, 45] = (COUNTER_HIT_SELECT_CLEAN_MIN_REACH + 0.10) / 8.0

    a = mod(hist)
    assert a[0, 2] == 0.0
    assert a[0, 5] == 0.0
    assert a[0, 6] == 1.0

    hist[:, -1, 45] = ((COUNTER_CLOSE_COUNTER_REACH + COUNTER_HIT_SELECT_MIN_REACH) * 0.5) / 8.0
    a = mod(hist)
    assert a[0, 2] == 1.0
    assert a[0, 5] == 1.0
    assert a[0, 6] == 0.0


def test_export_direct_lock_waits_for_under_combo_recovery_timing(tmp_path, monkeypatch):
    from train.export import export
    monkeypatch.chdir(tmp_path)
    cfg = {
        **TINY,
        "policy": {
            **TINY["policy"],
            "direct_movement_lock": True,
        },
    }
    trainer = Trainer(cfg, device="cpu")
    with torch.no_grad():
        trainer.policy.fwd_head.bias[1] = 20.0
        trainer.policy.fwd_head.bias[2] = 0.0
        trainer.policy.bin_head.bias[2] = 10.0
    ckpt = trainer.save()
    out = export(str(ckpt), str(tmp_path / "direct_counter_timing.pts"))
    mod = torch.jit.load(str(out)).eval()

    hist = torch.zeros(1, 4, 48)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = COUNTER_RECOVERY_CLICK_HURT + 0.10
    hist[:, -1, 22] = 0.0
    hist[:, -1, 45] = (COUNTER_HIT_REACH - 0.05) / 8.0

    a = mod(hist)

    assert a[0, 2] == 1.0
    assert a[0, 5] == 1.0
    assert a[0, 6] == 0.0


def test_export_direct_lock_z_tap_attacks_ready_combo_rehit(tmp_path, monkeypatch):
    from train.export import export
    monkeypatch.chdir(tmp_path)
    cfg = {
        **TINY,
        "policy": {
            **TINY["policy"],
            "direct_movement_lock": True,
        },
    }
    trainer = Trainer(cfg, device="cpu")
    with torch.no_grad():
        trainer.policy.fwd_head.bias[1] = 0.0
        trainer.policy.fwd_head.bias[2] = 20.0
        trainer.policy.bin_head.bias[2] = -10.0
    ckpt = trainer.save()
    out = export(str(ckpt), str(tmp_path / "direct_ztap_rehit.pts"))
    mod = torch.jit.load(str(out)).eval()

    hist = torch.zeros(1, 4, 48)
    hist[:, -1, 12] = 1.0
    hist[:, -1, 21] = 0.0
    hist[:, -1, 22] = 0.40
    hist[:, -1, 40] = 1.0
    hist[:, -1, 43] = 1.0
    hist[:, -1, 45] = 3.25 / 8.0

    a = mod(hist)

    assert a[0, 2] == 0.0
    assert a[0, 5] == 0.0
    assert a[0, 6] == 1.0


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
    # values nulles -> adv = somme discountÃƒÆ’Ã‚Â©e des rewards jusqu'au done
    assert abs(buf.adv[2, 0].item() - 1.0) < 1e-6      # done : pas de suite
    assert abs(buf.adv[3, 0].item() - 1.9) < 1e-6      # 1 + 0.9 (fin de buffer)
    assert abs(buf.adv[0, 0].item() - 2.71) < 1e-6     # 1 + 0.9 * adv[1]


def test_windows_mask_pre_episode_history():
    """windows() reconstruit l'historique et masque les ticks d'AVANT le
    reset d'ÃƒÆ’Ã‚Â©pisode (age) ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â exactement ce que voit la policy au rollout."""
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
    # t=3 (age 1) : fenÃƒÆ’Ã‚Âªtre brute [2, 3, 4], le 1er tick prÃƒÆ’Ã‚Â©cÃƒÆ’Ã‚Â¨de le reset
    win = buf.windows(torch.tensor([3]), torch.tensor([0]))
    assert win.shape == (1, H, 1)
    np.testing.assert_allclose(win[0, :, 0].numpy(), [0.0, 3.0, 4.0])
    # t=1 (age 6 >= H) : fenÃƒÆ’Ã‚Âªtre complÃƒÆ’Ã‚Â¨te [prefix[-1], obs[0], obs[1]]
    win = buf.windows(torch.tensor([1]), torch.tensor([0]))
    np.testing.assert_allclose(win[0, :, 0].numpy(), [-1.0, 1.0, 2.0])


def test_export_parity_with_policy(tiny_trainer, tmp_path):
    """Le .pts exportÃƒÆ’Ã‚Â© doit produire EXACTEMENT l'action dÃƒÆ’Ã‚Â©terministe de la
    policy d'entraÃƒÆ’Ã‚Â®nement sur les mÃƒÆ’Ã‚Âªmes obs ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â paritÃƒÆ’Ã‚Â© train <-> infÃƒÆ’Ã‚Â©rence."""
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
        "export TorchScript != policy dÃƒÆ’Ã‚Â©terministe sur les mÃƒÆ’Ã‚Âªmes obs"
