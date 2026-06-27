"""Tests du daemon : protocole, LiveSession, endpoints REST."""

import hashlib
import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient            # noqa: E402

from serve import daemon                             # noqa: E402
from serve.daemon import app, live, _normalize_training_cfg  # noqa: E402
from serve.live import LiveSession                   # noqa: E402
from serve.protocol import ArenaCalib, action_to_msg, player_from_msg  # noqa: E402
from serve.training_manager import TrainingManager   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def state_msg(z_self=3.0, z_target=5.5):
    def p(z):
        return {"x": 109.0, "y": 64.0, "z": 100.0 + z, "vx": 0.0, "vy": 0.0,
                "vz": 0.0, "yaw": 0.0, "pitch": 0.0, "onGround": True,
                "sprinting": False, "hurtTime": 0, "hits": 0}
    return {"t": "state", "tick": 1, "self": p(z_self), "target": p(z_target)}


ARENA = ArenaCalib(origin_x=100.0, origin_z=100.0, size_x=18.0, size_z=18.0,
                   floor_y=64.0)


def write_combo_safe_contract(safe_path: Path, **overrides) -> Path:
    payload = {
        "score_schema": 8,
        "requires_chase_combo": True,
        "score": [0.8, 0.5, 0.4, 12.0, 0.25, 0.2, 70.0, 0.0],
        "combo_tap_frac": 0.25,
        "combo_s_tap_frac": 0.0,
        "combo_z_tap_frac": 0.25,
        "hit_wtap_frac": 0.90,
        "under_combo_counter_hit_frac": 0.20,
        "under_combo_hit_select_clean_frac": 0.25,
        "under_combo_hit_select_trade_frac": 0.05,
        "back_frac": 0.0,
        "strafe_frac": 0.60,
        "opener_strafe_frac": 0.80,
        "opener_strafe_hold_frac": 0.75,
        "opener_pressure_frac": 0.65,
        "safety_back_frac": 0.002,
        "safety_min_strafe_frac": 0.50,
        "safety_min_opener_strafe_frac": 0.75,
        "safety_min_opener_strafe_hold_frac": 0.70,
        "safety_min_opener_pressure_frac": 0.60,
        "safety_min_combo_tap_frac": 0.12,
        "safety_min_combo_z_tap_frac": 0.10,
        "safety_max_combo_s_tap_frac": 0.02,
        "safety_min_hit_wtap_frac": 0.75,
        "safety_min_under_combo_counter_hit_frac": 0.05,
        "safety_min_under_combo_hit_select_clean_frac": 0.20,
        "safety_max_under_combo_hit_select_trade_frac": 0.12,
        "safety_opener_ticks": 20,
    }
    payload.update(overrides)
    path = safe_path.with_name("safe_latest.meta.json")
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_player_from_msg_arena_frame():
    p = player_from_msg(state_msg()["self"], ARENA)
    assert p.x == 9.0 and p.z == 3.0 and p.y == 0.0
    assert p.on_ground


def test_hurt_time_rescaled():
    msg = state_msg()["self"]
    msg["hurtTime"] = 9
    assert player_from_msg(msg, ARENA).hurt_resistant_time == 18


def test_hurt_resistant_time_exact_when_present():
    # le mod (thread jeu) fournit le champ exact : il prime sur hurtTime*2
    msg = state_msg()["self"]
    msg["hurtTime"] = 9
    msg["hurtResistantTime"] = 14
    assert player_from_msg(msg, ARENA).hurt_resistant_time == 14


def test_jump_ticks_passthrough():
    msg = state_msg()["self"]
    msg["jumpTicks"] = 7
    assert player_from_msg(msg, ARENA).jump_ticks == 7
    # absent -> 0 (rétro-compatible)
    assert player_from_msg(state_msg()["self"], ARENA).jump_ticks == 0


def test_click_cooldown_passthrough_for_opponent_timing():
    msg = state_msg()["target"]
    msg["clickCooldown"] = 2
    assert player_from_msg(msg, ARENA).click_cooldown == 2
    assert player_from_msg(state_msg()["target"], ARENA).click_cooldown == 0


def test_build_obs_slot_37_exposes_opponent_click_cooldown():
    from sim.obs import build_obs
    from sim.config import SimConfig
    from sim_ref import HumanizationConfig

    own = player_from_msg(state_msg()["self"], ARENA)
    opp_msg = state_msg()["target"]
    opp_msg["clickCooldown"] = 2
    opp = player_from_msg(opp_msg, ARENA)

    obs = build_obs(own, opp, SimConfig(arena_size_x=18, arena_size_z=18),
                    HumanizationConfig(action_delay=7), [0.0] * 7, 0)

    assert obs[37] == pytest.approx(0.1)


def test_auto_frame_recenters_world_coords():
    """Box loin de l'origine monde : le repère AUTO recentre les joueurs ->
    murs (obs[25..28]) / hauteurs (obs[46,47]) restent in-distribution, au lieu
    de partir hors plage (cause de la visée au sol observée en jeu)."""
    from sim.obs import build_obs
    from sim.config import SimConfig
    from sim_ref import HumanizationConfig

    def world(x, y, z):
        return {"x": x, "y": y, "z": z, "vx": 0., "vy": 0., "vz": 0.,
                "yaw": 0., "pitch": 0., "onGround": True, "sprinting": False,
                "hurtTime": 0, "hits": 0}

    msg = {"self": world(109., 64., 100.), "target": world(109., 64., 103.)}
    s = LiveSession(device="cpu")
    s.params.arena = ArenaCalib(0.0, 0.0, 18.0, 18.0, 0.0)   # défaut NON calibré
    s.reset()

    arena = s._resolve_arena(msg)                # auto_frame=True par défaut
    own = player_from_msg(msg["self"], arena)
    opp = player_from_msg(msg["target"], arena)
    assert abs(own.x - 9.0) < 1.5                # recentré (~size/2), pas 109
    assert abs(own.y) < 1e-6 and abs(opp.y) < 1e-6   # sol = min Y

    obs = build_obs(own, opp, SimConfig(arena_size_x=18, arena_size_z=18),
                    HumanizationConfig(), [0.0] * 7, 0)
    for i in (25, 26, 27, 28):
        assert -1.0 <= obs[i] <= 3.0, f"mur obs[{i}]={obs[i]} hors distribution"
    assert abs(obs[46]) < 0.2 and abs(obs[47]) < 0.2

    # auto_frame désactivé -> calibration explicite respectée (pas de recentrage)
    s.auto_frame = False
    s._frame = None
    assert s._resolve_arena(msg) is s.params.arena


def test_auto_frame_prefers_mod_detected_arena_and_updates_size():
    def world(x, y, z):
        return {"x": x, "y": y, "z": z, "vx": 0., "vy": 0., "vz": 0.,
                "yaw": 0., "pitch": 0., "onGround": True, "sprinting": False,
                "hurtTime": 0, "hits": 0}

    msg = {"self": world(109., 64., 100.), "target": world(109., 64., 103.),
           "arena": {"origin_x": 96.0, "origin_z": 90.0,
                     "size_x": 28.0, "size_z": 32.0, "floor_y": 63.0}}
    s = LiveSession(device="cpu")
    s.params.arena = ArenaCalib(0.0, 0.0, 40.0, 40.0, 0.0)
    s.reset()

    arena = s._resolve_arena(msg)

    assert arena.origin_x == pytest.approx(96.0)
    assert arena.origin_z == pytest.approx(90.0)
    assert arena.size_x == pytest.approx(28.0)
    assert arena.size_z == pytest.approx(32.0)
    assert arena.floor_y == pytest.approx(63.0)
    own = player_from_msg(msg["self"], arena)
    assert own.x == pytest.approx(13.0)
    assert own.y == pytest.approx(1.0)

    msg["arena"] = {"origin_x": 104.0, "origin_z": 94.0,
                    "size_x": 18.0, "size_z": 22.0, "floor_y": 64.0}
    arena = s._resolve_arena(msg)

    assert arena.origin_x == pytest.approx(104.0)
    assert arena.size_x == pytest.approx(18.0)
    assert arena.size_z == pytest.approx(22.0)
    assert arena.floor_y == pytest.approx(64.0)


def test_action_to_msg_thresholds():
    m = action_to_msg([0.5, -0.2, 1.0, -1.0, 0.0, 1.0, 1.0])
    assert m["forward"] == 1 and m["strafe"] == -1
    assert m["jump"] is False and m["sprint"] is True and m["attack"] is True


def _session_with_model(tmp_path) -> LiveSession:
    """Session avec un vrai modèle TorchScript minuscule (via train.export)."""
    from train.export import export
    from train.model import JudasPolicy, PolicyConfig

    pol_cfg = PolicyConfig(history=4, d_model=32, n_heads=2, n_layers=1)
    pol = JudasPolicy(pol_cfg)
    ckpt = tmp_path / "ckpt.pt"
    torch.save({"policy": pol.state_dict(), "policy_cfg": pol_cfg.__dict__,
                "iter": 0}, ckpt)
    path = export(str(ckpt), str(tmp_path / "tiny.pts"))

    s = LiveSession(device="cpu")
    s.load(str(path))
    s.params.arena = ARENA
    return s


class ConstantActionModel(torch.nn.Module):
    def __init__(self, action):
        super().__init__()
        self.register_buffer("action", torch.tensor(action, dtype=torch.float32))

    def forward(self, hist):
        return self.action.repeat(hist.shape[0], 1)


def test_live_load_applies_exported_combo_runtime_metadata(tmp_path):
    from train.export import export
    from train.model import JudasPolicy, PolicyConfig

    pol_cfg = PolicyConfig(history=4, d_model=32, n_heads=2, n_layers=1)
    pol = JudasPolicy(pol_cfg)
    ckpt = tmp_path / "combo.pt"
    torch.save({
        "policy": pol.state_dict(),
        "policy_cfg": pol_cfg.__dict__,
        "iter": 0,
        "cfg": {"sim": {
            "cps_min": 18.0,
            "cps_max": 20.0,
            "rot_speed_min": 150.0,
            "rot_speed_max": 240.0,
            "arena_size_x": 40.0,
            "arena_size_z": 40.0,
            "aim_smooth_min": 0.0,
            "aim_smooth_max": 0.04,
        }},
    }, ckpt)
    model = export(str(ckpt), str(tmp_path / "combo.pts"))

    s = LiveSession(device="cpu")
    s.params.max_cps = 12.0
    s.params.max_rot_speed = 40.0
    s.params.arena = ArenaCalib(size_x=18.0, size_z=18.0)
    s.load(str(model))

    assert s.params.max_cps == 19.0
    assert s.params.max_rot_speed == 195.0
    assert s.params.arena.size_x == 40.0
    assert s.params.arena.size_z == 40.0
    assert s.aim_smooth == 0.02


def test_live_app_defaults_match_combo_runtime_contract():
    src = (ROOT / "app/src/pages/Live.jsx").read_text(encoding="utf-8")

    assert "judas:app:live:v47" in src
    assert "judas:app:live:v46" not in src
    assert "judas:app:live:v45" not in src
    assert "judas:app:live:v44" not in src
    assert "judas:app:live:v43" not in src
    assert "judas:app:live:v41" not in src
    assert "judas:app:live:v38" not in src
    assert "judas:app:live:v37" not in src
    assert "judas:app:live:v31" not in src
    assert "judas:app:live:v30" not in src
    assert "judas:app:live:v29" not in src
    assert "judas:app:live:v27" not in src
    assert "judas:app:live:v24" not in src
    assert "judas:app:live:v23" not in src
    assert "judas:app:live:v22" not in src
    assert "judas:app:live:v21" not in src
    assert "judas:app:live:v20" not in src
    assert "combo_god_leaderboard10_combo12-safe_latest" in src
    assert "combo_god_countertap96_combo12-safe_latest" in src
    assert "combo_god_directpad_lock_combo12-safe_latest" in src
    assert "cps: 10" in src
    assert "rot: 190" in src
    assert "counterAssist: false" in src
    assert "counter_assist" in src
    assert "counter assist" in src
    assert "aimSmoothing: false" in src
    assert "aimSmoothingStrength: 0.22" in src
    assert "aimSmoothingSnap: 0.55" in src
    assert "aim_smoothing" in src
    assert "aim smoothing" in src
    assert "smoothing strength" in src
    assert "smoothing snap" in src
    assert "hitSelectAssist: false" in src
    assert "hit_select_assist" in src
    assert "perfect hit-select" in src
    assert "hit-select assist" in src
    assert "autoGapple: true" in src
    assert "autoGappleCriticalHealth: 8" in src
    assert "autoGappleSafeDistance: 11.50" in src
    assert "autoGappleRetreat: true" in src
    assert "autoGappleRetreatDistance: 18" in src
    assert "autoGappleFastRetreat: true" in src
    assert "autoGappleRetreatHops: true" in src
    assert "autoGappleSprintHopHold: true" in src
    assert "autoGappleAvoidObstacles: true" in src
    assert "autoGappleRetreatStrafe: true" in src
    assert "autoGappleWallSlide: true" in src
    assert "autoGappleSpeedLock: true" in src
    assert "autoGappleVelocityAssist: true" in src
    assert "autoGappleSpeedFirst: true" in src
    assert "autoGappleFullSpeed: true" in src
    assert "autoGappleSpeedFloor: 4.50" in src
    assert "autoGappleMaxSpeed: 4.80" in src
    assert "autoGappleAccel: 5.50" in src
    assert "autoGappleSprintRetap: true" in src
    assert "autoGappleSprintRetapTicks: 2" in src
    assert "autoGappleAirControl: true" in src
    assert "autoGappleStepAssist: true" in src
    assert "autoGappleStepHeight: 1.20" in src
    assert "autoGappleFallbackRetreat: true" in src
    assert "autoGappleRetreatInputLock: true" in src
    assert "autoGappleForceSprintRetreat: true" in src
    assert "autoGappleReleaseRetreatOnHit: true" in src
    assert "autoGappleCriticalRearmOnly: true" in src
    assert "autoGappleCriticalTrappedEat: true" in src
    assert "autoGappleRetreatTurnDeg: 360" in src
    assert "autoGappleEatingRetreatTurnDeg: 360" in src
    assert "autoGappleRetreatPathHoldTicks: 2" in src
    assert "autoGappleRetreatStuckAbortTicks: 4" in src
    assert "autoGappleRetreatMinTicks: 0" in src
    assert "autoGappleRetreatMaxTicks: 64" in src
    assert "autoGappleCriticalRetreatMaxTicks: 6" in src
    assert "autoGappleCriticalEatCommitTicks: 12" in src
    assert "autoGappleCombatRecoveryTicks: 6" in src
    assert "autoGappleRetreatStrafeHoldTicks: 5" in src
    assert "autoGappleRetreatObstacleJumpHoldTicks: 60" in src
    assert "autoGappleRetreatObstacleEscapeTicks: 120" in src
    assert "autoGappleRetreatPanicSpeed: true" in src
    assert "autoGappleRetreatObstacleLookahead: 24.00" in src
    assert "autoGappleCriticalTrappedStuckTicks: 2" in src
    assert "auto_gapple" in src
    assert "critical_health_threshold" in src
    assert "safe_distance" in src
    assert "retreat_enabled" in src
    assert "retreat_distance" in src
    assert "fast_retreat" in src
    assert "retreat_hops" in src
    assert "sprint_hop_hold" in src
    assert "avoid_obstacles" in src
    assert "retreat_strafe" in src
    assert "wall_slide" in src
    assert "retreat_speed_lock" in src
    assert "retreat_velocity_assist" in src
    assert "retreat_speed_first" in src
    assert "retreat_full_speed" in src
    assert "retreat_speed_floor" in src
    assert "retreat_max_speed" in src
    assert "retreat_accel" in src
    assert "retreat_sprint_retap" in src
    assert "retreat_sprint_retap_ticks" in src
    assert "retreat_air_control" in src
    assert "retreat_step_assist" in src
    assert "retreat_step_height" in src
    assert "fallback_retreat" in src
    assert "retreat_input_lock" in src
    assert "force_sprint_retreat" in src
    assert "release_retreat_on_hit" in src
    assert "critical_rearm_only" in src
    assert "critical_trapped_eat" in src
    assert "retreat_turn_limit_deg" in src
    assert "eating_retreat_turn_limit_deg" in src
    assert "retreat_path_hold_ticks" in src
    assert "retreat_stuck_abort_ticks" in src
    assert "retreat_min_ticks" in src
    assert "retreat_max_ticks" in src
    assert "critical_retreat_max_ticks" in src
    assert "critical_eat_commit_ticks" in src
    assert "combat_recovery_ticks" in src
    assert "retreat_strafe_hold_ticks" in src
    assert "retreat_obstacle_jump_hold_ticks" in src
    assert "retreat_obstacle_escape_ticks" in src
    assert "critical_trapped_stuck_ticks" in src
    assert "retreat before gapple" in src
    assert "fast retreat" in src
    assert "retreat hops" in src
    assert "hold sprint-hop" in src
    assert "avoid obstacles" in src
    assert "retreat strafe" in src
    assert "wall slide" in src
    assert "retreat speed lock" in src
    assert "retreat velocity assist" in src
    assert "speed-first retreat" in src
    assert "full-speed retreat" in src
    assert "speed floor" in src
    assert "max retreat speed" in src
    assert "retreat accel" in src
    assert "sprint retap" in src
    assert "air control" in src
    assert "step assist" in src
    assert "step height" in src
    assert "fallback retreat" in src
    assert "retreat input lock" in src
    assert "force sprint retreat" in src
    assert "release retreat on hit" in src
    assert "critical trapped eat" in src
    assert "retreat turn deg" in src
    assert "eat turn deg" in src
    assert "path hold ticks" in src
    assert "stuck abort ticks" in src
    assert "min retreat ticks" in src
    assert "max retreat ticks" in src
    assert "critical retreat ticks" in src
    assert "combat recovery ticks" in src
    assert "strafe hold ticks" in src
    assert "obstacle jump hold ticks" in src
    assert "obstacle escape ticks" in src
    assert "trapped stuck ticks" in src
    assert "autoJump: false" in src
    assert "auto_jump" in src
    assert "auto jump" in src
    assert "knockbackDump: false" in src
    assert "knockback_dump" in src
    assert "kb dump" in src
    assert "friendMode: false" in src
    assert "friend mode" in src
    assert "size_x: 40" in src and "size_z: 40" in src
    assert "max={260}" in src

def test_live_direct_aim_guard_prevents_pitch_drift_upward():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, -1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    s.aim_smooth = 0.0

    a = s.on_state(state_msg())

    assert a["dpitch"] > 0.0      # target is below eye line; never keep aiming up
    assert s.last_action[1] > 0.0 # next observation records the corrected action


def test_live_direct_aim_guard_prevents_yaw_from_turning_away():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([-1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    s.aim_smooth = 0.0
    msg = state_msg()
    msg["target"]["x"] += 2.5   # yaw error is negative in Judas' convention

    a = s.on_state(msg)

    assert a["dyaw"] < 0.0

def test_live_direct_aim_lock_has_no_smoothing_delay_on_lateral_target():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([-0.05, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    s.aim_smooth = 0.85
    msg = state_msg()
    msg["target"]["x"] += 0.35
    own = player_from_msg(msg["self"], ARENA)
    opp = player_from_msg(msg["target"], ARENA)
    yaw_err, _ = s._aim_errors_deg(own, opp)

    a = s.on_state(msg)

    assert 1.0 < abs(yaw_err) < s.params.max_rot_speed
    assert a["dyaw"] == pytest.approx(yaw_err, abs=1.0e-5)


def test_live_direct_aim_guard_caps_pitch_to_touchable_hitbox():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, -1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    s.aim_smooth = 0.0
    msg = state_msg()
    msg["target"]["y"] += 2.0
    own = player_from_msg(msg["self"], ARENA)
    opp = player_from_msg(msg["target"], ARENA)
    _, pitch_err = s._aim_errors_deg(own, opp)

    a = s.on_state(msg)

    assert a["dpitch"] < 0.0
    assert abs(a["dpitch"] - pitch_err) < 1.0e-5
    assert abs(a["dpitch"]) < s.params.max_rot_speed


def test_live_direct_aim_guard_does_not_pitch_sky_at_point_blank():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, -1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 195.0
    s.aim_smooth = 0.0
    msg = state_msg(z_self=3.0, z_target=3.06)

    a = s.on_state(msg)

    assert 0.0 < a["dpitch"] < 8.0
    assert abs(s.last_action[1]) < 0.05


def test_live_combo_direct_pad_guard_blocks_escape_and_forces_pressure():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, -1.0, 1.0, 1.0, 0.0, 0.0])
    s.model_path = "models/combo_god_directpad_lock_combo12-safe_latest.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0

    a = s.on_state(state_msg())

    assert a["forward"] == 1
    assert abs(a["strafe"]) == 1
    assert a["jump"] is False
    assert a["sprint"] is True
    assert a["attack"] is True
    assert s.last_action[2] == 1.0
    assert abs(s.last_action[3]) == 1.0
    assert s.last_action[4] == 0.0


def test_live_leaderboard_boxing_forces_real_opener_strafe_even_off_angle():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    s.model_path = "models/combo_god_leaderboard10_combo12-safe_latest.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0

    msg = state_msg(z_self=3.0, z_target=7.0)
    msg["self"]["yaw"] = 100.0

    actions = [s.on_state(msg) for _ in range(20)]

    assert all(a["forward"] == 1 for a in actions)
    assert all(a["sprint"] is True for a in actions)
    assert all(abs(a["strafe"]) == 1 for a in actions)
    assert len({a["strafe"] for a in actions}) == 1
    assert all(a["jump"] is False for a in actions)
    assert all(a["forward"] >= 0 for a in actions)


def test_live_attn96_uses_leaderboard_boxing_guard():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0])
    s.model_path = "models/combo_god_attn96_combo12-best.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0

    a = s.on_state(state_msg(z_self=3.0, z_target=7.0))

    assert a["forward"] == 1
    assert abs(a["strafe"]) == 1
    assert a["jump"] is False
    assert a["sprint"] is True


def test_live_leaderboard_boxing_keeps_strafe_after_opener():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    s.model_path = "models/combo_god_leaderboard10_combo12-safe_latest.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    s.tick = 60

    msg = state_msg(z_self=3.0, z_target=8.7)
    actions = [s.on_state(msg) for _ in range(20)]

    assert all(a["forward"] == 1 for a in actions)
    assert all(a["sprint"] is True for a in actions)
    assert all(abs(a["strafe"]) == 1 for a in actions)
    assert len({a["strafe"] for a in actions}) == 1


def test_live_leaderboard_boxing_keeps_strafe_after_opener_off_angle():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    s.model_path = "models/combo_god_leaderboard10_combo12-safe_latest.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    s.tick = 70
    s.last_action[3] = -1.0

    msg = state_msg(z_self=3.0, z_target=8.2)
    msg["self"]["yaw"] = 170.0
    actions = [s.on_state(msg) for _ in range(6)]

    assert all(a["forward"] == 1 for a in actions)
    assert all(a["sprint"] is True for a in actions)
    assert all(abs(a["strafe"]) == 1 for a in actions)
    assert any(a["strafe"] == 1 for a in actions)
    assert all(a["jump"] is False for a in actions)


def test_live_leaderboard_boxing_caps_neutral_reset_after_opener():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    s.model_path = "models/combo_god_leaderboard10_combo12-safe_latest.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    s.tick = 60
    s.last_action[2] = 1.0
    s.last_action[5] = 1.0
    msg = state_msg(z_self=3.0, z_target=6.20)
    msg["target"]["hurtResistantTime"] = 12

    actions = [s.on_state(msg) for _ in range(5)]

    assert [a["forward"] for a in actions] == [0, 1, 1, 1, 1]
    assert actions[4]["sprint"] is True
    assert all(abs(a["strafe"]) == 1 for a in actions)


def test_live_leaderboard_boxing_w_taps_on_hit_counter_even_before_hurt_state():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    s.model_path = "models/combo_god_leaderboard10_combo12-safe_latest.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    s.tick = 60
    s._last_hits = (0, 0)
    s.last_action[2] = 1.0
    s.last_action[5] = 1.0
    msg = state_msg(z_self=3.0, z_target=5.80)
    msg["self"]["hits"] = 1
    msg["target"]["hurtResistantTime"] = 0

    a = s.on_state(msg)
    a2 = s.on_state(msg)

    assert a["forward"] == 0
    assert a["sprint"] is False
    assert abs(a["strafe"]) == 1
    assert a["jump"] is False
    assert a2["forward"] == 1
    assert a2["sprint"] is True
    assert abs(a2["strafe"]) == 1
    assert a2["jump"] is False


def test_live_leaderboard_boxing_wtap_overrides_opener_drive_after_hit():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    s.model_path = "models/combo_god_attn96_combo12-best.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    s.tick = 8
    s._last_hits = (0, 0)
    s.last_action[2] = 1.0
    s.last_action[5] = 1.0
    msg = state_msg(z_self=3.0, z_target=5.80)
    msg["self"]["hits"] = 1
    msg["target"]["hurtResistantTime"] = 10

    a = s.on_state(msg)
    a2 = s.on_state(msg)

    assert a["forward"] == 0
    assert a["sprint"] is False
    assert abs(a["strafe"]) == 1
    assert a["jump"] is False
    assert a2["forward"] == 1
    assert a2["sprint"] is True
    assert abs(a2["strafe"]) == 1


def test_live_combo_direct_pad_guard_uses_short_reset_after_landed_hit():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    s.model_path = "models/combo_god_directpad_lock_combo12-safe_latest.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    s._last_hits = (0, 0)
    msg = state_msg(z_self=3.0, z_target=5.80)
    msg["self"]["hits"] = 1
    msg["target"]["hurtResistantTime"] = 12

    a = s.on_state(msg)

    assert a["forward"] == 0
    assert a["sprint"] is False
    assert abs(a["strafe"]) == 1
    assert a["jump"] is False

    a2 = s.on_state(msg)
    assert a2["forward"] == 0
    assert a2["sprint"] is False

    ready = dict(msg)
    ready["target"] = dict(msg["target"])
    ready["target"]["hurtResistantTime"] = 10
    a3 = s.on_state(ready)
    assert a3["forward"] == 1
    assert a3["sprint"] is True
    assert a3["attack"] is True

    far = state_msg(z_self=3.0, z_target=7.60)
    far["self"]["hits"] = 1
    far["target"]["hurtResistantTime"] = 12
    a4 = s.on_state(far)
    assert a4["forward"] == 1
    assert a4["sprint"] is True


def test_live_combo_direct_pad_guard_presses_combo_rehit_without_new_hit():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0])
    s.model_path = "models/combo_god_directpad_lock_combo12-safe_latest.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    s.last_action[2] = 1.0
    s.last_action[5] = 1.0
    msg = state_msg(z_self=3.0, z_target=5.80)
    msg["target"]["hurtResistantTime"] = 11

    a = s.on_state(msg)

    assert a["forward"] == 1
    assert a["sprint"] is True
    assert abs(a["strafe"]) == 1
    assert a["jump"] is False
    assert a["attack"] is True


def test_live_combo_direct_pad_guard_brakes_before_rehit_edge():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    s.model_path = "models/combo_god_directpad_lock_combo12-safe_latest.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    msg = state_msg(z_self=3.0, z_target=6.70)
    msg["target"]["hurtResistantTime"] = 10

    a = s.on_state(msg)

    assert a["forward"] == 0
    assert a["sprint"] is False
    assert abs(a["strafe"]) == 1
    assert a["jump"] is False
    assert a["attack"] is False


def test_live_combo_direct_pad_guard_edge_pokes_with_s_brake():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    s.model_path = "models/combo_god_directpad_lock_combo12-safe_latest.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    msg = state_msg(z_self=3.0, z_target=6.36)
    msg["target"]["hurtResistantTime"] = 10

    a = s.on_state(msg)

    assert a["forward"] == 0
    assert a["sprint"] is False
    assert abs(a["strafe"]) == 1
    assert a["jump"] is False
    assert a["attack"] is True


def test_live_combo_direct_pad_guard_waits_before_rehit_is_legal():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    s.model_path = "models/combo_god_directpad_lock_combo12-safe_latest.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    s.last_action[2] = 1.0
    s.last_action[5] = 1.0
    msg = state_msg(z_self=3.0, z_target=5.80)
    msg["target"]["hurtResistantTime"] = 16

    a = s.on_state(msg)

    assert a["forward"] == 0
    assert a["sprint"] is False
    assert a["attack"] is False


def test_live_combo_direct_pad_guard_represses_after_close_combo_tap():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    s.model_path = "models/combo_god_directpad_lock_combo12-safe_latest.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    s.last_action[2] = 1.0
    s.last_action[5] = 1.0
    msg = state_msg(z_self=3.0, z_target=5.10)
    msg["target"]["hurtResistantTime"] = 12

    a = s.on_state(msg)
    assert a["forward"] == 0
    assert a["sprint"] is False
    assert a["attack"] is False

    ready = dict(msg)
    ready["target"] = dict(msg["target"])
    ready["target"]["hurtResistantTime"] = 10
    a2 = s.on_state(ready)
    assert a2["forward"] == 1
    assert a2["sprint"] is True
    assert a2["attack"] is True


def test_live_combo_direct_pad_guard_s_taps_when_mirror_gets_point_blank():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    s.model_path = "models/combo_god_directpad_lock_combo12-safe_latest.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    s.last_action[2] = 1.0
    s.last_action[5] = 1.0
    msg = state_msg(z_self=3.0, z_target=4.20)

    a = s.on_state(msg)
    assert a["forward"] == 0
    assert a["sprint"] is False
    assert abs(a["strafe"]) == 1
    assert a["jump"] is False
    assert a["attack"] is True

    a2 = s.on_state(msg)
    assert a2["forward"] == 1
    assert a2["sprint"] is True
    assert abs(a2["strafe"]) == 1
    assert a2["jump"] is False

    far = state_msg(z_self=3.0, z_target=7.60)
    far["self"]["hits"] = 1
    far["target"]["hurtResistantTime"] = 12
    a3 = s.on_state(far)
    assert a3["forward"] == 1
    assert a3["sprint"] is True


def test_live_combo_direct_pad_guard_forces_under_combo_counter_hit():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    s.model_path = "models/combo_god_directpad_lock_combo12-safe_latest.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    msg = state_msg(z_self=3.0, z_target=5.60)
    msg["self"]["hurtResistantTime"] = 7

    a = s.on_state(msg)

    assert a["forward"] == 1
    assert a["sprint"] is True
    assert abs(a["strafe"]) == 1
    assert a["jump"] is False
    assert a["attack"] is True


def test_live_combo_direct_pad_guard_forces_reach_under_combo_counter_hit():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    s.model_path = "models/combo_god_directpad_lock_combo12-safe_latest.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    msg = state_msg(z_self=3.0, z_target=6.36)
    msg["self"]["hurtResistantTime"] = 10

    a = s.on_state(msg)

    assert a["forward"] == 1
    assert a["sprint"] is True
    assert abs(a["strafe"]) == 1
    assert a["jump"] is False
    assert a["attack"] is True


def test_live_combo_direct_pad_guard_forces_far_trade_under_combo_counter_hit():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    s.model_path = "models/combo_god_directpad_lock_combo12-safe_latest.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    msg = state_msg(z_self=3.0, z_target=6.52)
    msg["self"]["hurtResistantTime"] = 12

    a = s.on_state(msg)

    assert a["forward"] == 1
    assert a["sprint"] is True
    assert a["jump"] is False
    assert a["attack"] is True


def test_live_combo_direct_pad_guard_counter_clicks_while_aim_catches_up():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    s.model_path = "models/combo_god_directpad_lock_combo12-safe_latest.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 20.0
    msg = state_msg(z_self=3.0, z_target=5.60)
    msg["self"]["hurtResistantTime"] = 12
    msg["self"]["yaw"] = 90.0

    a = s.on_state(msg)

    assert a["forward"] == 1
    assert a["sprint"] is True
    assert a["attack"] is True
    assert a["dyaw"] < 0.0


def test_live_combo_direct_pad_guard_counter_clicks_respect_10_cps(tmp_path):
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    s.model_path = "models/combo_god_leaderboard10_combo12-safe_latest.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    s.params.max_cps = 10.0
    msg = state_msg(z_self=3.0, z_target=5.60)
    msg["self"]["hurtResistantTime"] = 12

    clicks = 0
    for _ in range(20):
        a = s.on_state(msg)
        clicks += int(bool(a["attack"]))

    assert clicks <= 10
    assert clicks >= 8


def test_live_counter_assist_forces_idle_under_combo_click():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    s.model_path = "models/plain.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    s.params.counter_assist.enabled = True
    msg = state_msg(z_self=3.0, z_target=5.60)
    msg["self"]["hurtResistantTime"] = 12

    a = s.on_state(msg)

    assert a["attack"] is True


def test_live_counter_assist_stays_idle_without_combo_disadvantage():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    s.model_path = "models/plain.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    s.params.counter_assist.enabled = True

    a = s.on_state(state_msg(z_self=3.0, z_target=5.60))

    assert a["attack"] is False


def test_live_counter_assist_clicks_when_direct_turn_can_reach_target():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    s.model_path = "models/plain.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 195.0
    s.params.counter_assist.enabled = True
    msg = state_msg(z_self=3.0, z_target=5.60)
    msg["self"]["hurtResistantTime"] = 12
    msg["self"]["yaw"] = 90.0

    a = s.on_state(msg)

    assert a["attack"] is True
    assert a["dyaw"] < 0.0


def test_live_counter_assist_clicks_at_five_block_recovery_range():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    s.model_path = "models/plain.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 195.0
    s.params.counter_assist.enabled = True
    msg = state_msg(z_self=3.0, z_target=8.0)
    msg["self"]["hurtResistantTime"] = 12

    a = s.on_state(msg)

    assert a["attack"] is True


def test_live_counter_assist_waits_when_direct_turn_cannot_reach_target():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    s.model_path = "models/plain.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    s.params.counter_assist.enabled = True
    msg = state_msg(z_self=3.0, z_target=5.60)
    msg["self"]["hurtResistantTime"] = 12
    msg["self"]["yaw"] = 90.0

    a = s.on_state(msg)

    assert a["attack"] is False
    assert a["dyaw"] == -40.0


def test_live_counter_assist_respects_10_cps():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    s.model_path = "models/plain.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    s.params.max_cps = 10.0
    s.params.counter_assist.enabled = True
    s._rng.seed(1)
    msg = state_msg(z_self=3.0, z_target=5.60)
    msg["self"]["hurtResistantTime"] = 12

    clicks = 0
    for _ in range(20):
        a = s.on_state(msg)
        clicks += int(bool(a["attack"]))

    assert clicks <= 10
    assert clicks >= 8


def test_live_counter_assist_varies_near_10_cps():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    s.model_path = "models/plain.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    s.params.max_cps = 10.0
    s.params.counter_assist.enabled = True
    s._rng.seed(1)
    msg = state_msg(z_self=3.0, z_target=5.60)
    msg["self"]["hurtResistantTime"] = 12

    ticks = []
    for i in range(60):
        a = s.on_state(msg)
        if a["attack"]:
            ticks.append(i)

    gaps = [b - a for a, b in zip(ticks, ticks[1:])]
    assert 25 <= len(ticks) <= 29
    assert all(gap in (2, 3, 4) for gap in gaps)
    assert 2 in gaps and 3 in gaps


def test_live_hit_select_assist_forces_clean_timed_click():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    s.model_path = "models/plain.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    s.params.hit_select_assist.enabled = True
    msg = state_msg(z_self=3.0, z_target=5.85)
    msg["self"]["hurtResistantTime"] = 18
    msg["target"]["clickCooldown"] = 2

    a = s.on_state(msg)

    assert a["forward"] == 0
    assert a["sprint"] is False
    assert a["attack"] is True
    assert a["hit_select_assist"]["enabled"] is True


def test_live_hit_select_assist_clicks_once_per_clean_window():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    s.model_path = "models/plain.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    s.params.hit_select_assist.enabled = True
    msg = state_msg(z_self=3.0, z_target=5.85)
    msg["self"]["hurtResistantTime"] = 18
    msg["target"]["clickCooldown"] = 2

    attacks = [s.on_state(msg)["attack"] for _ in range(12)]

    assert attacks.count(True) == 1
    assert attacks[0] is True


def test_live_hit_select_assist_blocks_early_trade_click():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    s.model_path = "models/plain.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    s.params.hit_select_assist.enabled = True
    msg = state_msg(z_self=3.0, z_target=5.85)
    msg["self"]["hurtResistantTime"] = 18
    msg["target"]["clickCooldown"] = 0

    a = s.on_state(msg)

    assert a["forward"] == 0
    assert a["sprint"] is False
    assert a["attack"] is False


def test_live_hit_select_assist_blocks_counter_assist_until_clean_window():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    s.model_path = "models/plain.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    s.params.hit_select_assist.enabled = True
    s.params.counter_assist.enabled = True
    msg = state_msg(z_self=3.0, z_target=5.85)
    msg["self"]["hurtResistantTime"] = 18
    msg["target"]["clickCooldown"] = 0

    a = s.on_state(msg)

    assert a["attack"] is False


def test_live_aim_smoothing_softens_direct_lock_when_enabled():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    s.model_path = "models/plain.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    s.params.aim_smoothing.enabled = True
    s.params.aim_smoothing.strength = 0.50
    msg = state_msg()
    msg["target"]["x"] += 0.35
    own = player_from_msg(msg["self"], ARENA)
    opp = player_from_msg(msg["target"], ARENA)
    yaw_err, _ = s._aim_errors_deg(own, opp)

    a = s.on_state(msg)

    assert 1.0 < abs(a["dyaw"]) < abs(yaw_err)
    assert a["aim_smoothing"]["enabled"] is True


def test_live_action_includes_auto_gapple_auto_jump_and_friend_config():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    s.model_path = "models/plain.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.auto_gapple.enabled = True
    s.params.auto_gapple.health_threshold = 12.0
    s.params.auto_gapple.critical_health_threshold = 7.0
    s.params.auto_gapple.safe_distance = 4.5
    s.params.auto_gapple.retreat_enabled = True
    s.params.auto_gapple.retreat_distance = 6.5
    s.params.auto_gapple.fast_retreat = True
    s.params.auto_gapple.retreat_hops = True
    s.params.auto_gapple.sprint_hop_hold = True
    s.params.auto_gapple.avoid_obstacles = True
    s.params.auto_gapple.retreat_strafe = True
    s.params.auto_gapple.wall_slide = True
    s.params.auto_gapple.retreat_speed_lock = True
    s.params.auto_gapple.retreat_velocity_assist = True
    s.params.auto_gapple.retreat_speed_first = True
    s.params.auto_gapple.retreat_full_speed = True
    s.params.auto_gapple.retreat_speed_floor = 0.25
    s.params.auto_gapple.retreat_max_speed = 0.36
    s.params.auto_gapple.retreat_accel = 0.09
    s.params.auto_gapple.fallback_retreat = True
    s.params.auto_gapple.retreat_input_lock = True
    s.params.auto_gapple.force_sprint_retreat = True
    s.params.auto_gapple.release_retreat_on_hit = True
    s.params.auto_gapple.critical_rearm_only = True
    s.params.auto_gapple.retreat_turn_limit_deg = 210.0
    s.params.auto_gapple.eating_retreat_turn_limit_deg = 150.0
    s.params.auto_gapple.retreat_path_hold_ticks = 3
    s.params.auto_gapple.retreat_stuck_abort_ticks = 7
    s.params.auto_gapple.retreat_min_ticks = 5
    s.params.auto_gapple.retreat_max_ticks = 70
    s.params.auto_gapple.critical_retreat_max_ticks = 12
    s.params.auto_gapple.combat_recovery_ticks = 19
    s.params.auto_gapple.retreat_strafe_hold_ticks = 5
    s.params.auto_gapple.retreat_obstacle_jump_hold_ticks = 6
    s.params.auto_gapple.retreat_obstacle_escape_ticks = 9
    s.params.aim_smoothing.enabled = True
    s.params.hit_select_assist.enabled = True
    s.params.auto_jump.enabled = True
    s.params.knockback_dump.enabled = True
    s.params.friends.enabled = True
    s.params.friends.names = ["Alice", "Bob"]

    a = s.on_state(state_msg())

    assert a["aim_smoothing"]["enabled"] is True
    assert a["hit_select_assist"]["enabled"] is True
    assert a["auto_gapple"]["enabled"] is True
    assert a["auto_gapple"]["health_threshold"] == 12.0
    assert a["auto_gapple"]["critical_health_threshold"] == 7.0
    assert a["auto_gapple"]["safe_distance"] == 4.5
    assert a["auto_gapple"]["retreat_enabled"] is True
    assert a["auto_gapple"]["retreat_distance"] == 6.5
    assert a["auto_gapple"]["fast_retreat"] is True
    assert a["auto_gapple"]["retreat_hops"] is True
    assert a["auto_gapple"]["sprint_hop_hold"] is True
    assert a["auto_gapple"]["avoid_obstacles"] is True
    assert a["auto_gapple"]["retreat_strafe"] is True
    assert a["auto_gapple"]["wall_slide"] is True
    assert a["auto_gapple"]["retreat_speed_lock"] is True
    assert a["auto_gapple"]["retreat_velocity_assist"] is True
    assert a["auto_gapple"]["retreat_speed_first"] is True
    assert a["auto_gapple"]["retreat_full_speed"] is True
    assert a["auto_gapple"]["retreat_speed_floor"] == 0.25
    assert a["auto_gapple"]["retreat_max_speed"] == 0.36
    assert a["auto_gapple"]["retreat_accel"] == 0.09
    assert a["auto_gapple"]["fallback_retreat"] is True
    assert a["auto_gapple"]["retreat_input_lock"] is True
    assert a["auto_gapple"]["force_sprint_retreat"] is True
    assert a["auto_gapple"]["release_retreat_on_hit"] is True
    assert a["auto_gapple"]["critical_rearm_only"] is True
    assert a["auto_gapple"]["retreat_turn_limit_deg"] == 210.0
    assert a["auto_gapple"]["eating_retreat_turn_limit_deg"] == 150.0
    assert a["auto_gapple"]["retreat_path_hold_ticks"] == 3
    assert a["auto_gapple"]["retreat_stuck_abort_ticks"] == 7
    assert a["auto_gapple"]["retreat_min_ticks"] == 5
    assert a["auto_gapple"]["retreat_max_ticks"] == 70
    assert a["auto_gapple"]["critical_retreat_max_ticks"] == 12
    assert a["auto_gapple"]["combat_recovery_ticks"] == 19
    assert a["auto_gapple"]["retreat_strafe_hold_ticks"] == 5
    assert a["auto_gapple"]["retreat_obstacle_jump_hold_ticks"] == 6
    assert a["auto_gapple"]["retreat_obstacle_escape_ticks"] == 9
    assert a["auto_jump"]["enabled"] is True
    assert a["knockback_dump"]["enabled"] is True
    assert a["friends"]["enabled"] is True
    assert a["friends"]["names"] == ["Alice", "Bob"]


def test_live_leaderboard_boxing_pressure_clicks_at_10_cps_edge_range():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    s.model_path = "models/combo_god_leaderboard10_combo12-safe_latest.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    s.params.max_cps = 10.0
    s.tick = 70
    msg = state_msg(z_self=3.0, z_target=6.78)

    clicks = 0
    actions = []
    for _ in range(20):
        a = s.on_state(msg)
        actions.append(a)
        clicks += int(bool(a["attack"]))

    assert clicks <= 10
    assert clicks >= 8
    assert all(a["forward"] == 1 for a in actions)
    assert all(a["sprint"] is True for a in actions)
    assert all(a["forward"] >= 0 for a in actions)


def test_live_leaderboard_boxing_pressure_clicks_while_approaching():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    s.model_path = "models/combo_god_leaderboard10_combo12-safe_latest.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    s.params.max_cps = 10.0
    s.tick = 70
    msg = state_msg(z_self=3.0, z_target=8.95)

    clicks = 0
    actions = []
    for _ in range(20):
        a = s.on_state(msg)
        actions.append(a)
        clicks += int(bool(a["attack"]))

    assert clicks <= 10
    assert clicks >= 8
    assert all(a["forward"] == 1 for a in actions)
    assert all(a["sprint"] is True for a in actions)
    assert all(abs(a["strafe"]) == 1 for a in actions)


def test_live_leaderboard_boxing_keeps_clicking_during_combo_cooldown():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    s.model_path = "models/combo_god_leaderboard10_combo12-safe_latest.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    s.params.max_cps = 10.0
    s.tick = 70
    msg = state_msg(z_self=3.0, z_target=6.70)
    msg["target"]["hurtResistantTime"] = 16

    actions = [s.on_state(msg) for _ in range(4)]

    assert any(a["attack"] is True for a in actions)
    assert all(a["jump"] is False for a in actions)
    assert all(a["forward"] >= 0 for a in actions)


def test_live_leaderboard_boxing_point_blank_combo_does_not_repress_w():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    s.model_path = "models/combo_god_leaderboard10_combo12-safe_latest.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    s.params.max_cps = 10.0
    s.tick = 70
    s.last_action[2] = 1.0
    s.last_action[5] = 1.0
    s._leaderboard_neutral_ticks = 99
    msg = state_msg(z_self=3.0, z_target=4.10)
    msg["target"]["hurtResistantTime"] = 18

    a = s.on_state(msg)

    assert a["forward"] == 0
    assert a["sprint"] is False
    assert a["jump"] is False
    assert a["forward"] >= 0


def test_live_leaderboard_boxing_point_blank_clinch_keeps_lateral_strafe():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    s.model_path = "models/combo_god_leaderboard10_combo12-safe_latest.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    s.params.max_cps = 10.0
    s.tick = 70
    s.last_action[3] = 1.0
    msg = state_msg(z_self=3.0, z_target=3.70)

    actions = [s.on_state(msg) for _ in range(4)]

    assert all(a["strafe"] == 1 for a in actions)
    assert all(a["jump"] is False for a in actions)
    assert all(a["forward"] >= 0 for a in actions)


def test_live_leaderboard_boxing_combo_pocket_does_not_repress_w_after_cap():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    s.model_path = "models/combo_god_leaderboard10_combo12-safe_latest.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    s.params.max_cps = 10.0
    s.tick = 120
    s.last_action[2] = 0.0
    s.last_action[5] = 0.0
    s._leaderboard_neutral_ticks = 99
    s._own_combo_streak = 6
    msg = state_msg(z_self=3.0, z_target=5.45)
    msg["target"]["hurtResistantTime"] = 18

    a = s.on_state(msg)

    assert a["forward"] == 0
    assert a["sprint"] is False
    assert abs(a["strafe"]) == 1
    assert a["jump"] is False
    assert a["attack"] is True


def test_live_leaderboard_boxing_rehit_brakes_before_contact_trade():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    s.model_path = "models/combo_god_leaderboard10_combo12-safe_latest.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    s.params.max_cps = 10.0
    s.tick = 120
    s.last_action[2] = 1.0
    s.last_action[5] = 1.0
    s._leaderboard_neutral_ticks = 99
    s._own_combo_streak = 6
    msg = state_msg(z_self=3.0, z_target=6.35)
    msg["target"]["hurtResistantTime"] = 8

    a = s.on_state(msg)

    assert a["forward"] == 0
    assert a["sprint"] is False
    assert abs(a["strafe"]) == 1
    assert a["jump"] is False
    assert a["attack"] is True


def test_live_leaderboard_boxing_counter_reenters_without_clicking_outside_hit_reach():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0])
    s.model_path = "models/combo_god_leaderboard10_combo12-safe_latest.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    s.params.max_cps = 10.0
    s.tick = 70
    msg = state_msg(z_self=3.0, z_target=6.52)
    msg["self"]["hurtResistantTime"] = 12

    a = s.on_state(msg)

    assert a["forward"] == 1
    assert a["sprint"] is True
    assert a["strafe"] == 0
    assert a["jump"] is False
    assert a["attack"] is False


def test_live_leaderboard_boxing_does_not_force_midrange_counter_trade_click():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0])
    s.model_path = "models/combo_god_leaderboard10_combo12-safe_latest.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    s.params.max_cps = 10.0
    s.tick = 70
    msg = state_msg(z_self=3.0, z_target=6.35)
    msg["self"]["hurtResistantTime"] = 12

    a = s.on_state(msg)

    assert a["forward"] == 1
    assert a["sprint"] is True
    assert a["jump"] is False
    assert a["attack"] is False


def test_live_leaderboard_boxing_keeps_close_counter_click_for_recovery():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0])
    s.model_path = "models/combo_god_leaderboard10_combo12-safe_latest.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    s.params.max_cps = 10.0
    s.tick = 70
    msg = state_msg(z_self=3.0, z_target=5.05)
    msg["self"]["hurtResistantTime"] = 8

    a = s.on_state(msg)

    assert a["jump"] is False
    assert a["attack"] is True


def test_live_leaderboard_boxing_z_tap_attacks_ready_combo_rehit():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
    s.model_path = "models/combo_god_leaderboard10_combo12-safe_latest.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    s.params.max_cps = 10.0
    s.tick = 90
    s.last_action[2] = 1.0
    s.last_action[5] = 1.0
    msg = state_msg(z_self=3.0, z_target=6.25)
    msg["target"]["hurtResistantTime"] = 8

    a = s.on_state(msg)

    assert a["forward"] == 0
    assert a["sprint"] is False
    assert abs(a["strafe"]) == 1
    assert a["jump"] is False
    assert a["attack"] is True


def test_live_leaderboard_boxing_counter_drives_straight_when_pushed_out():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0])
    s.model_path = "models/combo_god_leaderboard10_combo12-safe_latest.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    s.params.max_cps = 10.0
    s.tick = 70
    msg = state_msg(z_self=3.0, z_target=8.85)
    msg["self"]["hurtResistantTime"] = 12

    a = s.on_state(msg)

    assert a["forward"] == 1
    assert a["sprint"] is True
    assert a["strafe"] == 0
    assert a["jump"] is False
    assert a["attack"] is False


def test_live_leaderboard_boxing_far_under_combo_reenters_without_circling():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 1.0])
    s.model_path = "models/combo_god_leaderboard10_combo12-safe_latest.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    s.params.max_cps = 10.0
    s.tick = 70
    s._last_hits = (0, 3)
    s._opp_combo_streak = 3
    s._opp_combo_break_ticks = 5
    s._opp_combo_break_sign = 1.0
    s.last_action[3] = 1.0
    msg = state_msg(z_self=3.0, z_target=9.60)
    msg["self"]["hurtResistantTime"] = 12
    msg["target"]["hits"] = 4

    a = s.on_state(msg)

    assert a["forward"] == 1
    assert a["sprint"] is True
    assert a["strafe"] == 0
    assert a["jump"] is False
    assert a["attack"] is False


def test_live_leaderboard_boxing_counter_reacquires_lateral_target():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0])
    s.model_path = "models/combo_god_leaderboard10_combo12-safe_latest.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    s.params.max_cps = 10.0
    s.tick = 70
    s.last_action[3] = -1.0
    msg = state_msg(z_self=3.0, z_target=5.80)
    msg["target"]["x"] = 110.2
    msg["target"]["clickCooldown"] = 2
    msg["self"]["hurtResistantTime"] = 7

    a = s.on_state(msg)

    assert a["forward"] == 0
    assert a["sprint"] is False
    assert a["strafe"] == 0
    assert a["jump"] is False
    assert a["attack"] is False


def test_live_leaderboard_boxing_counter_breaker_reenters_straight_outside_hit_reach():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0])
    s.model_path = "models/combo_god_leaderboard10_combo12-safe_latest.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    s.params.max_cps = 10.0
    s.tick = 70
    s._last_hits = (0, 2)
    s._opp_combo_streak = 2
    s.last_action[3] = -1.0
    msg = state_msg(z_self=3.0, z_target=6.90)
    msg["self"]["hurtResistantTime"] = 12
    msg["target"]["hits"] = 3

    a = s.on_state(msg)

    assert a["forward"] == 1
    assert a["sprint"] is True
    assert a["strafe"] == 0
    assert a["jump"] is False
    assert a["attack"] is False


def test_live_leaderboard_boxing_counter_breaker_keeps_signal_but_does_not_circle_outside_hit_reach():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0])
    s.model_path = "models/combo_god_leaderboard10_combo12-safe_latest.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    s.params.max_cps = 10.0
    s.tick = 90
    s._last_hits = (0, 3)
    s._opp_combo_streak = 3
    s._opp_combo_break_ticks = 5
    s._opp_combo_break_sign = 1.0
    s.last_action[3] = 1.0
    msg = state_msg(z_self=3.0, z_target=7.10)
    msg["self"]["hurtResistantTime"] = 12
    msg["target"]["hits"] = 4

    a = s.on_state(msg)

    assert a["forward"] == 1
    assert a["sprint"] is True
    assert a["strafe"] == 0
    assert s._opp_combo_break_sign == 1.0
    assert s._opp_combo_break_ticks == 12
    assert a["jump"] is False
    assert a["attack"] is False


def test_live_leaderboard_boxing_counter_drives_at_close_trade_range():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0])
    s.model_path = "models/combo_god_leaderboard10_combo12-safe_latest.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    s.params.max_cps = 10.0
    s.tick = 70
    s.last_action[3] = 1.0
    msg = state_msg(z_self=3.0, z_target=4.85)
    msg["self"]["hurtResistantTime"] = 10

    actions = [s.on_state(msg) for _ in range(4)]

    assert all(a["forward"] == 1 for a in actions)
    assert all(a["sprint"] is True for a in actions)
    assert all(a["jump"] is False for a in actions)
    assert all(a["strafe"] == 0 for a in actions)
    assert any(a["attack"] is True for a in actions)


def test_live_leaderboard_boxing_score_deficit_rescues_contact_trade():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0])
    s.model_path = "models/combo_god_leaderboard10_combo12-safe_latest.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    s.params.max_cps = 10.0
    s.tick = 70
    s.last_action[3] = -1.0
    msg = state_msg(z_self=3.0, z_target=5.15)
    msg["self"]["hits"] = 4
    msg["target"]["hits"] = 8

    actions = [s.on_state(msg) for _ in range(4)]

    assert all(a["forward"] == 1 for a in actions)
    assert all(a["sprint"] is True for a in actions)
    assert all(a["jump"] is False for a in actions)
    assert all(a["strafe"] == -1 for a in actions)
    assert any(a["attack"] is True for a in actions)


def test_live_combo_direct_pad_guard_blocks_far_under_combo_trade_click():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    s.model_path = "models/combo_god_directpad_lock_combo12-safe_latest.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0
    msg = state_msg(z_self=3.0, z_target=6.65)
    msg["self"]["hurtResistantTime"] = 12

    a = s.on_state(msg)

    assert a["forward"] == 1
    assert a["sprint"] is True
    assert abs(a["strafe"]) == 1
    assert a["jump"] is False
    assert a["attack"] is False


def test_live_combo_direct_pad_guard_is_scoped_to_combo_safe_model():
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, -1.0, 1.0, 1.0, 0.0, 0.0])
    s.model_path = "models/other.pts"
    s.params.arena = ARENA
    s.auto_frame = False
    s.params.max_rot_speed = 40.0

    a = s.on_state(state_msg())

    assert a["forward"] == -1
    assert a["strafe"] == 1
    assert a["jump"] is True
    assert a["sprint"] is False
    assert a["attack"] is False

def test_live_session_produces_action(tmp_path):
    s = _session_with_model(tmp_path)
    a = s.on_state(state_msg())
    assert a is not None and a["t"] == "action"
    assert abs(a["dyaw"]) <= s.params.max_rot_speed + 1e-6
    assert abs(a["dpitch"]) <= s.params.max_rot_speed + 1e-6
    assert s.last_latency_ms > 0


def test_live_session_writes_structured_action_log(tmp_path, monkeypatch):
    log = tmp_path / "live-actions.log"
    monkeypatch.setenv("JUDAS_LIVE_ACTION_LOG", str(log))
    s = LiveSession(device="cpu")
    s.history = 4
    s.hist = torch.zeros(1, s.history, 48)
    s.model = ConstantActionModel([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    s.model_path = "models/combo_god_directpad_lock_combo12-safe_latest.pts"
    s.params.arena = ARENA
    s.auto_frame = False

    a = s.on_state(state_msg())

    assert a["forward"] == 1 and abs(a["strafe"]) == 1 and a["jump"] is False
    text = log.read_text(encoding="utf-8")
    assert "model=models/combo_god_directpad_lock_combo12-safe_latest.pts" in text
    assert "forward=1" in text
    assert ("strafe=1" in text) or ("strafe=-1" in text)
    assert "jump=false" in text
    assert "ownPitch=0.000000" in text


def test_live_session_cps_limit(tmp_path):
    s = _session_with_model(tmp_path)
    s.params.max_cps = 10.0   # cooldown 2 ticks
    clicks = 0
    for _ in range(20):
        a = s.on_state(state_msg())
        clicks += 1 if a["attack"] else 0
    assert clicks <= 10  # jamais plus d'1 clic / 2 ticks


def test_live_disabled_returns_none(tmp_path):
    s = _session_with_model(tmp_path)
    s.params.enabled = False
    assert s.on_state(state_msg()) is None


def test_rest_status_and_params():
    client = TestClient(app)
    r = client.get("/status")
    assert r.status_code == 200
    body = r.json()
    assert "training" in body and "live" in body and "gpu" in body

    r = client.post("/live/params", json={"max_cps": 9.0,
                                          "aim_smoothing": {"enabled": True,
                                                            "strength": 0.33,
                                                            "snap_deg": 0.72},
                                          "hit_select_assist": {"enabled": True,
                                                                "release_movement": False},
                                          "counter_assist": {"enabled": True},
                                          "auto_gapple": {"enabled": True,
                                                          "health_threshold": 13.0,
                                                          "critical_health_threshold": 7.0,
                                                          "safe_distance": 4.5,
                                                          "retreat_enabled": True,
                                                          "retreat_distance": 6.5,
                                                          "fast_retreat": True,
                                                          "retreat_hops": True,
                                                          "sprint_hop_hold": True,
                                                          "avoid_obstacles": True,
                                                          "retreat_strafe": True,
                                                          "wall_slide": True,
                                                           "retreat_speed_lock": True,
                                                           "retreat_velocity_assist": True,
                                                           "retreat_speed_first": True,
                                                           "retreat_full_speed": True,
                                                           "retreat_speed_floor": 0.25,
                                                          "retreat_max_speed": 0.36,
                                                          "retreat_accel": 0.09,
                                                          "fallback_retreat": True,
                                                           "retreat_input_lock": True,
                                                           "force_sprint_retreat": True,
                                                           "release_retreat_on_hit": True,
                                                           "critical_rearm_only": True,
                                                           "retreat_turn_limit_deg": 210.0,
                                                          "eating_retreat_turn_limit_deg": 150.0,
                                                          "retreat_path_hold_ticks": 3,
                                                          "retreat_stuck_abort_ticks": 7,
                                                          "retreat_min_ticks": 5,
                                                          "retreat_max_ticks": 70,
                                                           "critical_retreat_max_ticks": 12,
                                                           "combat_recovery_ticks": 19,
                                                           "retreat_strafe_hold_ticks": 5,
                                                           "retreat_obstacle_jump_hold_ticks": 6,
                                                           "retreat_obstacle_escape_ticks": 9},
                                          "auto_jump": {"enabled": True},
                                          "knockback_dump": {"enabled": True},
                                          "friends": {"enabled": True,
                                                      "names": ["Alice", "Bob"]},
                                          "arena": {"origin_x": 1.0, "origin_z": 2.0,
                                                    "size_x": 20.0, "size_z": 20.0,
                                                    "floor_y": 60.0}})
    assert r.status_code == 200
    assert r.json()["max_cps"] == 9.0
    assert r.json()["aim_smoothing"]["enabled"] is True
    assert r.json()["aim_smoothing"]["strength"] == 0.33
    assert r.json()["aim_smoothing"]["snap_deg"] == 0.72
    assert r.json()["hit_select_assist"]["enabled"] is True
    assert r.json()["hit_select_assist"]["release_movement"] is False
    assert r.json()["counter_assist"]["enabled"] is True
    assert r.json()["auto_gapple"]["enabled"] is True
    assert r.json()["auto_gapple"]["health_threshold"] == 13.0
    assert r.json()["auto_gapple"]["critical_health_threshold"] == 7.0
    assert r.json()["auto_gapple"]["safe_distance"] == 4.5
    assert r.json()["auto_gapple"]["retreat_enabled"] is True
    assert r.json()["auto_gapple"]["retreat_distance"] == 6.5
    assert r.json()["auto_gapple"]["fast_retreat"] is True
    assert r.json()["auto_gapple"]["retreat_hops"] is True
    assert r.json()["auto_gapple"]["sprint_hop_hold"] is True
    assert r.json()["auto_gapple"]["avoid_obstacles"] is True
    assert r.json()["auto_gapple"]["retreat_strafe"] is True
    assert r.json()["auto_gapple"]["wall_slide"] is True
    assert r.json()["auto_gapple"]["retreat_speed_lock"] is True
    assert r.json()["auto_gapple"]["retreat_velocity_assist"] is True
    assert r.json()["auto_gapple"]["retreat_speed_first"] is True
    assert r.json()["auto_gapple"]["retreat_full_speed"] is True
    assert r.json()["auto_gapple"]["retreat_speed_floor"] == 0.25
    assert r.json()["auto_gapple"]["retreat_max_speed"] == 0.36
    assert r.json()["auto_gapple"]["retreat_accel"] == 0.09
    assert r.json()["auto_gapple"]["fallback_retreat"] is True
    assert r.json()["auto_gapple"]["retreat_input_lock"] is True
    assert r.json()["auto_gapple"]["force_sprint_retreat"] is True
    assert r.json()["auto_gapple"]["release_retreat_on_hit"] is True
    assert r.json()["auto_gapple"]["critical_rearm_only"] is True
    assert r.json()["auto_gapple"]["retreat_turn_limit_deg"] == 210.0
    assert r.json()["auto_gapple"]["eating_retreat_turn_limit_deg"] == 150.0
    assert r.json()["auto_gapple"]["retreat_path_hold_ticks"] == 3
    assert r.json()["auto_gapple"]["retreat_stuck_abort_ticks"] == 7
    assert r.json()["auto_gapple"]["retreat_min_ticks"] == 5
    assert r.json()["auto_gapple"]["retreat_max_ticks"] == 70
    assert r.json()["auto_gapple"]["critical_retreat_max_ticks"] == 12
    assert r.json()["auto_gapple"]["combat_recovery_ticks"] == 19
    assert r.json()["auto_gapple"]["retreat_strafe_hold_ticks"] == 5
    assert r.json()["auto_gapple"]["retreat_obstacle_jump_hold_ticks"] == 6
    assert r.json()["auto_gapple"]["retreat_obstacle_escape_ticks"] == 9
    assert r.json()["auto_jump"]["enabled"] is True
    assert r.json()["knockback_dump"]["enabled"] is True
    assert r.json()["friends"]["enabled"] is True
    assert r.json()["friends"]["names"] == ["Alice", "Bob"]
    assert live.params.arena.size_x == 20.0
    live.params.aim_smoothing.enabled = False
    live.params.aim_smoothing.strength = 0.22
    live.params.hit_select_assist.enabled = False
    live.params.hit_select_assist.release_movement = True
    live.params.counter_assist.enabled = False
    live.params.auto_gapple.enabled = False
    live.params.auto_jump.enabled = False
    live.params.knockback_dump.enabled = False
    live.params.friends.enabled = False
    live.params.friends.names = []

    r = client.post("/live/kill")
    assert r.json()["enabled"] is False


def test_rest_models_empty():
    client = TestClient(app)
    r = client.get("/models")
    assert r.status_code == 200
    assert "runs" in r.json()


def test_rest_payloads_replace_non_finite_numbers(monkeypatch):
    monkeypatch.setattr(daemon, "_gpu_status",
                        lambda: {"available": True, "mem_used_gb": float("nan")})
    monkeypatch.setattr(daemon.training, "metrics",
                        lambda *args, **kwargs: [{"reward_mean": float("inf")}])

    client = TestClient(app, raise_server_exceptions=False)

    r = client.get("/status")
    assert r.status_code == 200
    assert r.json()["gpu"]["mem_used_gb"] is None

    r = client.get("/training/metrics")
    assert r.status_code == 200
    assert r.json()[0]["reward_mean"] is None


def test_list_runs_reports_best_and_latest(tmp_path):
    """best.pt et latest.pt sont remontés par /models (flags dédiés) et ne
    polluent pas la liste des checkpoints numérotés."""
    run = tmp_path / "runs" / "boxing"
    run.mkdir(parents=True)
    (run / "ckpt_000025.pt").write_bytes(b"x")
    (run / "latest.pt").write_bytes(b"x")
    (run / "best.pt").write_bytes(b"x")
    (run / "safe_latest.pt").write_bytes(b"x")

    mgr = TrainingManager(repo_root=tmp_path)
    runs = mgr.list_runs()

    assert len(runs) == 1
    assert runs[0]["best"] is True
    assert runs[0]["latest"] is True
    assert runs[0]["safe"] is True
    assert runs[0]["checkpoints"] == ["ckpt_000025.pt"]


def test_daemon_prefers_combo_safe_model_paths(tmp_path, monkeypatch):
    run = tmp_path / "runs" / daemon.COMBO_SAFE_RUN
    run.mkdir(parents=True)
    latest = run / "latest.pt"
    ckpt = run / "ckpt_000616.pt"
    safe = run / "safe_latest.pt"
    latest.write_bytes(b"bad-latest")
    ckpt.write_bytes(b"raw-ckpt")
    safe.write_bytes(b"safe")
    export = tmp_path / daemon.COMBO_SAFE_EXPORT
    export.parent.mkdir(parents=True)
    export.write_bytes(b"export")
    monkeypatch.setattr(daemon.training, "root", tmp_path)

    assert daemon._prefer_combo_safe_model_path(
        f"runs/{daemon.COMBO_SAFE_RUN}/latest.pt"
    ) == f"runs/{daemon.COMBO_SAFE_RUN}/safe_latest.pt"
    assert daemon._prefer_combo_safe_model_path(
        f"runs/{daemon.COMBO_SAFE_RUN}/ckpt_000616.pt"
    ) == f"runs/{daemon.COMBO_SAFE_RUN}/safe_latest.pt"
    assert daemon._prefer_combo_safe_model_path(
        str(latest)
    ) == str(safe)
    assert daemon._prefer_combo_safe_model_path(
        f"runs/{daemon.COMBO_SAFE_RUN}/latest.pt", live_export=True
    ) == daemon.COMBO_SAFE_EXPORT
    assert daemon._prefer_combo_safe_model_path(
        "runs/other/latest.pt"
    ) == "runs/other/latest.pt"
    assert daemon._prefer_combo_safe_model_path("__combo_pad__") == "__combo_pad__"


def test_daemon_load_endpoints_redirect_combo_latest_to_safe(tmp_path, monkeypatch):
    run = tmp_path / "runs" / daemon.COMBO_SAFE_RUN
    run.mkdir(parents=True)
    safe = run / "safe_latest.pt"
    safe.write_bytes(b"safe")
    write_combo_safe_contract(safe)
    export = tmp_path / daemon.COMBO_SAFE_EXPORT
    export.parent.mkdir(parents=True)
    export.write_bytes(b"export")
    export.with_suffix(".json").write_text(json.dumps({
        "source": f"runs/{daemon.COMBO_SAFE_RUN}/safe_latest.pt",
        "source_size": safe.stat().st_size,
        "source_sha256": hashlib.sha256(safe.read_bytes()).hexdigest(),
    }))
    monkeypatch.setattr(daemon.training, "root", tmp_path)
    captured = {}

    def fake_live_load(model):
        captured["live"] = model
        return {"model": model, "history": 8}

    def fake_arena_load(model_a, model_b, **kwargs):
        captured["arena"] = (model_a, model_b)
        return {"ready": True, "models": [model_a, model_b], "sample": kwargs["sample"]}

    monkeypatch.setattr(daemon.live, "load", fake_live_load)
    monkeypatch.setattr(daemon.arena, "load", fake_arena_load)
    client = TestClient(app)

    r = client.post("/live/load", json={
        "model": f"runs/{daemon.COMBO_SAFE_RUN}/latest.pt",
    })
    assert r.status_code == 200
    assert captured["live"] == daemon.COMBO_SAFE_EXPORT

    r = client.post("/arena/load", json={
        "model_a": f"runs/{daemon.COMBO_SAFE_RUN}/latest.pt",
        "model_b": f"runs/{daemon.COMBO_SAFE_RUN}/ckpt_000616.pt",
    })
    assert r.status_code == 200
    assert captured["arena"] == (
        f"runs/{daemon.COMBO_SAFE_RUN}/safe_latest.pt",
        f"runs/{daemon.COMBO_SAFE_RUN}/safe_latest.pt",
    )


def test_daemon_live_load_rejects_stale_combo_safe_export(tmp_path, monkeypatch):
    run = tmp_path / "runs" / daemon.COMBO_SAFE_RUN
    run.mkdir(parents=True)
    safe = run / "safe_latest.pt"
    safe.write_bytes(b"safe")
    write_combo_safe_contract(safe)
    export = tmp_path / daemon.COMBO_SAFE_EXPORT
    export.parent.mkdir(parents=True)
    export.write_bytes(b"stale-export")
    export.with_suffix(".json").write_text(json.dumps({
        "source": f"runs/{daemon.COMBO_SAFE_RUN}/safe_latest.pt",
        "source_size": safe.stat().st_size,
        "source_sha256": "0" * 64,
    }))
    monkeypatch.setattr(daemon.training, "root", tmp_path)
    called = {"live": False}

    def fake_live_load(model):
        called["live"] = True
        return {"model": model, "history": 8}

    monkeypatch.setattr(daemon.live, "load", fake_live_load)
    client = TestClient(app)

    r = client.post("/live/load", json={
        "model": f"runs/{daemon.COMBO_SAFE_RUN}/latest.pt",
    })

    assert r.status_code == 404
    assert "hash mismatch" in r.json()["detail"]
    assert called["live"] is False


def test_daemon_live_load_rejects_combo_safe_without_contract_meta(tmp_path, monkeypatch):
    run = tmp_path / "runs" / daemon.COMBO_SAFE_RUN
    run.mkdir(parents=True)
    safe = run / "safe_latest.pt"
    safe.write_bytes(b"safe")
    export = tmp_path / daemon.COMBO_SAFE_EXPORT
    export.parent.mkdir(parents=True)
    export.write_bytes(b"export")
    export.with_suffix(".json").write_text(json.dumps({
        "source": f"runs/{daemon.COMBO_SAFE_RUN}/safe_latest.pt",
        "source_size": safe.stat().st_size,
        "source_sha256": hashlib.sha256(safe.read_bytes()).hexdigest(),
    }))
    monkeypatch.setattr(daemon.training, "root", tmp_path)
    called = {"live": False}

    def fake_live_load(model):
        called["live"] = True
        return {"model": model, "history": 8}

    monkeypatch.setattr(daemon.live, "load", fake_live_load)
    client = TestClient(app)

    r = client.post("/live/load", json={
        "model": f"runs/{daemon.COMBO_SAFE_RUN}/latest.pt",
    })

    assert r.status_code == 404
    assert "missing combo safe metadata" in r.json()["detail"]
    assert called["live"] is False


def test_models_endpoint_marks_combo_safe_export_freshness(tmp_path, monkeypatch):
    run = tmp_path / "runs" / daemon.COMBO_SAFE_RUN
    run.mkdir(parents=True)
    safe = run / "safe_latest.pt"
    safe.write_bytes(b"safe")
    write_combo_safe_contract(safe)
    models = tmp_path / "models"
    models.mkdir()
    export = models / Path(daemon.COMBO_SAFE_EXPORT).name
    export.write_bytes(b"export")
    export.with_suffix(".json").write_text(json.dumps({
        "source": f"runs/{daemon.COMBO_SAFE_RUN}/safe_latest.pt",
        "source_size": safe.stat().st_size,
        "source_sha256": hashlib.sha256(safe.read_bytes()).hexdigest(),
    }))
    stale = models / "other.pts"
    stale.write_bytes(b"other")
    monkeypatch.setattr(daemon.training, "root", tmp_path)
    client = TestClient(app)

    r = client.get("/models")

    assert r.status_code == 200
    exported = {Path(m["path"]).name: m for m in r.json()["exported"]}
    assert exported[Path(daemon.COMBO_SAFE_EXPORT).name]["export_fresh"] is True
    assert exported[Path(daemon.COMBO_SAFE_EXPORT).name]["export_status"] == "fresh"
    assert "export_fresh" not in exported["other.pts"]


def test_models_endpoint_marks_combo_safe_run_contract(tmp_path, monkeypatch):
    stale_run = tmp_path / "runs" / daemon.COMBO_SAFE_RUN
    stale_run.mkdir(parents=True)
    (stale_run / "safe_latest.pt").write_bytes(b"stale-safe")
    fresh_run = tmp_path / "runs" / daemon.COMBO_COUNTER_SAFE_RUN
    fresh_run.mkdir(parents=True)
    fresh_safe = fresh_run / "safe_latest.pt"
    fresh_safe.write_bytes(b"fresh-safe")
    write_combo_safe_contract(fresh_safe)
    missing_run = tmp_path / "runs" / daemon.COMBO_LEGACY_SAFE_RUN
    missing_run.mkdir(parents=True)
    monkeypatch.setattr(daemon.training, "root", tmp_path)
    client = TestClient(app)

    r = client.get("/models")

    assert r.status_code == 200
    runs = {row["name"]: row for row in r.json()["runs"]}
    assert runs[daemon.COMBO_SAFE_RUN]["combo_safe"] is True
    assert runs[daemon.COMBO_SAFE_RUN]["safe_status"] == "stale"
    assert "missing combo safe metadata" in runs[daemon.COMBO_SAFE_RUN]["safe_error"]
    assert runs[daemon.COMBO_COUNTER_SAFE_RUN]["safe_status"] == "fresh"
    assert runs[daemon.COMBO_LEGACY_SAFE_RUN]["safe_status"] == "missing"


def test_combo_safe_contract_rejects_low_combo_tap(tmp_path, monkeypatch):
    run = tmp_path / "runs" / daemon.COMBO_SAFE_RUN
    run.mkdir(parents=True)
    safe = run / "safe_latest.pt"
    safe.write_bytes(b"low-tap-safe")
    write_combo_safe_contract(safe, combo_tap_frac=0.02)
    monkeypatch.setattr(daemon.training, "root", tmp_path)

    with pytest.raises(ValueError, match="combo safe combo_tap_frac=0.02<0.12"):
        daemon._assert_combo_safe_checkpoint_contract(safe)


def test_combo_safe_contract_rejects_s_tap_disguised_as_combo_tap(tmp_path, monkeypatch):
    run = tmp_path / "runs" / daemon.COMBO_SAFE_RUN
    run.mkdir(parents=True)
    safe = run / "safe_latest.pt"
    safe.write_bytes(b"s-tap-safe")
    write_combo_safe_contract(
        safe,
        combo_tap_frac=0.18,
        combo_s_tap_frac=0.16,
        combo_z_tap_frac=0.02,
    )
    monkeypatch.setattr(daemon.training, "root", tmp_path)

    with pytest.raises(ValueError, match="combo safe combo_z_tap_frac=0.02<0.1"):
        daemon._assert_combo_safe_checkpoint_contract(safe)


def test_combo_safe_contract_rejects_low_hit_wtap(tmp_path, monkeypatch):
    run = tmp_path / "runs" / daemon.COMBO_SAFE_RUN
    run.mkdir(parents=True)
    safe = run / "safe_latest.pt"
    safe.write_bytes(b"low-hit-wtap-safe")
    write_combo_safe_contract(safe, hit_wtap_frac=0.20)
    monkeypatch.setattr(daemon.training, "root", tmp_path)

    with pytest.raises(ValueError, match="combo safe hit_wtap_frac=0.2<0.75"):
        daemon._assert_combo_safe_checkpoint_contract(safe)


def test_combo_safe_contract_rejects_low_under_combo_counter(tmp_path, monkeypatch):
    run = tmp_path / "runs" / daemon.COMBO_SAFE_RUN
    run.mkdir(parents=True)
    safe = run / "safe_latest.pt"
    safe.write_bytes(b"low-counter-safe")
    write_combo_safe_contract(safe, under_combo_counter_hit_frac=0.0)
    monkeypatch.setattr(daemon.training, "root", tmp_path)

    with pytest.raises(
        ValueError,
        match="combo safe under_combo_counter_hit_frac=0<0.05",
    ):
        daemon._assert_combo_safe_checkpoint_contract(safe)


def test_combo_safe_contract_accepts_low_under_combo_counter_with_avoidance_bonus(tmp_path, monkeypatch):
    run = tmp_path / "runs" / daemon.COMBO_SAFE_RUN
    run.mkdir(parents=True)
    safe = run / "safe_latest.pt"
    safe.write_bytes(b"avoidance-counter-safe")
    write_combo_safe_contract(
        safe,
        under_combo_counter_hit_frac=0.0,
        under_combo_avoidance_score_bonus=0.015,
    )
    monkeypatch.setattr(daemon.training, "root", tmp_path)

    daemon._assert_combo_safe_checkpoint_contract(safe)


def test_combo_safe_contract_rejects_dirty_counter_recovery_when_required(tmp_path, monkeypatch):
    run = tmp_path / "runs" / daemon.COMBO_SAFE_RUN
    run.mkdir(parents=True)
    safe = run / "safe_latest.pt"
    safe.write_bytes(b"dirty-counter-recovery-safe")
    write_combo_safe_contract(
        safe,
        requires_counter_recovery=True,
        under_combo_hit_select_clean_frac=0.0,
        under_combo_hit_select_trade_frac=1.0,
    )
    monkeypatch.setattr(daemon.training, "root", tmp_path)

    with pytest.raises(
        ValueError,
        match="combo safe under_combo_hit_select_clean_frac=0<0.2",
    ):
        daemon._assert_combo_safe_checkpoint_contract(safe)


def test_combo_safe_contract_rejects_passive_lateral_opener(tmp_path, monkeypatch):
    run = tmp_path / "runs" / daemon.COMBO_SAFE_RUN
    run.mkdir(parents=True)
    safe = run / "safe_latest.pt"
    safe.write_bytes(b"passive-opener-safe")
    write_combo_safe_contract(
        safe,
        opener_strafe_frac=0.90,
        opener_pressure_frac=0.25,
    )
    monkeypatch.setattr(daemon.training, "root", tmp_path)

    with pytest.raises(ValueError, match="combo safe opener_pressure_frac=0.25<0.6"):
        daemon._assert_combo_safe_checkpoint_contract(safe)


def test_combo_safe_contract_rejects_jitter_lateral_opener(tmp_path, monkeypatch):
    run = tmp_path / "runs" / daemon.COMBO_SAFE_RUN
    run.mkdir(parents=True)
    safe = run / "safe_latest.pt"
    safe.write_bytes(b"jitter-opener-safe")
    write_combo_safe_contract(
        safe,
        opener_strafe_frac=0.90,
        opener_strafe_hold_frac=0.45,
        opener_pressure_frac=0.80,
    )
    monkeypatch.setattr(daemon.training, "root", tmp_path)

    with pytest.raises(ValueError, match="combo safe opener_strafe_hold_frac=0.45<0.7"):
        daemon._assert_combo_safe_checkpoint_contract(safe)


def test_daemon_export_redirects_combo_raw_checkpoint_to_safe(tmp_path, monkeypatch):
    run = tmp_path / "runs" / daemon.COMBO_SAFE_RUN
    run.mkdir(parents=True)
    safe = run / "safe_latest.pt"
    raw = run / "ckpt_000616.pt"
    safe.write_bytes(b"safe")
    write_combo_safe_contract(safe)
    raw.write_bytes(b"raw")
    monkeypatch.setattr(daemon.training, "root", tmp_path)
    captured = {}

    def fake_export(ckpt, out):
        captured["ckpt"] = ckpt
        captured["out"] = out
        return out

    monkeypatch.setattr("train.export.export", fake_export)
    client = TestClient(app)

    r = client.post("/models/export", json={
        "ckpt": f"runs/{daemon.COMBO_SAFE_RUN}/ckpt_000616.pt",
        "out": "models/manual.pts",
    })

    assert r.status_code == 200
    assert captured["ckpt"] == f"runs/{daemon.COMBO_SAFE_RUN}/safe_latest.pt"
    assert captured["out"] == "models/manual.pts"
    assert r.json()["exported"] == "models/manual.pts"


def test_daemon_export_rejects_stale_combo_safe_contract(tmp_path, monkeypatch):
    run = tmp_path / "runs" / daemon.COMBO_SAFE_RUN
    run.mkdir(parents=True)
    (run / "safe_latest.pt").write_bytes(b"stale-safe")
    raw = run / "ckpt_000616.pt"
    raw.write_bytes(b"raw")
    monkeypatch.setattr(daemon.training, "root", tmp_path)
    client = TestClient(app)

    r = client.post("/models/export", json={
        "ckpt": f"runs/{daemon.COMBO_SAFE_RUN}/ckpt_000616.pt",
        "out": "models/manual.pts",
    })

    assert r.status_code == 404
    assert "missing combo safe metadata" in r.json()["detail"]


def test_training_manager_prefers_repo_venv_python(tmp_path):
    venv_python = tmp_path / ".venv" / "Scripts" / "python.exe"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("")

    mgr = TrainingManager(repo_root=tmp_path)

    assert mgr.python == venv_python
    assert mgr.status()["python"] == str(venv_python)
    assert str(venv_python.parent) in mgr._training_env()["PATH"].split(os.pathsep)


def test_training_manager_preserves_windows_path_key(tmp_path, monkeypatch):
    venv_python = tmp_path / ".venv" / "Scripts" / "python.exe"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("")
    msvc_path = str(tmp_path / "BuildTools" / "VC" / "bin")

    mgr = TrainingManager(repo_root=tmp_path)
    monkeypatch.setattr(mgr, "_env_from_script",
                        lambda: {"Path": msvc_path, "TORCH_CUDA_ARCH_LIST": "8.6"})

    env = mgr._training_env()

    assert "Path" not in env
    assert env["PATH"].split(os.pathsep)[:2] == [str(venv_python.parent), msvc_path]


def test_training_manager_ignores_missing_resume(tmp_path, monkeypatch):
    venv_python = tmp_path / ".venv" / "Scripts" / "python.exe"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("")
    launched = {}

    class FakeProc:
        pid = 123

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            pass

    def fake_popen(cmd, **kwargs):
        launched["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr("serve.training_manager.subprocess.Popen", fake_popen)
    monkeypatch.setattr(TrainingManager, "_training_env", lambda self: {})

    mgr = TrainingManager(repo_root=tmp_path)
    mgr.start({"name": "fresh"}, resume="runs/fresh/latest.pt",
              autorestart=False)

    assert "--resume" not in launched["cmd"]
    log = tmp_path / "runs" / "fresh" / "train.log"
    assert "resume ignoré" in log.read_text(encoding="utf-8")


def test_training_manager_prefers_safe_for_restart_but_respects_explicit_latest(tmp_path, monkeypatch):
    venv_python = tmp_path / ".venv" / "Scripts" / "python.exe"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("")
    run = tmp_path / "runs" / "boxing"
    run.mkdir(parents=True)
    latest = run / "latest.pt"
    safe = run / "safe_latest.pt"
    latest.write_bytes(b"bad-latest")
    safe.write_bytes(b"safe")
    launched = {}

    class FakeProc:
        pid = 123

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            pass

    def fake_popen(cmd, **kwargs):
        launched["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr("serve.training_manager.subprocess.Popen", fake_popen)
    monkeypatch.setattr(TrainingManager, "_training_env", lambda self: {})

    mgr = TrainingManager(repo_root=tmp_path)
    assert mgr._preferred_restart_checkpoint("boxing") == safe
    mgr.start({"name": "boxing"}, resume="runs/boxing/latest.pt",
              autorestart=False)

    resume_index = launched["cmd"].index("--resume") + 1
    assert launched["cmd"][resume_index] == str(latest)


def test_training_manager_uses_safe_when_latest_resume_missing(tmp_path, monkeypatch):
    venv_python = tmp_path / ".venv" / "Scripts" / "python.exe"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("")
    run = tmp_path / "runs" / "boxing"
    run.mkdir(parents=True)
    safe = run / "safe_latest.pt"
    safe.write_bytes(b"safe-only")
    launched = {}

    class FakeProc:
        pid = 123

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            pass

    def fake_popen(cmd, **kwargs):
        launched["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr("serve.training_manager.subprocess.Popen", fake_popen)
    monkeypatch.setattr(TrainingManager, "_training_env", lambda self: {})

    mgr = TrainingManager(repo_root=tmp_path)
    mgr.start({"name": "boxing"}, resume="runs/boxing/latest.pt",
              autorestart=False)

    resume_index = launched["cmd"].index("--resume") + 1
    assert launched["cmd"][resume_index] == str(safe)


def test_training_manager_countertap_falls_back_to_legacy_safe_seed(tmp_path, monkeypatch):
    venv_python = tmp_path / ".venv" / "Scripts" / "python.exe"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("")
    legacy = tmp_path / "runs" / "combo_god_directpad_lock_combo12" / "safe_latest.pt"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"legacy-safe")
    launched = {}

    class FakeProc:
        pid = 123

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            pass

    def fake_popen(cmd, **kwargs):
        launched["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr("serve.training_manager.subprocess.Popen", fake_popen)
    monkeypatch.setattr(TrainingManager, "_training_env", lambda self: {})

    mgr = TrainingManager(repo_root=tmp_path)
    mgr.start(
        {"name": "combo_god_countertap96_combo12", "resume_as_seed": True},
        resume="runs/combo_god_countertap96_combo12/safe_latest.pt",
        autorestart=False,
    )

    resume_index = launched["cmd"].index("--resume") + 1
    assert launched["cmd"][resume_index] == str(legacy)


def test_training_manager_leaderboard_falls_back_to_countertap_safe_seed(tmp_path, monkeypatch):
    venv_python = tmp_path / ".venv" / "Scripts" / "python.exe"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("")
    counter = tmp_path / "runs" / "combo_god_countertap96_combo12" / "safe_latest.pt"
    counter.parent.mkdir(parents=True)
    counter.write_bytes(b"counter-safe")
    launched = {}

    class FakeProc:
        pid = 123

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            pass

    def fake_popen(cmd, **kwargs):
        launched["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr("serve.training_manager.subprocess.Popen", fake_popen)
    monkeypatch.setattr(TrainingManager, "_training_env", lambda self: {})

    mgr = TrainingManager(repo_root=tmp_path)
    mgr.start(
        {"name": "combo_god_leaderboard10_combo12", "resume_as_seed": True},
        resume="runs/combo_god_leaderboard10_combo12/safe_latest.pt",
        autorestart=False,
    )

    resume_index = launched["cmd"].index("--resume") + 1
    assert launched["cmd"][resume_index] == str(counter)


def test_training_start_normalizes_reward_hurt_sign():
    cfg = {"sim": {"reward_hurt": 1.2}}
    assert _normalize_training_cfg(cfg)["sim"]["reward_hurt"] == -1.2
    cfg = {"sim": {"reward_hurt": -1.2}}
    assert _normalize_training_cfg(cfg)["sim"]["reward_hurt"] == -1.2
