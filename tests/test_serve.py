"""Tests du daemon : protocole, LiveSession, endpoints REST."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient            # noqa: E402

from serve.daemon import app, live                   # noqa: E402
from serve.live import LiveSession                   # noqa: E402
from serve.protocol import ArenaCalib, action_to_msg, player_from_msg  # noqa: E402


def state_msg(z_self=3.0, z_target=5.5):
    def p(z):
        return {"x": 109.0, "y": 64.0, "z": 100.0 + z, "vx": 0.0, "vy": 0.0,
                "vz": 0.0, "yaw": 0.0, "pitch": 0.0, "onGround": True,
                "sprinting": False, "hurtTime": 0, "hits": 0}
    return {"t": "state", "tick": 1, "self": p(z_self), "target": p(z_target)}


ARENA = ArenaCalib(origin_x=100.0, origin_z=100.0, size_x=18.0, size_z=18.0,
                   floor_y=64.0)


def test_player_from_msg_arena_frame():
    p = player_from_msg(state_msg()["self"], ARENA)
    assert p.x == 9.0 and p.z == 3.0 and p.y == 0.0
    assert p.on_ground


def test_hurt_time_rescaled():
    msg = state_msg()["self"]
    msg["hurtTime"] = 9
    assert player_from_msg(msg, ARENA).hurt_resistant_time == 18


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


def test_live_session_produces_action(tmp_path):
    s = _session_with_model(tmp_path)
    a = s.on_state(state_msg())
    assert a is not None and a["t"] == "action"
    assert abs(a["dyaw"]) <= s.params.max_rot_speed + 1e-6
    assert abs(a["dpitch"]) <= s.params.max_rot_speed + 1e-6
    assert s.last_latency_ms > 0


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
                                          "arena": {"origin_x": 1.0, "origin_z": 2.0,
                                                    "size_x": 20.0, "size_z": 20.0,
                                                    "floor_y": 60.0}})
    assert r.status_code == 200
    assert r.json()["max_cps"] == 9.0
    assert live.params.arena.size_x == 20.0

    r = client.post("/live/kill")
    assert r.json()["enabled"] is False


def test_rest_models_empty():
    client = TestClient(app)
    r = client.get("/models")
    assert r.status_code == 200
    assert "runs" in r.json()
