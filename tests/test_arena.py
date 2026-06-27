"""Tests de l'arène IA vs IA (visualiseur)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fastapi.testclient import TestClient        # noqa: E402

from serve.arena import ArenaSession, _stabilize_axis_norm             # noqa: E402
from sim import OBS_DIM                           # noqa: E402
from sim_ref.player import PlayerState            # noqa: E402
from serve.daemon import app, arena as daemon_arena  # noqa: E402


@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory):
    from train.model import JudasPolicy, PolicyConfig
    cfg = PolicyConfig(history=4, d_model=32, n_heads=2, n_layers=1)
    pol = JudasPolicy(cfg)
    path = tmp_path_factory.mktemp("arena") / "ckpt.pt"
    torch.save({"policy": pol.state_dict(), "policy_cfg": cfg.__dict__,
                "iter": 0}, path)
    return str(path)


def test_load_and_step(tiny_ckpt):
    s = ArenaSession()
    st = s.load(tiny_ckpt, tiny_ckpt, target_hits=5, arena_size=12.0,
                kb_h=0.9055, kb_v=0.8835, kb_idle=0.6)
    assert st["ready"] and st["models"][0] == st["models"][1]
    assert s.cfg.kb_h_mult == 0.9055
    assert s.cfg.kb_v_mult == 0.8835
    assert s.cfg.kb_idle_mult == 0.6

    state = s.step()
    assert state["ready"]
    assert len(state["players"]) == 2
    p = state["players"][0]
    for k in ("x", "y", "z", "yaw", "pitch", "sprint", "hurt", "hits",
              "combo", "max_combo", "swing"):
        assert k in p
    assert state["arena"]["sx"] == 12.0
    assert "max_combo" in state


def test_match_completes_and_counts(tiny_ckpt):
    s = ArenaSession()
    s.load(tiny_ckpt, tiny_ckpt, target_hits=2, arena_size=10.0)
    s.cfg.max_ticks = 50
    s.sim.cfg.max_ticks = 50          # force un timeout rapide
    for m in s.sim._matches:
        m.cfg.max_ticks = 50
    done_seen = False
    for _ in range(120):
        st = s.step()
        if st["done"]:
            done_seen = True
            assert st["winner"] in (-1, 0, 1)
            break
    assert done_seen
    assert s.matches == 1
    assert s.tick <= 1                # historique reparti de zéro


def test_reset(tiny_ckpt):
    s = ArenaSession()
    s.load(tiny_ckpt, tiny_ckpt)
    for _ in range(5):
        s.step()
    s.reset()
    assert s.tick == 0


def test_chase_bot_can_be_loaded(tiny_ckpt):
    s = ArenaSession()
    st = s.load(tiny_ckpt, "__chase_bot__", target_hits=3, arena_size=12.0,
                spawn_gap=1.0)
    assert st["ready"]
    assert st["models"][1] == "chase-bot"
    assert s.step()["ready"]




def test_arena_defaults_match_combo_visualizer_contract(tiny_ckpt):
    s = ArenaSession()
    st = s.load(tiny_ckpt, "__chase_bot__")

    assert st["sample"] is True
    assert s.cfg.cps_min == 10.0
    assert s.cfg.cps_max == 10.0
    assert s.cfg.rot_speed_min == 190.0
    assert s.cfg.rot_speed_max == 190.0
    assert s.cfg.arena_size_x == 40.0
    assert s.cfg.arena_size_z == 40.0
    assert s.cfg.spawn_gap == 8.0
    assert s.cfg.target_hits == 50
    assert s.cfg.kb_h_mult == 0.92
    assert s.cfg.kb_v_mult == 0.90
    assert s.cfg.kb_idle_mult == 0.6
    assert s.cfg.aim_smooth_min == 0.02
    assert s.cfg.aim_smooth_max == 0.02
    assert s.cfg.post_sprint_hit_stop is True
    assert st["post_sprint_hit_stop"] is True


def test_arena_forces_sampling_for_same_policy_mirror(tiny_ckpt):
    s = ArenaSession()
    st = s.load(tiny_ckpt, tiny_ckpt, sample=False)

    assert st["sample"] is True
    assert st["mirror_sample_forced"] is True
    assert st["mirror_desync"] is True
    assert s.sample is True


def test_arena_same_export_uses_source_policy_for_mirror_sampling(tiny_ckpt, tmp_path):
    from train.export import export

    exported = export(tiny_ckpt, str(tmp_path / "tiny.pts"))
    s = ArenaSession()
    st = s.load(str(exported), str(exported), sample=True)

    assert st["sample"] is True
    assert st["mirror_desync"] is True
    assert s.agents[0].sample_policy is not None
    assert s.agents[1].sample_policy is not None
    assert s.agents[0].sample_history > 0
    assert s.agents[1].sample_history > 0


def test_arena_mirror_opening_desync_suppresses_one_initial_trade():
    class Match:
        def __init__(self):
            self.players = [PlayerState(), PlayerState()]
            self.players[0].x = 0.0
            self.players[0].z = 0.0
            self.players[1].x = 3.0
            self.players[1].z = 0.0

    class FakeSim:
        def __init__(self):
            self._matches = [Match()]

    s = ArenaSession()
    s.sim = FakeSim()
    s.matches = 0
    actions = np.zeros((2, 7), dtype=np.float32)
    actions[:, 6] = 1.0

    s._apply_mirror_opening_desync(actions)

    assert actions[0, 6] == 1.0
    assert actions[1, 6] == 0.0
    assert s._mirror_opened is True


def test_arena_mirror_opening_desync_alternates_receiver():
    class Match:
        def __init__(self):
            self.players = [PlayerState(), PlayerState()]
            self.players[0].x = 0.0
            self.players[0].z = 0.0
            self.players[1].x = 3.0
            self.players[1].z = 0.0

    class FakeSim:
        def __init__(self):
            self._matches = [Match()]

    s = ArenaSession()
    s.sim = FakeSim()
    s.matches = 1
    actions = np.zeros((2, 7), dtype=np.float32)
    actions[:, 6] = 1.0

    s._apply_mirror_opening_desync(actions)

    assert actions[0, 6] == 0.0
    assert actions[1, 6] == 1.0


def test_arena_stabilized_aim_replaces_raw_sky_overshoot():
    assert _stabilize_axis_norm(1.0, 9.5, 190.0) == pytest.approx(9.5 / 190.0)
    assert _stabilize_axis_norm(-1.0, -7.0, 190.0) == pytest.approx(-7.0 / 190.0)
    assert _stabilize_axis_norm(0.8, 0.01, 190.0) == 0.0
    assert abs(_stabilize_axis_norm(1.0, 500.0, 190.0)) <= 1.0




def test_arena_uses_sim_combo_info_without_leaking_combo_across_matches():
    class Agent:
        history = 8

        def act(self, hist, sample):
            return np.zeros(7, dtype=np.float32)

    class Match:
        def __init__(self):
            self.players = [PlayerState(), PlayerState()]

    class FakeSim:
        cfg = type("Cfg", (), {"arena_size_x": 40.0, "arena_size_z": 40.0,
                               "spawn_gap": 8.0, "target_hits": 50})()

        def __init__(self):
            self._matches = [Match()]

        def step(self, actions):
            obs = np.zeros((1, 2, OBS_DIM), dtype=np.float32)
            reward = np.zeros((1, 2), dtype=np.float32)
            done = np.array([True])
            info = {
                "winner": np.array([0], dtype=np.int32),
                "dealt": np.array([[1, 0]], dtype=np.int32),
                "combo": np.array([[12, 0]], dtype=np.int32),
            }
            self._matches = [Match()]  # sim_ref auto-reset masquerait l'etat final
            return obs, reward, done, info

    s = ArenaSession()
    s.agents = [Agent(), Agent()]
    s.sim = FakeSim()
    s._hist = np.zeros((2, 8, OBS_DIM), dtype=np.float32)

    st = s.step()

    assert st["done"] is True
    assert st["winner"] == 0
    assert st["players"][0]["hits"] == 1
    assert st["combo"] == [12, 0]
    assert st["max_combo"] == [12, 0]
    assert s.combo == [0, 0]
def test_arena_rest_endpoints(tiny_ckpt):
    client = TestClient(app)
    r = client.get("/arena/status")
    assert r.status_code == 200

    r = client.post("/arena/load", json={"model_a": tiny_ckpt,
                                         "model_b": "__chase_bot__"})
    assert r.status_code == 200 and r.json()["ready"]
    assert r.json()["sample"] is True
    assert r.json()["mirror_sample_forced"] is False
    assert daemon_arena.cfg.target_hits == 50
    assert daemon_arena.cfg.arena_size_x == 40.0
    assert daemon_arena.cfg.spawn_gap == 8.0
    assert daemon_arena.cfg.kb_h_mult == 0.92
    assert daemon_arena.cfg.kb_v_mult == 0.90
    assert daemon_arena.cfg.kb_idle_mult == 0.6

    r = client.post("/arena/load", json={"model_a": tiny_ckpt,
                                         "model_b": tiny_ckpt,
                                         "sample": False,
                                         "target_hits": 5,
                                         "kb_h": 0.91,
                                         "kb_v": 0.88,
                                         "kb_idle": 0.6})
    assert r.status_code == 200 and r.json()["ready"]
    assert r.json()["sample"] is True
    assert r.json()["mirror_sample_forced"] is True
    assert daemon_arena.cfg.kb_h_mult == 0.91
    assert daemon_arena.cfg.kb_v_mult == 0.88
    assert daemon_arena.cfg.kb_idle_mult == 0.6

    r = client.post("/arena/control", json={"speed": 4.0})
    assert r.json()["speed"] == 4.0

    r = client.post("/arena/control", json={"reset": True})
    assert r.status_code == 200
