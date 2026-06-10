"""Tests de l'arène IA vs IA (visualiseur)."""

import pytest

torch = pytest.importorskip("torch")

from fastapi.testclient import TestClient        # noqa: E402

from serve.arena import ArenaSession             # noqa: E402
from serve.daemon import app                     # noqa: E402


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
    st = s.load(tiny_ckpt, tiny_ckpt, target_hits=5, arena_size=12.0)
    assert st["ready"] and st["models"][0] == st["models"][1]

    state = s.step()
    assert state["ready"]
    assert len(state["players"]) == 2
    p = state["players"][0]
    for k in ("x", "y", "z", "yaw", "pitch", "sprint", "hurt", "hits", "swing"):
        assert k in p
    assert state["arena"]["sx"] == 12.0


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


def test_arena_rest_endpoints(tiny_ckpt):
    client = TestClient(app)
    r = client.get("/arena/status")
    assert r.status_code == 200

    r = client.post("/arena/load", json={"model_a": tiny_ckpt,
                                         "model_b": tiny_ckpt,
                                         "target_hits": 5})
    assert r.status_code == 200 and r.json()["ready"]

    r = client.post("/arena/control", json={"speed": 4.0})
    assert r.json()["speed"] == 4.0

    r = client.post("/arena/control", json={"reset": True})
    assert r.status_code == 200
