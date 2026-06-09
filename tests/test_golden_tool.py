"""Round-trip de l'outil golden : une trace générée PAR sim_ref doit être
rejouée avec une erreur nulle (vérifie l'outil lui-même)."""

import json

from sim_ref.physics import living_update_movement
from sim_ref.player import PlayerState

from tools.golden_compare import load_trace, replay

ARENA = 1.0e6


def synth_trace(path, n=200):
    p = PlayerState(x=10.0, y=0.0, z=10.0)
    records = []

    def rec(tick, fwd, jump, sprint, left):
        records.append({
            "tick": tick,
            "x": p.x - ARENA / 2, "y": p.y + 64.0, "z": p.z - ARENA / 2,
            "vx": p.vx, "vy": p.vy, "vz": p.vz,
            "yaw": p.yaw, "pitch": p.pitch,
            "onGround": p.on_ground, "sprinting": p.sprinting,
            "in_fwd": fwd, "in_jump": jump, "in_sprint": sprint, "in_left": left,
        })

    # important : les coordonnées de la trace sont "monde réel" (offset),
    # golden_compare recale tout dans son propre repère
    p.x += ARENA / 2
    p.z += ARENA / 2
    rec(0, False, False, False, False)
    for t in range(1, n):
        fwd = t > 20
        jump = 100 < t < 160
        sprint = t > 60
        left = t > 170
        p.sprinting = sprint and fwd
        living_update_movement(p, 1.0 if left else 0.0, 1.0 if fwd else 0.0,
                               jump, ARENA, ARENA, -1)
        rec(t, fwd, jump, sprint, left)

    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_roundtrip_zero_error(tmp_path):
    path = tmp_path / "trace.jsonl"
    synth_trace(path)
    res = replay(load_trace(str(path)))
    assert res["worst"]["pos"] < 1e-9
    assert res["worst"]["vel"] < 1e-9
