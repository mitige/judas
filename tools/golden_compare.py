"""Compare une trace golden du mod Forge (touche J in-game) à sim_ref.

    python tools/golden_compare.py .minecraft/judas-traces/trace-XXX.jsonl [--tol 1e-4]

Rejoue les inputs enregistrés dans la physique de référence et mesure l'écart
position/vitesse tick par tick. C'est LE juge de paix de l'exactitude du
simulateur face au vrai client 1.8.9.

La trace doit être enregistrée sur sol plat, loin des murs (la séquence
scriptée du mod s'en charge).
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim_ref.physics import living_update_movement  # noqa: E402
from sim_ref.player import PlayerState               # noqa: E402

ARENA = 1.0e6   # pas de murs


def load_trace(path: str) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def replay(trace: list[dict]) -> dict:
    first = trace[0]
    y0 = first["y"]
    p = PlayerState(
        x=first["x"] + ARENA / 2, y=first["y"] - y0, z=first["z"] + ARENA / 2,
        vx=first["vx"], vy=first["vy"], vz=first["vz"],
        yaw=first["yaw"], pitch=first["pitch"],
        on_ground=first["onGround"], sprinting=first["sprinting"],
    )
    worst = {"pos": 0.0, "vel": 0.0, "tick": -1}
    per_tick = []
    for rec in trace[1:]:
        # état de sprint : on fait confiance au client (double-tap, etc.)
        p.sprinting = rec["in_sprint"] and rec["in_fwd"]
        strafe = 1.0 if rec.get("in_left") else 0.0
        forward = 1.0 if rec["in_fwd"] else 0.0
        living_update_movement(p, strafe, forward, rec["in_jump"],
                               ARENA, ARENA, -1)
        ex = abs(p.x - (rec["x"] + ARENA / 2))
        ey = abs(p.y - (rec["y"] - y0))
        ez = abs(p.z - (rec["z"] + ARENA / 2))
        ev = math.sqrt((p.vx - rec["vx"]) ** 2 + (p.vy - rec["vy"]) ** 2
                       + (p.vz - rec["vz"]) ** 2)
        ep = math.sqrt(ex * ex + ey * ey + ez * ez)
        per_tick.append((rec["tick"], ep, ev))
        if ep > worst["pos"]:
            worst.update(pos=ep, tick=rec["tick"])
        worst["vel"] = max(worst["vel"], ev)
        # re-synchronise pour mesurer l'erreur PAR TICK (pas cumulée)
        p.x = rec["x"] + ARENA / 2
        p.y = rec["y"] - y0
        p.z = rec["z"] + ARENA / 2
        p.vx, p.vy, p.vz = rec["vx"], rec["vy"], rec["vz"]
        p.yaw, p.pitch = rec["yaw"], rec["pitch"]
        p.on_ground = rec["onGround"]
    return {"worst": worst, "per_tick": per_tick}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="tolérance position par tick (blocs)")
    args = ap.parse_args()

    trace = load_trace(args.trace)
    if len(trace) < 10:
        print("trace trop courte")
        sys.exit(1)
    res = replay(trace)
    w = res["worst"]
    bad = [t for t, ep, _ in res["per_tick"] if ep > args.tol]
    print(f"{len(trace)} ticks rejoués")
    print(f"écart position max / tick : {w['pos']:.3e} blocs (tick {w['tick']})")
    print(f"écart vitesse  max / tick : {w['vel']:.3e}")
    if bad:
        print(f"ECHEC : {len(bad)} ticks au-dessus de la tolérance {args.tol:g}")
        print(f"premiers ticks fautifs : {bad[:10]}")
        sys.exit(1)
    print(f"OK — physique sim_ref fidèle au client 1.8.9 (tol {args.tol:g})")


if __name__ == "__main__":
    main()
