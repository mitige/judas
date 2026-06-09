"""Protocole mod <-> daemon : conversion des messages JSON du mod Forge
vers des PlayerState sim_ref (repère arène) et des actions vers le mod.

Voir mod/README.md pour le format exact.
"""

from dataclasses import dataclass

from sim_ref import PlayerState


@dataclass
class ArenaCalib:
    """Position de l'arène boxing dans le monde réel (configurée via l'app)."""
    origin_x: float = 0.0
    origin_z: float = 0.0
    size_x: float = 18.0
    size_z: float = 18.0
    floor_y: float = 0.0


def player_from_msg(p: dict, arena: ArenaCalib) -> PlayerState:
    """JSON du mod -> PlayerState dans le repère arène.

    hurtTime (0..10, animation client) est converti vers l'échelle
    hurtResistantTime (0..20) utilisée par l'observation : x2.
    """
    return PlayerState(
        x=float(p["x"]) - arena.origin_x,
        y=float(p["y"]) - arena.floor_y,
        z=float(p["z"]) - arena.origin_z,
        vx=float(p["vx"]),
        vy=float(p["vy"]),
        vz=float(p["vz"]),
        yaw=float(p["yaw"]),
        pitch=float(p["pitch"]),
        on_ground=bool(p["onGround"]),
        sprinting=bool(p["sprinting"]),
        hurt_resistant_time=min(20, int(p["hurtTime"]) * 2),
        hits=int(p.get("hits", 0)),
    )


def action_to_msg(action7) -> dict:
    """Tenseur/séquence sim [7] (déjà humanisé) -> message JSON pour le mod."""
    a = [float(v) for v in action7]
    return {
        "t": "action",
        "dyaw": a[0],
        "dpitch": a[1],
        "forward": 1 if a[2] > 0.5 else (-1 if a[2] < -0.5 else 0),
        "strafe": 1 if a[3] > 0.5 else (-1 if a[3] < -0.5 else 0),
        "jump": a[4] > 0.5,
        "sprint": a[5] > 0.5,
        "attack": a[6] > 0.5,
    }
