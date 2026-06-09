"""Collisions Entity.moveEntity portées pour une arène plate à 4 murs.

Vanilla résout les collisions axe par axe dans l'ordre Y, X, Z contre les AABB
du monde. Ici le monde est : sol y=0, murs x=0 / x=Lx, z=0 / z=Lz (hauteur
infinie). Le joueur est une AABB de demi-largeur 0.3.

Effets de bord vanilla reproduits :
- onGround = collision verticale avec dy initial < 0
- l'axe qui collisionne voit son motion mis à 0
- collided_horizontally coupe le sprint (géré dans match.py)
"""

from .constants import PLAYER_HALF_WIDTH
from .player import PlayerState


def move_entity(p: PlayerState, dx: float, dy: float, dz: float,
                arena_x: float, arena_z: float) -> None:
    dx0, dy0, dz0 = dx, dy, dz

    # --- Axe Y (sol) ---
    new_y = p.y + dy
    if new_y < 0.0:
        new_y = 0.0
        dy = new_y - p.y
    p.y = new_y

    # --- Axe X (murs) ---
    lo = PLAYER_HALF_WIDTH
    hi_x = arena_x - PLAYER_HALF_WIDTH
    new_x = p.x + dx
    if new_x < lo:
        new_x = lo
        dx = new_x - p.x
    elif new_x > hi_x:
        new_x = hi_x
        dx = new_x - p.x
    p.x = new_x

    # --- Axe Z (murs) ---
    hi_z = arena_z - PLAYER_HALF_WIDTH
    new_z = p.z + dz
    if new_z < lo:
        new_z = lo
        dz = new_z - p.z
    elif new_z > hi_z:
        new_z = hi_z
        dz = new_z - p.z
    p.z = new_z

    collided_x = dx0 != dx
    collided_y = dy0 != dy
    collided_z = dz0 != dz

    p.collided_horizontally = collided_x or collided_z
    p.on_ground = collided_y and dy0 < 0.0

    # Vanilla : l'axe bloqué annule son motion
    if collided_x:
        p.vx = 0.0
    if collided_y:
        p.vy = 0.0
    if collided_z:
        p.vz = 0.0
