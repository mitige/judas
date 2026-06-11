"""Tests de la physique 1.8.9 contre les valeurs connues du jeu vanilla."""

import math

from sim_ref import PlayerState
from sim_ref.constants import move_speed
from sim_ref.physics import living_update_movement

ARENA = 1000.0  # arène géante : pas de murs dans les tests de physique pure


def run_ticks(p, n, strafe=0.0, forward=0.0, jump=False, amp=-1):
    for _ in range(n):
        living_update_movement(p, strafe, forward, jump, ARENA, ARENA, amp)


def spawn(**kw):
    return PlayerState(x=500.0, y=0.0, z=500.0, **kw)


# ---------------------------------------------------------------- vitesses

def block_per_tick(p, forward=1.0, sprint=False, amp=-1, ticks=200):
    if sprint:
        p.sprinting = True
    z0 = None
    for _ in range(ticks):
        living_update_movement(p, 0.0, forward, False, ARENA, ARENA, amp)
    z0 = p.z
    living_update_movement(p, 0.0, forward, False, ARENA, ARENA, amp)
    return p.z - z0


def test_walk_speed_vanilla():
    """Marche : 4.317 blocs/s (valeur vanilla bien connue)."""
    bpt = block_per_tick(spawn(), amp=-1)
    assert abs(bpt * 20 - 4.3170) < 2e-3


def test_sprint_speed_vanilla():
    """Sprint : 5.612 blocs/s."""
    bpt = block_per_tick(spawn(), sprint=True, amp=-1)
    assert abs(bpt * 20 - 5.6121) < 2e-3


def test_sprint_speed2_boxing():
    """Sprint + Speed II (boxing) : ~7.86 blocs/s."""
    bpt = block_per_tick(spawn(), sprint=True, amp=1)
    assert abs(bpt * 20 - 7.8557) < 5e-3


def test_move_speed_attribute():
    assert abs(move_speed(False, -1) - 0.1) < 1e-6
    assert abs(move_speed(True, -1) - 0.13) < 1e-6
    assert abs(move_speed(False, 1) - 0.14) < 1e-6      # Speed II
    assert abs(move_speed(True, 1) - 0.182) < 1e-6      # sprint + Speed II


# ------------------------------------------------------------------- saut

def test_jump_peak_height_vanilla():
    """Hauteur max de saut 1.8.9 réelle : 1.2492 blocs.

    Le seuil vanilla |motionY| < 0.005 -> 0 (tête d'onLivingUpdate) annule le
    dernier micro-pas de 0.0030 à l'apex : la valeur naïve 1.2522 (calculée
    sans le seuil) n'est jamais atteinte en jeu."""
    p = spawn()
    peak = 0.0
    living_update_movement(p, 0.0, 0.0, True, ARENA, ARENA, -1)
    for _ in range(20):
        living_update_movement(p, 0.0, 0.0, False, ARENA, ARENA, -1)
        peak = max(peak, p.y)
    assert abs(peak - 1.2491871) < 1e-4


def test_jump_airtime_vanilla():
    """Un saut vanilla dure 12 ticks avant de retoucher le sol."""
    p = spawn()
    living_update_movement(p, 0.0, 0.0, True, ARENA, ARENA, -1)
    airborne = 1
    while not p.on_ground and airborne < 40:
        living_update_movement(p, 0.0, 0.0, False, ARENA, ARENA, -1)
        airborne += 1
    assert airborne == 12


def test_jump_cooldown_value():
    """jumpTicks = 10 posé au saut, remis à 0 quand la touche est relâchée."""
    p = spawn()
    living_update_movement(p, 0.0, 0.0, True, ARENA, ARENA, -1)
    assert p.jump_ticks == 10
    living_update_movement(p, 0.0, 0.0, False, ARENA, ARENA, -1)
    assert p.jump_ticks == 0  # touche relâchée -> reset vanilla


def test_sprint_jump_boost():
    """Le sprint-jump ajoute 0.2 dans la direction du regard (yaw 0 -> +Z)."""
    p1, p2 = spawn(), spawn()
    p1.sprinting = p2.sprinting = True
    living_update_movement(p1, 0.0, 1.0, False, ARENA, ARENA, -1)
    living_update_movement(p2, 0.0, 1.0, True, ARENA, ARENA, -1)
    # delta vz immédiat = 0.2 * f4 (0.546) entre saut et non-saut au 1er tick
    assert p2.vz - p1.vz > 0.1


# ------------------------------------------------------------- friction/air

def test_ground_friction_decay():
    """Sans input, la vitesse au sol décroît par x0.546 par tick."""
    p = spawn()
    p.vz = 0.5
    living_update_movement(p, 0.0, 0.0, False, ARENA, ARENA, -1)
    assert abs(p.vz - 0.5 * 0.546) < 1e-9


def test_air_drag():
    """En l'air, drag horizontal x0.91 et gravité (vy-0.08)*0.98."""
    p = spawn()
    p.on_ground = False
    p.y = 10.0
    p.vz = 0.5
    p.vy = 0.0
    living_update_movement(p, 0.0, 0.0, False, ARENA, ARENA, -1)
    assert abs(p.vz - 0.5 * 0.91) < 1e-9
    assert abs(p.vy - (0.0 - 0.08) * 0.98) < 1e-9


def test_gravity_steady_on_ground():
    """Au sol, motionY est annulé par la collision puis vaut -0.0784.

    2 ticks : le premier fait converger l'état spawn (vy=0) vers le régime
    vanilla (la gravité plaque l'entité au sol à chaque tick)."""
    p = spawn()
    living_update_movement(p, 0.0, 0.0, False, ARENA, ARENA, -1)
    living_update_movement(p, 0.0, 0.0, False, ARENA, ARENA, -1)
    assert p.on_ground
    assert abs(p.vy - (-0.08 * 0.98)) < 1e-9


# ------------------------------------------------------------------- murs

def test_wall_collision_clamps_and_cuts_motion():
    p = PlayerState(x=5.0, y=0.0, z=1.0, yaw=180.0)  # yaw 180 -> -Z
    for _ in range(20):
        living_update_movement(p, 0.0, 1.0, False, 10.0, 10.0, 1)
    assert abs(p.z - 0.3) < 1e-9          # demi-largeur contre le mur
    assert p.collided_horizontally
    assert p.vz == 0.0


def test_diagonal_slightly_faster_vanilla_quirk():
    """Quirk vanilla : la diagonale est ~2% plus rapide qu'avancer tout droit.

    Les inputs sont multipliés par 0.98, mais moveFlying normalise le vecteur
    quand sa norme dépasse 1 — en diagonale (norme 1.386) le 0.98 disparaît,
    tout droit (norme 0.98) il subsiste. Rapport attendu : 1/0.98 ~= 1.0204."""
    p1, p2 = spawn(), spawn()
    for _ in range(100):
        living_update_movement(p1, 0.0, 1.0, False, ARENA, ARENA, -1)
        living_update_movement(p2, 1.0, 1.0, False, ARENA, ARENA, -1)
    d1 = math.hypot(p1.x - 500.0, p1.z - 500.0)
    d2 = math.hypot(p2.x - 500.0, p2.z - 500.0)
    assert d2 > d1
    assert abs(d2 / d1 - 1.0 / 0.98) < 1e-3
