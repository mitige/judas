"""Combat 1.8.9 : visée (raycast), portée, knockback, invulnérabilité.

Chaîne vanilla d'un hit :
  client  : EntityRenderer.getMouseOver — raycast œil -> regard, portée 3.0,
            contre l'AABB de la cible étendue de getCollisionBorderSize() (0.1)
  serveur : EntityPlayer.attackTargetEntityWithCurrentItem
            -> EntityLivingBase.attackEntityFrom (hurtResistantTime)
            -> EntityLivingBase.knockBack (0.4 horizontal, +0.4 vertical
               inconditionnel en 1.8.9 — la garde onGround est 1.9+)
            -> bonus sprint : addVelocity(-sin(yaw)*0.5, 0.1, cos(yaw)*0.5),
               attaquant ralenti x0.6 et sprint coupé (sprint reset)
"""

import math
from dataclasses import dataclass

from . import constants as C
from .player import PlayerState


def look_vector(yaw: float, pitch: float) -> tuple:
    """Entity.getLook (vecteur unitaire de visée)."""
    yaw_rad = yaw * C.DEG_TO_RAD
    pitch_rad = pitch * C.DEG_TO_RAD
    cp = math.cos(pitch_rad)
    return (-math.sin(yaw_rad) * cp, -math.sin(pitch_rad), math.cos(yaw_rad) * cp)


def ray_intersects_aabb(ox: float, oy: float, oz: float,
                        dx: float, dy: float, dz: float,
                        min_x: float, min_y: float, min_z: float,
                        max_x: float, max_y: float, max_z: float,
                        max_dist: float) -> float:
    """Slab method. Retourne la distance d'impact, ou -1 si pas d'impact <= max_dist.

    Si l'origine est déjà dans l'AABB, retourne 0 (vanilla touche aussi)."""
    if min_x <= ox <= max_x and min_y <= oy <= max_y and min_z <= oz <= max_z:
        return 0.0

    t_min, t_max = 0.0, max_dist
    for o, d, lo, hi in ((ox, dx, min_x, max_x),
                         (oy, dy, min_y, max_y),
                         (oz, dz, min_z, max_z)):
        if abs(d) < 1.0e-12:
            if o < lo or o > hi:
                return -1.0
        else:
            inv = 1.0 / d
            t1 = (lo - o) * inv
            t2 = (hi - o) * inv
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return -1.0
    return t_min


def can_hit(attacker: PlayerState, target: PlayerState) -> bool:
    """Le raycast de visée de l'attaquant touche-t-il la cible à <= 3.0 blocs ?"""
    ex = attacker.x
    ey = attacker.y + C.PLAYER_EYE_HEIGHT
    ez = attacker.z
    lx, ly, lz = look_vector(attacker.yaw, attacker.pitch)
    b = C.COLLISION_BORDER
    hw = C.PLAYER_HALF_WIDTH
    dist = ray_intersects_aabb(
        ex, ey, ez, lx, ly, lz,
        target.x - hw - b, target.y - b, target.z - hw - b,
        target.x + hw + b, target.y + C.PLAYER_HEIGHT + b, target.z + hw + b,
        C.ATTACK_REACH,
    )
    return dist >= 0.0


@dataclass
class AttackResult:
    landed: bool = False           # hit comptabilisé
    swung: bool = False            # clic consommé (même raté)


def knock_back(target: PlayerState, ratio_x: float, ratio_z: float,
               kb_h: float = 1.0, kb_v: float = 1.0) -> None:
    """EntityLivingBase.knockBack(attacker, 0.4, dx, dz), avec multiplicateurs
    custom (plugins serveur). kb_h/kb_v = 1.0 reproduit vanilla exactement.

    1.8.9 : motionX/Y/Z /= 2 puis le boost (+0.4 vertical, cap 0.4) sont
    INCONDITIONNELS — la garde onGround n'apparaît qu'en 1.9. C'est ce qui
    permet le juggle aérien des combos."""
    f = math.sqrt(ratio_x * ratio_x + ratio_z * ratio_z)
    if f < 1.0e-4:
        return  # vanilla randomise un epsilon ; cas quasi impossible à reach > 0
    target.vx /= 2.0
    target.vy /= 2.0
    target.vz /= 2.0
    target.vx -= ratio_x / f * C.KNOCKBACK_STRENGTH * kb_h
    target.vy += C.KNOCKBACK_STRENGTH * kb_v
    target.vz -= ratio_z / f * C.KNOCKBACK_STRENGTH * kb_h
    if target.vy > C.KNOCKBACK_Y_CAP * kb_v:
        target.vy = C.KNOCKBACK_Y_CAP * kb_v


def apply_entity_collision(a: PlayerState, b: PlayerState) -> None:
    """Entity.applyEntityCollision — les joueurs qui se chevauchent se poussent.

    Appelé pour chaque paire ordonnée (vanilla : chaque entité pousse lors de
    son collideWithNearbyEntities). Quirk vanilla conservé : d2 = sqrt(max(|dx|,|dz|)).
    Déclenché si l'AABB de `a` étendue de (0.2, 0, 0.2) intersecte celle de `b`."""
    hw = C.PLAYER_HALF_WIDTH
    if (abs(b.x - a.x) >= 2.0 * hw + 0.2 or abs(b.z - a.z) >= 2.0 * hw + 0.2
            or b.y >= a.y + C.PLAYER_HEIGHT or b.y + C.PLAYER_HEIGHT <= a.y):
        return
    d0 = b.x - a.x
    d1 = b.z - a.z
    d2 = max(abs(d0), abs(d1))
    if d2 >= 0.01:
        d2 = math.sqrt(d2)
        d0 /= d2
        d1 /= d2
        d3 = 1.0 / d2
        if d3 > 1.0:
            d3 = 1.0
        d0 *= d3 * 0.05
        d1 *= d3 * 0.05
        a.vx -= d0
        a.vz -= d1
        b.vx += d0
        b.vz += d1


def try_attack(attacker: PlayerState, target: PlayerState,
               click_cooldown_ticks: int,
               kb_h: float = 1.0, kb_v: float = 1.0,
               kb_idle: float = 1.0, target_idle: bool = False) -> AttackResult:
    """Résout un clic d'attaque. Mutile attacker/target comme vanilla.

    kb_h/kb_v : multiplicateurs de knockback custom (1.0 = vanilla).
    kb_idle   : multiplicateur additionnel si la victime est immobile
                (aucun input de déplacement).

    En boxing les dégâts sont égaux entre hits -> un hit pendant
    hurtResistantTime > 10 est intégralement ignoré (attackEntityFrom
    retourne false si amount <= lastDamage)."""
    res = AttackResult()
    if attacker.click_cooldown > 0:
        return res

    attacker.click_cooldown = click_cooldown_ticks
    res.swung = True

    if not can_hit(attacker, target):
        return res
    if target.hurt_resistant_time > C.HURT_REHIT_THRESHOLD:
        return res

    res.landed = True
    target.hurt_resistant_time = C.MAX_HURT_RESISTANT_TIME
    attacker.hits += 1

    eff_h = kb_h * (kb_idle if target_idle else 1.0)
    eff_v = kb_v * (kb_idle if target_idle else 1.0)

    # Knockback de base : direction attaquant -> cible
    knock_back(target, attacker.x - target.x, attacker.z - target.z, eff_h, eff_v)

    # Bonus sprint (i = 1) + sprint reset de l'attaquant
    if attacker.sprinting:
        yaw_rad = attacker.yaw * C.DEG_TO_RAD
        target.vx += -math.sin(yaw_rad) * C.SPRINT_KB_H * eff_h
        target.vy += C.SPRINT_KB_Y * eff_v
        target.vz += math.cos(yaw_rad) * C.SPRINT_KB_H * eff_h
        attacker.vx *= C.ATTACKER_SLOWDOWN
        attacker.vz *= C.ATTACKER_SLOWDOWN
        attacker.sprinting = False

    return res
