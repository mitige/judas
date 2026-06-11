"""Mouvement EntityLivingBase 1.8.9 — port exact de moveEntityWithHeading.

Ordre vanilla d'un tick de mouvement (onLivingUpdate) :
  1. moveStrafing/moveForward *= 0.98
  2. saut si isJumping && onGround && jumpTicks == 0 (puis jumpTicks = 10)
  3. moveEntityWithHeading :
     a. f4 = friction (0.546 au sol, 0.91 en l'air) calculée AVANT le déplacement
     b. accel sol  = moveSpeed * 0.16277136 / f4^3
        accel air  = 0.02 (x1.3 si sprint)
     c. moveFlying(strafe, forward, accel)   -> ajoute à motionX/Z
     d. moveEntity(motion)                   -> collisions
     e. motionY = (motionY - 0.08) * 0.98
     f. motionX/Z *= f4 (friction d'AVANT le déplacement)
"""

import math

from . import constants as C
from .collision import move_entity
from .player import PlayerState


def jump(p: PlayerState) -> None:
    """EntityLivingBase.jump"""
    p.vy = C.JUMP_MOTION_Y
    if p.sprinting:
        yaw_rad = p.yaw * C.DEG_TO_RAD
        p.vx -= math.sin(yaw_rad) * C.SPRINT_JUMP_BOOST
        p.vz += math.cos(yaw_rad) * C.SPRINT_JUMP_BOOST


def move_flying(p: PlayerState, strafe: float, forward: float, friction: float) -> None:
    """Entity.moveFlying — conversion inputs locaux -> accélération monde."""
    f = strafe * strafe + forward * forward
    if f >= 1.0e-4:
        f = math.sqrt(f)
        if f < 1.0:
            f = 1.0
        f = friction / f
        strafe *= f
        forward *= f
        s = math.sin(p.yaw * C.DEG_TO_RAD)
        c = math.cos(p.yaw * C.DEG_TO_RAD)
        p.vx += strafe * c - forward * s
        p.vz += forward * c + strafe * s


def move_entity_with_heading(p: PlayerState, strafe: float, forward: float,
                             arena_x: float, arena_z: float,
                             speed_amplifier: int) -> None:
    """EntityLivingBase.moveEntityWithHeading (cas terrestre)."""
    # f4 et accel utilisent l'état onGround d'AVANT le déplacement
    if p.on_ground:
        f4 = C.GROUND_FRICTION
        accel = C.move_speed(p.sprinting, speed_amplifier) * (C.MAGIC_GROUND / (f4 * f4 * f4))
    else:
        f4 = C.AIR_DRAG_H
        accel = C.AIR_MOVE_FACTOR * (1.0 + C.SPRINT_AIR_BONUS if p.sprinting else 1.0)

    move_flying(p, strafe, forward, accel)

    move_entity(p, p.vx, p.vy, p.vz, arena_x, arena_z)

    p.vy = (p.vy - C.GRAVITY) * C.AIR_DRAG_V
    p.vx *= f4
    p.vz *= f4


def living_update_movement(p: PlayerState, strafe_in: float, forward_in: float,
                           jumping: bool, arena_x: float, arena_z: float,
                           speed_amplifier: int = 1) -> None:
    """Partie mouvement de EntityLivingBase.onLivingUpdate.

    strafe_in / forward_in dans {-1, 0, 1} (inputs clavier).
    """
    strafe = strafe_in * C.INPUT_FACTOR
    forward = forward_in * C.INPUT_FACTOR

    if p.jump_ticks > 0:
        p.jump_ticks -= 1

    # vanilla : les micro-vitesses sont annulées en tête d'onLivingUpdate
    if abs(p.vx) < C.MOTION_ZERO_THRESHOLD:
        p.vx = 0.0
    if abs(p.vy) < C.MOTION_ZERO_THRESHOLD:
        p.vy = 0.0
    if abs(p.vz) < C.MOTION_ZERO_THRESHOLD:
        p.vz = 0.0

    if jumping:
        if p.on_ground and p.jump_ticks == 0:
            jump(p)
            p.jump_ticks = C.JUMP_COOLDOWN_TICKS
    else:
        p.jump_ticks = 0

    move_entity_with_heading(p, strafe, forward, arena_x, arena_z, speed_amplifier)
