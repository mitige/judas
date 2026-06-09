"""Aides communes aux tests sim_ref."""

import math

from sim_ref import Action, BoxingConfig, BoxingMatch, HumanizationConfig
from sim_ref.constants import PLAYER_EYE_HEIGHT


def free_humanization() -> HumanizationConfig:
    """Aucune limite humaine (tests de physique pure)."""
    return HumanizationConfig(max_cps=20.0, max_rot_speed=360.0, action_delay=0)


def make_match(**cfg_kwargs) -> BoxingMatch:
    cfg = BoxingConfig(
        humanization=(free_humanization(), free_humanization()),
        **cfg_kwargs,
    )
    return BoxingMatch(cfg)


def aim_action(attacker, target, **kwargs) -> Action:
    """Action dont dyaw/dpitch pointent exactement le centre de la cible."""
    dx = target.x - attacker.x
    dz = target.z - attacker.z
    dy = (target.y + 0.9) - (attacker.y + PLAYER_EYE_HEIGHT)
    dist_h = math.sqrt(dx * dx + dz * dz)
    yaw_to = math.degrees(math.atan2(-dx, dz))
    pitch_to = math.degrees(-math.atan2(dy, dist_h))
    dyaw = _wrap_degrees(yaw_to - attacker.yaw)
    dpitch = pitch_to - attacker.pitch
    return Action(dyaw=dyaw, dpitch=dpitch, **kwargs)


def _wrap_degrees(angle: float) -> float:
    angle = math.fmod(angle, 360.0)
    if angle >= 180.0:
        angle -= 360.0
    if angle < -180.0:
        angle += 360.0
    return angle
