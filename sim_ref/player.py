"""État d'un joueur — miroir exact des champs vanilla utiles au boxing."""

from dataclasses import dataclass


@dataclass
class PlayerState:
    # Position des pieds (double en vanilla)
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    # motionX/Y/Z
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    # Rotations en degrés (float en vanilla)
    yaw: float = 0.0
    pitch: float = 0.0
    # Flags / timers vanilla
    on_ground: bool = True
    sprinting: bool = False
    collided_horizontally: bool = False
    hurt_resistant_time: int = 0     # EntityLivingBase.hurtResistantTime
    jump_ticks: int = 0              # EntityLivingBase.jumpTicks
    # Judas
    click_cooldown: int = 0          # ticks restants avant prochain clic autorisé
    hits: int = 0                    # hits infligés (score boxing)

    def copy(self) -> "PlayerState":
        return PlayerState(**self.__dict__)
