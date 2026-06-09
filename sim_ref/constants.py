"""Constantes physiques Minecraft 1.8.9, portées du code décompilé MCP.

Toutes les valeurs sont celles du jeu vanilla. Les références indiquent la
classe/méthode d'origine dans le code décompilé.
"""

import math

TICKS_PER_SECOND = 20

# --- Entité joueur ----------------------------------------------------------
PLAYER_WIDTH = 0.6            # Entity.setSize (EntityPlayer)
PLAYER_HALF_WIDTH = 0.3
PLAYER_HEIGHT = 1.8
PLAYER_EYE_HEIGHT = 1.62      # EntityPlayer.getEyeHeight (debout)

# --- Mouvement (EntityLivingBase.moveEntityWithHeading) ---------------------
INPUT_FACTOR = 0.98           # onLivingUpdate: moveStrafing/moveForward *= 0.98
BLOCK_SLIPPERINESS = 0.6      # Block.slipperiness (blocs normaux)
GROUND_FRICTION = BLOCK_SLIPPERINESS * 0.91          # = 0.546
AIR_DRAG_H = 0.91             # drag horizontal en l'air
MAGIC_GROUND = 0.16277136     # 0.546^3 ; accel sol = speed * MAGIC / f4^3
GRAVITY = 0.08                # motionY -= 0.08
AIR_DRAG_V = 0.98             # motionY *= 0.98
JUMP_MOTION_Y = 0.42          # EntityLivingBase.jump
SPRINT_JUMP_BOOST = 0.2       # boost directionnel au sprint-jump
JUMP_COOLDOWN_TICKS = 10      # jumpTicks (saut maintenu)
AIR_MOVE_FACTOR = 0.02        # jumpMovementFactor
SPRINT_AIR_BONUS = 0.3        # jumpMovementFactor *= 1.3 si sprint (0.02 -> 0.026)

# --- Vitesse (attribut movementSpeed, opérations multiplicatives) -----------
BASE_MOVE_SPEED = 0.10000000149011612   # SharedMonsterAttributes.movementSpeed (joueur)
SPRINT_MODIFIER = 0.3                   # ItemStack op 2 : x1.3
SPEED_POTION_PER_LEVEL = 0.20000000298023224  # Potion.moveSpeed, op 2, x(1 + 0.2*(ampli+1))


def move_speed(sprinting: bool, speed_amplifier: int = 1) -> float:
    """Vitesse au sol effective. speed_amplifier=1 -> Speed II (boxing)."""
    speed = BASE_MOVE_SPEED
    if speed_amplifier >= 0:
        speed *= 1.0 + SPEED_POTION_PER_LEVEL * (speed_amplifier + 1)
    if sprinting:
        speed *= 1.0 + SPRINT_MODIFIER
    return speed


# --- Combat ------------------------------------------------------------------
ATTACK_REACH = 3.0            # EntityRenderer.getMouseOver (survie)
COLLISION_BORDER = 0.1        # Entity.getCollisionBorderSize
KNOCKBACK_STRENGTH = 0.4      # EntityLivingBase.knockBack (f1)
KNOCKBACK_Y_CAP = 0.4
SPRINT_KB_H = 0.5             # EntityPlayer.attackTargetEntityWithCurrentItem (i * 0.6 ... 0.5)
SPRINT_KB_Y = 0.1
ATTACKER_SLOWDOWN = 0.6       # motionX/Z *= 0.6 après KB sprint
MAX_HURT_RESISTANT_TIME = 20  # EntityLivingBase.maxHurtResistantTime
HURT_REHIT_THRESHOLD = 10     # re-hit bloqué tant que hurtResistantTime > max/2

DEG_TO_RAD = math.pi / 180.0
