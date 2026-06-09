"""Layout de l'observation — SOURCE DE VÉRITÉ.

Le kernel CUDA (sim/csrc/boxing_kernel.cu, fonction write_obs) reproduit
exactement ce calcul ; tests/test_equivalence.py vérifie l'égalité.
Utilisé aussi par serve/ pour construire l'observation live depuis le mod.

Repère égocentrique : yaw 0 regarde +Z.
  forward = (-sin(yaw), cos(yaw)) ; side (strafe +) = (cos(yaw), sin(yaw))
"""

import math

OBS_DIM = 48

DEG = math.pi / 180.0


def wrap_degrees(a: float) -> float:
    a = math.fmod(a, 360.0)
    if a >= 180.0:
        a -= 360.0
    if a < -180.0:
        a += 360.0
    return a


def build_obs(own, opp, cfg, h, last_action, tick_count: int) -> list:
    """Observation d'un agent.

    own/opp : sim_ref.PlayerState ; cfg : SimConfig ; h : HumanizationConfig ;
    last_action : liste de 7 floats (action appliquée au tick précédent).
    """
    sy = math.sin(own.yaw * DEG)
    cy = math.cos(own.yaw * DEG)

    def ego(wx: float, wz: float) -> tuple:
        return (wx * -sy + wz * cy, wx * cy + wz * sy)  # (along, side)

    dx = opp.x - own.x
    dy = opp.y - own.y
    dz = opp.z - own.z
    along, side = ego(dx, dz)
    dist_h = math.sqrt(dx * dx + dz * dz)
    dist3 = math.sqrt(dx * dx + dy * dy + dz * dz)

    # Erreurs de visée (œil -> centre de la cible)
    eye_dy = (opp.y + 0.9) - (own.y + 1.62)
    yaw_to = math.degrees(math.atan2(-dx, dz)) if dist_h > 1e-9 else own.yaw
    yaw_err = wrap_degrees(yaw_to - own.yaw) * DEG
    pitch_to = math.degrees(-math.atan2(eye_dy, dist_h)) if dist_h > 1e-9 else 0.0
    pitch_err = pitch_to - own.pitch

    ov_along, ov_side = ego(opp.vx, opp.vz)
    sv_along, sv_side = ego(own.vx, own.vz)
    dyaw_rel = wrap_degrees(opp.yaw - own.yaw) * DEG

    o = [0.0] * OBS_DIM
    o[0] = along / 8.0
    o[1] = side / 8.0
    o[2] = dy / 4.0
    o[3] = dist3 / 8.0
    o[4] = ov_along
    o[5] = ov_side
    o[6] = opp.vy
    o[7] = sv_along
    o[8] = sv_side
    o[9] = own.vy
    o[10] = own.pitch / 90.0
    o[11] = math.sin(yaw_err)
    o[12] = math.cos(yaw_err)
    o[13] = pitch_err / 90.0
    o[14] = math.sin(dyaw_rel)
    o[15] = math.cos(dyaw_rel)
    o[16] = opp.pitch / 90.0
    o[17] = 1.0 if own.on_ground else 0.0
    o[18] = 1.0 if opp.on_ground else 0.0
    o[19] = 1.0 if own.sprinting else 0.0
    o[20] = 1.0 if opp.sprinting else 0.0
    o[21] = own.hurt_resistant_time / 20.0
    o[22] = opp.hurt_resistant_time / 20.0
    o[23] = own.click_cooldown / 20.0
    o[24] = own.jump_ticks / 10.0
    o[25] = (cfg.arena_size_x - 0.3 - own.x) / 8.0
    o[26] = (own.x - 0.3) / 8.0
    o[27] = (cfg.arena_size_z - 0.3 - own.z) / 8.0
    o[28] = (own.z - 0.3) / 8.0
    o[29] = sy
    o[30] = cy
    o[31] = own.hits / 100.0
    o[32] = opp.hits / 100.0
    o[33] = (own.hits - opp.hits) / 20.0
    o[34] = (cfg.max_ticks - tick_count) / float(cfg.max_ticks)
    o[35] = h.max_cps / 20.0
    o[36] = h.max_rot_speed / 180.0
    o[37] = h.action_delay / 8.0
    for k in range(7):
        o[38 + k] = float(last_action[k])
    o[45] = dist_h / 8.0
    o[46] = own.y / 4.0
    o[47] = opp.y / 4.0
    return o
