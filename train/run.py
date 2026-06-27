"""Boucle d'entraÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â®nement Judas : PPO self-play league sur le simulateur.

    python -m train.run --config train/configs/boxing.json [--resume runs/x/ckpt.pt]

Sorties dans runs/<name>/ :
  metrics.jsonl   (1 ligne JSON / itÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©ration ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â consommÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â© par serve/ et l'app)
  tb/             (TensorBoard)
  ckpt_*.pt       (checkpoints complets)
  latest.pt       (lien logique vers le dernier)
"""

import argparse
import json
import random
import shutil
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from sim import OBS_DIM, SimConfig, make_sim

from .buffer import RolloutBuffer
from .league import League
from .model import (
    COMBO_PRESS_REACH,
    COMBO_REHIT_ATTACK_REACH,
    COMBO_REHIT_CLICK_HURT,
    COMBO_REHIT_COAST_REACH,
    COMBO_REHIT_EDGE_BRAKE_REACH,
    COMBO_REHIT_PRESS_HURT,
    COMBO_S_TAP_REACH,
    COMBO_TAP_REACH,
    COUNTER_CLOSE_COUNTER_REACH,
    COUNTER_CLOSE_RECOVERY_CLICK_HURT,
    COUNTER_FAR_TRADE_REACH,
    COUNTER_HIT_REACH,
    COUNTER_HIT_SELECT_CLEAN_HURT,
    COUNTER_HIT_SELECT_CLEAN_MAX_REACH,
    COUNTER_HIT_SELECT_CLEAN_MIN_REACH,
    COUNTER_HIT_SELECT_MIN_REACH,
    COUNTER_HIT_SELECT_MIN_OWN_HURT,
    COUNTER_HIT_SELECT_OPP_COOLDOWN,
    COUNTER_RECOVERY_CLICK_HURT,
    JudasPolicy,
    PolicyConfig,
    to_sim_actions,
)
from .pbt import (DEFAULT_PBT, Member, apply_hypers, elo_delta,
                  exploit_explore, perturb_hypers, slice_envs)
from .ppo import PPO, PPOConfig

DEFAULT_CFG = {
    "name": "boxing",
    "total_iters": 300,
    "n_envs": 4096,
    "rollout_ticks": 128,
    "league_frac": 0.3,
    "league_pad_bot_frac": 0.0,
    "league_spar_bot_frac": 0.0,
    "league_rehit_bot_frac": 0.0,
    "league_pressure_bot_frac": 0.0,
    "league_combo_chase_bot_frac": 0.0,
    "league_counter_bot_frac": 0.0,
    "pool_every": 25,
    "save_every": 25,
    "keep_ckpts": 10,
    "safety_stop_on_regression": False,
    "safety_under_combo_escape": 0.02,
    "safety_back_frac": 0.002,
    "safety_min_strafe_frac": 0.50,
    "safety_min_opener_strafe_frac": 0.75,
    "safety_min_opener_strafe_hold_frac": -1.0,
    "safety_opener_ticks": 20,
    "safety_min_opener_pressure_frac": -1.0,
    "safety_min_combo_tap_frac": -1.0,
    "safety_min_combo_z_tap_frac": -1.0,
    "safety_max_combo_s_tap_frac": -1.0,
    "safety_min_hit_wtap_frac": -1.0,
    "safety_min_chase_hit_wtap_frac": -1.0,
    "safety_rollout_hit_wtap_slack": 0.0,
    "safety_hit_wtap_blocks_promotion": False,
    "safety_min_under_combo_counter_hit_frac": -1.0,
    "safety_under_combo_avoid_frac": -1.0,
    "safety_under_combo_avoid_min_combo12": -1.0,
    "safety_under_combo_avoid_min_hit_rate": -1.0,
    "score_under_combo_avoid_target": -1.0,
    "score_under_combo_avoid_weight": 0.0,
    "score_under_combo_avoid_cap": 0.0,
    "safety_min_under_combo_hit_select_clean_frac": -1.0,
    "safety_max_under_combo_hit_select_trade_frac": -1.0,
    "safety_strafe_frac": 1.0,
    "safety_sky_frac": 0.08,
    "safety_min_hit_rate": 60.0,
    "safety_fresh_min_hit_rate": -1.0,
    "safety_promote_min_combo_max": -1.0,
    "safety_restore_on_low_combo": False,
    "eval_every": 25,            # matchs d'ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©val auto vs anciens snapshots
    "eval_envs": 128,
    "eval_target_hits": 15,      # 15 hits / 900 ticks : mÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Âªmes signaux, taxe
    "eval_max_ticks": 900,
    "combo_eval_every": 0,
    "combo_eval_envs": 128,
    "combo_eval_ticks": 1200,       # d'ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©val ~9% au lieu de ~15% du temps total
    "combo_eval_chase": True,
    "combo_eval_chase_envs": 64,
    "combo_eval_chase_ticks": 900,
    "combo_eval_spar": True,
    "combo_eval_spar_envs": 64,
    "combo_eval_spar_ticks": 900,
    "combo_eval_rehit": True,
    "combo_eval_rehit_envs": 64,
    "combo_eval_rehit_ticks": 900,
    "combo_eval_pressure": True,
    "combo_eval_pressure_envs": 64,
    "combo_eval_pressure_ticks": 900,
    "combo_eval_counter": False,
    "combo_eval_counter_envs": 64,
    "combo_eval_counter_ticks": 900,
    "safety_require_chase_combo": False,
    "safety_require_counter_recovery": False,
    "shaping_hit_rate": 5.0,     # hits/min dÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©clenchant la rampe (shaping + spawn)
    "shaping_decay_iters": 100,
    "shaping_floor_frac": 0.0,   # plancher du shaping distance en fin de rampe
                                 # (0 = s'ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©teint ; 0.25 = pression permanente)
    "curriculum_gap": 2.0,       # spawn proche au dÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©but (0 = dÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©sactivÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©)
    "snapshot_gate": True,       # snapshot league seulement s'il bat le prÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©cÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©dent
    "league_bot_frac": 0.25,     # part d'envs vs chase-bot (annealÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©e -> 0.05)
    "cuda_graphs": True,         # capture le forward de rollout (fallback eager)
    "seed": 0,
    "sim": {"target_hits": 50},
    "policy": {"history": 8, "d_model": 96, "n_heads": 4, "n_layers": 2},
    "ppo": {},
    "pbt": {},                   # voir train/pbt.py::DEFAULT_PBT
}



BEHAVIOR_REWARD_KEYS = (
    "reward_aim",
    "reward_bad_pitch",
    "reward_chase",
    "reward_turn_aim",
    "reward_aggression",
    "reward_no_escape",
    "reward_combo_focus",
    "reward_combo_tap",
    "reward_opener_strafe",
    "reward_hit_wtap",
    "reward_counter_hit",
    "reward_hit_select",
    "reward_chase_rechain",
    "reward_chase_hit_select",
    "reward_chase_close_counter",
    "reward_chase_counter",
    "reward_spar_counter",
)


def _behavior_reward_bonus(obs: torch.Tensor, actions: torch.Tensor,
                           cfg: dict[str, float],
                           combo_lengths: torch.Tensor | None = None) -> torch.Tensor:
    """Trainer-side dense shaping for body aim, sane pitch, and pressure.

    This is intentionally action-aware. The bad policy we are fighting can sit
    near the opponent while choosing tiny turn deltas, lateral strafe, backsteps,
    and intermittent attacks; state-only shaping does not assign that blame to
    the current action clearly enough for PPO.
    """
    coeff_aim = float(cfg.get("reward_aim", 0.0))
    coeff_bad_pitch = float(cfg.get("reward_bad_pitch", 0.0))
    coeff_chase = float(cfg.get("reward_chase", 0.0))
    coeff_turn = float(cfg.get("reward_turn_aim", 0.0))
    coeff_aggression = float(cfg.get("reward_aggression", 0.0))
    coeff_no_escape = float(cfg.get("reward_no_escape", 0.0))
    coeff_combo_focus = float(cfg.get("reward_combo_focus", 0.0))
    coeff_combo_tap = float(cfg.get("reward_combo_tap", 0.0))
    coeff_counter_hit = float(cfg.get("reward_counter_hit", 0.0))
    coeff_hit_select = float(cfg.get("reward_hit_select", 0.0))
    if (coeff_aim == 0.0 and coeff_bad_pitch == 0.0
            and coeff_chase == 0.0 and coeff_turn == 0.0
            and coeff_aggression == 0.0 and coeff_no_escape == 0.0
            and coeff_combo_focus == 0.0 and coeff_combo_tap == 0.0
            and coeff_counter_hit == 0.0 and coeff_hit_select == 0.0):
        return torch.zeros(obs.shape[0], dtype=obs.dtype, device=obs.device)

    bonus = torch.zeros(obs.shape[0], dtype=torch.float32, device=obs.device)
    actions_f = actions.float()
    yaw_err_deg = torch.rad2deg(torch.atan2(obs[:, 11], obs[:, 12]))
    pitch_err_deg = obs[:, 13] * 90.0
    yaw_err_abs = yaw_err_deg.abs()
    pitch_err_abs = pitch_err_deg.abs()
    yaw_score = (1.0 - yaw_err_abs / 90.0).clamp(0.0, 1.0)
    pitch_score = (1.0 - pitch_err_abs / 45.0).clamp(0.0, 1.0)
    aim_score = yaw_score * pitch_score
    dist = (obs[:, 45] * 8.0).clamp_min(1.0e-6)
    sweet_range = (1.0 - (dist - 3.05).abs() / 0.95).clamp(0.0, 1.0)
    too_close = ((2.55 - dist) / 0.95).clamp(0.0, 1.0)
    approach = ((dist - 2.75) / 2.0).clamp(0.0, 1.0)
    near = ((4.8 - dist) / 2.6).clamp(0.0, 1.0)
    far = ((dist - 3.0) / 5.0).clamp(0.0, 1.0)
    counter_close = (1.0 - (dist - 2.85).abs() / 0.65).clamp(0.0, 1.0)
    counter_too_far = ((dist - COUNTER_FAR_TRADE_REACH) / 0.45).clamp(0.0, 1.0)
    fwd_dir = actions_f[:, 2].clamp(-1.0, 1.0)
    strafe_dir = actions_f[:, 3].clamp(-1.0, 1.0)
    strafe_abs = strafe_dir.abs()
    attack = actions_f[:, 6].clamp(0.0, 1.0)
    sprint = actions_f[:, 5].clamp(0.0, 1.0)
    own_hurt = obs[:, 21].clamp(0.0, 1.0)
    opp_hurt = obs[:, 22].clamp(0.0, 1.0)
    own_cooldown = obs[:, 23].clamp(0.0, 1.0)
    opp_cooldown = obs[:, 37].clamp(0.0, 1.0)
    combo_adv = (opp_hurt - own_hurt).clamp(0.0, 1.0)
    combo_disadv = (own_hurt - opp_hurt).clamp(0.0, 1.0)
    combo_disadv_active = (combo_disadv > 0.05).float()
    rehit_press_ready = (opp_hurt <= COMBO_REHIT_PRESS_HURT).float()
    rehit_click_ready = (opp_hurt <= COMBO_REHIT_CLICK_HURT).float()
    click_ready = (own_cooldown <= 0.05).float()
    counter_click_ready = click_ready * rehit_click_ready
    hit_select_hurt_ready = (
        (own_hurt >= COUNTER_HIT_SELECT_MIN_OWN_HURT)
        & (own_hurt <= COUNTER_HIT_SELECT_CLEAN_HURT)
        & (opp_cooldown >= COUNTER_HIT_SELECT_OPP_COOLDOWN)
    ).float()
    close_counter_ready = counter_click_ready * (
        own_hurt <= COUNTER_CLOSE_RECOVERY_CLICK_HURT).float()
    clean_hit_select_range = (
        (dist >= COUNTER_HIT_SELECT_CLEAN_MIN_REACH)
        & (dist <= COUNTER_HIT_SELECT_CLEAN_MAX_REACH)
    ).float()
    hit_select_counter_ready = (
        counter_click_ready * clean_hit_select_range * hit_select_hurt_ready)
    reach_recovery_range = (
        (dist >= COUNTER_HIT_SELECT_MIN_REACH)
        & (dist <= COUNTER_HIT_REACH)
    ).float()
    reach_recovery_ready = (
        counter_click_ready
        * reach_recovery_range
        * (own_hurt <= COUNTER_RECOVERY_CLICK_HURT).float()
    )
    opp_facing = ((1.0 - obs[:, 15]) * 0.5).clamp(0.0, 1.0)

    if coeff_aim != 0.0:
        bonus += coeff_aim * (2.0 * aim_score - 1.0)
    if coeff_bad_pitch != 0.0:
        bad_pitch = ((obs[:, 10] * 90.0).abs() - 18.0).div(35.0).clamp(0.0, 1.0)
        bonus -= coeff_bad_pitch * bad_pitch
    if coeff_chase != 0.0:
        along = obs[:, 0] * 8.0
        front = (along / dist).clamp(-1.0, 1.0)
        chase = torch.where(
            fwd_dir > 0.5,
            far * front.clamp_min(0.0),
            torch.where(fwd_dir < -0.5, -far, -0.25 * far),
        )
        bonus += coeff_chase * chase
    if coeff_turn != 0.0:
        yaw_need = (yaw_err_abs / 45.0).clamp(0.0, 1.0)
        pitch_need = (pitch_err_abs / 45.0).clamp(0.0, 1.0)
        yaw_align = torch.sign(yaw_err_deg) * actions_f[:, 0].clamp(-1.0, 1.0)
        pitch_align = torch.sign(pitch_err_deg) * actions_f[:, 1].clamp(-1.0, 1.0)
        turn_score = 0.5 * (yaw_need * yaw_align + pitch_need * pitch_align)
        bonus += coeff_turn * turn_score.clamp(-1.0, 1.0)
    if coeff_aggression != 0.0:
        hit_band = (1.0 - (dist - 2.80).abs() / 0.75).clamp(0.0, 1.0)
        should_attack = hit_band * (0.45 + 0.55 * aim_score)
        attack_score = should_attack * (2.0 * attack - 1.0)
        fwd_pos = fwd_dir.clamp_min(0.0)
        sprint_pressure = (approach + 0.65 * combo_adv).clamp(0.0, 1.0) * fwd_pos * sprint
        hold_range = sweet_range * aim_score
        overrun_risk = too_close * opp_facing * (0.65 + 0.35 * combo_adv) * (0.50 + 0.50 * fwd_pos)
        close_trade = too_close * opp_facing * attack
        bonus += coeff_aggression * (
            attack_score + 0.45 * sprint_pressure + 0.65 * hold_range
            - 0.85 * overrun_risk - 0.20 * close_trade
        )
    if coeff_no_escape != 0.0:
        fwd_pos = fwd_dir.clamp_min(0.0)
        back = (-fwd_dir).clamp_min(0.0)
        idle = (1.0 - fwd_dir.abs()).clamp(0.0, 1.0)
        strafe_abs = strafe_dir.abs()
        circle = strafe_abs * (near + 0.50 * sweet_range).clamp(0.0, 1.0)
        chain_pressure = (sweet_range + 0.75 * approach + 0.85 * combo_adv).clamp(0.0, 1.0)
        forward_pressure = fwd_pos * (0.75 * approach + 0.35 * combo_adv)
        sprint_forward = fwd_pos * sprint * chain_pressure
        tap_control = too_close * combo_adv
        hold_range = sweet_range * aim_score
        overrun_risk = too_close * fwd_pos * opp_facing * (0.75 + 0.25 * combo_adv)
        back_escape = back * (0.85 + far + 0.55 * combo_adv + 0.60 * combo_disadv) * (1.0 - 0.75 * tap_control)
        line_drive = fwd_pos * (1.0 - strafe_abs).clamp(0.0, 1.0) * chain_pressure
        attack_hold = attack * aim_score * (sweet_range + 0.45 * combo_adv).clamp(0.0, 1.0)
        under_combo_band = combo_disadv * (near + 0.35 * sweet_range).clamp(0.0, 1.0)
        under_combo_hold = (under_combo_band * (1.0 - attack)
                            * (1.0 - strafe_abs).clamp(0.0, 1.0)
                            * (idle + 0.35 * fwd_pos).clamp(0.0, 1.0))
        counter_spacing_ready = (
            ((dist < COUNTER_CLOSE_COUNTER_REACH).float() * close_counter_ready)
            + (clean_hit_select_range
               * hit_select_counter_ready)
        ).clamp(0.0, 1.0)
        under_combo_counter = (combo_disadv * counter_close * attack
                               * counter_spacing_ready
                               * (0.35 + 0.65 * aim_score)
                               * (1.0 - strafe_abs).clamp(0.0, 1.0)
                               * (0.35 + 0.65 * (idle + fwd_pos).clamp(0.0, 1.0)))
        under_combo_early_counter = (combo_disadv * counter_close * attack
                                     * (1.0 - counter_spacing_ready)
                                     * (0.35 + 0.65 * aim_score)
                                     * (1.0 - strafe_abs).clamp(0.0, 1.0))
        under_combo_far_trade = (combo_disadv * counter_too_far * attack
                                 * (0.35 + 0.65 * aim_score)
                                 * (1.0 - strafe_abs).clamp(0.0, 1.0))
        panic_escape = (combo_disadv * (1.35 * back + 1.50 * strafe_abs + 0.20 * idle)
                        * (near + 0.35 * sweet_range).clamp(0.0, 1.0))
        circle_escape = circle * (0.65 + 0.35 * (1.0 - attack) + 0.25 * far) * (1.0 - 0.25 * combo_adv)
        far_not_pressing = far * (1.0 - fwd_pos)
        close_trade = too_close * opp_facing * attack
        bonus += coeff_no_escape * (
            0.85 * forward_pressure
            + 0.35 * sprint_forward
            + 0.45 * hold_range
            + 0.55 * line_drive
            + 0.35 * attack_hold
            + 0.15 * under_combo_hold
            + 1.35 * under_combo_counter
            + 0.25 * tap_control * idle
            - 1.15 * under_combo_early_counter
            - 2.25 * back_escape
            - 4.00 * panic_escape
            - 3.25 * under_combo_far_trade
            - 0.50 * idle * (far + 0.50 * approach)
            - 2.25 * circle_escape
            - 1.10 * overrun_risk
            - 0.20 * close_trade
            - 0.75 * far_not_pressing
        )
    if coeff_combo_focus != 0.0 and combo_lengths is not None:
        chain = (combo_lengths.float() / 12.0).clamp(0.0, 1.0)
        sky = ((obs[:, 10] * 90.0).abs() - 24.0).div(36.0).clamp(0.0, 1.0)
        pitch_miss = (pitch_err_abs / 45.0).clamp(0.0, 1.0)
        fwd_pos = fwd_dir.clamp_min(0.0)
        strafe_abs = strafe_dir.abs()
        line_drive = fwd_pos * (1.0 - strafe_abs).clamp(0.0, 1.0)
        sprint_line = line_drive * sprint
        attack_body = attack * aim_score
        combo_control = (
            1.25 * aim_score
            + 0.45 * sprint_line
            + 0.35 * attack_body
            - 1.65 * sky
            - 0.65 * pitch_miss
            - 0.50 * strafe_abs
            - 0.25 * (1.0 - fwd_pos)
        )
        bonus += coeff_combo_focus * chain * combo_control
    if coeff_combo_tap != 0.0:
        close_combo = combo_adv * (too_close + 0.75 * sweet_range).clamp(0.0, 1.0)
        wait_rehit = close_combo * (1.0 - rehit_press_ready)
        no_sprint = (1.0 - sprint).clamp(0.0, 1.0)
        s_tap = (-fwd_dir).clamp_min(0.0)
        z_tap = (1.0 - fwd_dir.abs()).clamp(0.0, 1.0)
        last_fwd = obs[:, 40].clamp(-1.0, 1.0)
        last_sprint = obs[:, 43].clamp(0.0, 1.0)
        last_pressed = (last_fwd.clamp_min(0.0) * last_sprint).clamp(0.0, 1.0)
        last_tapped = ((1.0 - last_fwd.clamp_min(0.0)) * (1.0 - last_sprint)).clamp(0.0, 1.0)
        line_control = (1.0 - strafe_abs).clamp(0.0, 1.0)
        tap_reset = close_combo * last_pressed * aim_score * no_sprint * line_control
        close_wait_tap = tap_reset * (1.0 - rehit_press_ready)
        repress = (close_combo * rehit_press_ready * last_tapped
                   * fwd_dir.clamp_min(0.0) * sprint * line_control)
        reset_hold = close_combo * last_tapped * too_close * z_tap * no_sprint * line_control
        wait_release = wait_rehit * no_sprint * line_control * (z_tap + 0.15 * s_tap).clamp(0.0, 1.0)
        early_click = wait_rehit * attack
        ready_attack = (close_combo * rehit_click_ready * attack * aim_score
                        * fwd_dir.clamp_min(0.0) * sprint * line_control)
        repeat_tap = (close_combo * rehit_press_ready * last_tapped
                      * no_sprint * line_control * (1.0 - too_close))
        hold_sprint_close = close_combo * last_pressed * fwd_dir.clamp_min(0.0) * sprint * opp_facing
        bonus += coeff_combo_tap * (
            1.55 * tap_reset * z_tap
            + 1.20 * close_wait_tap * z_tap
            - 0.85 * tap_reset * s_tap
            - 0.70 * close_wait_tap * s_tap
            + 0.40 * reset_hold
            + 0.70 * wait_release
            + 2.20 * repress
            + 1.80 * ready_attack
            - 0.45 * hold_sprint_close
            - 1.15 * early_click
            - 0.65 * repeat_tap
        )
    if coeff_counter_hit != 0.0:
        counter_band = combo_disadv * counter_close
        line_control = (1.0 - strafe_abs).clamp(0.0, 1.0)
        no_back = (1.0 - (-fwd_dir).clamp_min(0.0)).clamp(0.0, 1.0)
        idle = (1.0 - fwd_dir.abs()).clamp(0.0, 1.0)
        forward_or_hold = (0.35 + 0.65 * (fwd_dir.clamp_min(0.0) + idle)).clamp(0.0, 1.0)
        active_control = (0.55 + 0.45 * line_control) * no_back * forward_or_hold
        ready_counter_band = counter_band * counter_click_ready
        counter = ready_counter_band * attack * aim_score * active_control
        close_counter_spacing = (1.0 - (dist - 1.85).abs() / 0.45).clamp(0.0, 1.0)
        close_counter_control = (0.35 + 0.65 * idle) * (1.0 - sprint).clamp(0.0, 1.0) * line_control * no_back
        timed_close_counter = (
            combo_disadv
            * close_counter_spacing
            * close_counter_ready
            * attack
            * aim_score
            * close_counter_control
        )
        timed_select_counter = (
            combo_disadv
            * clean_hit_select_range
            * hit_select_counter_ready
            * attack
            * aim_score
            * active_control
        )
        timed_reach_counter = (
            combo_disadv
            * reach_recovery_ready
            * attack
            * aim_score
            * active_control
        )
        missed_timed_counter = (
            combo_disadv
            * (close_counter_spacing * close_counter_ready
               + clean_hit_select_range * hit_select_counter_ready
               + reach_recovery_ready).clamp(0.0, 1.0)
            * (1.0 - attack)
            * (0.35 + 0.65 * aim_score)
        )
        freeze = ready_counter_band * (1.0 - attack) * (0.35 + 0.65 * (1.0 - fwd_dir.abs()).clamp(0.0, 1.0))
        panic_window = (counter_band + close_counter_spacing * close_counter_ready).clamp(0.0, 1.0)
        panic = panic_window * ((-fwd_dir).clamp_min(0.0) + 0.85 * strafe_abs)
        cooldown_spam = counter_band * attack * (1.0 - counter_click_ready) * (0.35 + 0.65 * aim_score) * active_control
        far_trade = combo_disadv * counter_too_far * attack * aim_score * active_control
        bonus += coeff_counter_hit * (
            3.50 * counter
            + 2.80 * timed_close_counter
            + 1.75 * timed_select_counter
            + 2.70 * timed_reach_counter
            - 2.40 * missed_timed_counter
            - 1.55 * freeze
            - 1.70 * panic
            - 1.95 * cooldown_spam
            - 3.85 * far_trade
        )
    if coeff_hit_select != 0.0:
        line_control = (1.0 - strafe_abs).clamp(0.0, 1.0)
        no_back = (1.0 - (-fwd_dir).clamp_min(0.0)).clamp(0.0, 1.0)
        idle = (1.0 - fwd_dir.abs()).clamp(0.0, 1.0)
        sprint_off = (1.0 - sprint).clamp(0.0, 1.0)
        release_control = idle * sprint_off * line_control * no_back
        active_control = (0.55 + 0.45 * line_control) * no_back
        fwd_pos = fwd_dir.clamp_min(0.0)
        back = (-fwd_dir).clamp_min(0.0)
        select_window = combo_disadv_active * clean_hit_select_range
        close_jam = combo_disadv_active * ((2.25 - dist) / 0.55).clamp(0.0, 1.0)
        far_spam = combo_disadv_active * ((dist - COUNTER_HIT_REACH) / 0.55).clamp(0.0, 1.0)
        recovery_ready = (
            counter_click_ready * clean_hit_select_range * hit_select_hurt_ready)
        ready = select_window * recovery_ready * (0.35 + 0.65 * aim_score)
        timed_click = ready * attack * release_control
        missed_click = ready * (1.0 - attack)
        bad_aim_spam = combo_disadv_active * attack * (1.0 - aim_score).clamp(0.0, 1.0)
        cooldown_spam = select_window * attack * (1.0 - recovery_ready) * (
            0.35 + 0.65 * aim_score) * active_control
        range_spam = attack * (far_spam + 0.75 * close_jam)
        reenter = combo_disadv_active * fwd_pos * sprint * active_control * (
            (dist - 3.05) / 1.20).clamp(0.0, 1.0)
        clean_release = ready * release_control
        overrun_select = ready * fwd_pos * sprint * (0.35 + 0.65 * aim_score)
        panic_move = combo_disadv_active * (back + 0.35 * strafe_abs) * (
            near + 0.35 * sweet_range).clamp(0.0, 1.0)
        bonus += coeff_hit_select * (
            1.10 * timed_click
            + 0.95 * clean_release
            + 0.40 * reenter
            - 1.10 * missed_click
            - 2.15 * bad_aim_spam
            - 2.75 * cooldown_spam
            - 2.85 * overrun_select
            - 3.25 * range_spam
            - 2.05 * panic_move
        )
    return bonus.to(dtype=obs.dtype)


def _hit_event_reward_bonus(obs: torch.Tensor, actions: torch.Tensor,
                            dealt: torch.Tensor, taken: torch.Tensor,
                            cfg: dict[str, float]) -> torch.Tensor:
    coeff_combo_tap = float(cfg.get("reward_combo_tap", 0.0))
    coeff_counter_hit = float(cfg.get("reward_counter_hit", 0.0))
    coeff_hit_select = float(cfg.get("reward_hit_select", 0.0))
    coeff_hit_wtap = float(cfg.get("reward_hit_wtap", 0.0))
    if (coeff_combo_tap == 0.0 and coeff_counter_hit == 0.0
            and coeff_hit_select == 0.0 and coeff_hit_wtap == 0.0):
        return torch.zeros(obs.shape[0], dtype=obs.dtype, device=obs.device)

    actions_f = actions.float()
    dist = (obs[:, 45] * 8.0).clamp_min(1.0e-6)
    yaw_err_deg = torch.rad2deg(torch.atan2(obs[:, 11], obs[:, 12])).abs()
    pitch_err_deg = (obs[:, 13] * 90.0).abs()
    aim_score = ((1.0 - yaw_err_deg / 90.0).clamp(0.0, 1.0)
                 * (1.0 - pitch_err_deg / 45.0).clamp(0.0, 1.0))
    strafe_abs = actions_f[:, 3].clamp(-1.0, 1.0).abs()
    line_control = (1.0 - strafe_abs).clamp(0.0, 1.0)
    strafe_control = (strafe_abs - 0.45).div(0.55).clamp(0.0, 1.0)
    posture = (0.35 + 0.65 * aim_score) * line_control
    active_posture = (0.35 + 0.65 * aim_score) * (0.55 + 0.45 * line_control)
    clean_dealt = dealt.bool() & (~taken.bool())
    took_hit = taken.bool()
    combo_adv = obs[:, 22] > obs[:, 21] + 0.05
    combo_disadv = obs[:, 21] > obs[:, 22] + 0.05
    own_hurt = obs[:, 21].clamp(0.0, 1.0)
    rehit_window = combo_adv & (dist <= COMBO_PRESS_REACH + 0.35)
    counter_window = combo_disadv & (dist <= COUNTER_HIT_REACH)
    combo_break_window = combo_adv & (dist <= 3.65)
    under_combo_window = combo_disadv & (dist <= 3.65)

    bonus = torch.zeros(obs.shape[0], dtype=torch.float32, device=obs.device)
    if coeff_combo_tap != 0.0:
        rehit = clean_dealt.float() * rehit_window.float() * posture
        broken = took_hit.float() * combo_break_window.float()
        bonus += coeff_combo_tap * (5.00 * rehit - 6.00 * broken)
    if coeff_counter_hit != 0.0:
        counter = clean_dealt.float() * counter_window.float() * active_posture
        eaten = took_hit.float() * under_combo_window.float()
        bonus += coeff_counter_hit * (3.15 * counter - 1.85 * eaten)
    if coeff_hit_select != 0.0:
        attack = actions_f[:, 6].clamp(0.0, 1.0)
        select_window = (
            combo_disadv
            & (dist >= COUNTER_HIT_SELECT_CLEAN_MIN_REACH)
            & (dist <= COUNTER_HIT_SELECT_CLEAN_MAX_REACH)
            & (own_hurt >= COUNTER_HIT_SELECT_MIN_OWN_HURT)
            & (own_hurt <= COUNTER_HIT_SELECT_CLEAN_HURT)
            & (obs[:, 37].clamp(0.0, 1.0) >= COUNTER_HIT_SELECT_OPP_COOLDOWN)
        )
        far_select_spam = combo_disadv & (dist > COUNTER_HIT_REACH) & (attack > 0.5)
        clean_select = clean_dealt.float() * select_window.float() * active_posture
        stolen_select = took_hit.float() * select_window.float()
        trade_select = (dealt.bool() & taken.bool()).float() * select_window.float()
        bonus += coeff_hit_select * (
            5.45 * clean_select
            - 4.85 * stolen_select
            - 7.40 * trade_select
            - 2.05 * far_select_spam.float()
        )
    if coeff_hit_wtap != 0.0:
        fwd = actions_f[:, 2].clamp(-1.0, 1.0)
        sprint = actions_f[:, 5].clamp(0.0, 1.0)
        z_release = (1.0 - fwd.abs()).clamp(0.0, 1.0) * (1.0 - sprint)
        strafe_control = (strafe_abs - 0.45).div(0.55).clamp(0.0, 1.0)
        wtap_hit = z_release * strafe_control * (0.35 + 0.65 * aim_score)
        missed_wtap = (1.0 - wtap_hit).clamp(0.0, 1.0)
        bonus += coeff_hit_wtap * clean_dealt.float() * (
            2.40 * wtap_hit - 1.10 * missed_wtap)
    return bonus.to(dtype=obs.dtype)


def _post_hit_wtap_reward_bonus(hist: torch.Tensor, actions: torch.Tensor,
                                cfg: dict[str, float]) -> torch.Tensor:
    coeff = float(cfg.get("reward_hit_wtap", 0.0))
    if coeff == 0.0:
        return torch.zeros(hist.shape[0], dtype=hist.dtype, device=hist.device)

    obs = hist[:, -1].float()
    prev = hist[:, -2].float() if hist.shape[1] > 1 else obs
    landed_hit = obs[:, 31] > prev[:, 31] + 0.004
    actions_f = actions.float()
    dist = (obs[:, 45] * 8.0).clamp_min(1.0e-6)
    yaw_err_deg = torch.rad2deg(torch.atan2(obs[:, 11], obs[:, 12])).abs()
    pitch_err_deg = (obs[:, 13] * 90.0).abs()
    aim_score = ((1.0 - yaw_err_deg / 90.0).clamp(0.0, 1.0)
                 * (1.0 - pitch_err_deg / 45.0).clamp(0.0, 1.0))
    fwd = actions_f[:, 2].clamp(-1.0, 1.0)
    strafe_abs = actions_f[:, 3].clamp(-1.0, 1.0).abs()
    sprint = actions_f[:, 5].clamp(0.0, 1.0)
    attack = actions_f[:, 6].clamp(0.0, 1.0)
    z_release = (1.0 - fwd.abs()).clamp(0.0, 1.0) * (1.0 - sprint)
    strafe_control = (strafe_abs - 0.45).div(0.55).clamp(0.0, 1.0)
    wtap = z_release * strafe_control * (0.35 + 0.65 * aim_score)
    reset_window = landed_hit.float() * ((4.15 - dist) / 1.65).clamp(0.0, 1.0)
    hold_forward = fwd.clamp_min(0.0) * sprint
    idle_no_strafe = z_release * (1.0 - strafe_control)
    bad_posture = (0.75 * hold_forward + 0.45 * idle_no_strafe
                   + 0.30 * attack * (1.0 - wtap))
    bonus = coeff * reset_window * (2.20 * wtap - 1.05 * bad_posture)
    return bonus.to(dtype=hist.dtype)


def _opener_boxing_reward_bonus(age: torch.Tensor, obs: torch.Tensor,
                                actions: torch.Tensor, cfg: dict[str, float],
                                opener_ticks: int = 20) -> torch.Tensor:
    coeff = float(cfg.get("reward_opener_strafe", 0.0))
    if coeff == 0.0 or opener_ticks <= 0:
        return torch.zeros(obs.shape[0], dtype=obs.dtype, device=obs.device)

    actions_f = actions.float()
    dist = (obs[:, 45] * 8.0).clamp_min(1.0e-6)
    yaw_err_deg = torch.rad2deg(torch.atan2(obs[:, 11], obs[:, 12])).abs()
    pitch_err_deg = (obs[:, 13] * 90.0).abs()
    aim_score = ((1.0 - yaw_err_deg / 90.0).clamp(0.0, 1.0)
                 * (1.0 - pitch_err_deg / 45.0).clamp(0.0, 1.0))
    fwd = actions_f[:, 2].clamp(-1.0, 1.0)
    strafe_abs = actions_f[:, 3].clamp(-1.0, 1.0).abs()
    jump = actions_f[:, 4].clamp(0.0, 1.0)
    sprint = actions_f[:, 5].clamp(0.0, 1.0)
    attack = actions_f[:, 6].clamp(0.0, 1.0)
    active = ((age >= 0) & (age < opener_ticks)).to(dtype=torch.float32)
    approach_band = ((dist - 1.65) / 3.25).clamp(0.0, 1.0)
    approach_band = approach_band * ((8.25 - dist) / 2.0).clamp(0.0, 1.0)
    fwd_pressure = fwd.clamp_min(0.0) * sprint
    strafe_control = (strafe_abs - 0.45).div(0.55).clamp(0.0, 1.0)
    back = (-fwd).clamp_min(0.0)
    idle = (1.0 - fwd.abs()).clamp(0.0, 1.0)
    straightline = fwd_pressure * (1.0 - strafe_control)
    good_opener = fwd_pressure * strafe_control * (0.45 + 0.55 * aim_score)
    pressure_click = attack * aim_score * ((3.45 - dist).div(1.0).clamp(0.0, 1.0))
    passive_lateral = strafe_control * (1.0 - fwd_pressure)
    bad_opener = (
        1.60 * back
        + 1.10 * straightline
        + 0.60 * idle
        + 1.55 * passive_lateral
        + 0.85 * jump
    )
    bonus = coeff * active * approach_band * (
        2.75 * good_opener + 0.45 * pressure_click - bad_opener)
    return bonus.to(dtype=obs.dtype)


def _chase_transfer_reward_bonus(
    obs: torch.Tensor,
    actions: torch.Tensor,
    dealt: torch.Tensor,
    taken: torch.Tensor,
    rechain_learner_mask: torch.Tensor,
    counter_learner_mask: torch.Tensor,
    spar_counter_learner_mask: torch.Tensor,
    cfg: dict[str, float],
) -> torch.Tensor:
    """Extra pressure for live-like combo continuation and recovery lanes.

    Generic combo shaping is learnable against drills but too weak against the
    active chaser: one stolen hit often pays nearly as much as a clean rechain,
    and spar recovery needs the same timed counter-click signal as chase drills.
    """
    coeff_rechain = float(cfg.get("reward_chase_rechain", 0.0))
    coeff_chase_hit_select = float(cfg.get("reward_chase_hit_select", 0.0))
    coeff_chase_close_counter = float(cfg.get("reward_chase_close_counter", 0.0))
    coeff_counter = float(cfg.get("reward_chase_counter", 0.0))
    coeff_spar_counter = float(cfg.get("reward_spar_counter", coeff_counter))
    if (coeff_rechain == 0.0 and coeff_chase_hit_select == 0.0
            and coeff_chase_close_counter == 0.0 and coeff_counter == 0.0
            and coeff_spar_counter == 0.0):
        return torch.zeros(obs.shape[0], dtype=obs.dtype, device=obs.device)

    rechain_mask = rechain_learner_mask.to(device=obs.device, dtype=torch.float32)
    counter_mask = counter_learner_mask.to(device=obs.device, dtype=torch.float32)
    spar_counter_mask = spar_counter_learner_mask.to(device=obs.device, dtype=torch.float32)
    any_mask = rechain_mask.bool() | counter_mask.bool() | spar_counter_mask.bool()
    if not bool(any_mask.any().item()):
        return torch.zeros(obs.shape[0], dtype=obs.dtype, device=obs.device)

    actions_f = actions.float()
    dist = (obs[:, 45] * 8.0).clamp_min(1.0e-6)
    yaw_err_deg = torch.rad2deg(torch.atan2(obs[:, 11], obs[:, 12])).abs()
    pitch_err_deg = (obs[:, 13] * 90.0).abs()
    aim_score = ((1.0 - yaw_err_deg / 90.0).clamp(0.0, 1.0)
                 * (1.0 - pitch_err_deg / 45.0).clamp(0.0, 1.0))
    strafe_abs = actions_f[:, 3].clamp(-1.0, 1.0).abs()
    strafe_control = (strafe_abs - 0.45).div(0.55).clamp(0.0, 1.0)
    active_control = 0.65 + 0.35 * strafe_control
    posture = (0.35 + 0.65 * aim_score) * active_control
    attack = actions_f[:, 6].clamp(0.0, 1.0)
    fwd = actions_f[:, 2].clamp(-1.0, 1.0)
    sprint = actions_f[:, 5].clamp(0.0, 1.0)
    jump = actions_f[:, 4].clamp(0.0, 1.0)
    fwd_pos = fwd.clamp_min(0.0)
    back = (-fwd).clamp_min(0.0)
    line_control = (1.0 - strafe_abs).clamp(0.0, 1.0)
    counter_posture = (0.35 + 0.65 * aim_score) * (0.55 + 0.45 * line_control) * (
        1.0 - back).clamp(0.0, 1.0)
    z_release = (1.0 - fwd.abs()).clamp(0.0, 1.0) * (1.0 - sprint)
    own_hurt = obs[:, 21].clamp(0.0, 1.0)
    opp_hurt = obs[:, 22].clamp(0.0, 1.0)
    own_cooldown = obs[:, 23].clamp(0.0, 1.0)
    opp_cooldown = obs[:, 37].clamp(0.0, 1.0)
    combo_adv = opp_hurt > own_hurt + 0.05
    combo_disadv = own_hurt > opp_hurt + 0.05
    clean_dealt = dealt.bool() & (~taken.bool())
    traded = dealt.bool() & taken.bool()
    stolen = taken.bool() & (~dealt.bool())

    combo_window = combo_adv & (dist <= 3.65)
    ready_rehit = combo_adv & (opp_hurt <= COMBO_REHIT_CLICK_HURT) & (dist <= COMBO_REHIT_ATTACK_REACH)
    edge_brake = (
        combo_adv
        & (opp_hurt <= COMBO_REHIT_PRESS_HURT)
        & (dist > COMBO_REHIT_ATTACK_REACH)
        & (dist <= COMBO_REHIT_EDGE_BRAKE_REACH)
    )
    drive_rehit = (
        combo_adv
        & (dist > COMBO_REHIT_EDGE_BRAKE_REACH)
        & (dist <= COMBO_REHIT_COAST_REACH)
    )
    press_rehit = (
        combo_adv
        & (opp_hurt <= COMBO_REHIT_PRESS_HURT)
        & (dist > COMBO_REHIT_EDGE_BRAKE_REACH)
        & (dist <= COMBO_REHIT_COAST_REACH)
    )
    wait_rehit = (
        combo_adv
        & (opp_hurt > COMBO_REHIT_PRESS_HURT)
        & (dist <= COMBO_S_TAP_REACH)
    )
    stale_release = (
        combo_adv
        & (opp_hurt <= COMBO_REHIT_CLICK_HURT)
        & (dist > 1.75)
        & (dist <= COMBO_REHIT_ATTACK_REACH)
        & (attack <= 0.5)
        & (z_release > 0.5)
    )
    under_window = combo_disadv & (dist <= COUNTER_HIT_REACH)
    under_close = combo_disadv & (dist <= 3.65)
    hit_select_window = (
        under_window
        & (dist >= COUNTER_HIT_SELECT_CLEAN_MIN_REACH)
        & (dist <= COUNTER_HIT_SELECT_CLEAN_MAX_REACH)
        & (own_hurt >= COUNTER_HIT_SELECT_MIN_OWN_HURT)
        & (own_hurt <= COUNTER_HIT_SELECT_CLEAN_HURT)
        & (opp_cooldown >= COUNTER_HIT_SELECT_OPP_COOLDOWN)
    )
    close_counter_window = (
        under_window
        & (dist < COUNTER_CLOSE_COUNTER_REACH)
        & (own_hurt <= COUNTER_CLOSE_RECOVERY_CLICK_HURT)
    )
    reach_counter_window = (
        under_window
        & (dist >= COUNTER_HIT_SELECT_MIN_REACH)
        & (dist <= COUNTER_HIT_REACH)
        & (own_hurt <= COUNTER_RECOVERY_CLICK_HURT)
    )
    counter_click_ready = (
        under_window
        & (own_cooldown <= 0.05)
        & (opp_hurt <= COMBO_REHIT_CLICK_HURT)
    )
    counter_timed_ready = counter_click_ready & (
        hit_select_window | close_counter_window | reach_counter_window)
    hit_select_ready = counter_click_ready & hit_select_window
    close_counter_ready = counter_click_ready & close_counter_window
    counter_drive_window = combo_disadv & (dist > COUNTER_HIT_REACH) & (dist <= COUNTER_FAR_TRADE_REACH)
    far_under_click = combo_disadv & (dist > COUNTER_FAR_TRADE_REACH) & (attack > 0.5)

    rehit_clean = clean_dealt.float() * combo_window.float() * posture
    rehit_stolen = stolen.float() * combo_window.float()
    rehit_trade = traded.float() * combo_window.float()
    rehit_action = ready_rehit.float() * attack * posture
    rehit_z_select = ready_rehit.float() * attack * z_release * posture
    rehit_drive = (drive_rehit | press_rehit).float() * fwd_pos * sprint * posture
    rehit_edge_release = edge_brake.float() * z_release * (
        0.35 + 0.65 * aim_score) * (0.75 + 0.25 * strafe_control)
    rehit_edge_overrun = edge_brake.float() * fwd_pos * sprint
    rehit_edge_prefire = edge_brake.float() * attack
    rehit_clean_select = rehit_clean * z_release
    rehit_wait_release = wait_rehit.float() * z_release * (0.65 + 0.35 * strafe_control) * (
        0.35 + 0.65 * aim_score)
    rehit_early_click = (
        combo_adv
        & (opp_hurt > COMBO_REHIT_CLICK_HURT)
        & (dist <= COMBO_REHIT_ATTACK_REACH)
        & (attack > 0.5)
    ).float()
    rehit_far_click = (
        combo_adv
        & (dist > COMBO_REHIT_ATTACK_REACH)
        & (attack > 0.5)
    ).float()
    rehit_bad_move = combo_window.float() * (1.65 * back + 0.55 * jump)
    rehit_idle_far = drive_rehit.float() * (1.0 - fwd_pos * sprint)
    stale_ready_release = stale_release.float() * (0.35 + 0.65 * aim_score)
    missed_rehit = ready_rehit.float() * (1.0 - attack) * (0.35 + 0.65 * aim_score)

    counter_clean = (
        clean_dealt.float()
        * (hit_select_window | close_counter_window | reach_counter_window).float()
        * counter_posture
    )
    counter_trade = traded.float() * under_close.float()
    counter_stolen = stolen.float() * under_close.float()
    counter_action = counter_timed_ready.float() * attack * counter_posture
    hit_select_clean = clean_dealt.float() * hit_select_window.float() * counter_posture
    hit_select_trade = traded.float() * hit_select_window.float()
    hit_select_stolen = stolen.float() * hit_select_window.float()
    hit_select_action = hit_select_ready.float() * attack * counter_posture
    close_counter_clean = clean_dealt.float() * close_counter_window.float() * counter_posture
    close_counter_trade = traded.float() * close_counter_window.float()
    close_counter_stolen = stolen.float() * close_counter_window.float()
    close_counter_action = close_counter_ready.float() * attack * counter_posture
    counter_drive = counter_drive_window.float() * fwd_pos * sprint * (
        0.35 + 0.65 * aim_score) * (0.55 + 0.45 * line_control)
    missed_counter = counter_timed_ready.float() * (1.0 - attack) * (0.35 + 0.65 * aim_score)
    missed_hit_select = hit_select_ready.float() * (1.0 - attack) * (0.35 + 0.65 * aim_score)
    missed_close_counter = close_counter_ready.float() * (1.0 - attack) * (
        0.35 + 0.65 * aim_score)
    early_counter_click = (
        under_window
        & (~counter_timed_ready)
        & (attack > 0.5)
    ).float() * (0.35 + 0.65 * aim_score) * counter_posture
    early_hit_select_click = (
        hit_select_window
        & (~hit_select_ready)
        & (attack > 0.5)
    ).float() * (0.35 + 0.65 * aim_score) * counter_posture
    panic_escape = under_close.float() * (1.75 * back + 1.20 * strafe_abs + 0.70 * jump)
    hit_select_escape = hit_select_window.float() * (1.30 * back + 0.90 * jump)
    close_counter_escape = close_counter_window.float() * (
        1.50 * back + 0.70 * jump + 0.35 * strafe_abs)

    bonus = torch.zeros(obs.shape[0], dtype=torch.float32, device=obs.device)
    if coeff_rechain != 0.0:
        rechain_bonus = coeff_rechain * (
            9.50 * rehit_clean
            + 1.20 * rehit_clean_select
            + 1.55 * rehit_action
            + 1.10 * rehit_z_select
            + 0.70 * rehit_drive
            + 1.80 * rehit_edge_release
            + 1.20 * rehit_wait_release
            - 11.50 * rehit_stolen
            - 6.50 * rehit_trade
            - 1.65 * missed_rehit
            - 1.85 * stale_ready_release
            - 0.90 * rehit_idle_far
            - 1.35 * rehit_edge_overrun
            - 1.20 * rehit_edge_prefire
            - 1.55 * rehit_early_click
            - 1.15 * rehit_far_click
            - 1.20 * rehit_bad_move
        )
        bonus += rechain_bonus * rechain_mask
    if coeff_chase_hit_select != 0.0:
        chase_hit_select_bonus = coeff_chase_hit_select * (
            10.40 * hit_select_clean
            + 1.10 * hit_select_action
            - 18.80 * hit_select_trade
            - 12.40 * hit_select_stolen
            - 4.15 * missed_hit_select
            - 2.80 * early_hit_select_click
            - 1.60 * hit_select_escape
        )
        bonus += chase_hit_select_bonus * counter_mask
    if coeff_chase_close_counter != 0.0:
        chase_close_counter_bonus = coeff_chase_close_counter * (
            9.40 * close_counter_clean
            + 3.65 * close_counter_action
            - 13.20 * close_counter_trade
            - 8.40 * close_counter_stolen
            - 4.35 * missed_close_counter
            - 2.05 * close_counter_escape
        )
        bonus += chase_close_counter_bonus * counter_mask
    if coeff_counter != 0.0:
        counter_bonus = coeff_counter * (
            14.80 * counter_clean
            + 3.25 * counter_action
            + 1.20 * counter_drive
            - 6.70 * counter_stolen
            - 8.40 * counter_trade
            - 4.15 * missed_counter
            - 3.05 * early_counter_click
            - 3.05 * far_under_click.float()
            - 2.25 * panic_escape
        )
        bonus += counter_bonus * counter_mask
    if coeff_spar_counter != 0.0:
        spar_counter_bonus = coeff_spar_counter * (
            14.80 * counter_clean
            + 3.25 * counter_action
            + 1.20 * counter_drive
            - 6.70 * counter_stolen
            - 8.40 * counter_trade
            - 4.15 * missed_counter
            - 3.05 * early_counter_click
            - 3.05 * far_under_click.float()
            - 2.25 * panic_escape
        )
        bonus += spar_counter_bonus * spar_counter_mask
    return bonus.to(dtype=obs.dtype)


def _chain_followup_stats(start: torch.Tensor, response: torch.Tensor,
                          blocker: torch.Tensor, done: torch.Tensor,
                          learner_mask: torch.Tensor, window: int) -> dict:
    start_np = start.detach().bool().cpu().numpy()
    response_np = response.detach().bool().cpu().numpy()
    blocker_np = blocker.detach().bool().cpu().numpy()
    done_np = done.detach().bool().cpu().numpy()
    mask_np = learner_mask.detach().bool().cpu().numpy()
    idxs = np.flatnonzero(mask_np).tolist()
    opportunities = 0
    successes = 0
    blocked = 0
    missed = 0
    for b in idxs:
        pending = False
        gap = window + 1
        for t in range(start_np.shape[0]):
            if done_np[t, b]:
                pending = False
                gap = window + 1
                continue
            if pending:
                gap += 1
                if gap > window:
                    missed += 1
                    pending = False
                elif blocker_np[t, b]:
                    blocked += 1
                    pending = False
                    continue
                elif response_np[t, b]:
                    successes += 1
                    pending = False
            if start_np[t, b]:
                opportunities += 1
                pending = True
                gap = 0
    denom = max(opportunities, 1)
    return {
        "opps": opportunities,
        "hit_frac": float(successes / denom),
        "taken_frac": float(blocked / denom),
        "miss_frac": float(missed / denom),
    }


def _combo_rollout_stats(dealt: torch.Tensor, taken: torch.Tensor,
                         done: torch.Tensor, learner_mask: torch.Tensor,
                         window: int, threshold: int = 12) -> dict:
    dealt_np = dealt.detach().bool().cpu().numpy()
    taken_np = taken.detach().bool().cpu().numpy()
    done_np = done.detach().bool().cpu().numpy()
    mask_np = learner_mask.detach().bool().cpu().numpy()
    idxs = np.flatnonzero(mask_np).tolist()
    max_combo = 0
    chain_values: list[int] = []
    clean_hits = 0
    combo5 = combo8 = combo12 = combo_t = 0
    for b in idxs:
        chain = 0
        gap = window + 1
        for t in range(dealt_np.shape[0]):
            if done_np[t, b]:
                chain = 0
                gap = window + 1
                continue
            gap += 1
            hurt = bool(taken_np[t, b])
            hit = bool(dealt_np[t, b])
            if hurt:
                chain = 0
                gap = window + 1
            if hit and not hurt:
                if gap > window:
                    chain = 0
                chain += 1
                gap = 0
                clean_hits += 1
                chain_values.append(chain)
                max_combo = max(max_combo, chain)
                combo5 += int(chain >= 5)
                combo8 += int(chain >= 8)
                combo12 += int(chain >= 12)
                combo_t += int(chain >= threshold)
    denom = max(clean_hits, 1)
    return {
        "combo_max": max_combo,
        "combo_mean": float(sum(chain_values) / max(len(chain_values), 1)),
        "combo5_hits": float(combo5 / denom),
        "combo8_hits": float(combo8 / denom),
        "combo12_hits": float(combo12 / denom),
        f"combo{threshold}_hits": float(combo_t / denom),
    }


def _round_train_stat(key: str, value: float) -> float:
    return round(value, 8 if key == "lr" else 5)


class Trainer:
    def __init__(self, cfg: dict, device: str | None = None):
        self.cfg = {**DEFAULT_CFG, **cfg}
        self.cfg["sim"] = {**DEFAULT_CFG["sim"], **cfg.get("sim", {})}
        self.cfg["pbt"] = {**DEFAULT_PBT, **cfg.get("pbt", {})}
        torch.manual_seed(self.cfg["seed"])
        np.random.seed(self.cfg["seed"])
        random.seed(self.cfg["seed"])          # league.sample (random.choices)

        # Ampere+ : TF32 pour les matmuls (gros gain, prÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©cision suffisante en RL)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.N = self.cfg["n_envs"]
        self.B = self.N * 2
        self.T = self.cfg["rollout_ticks"]

        self._behavior_reward = {
            key: float(self.cfg["sim"].pop(key, 0.0))
            for key in BEHAVIOR_REWARD_KEYS
        }
        self._behavior_reward_enabled = any(
            value != 0.0 for value in self._behavior_reward.values()
        )
        self.cfg["behavior_reward"] = dict(self._behavior_reward)
        self.sim_cfg = SimConfig(**self.cfg["sim"])
        # curriculum : spawn proche tant que les agents ne se battent pas
        if self.cfg["curriculum_gap"] > 0 and self.sim_cfg.spawn_gap == 0:
            self.sim_cfg.spawn_gap = self.cfg["curriculum_gap"]
        self.sim = make_sim(self.N, self.sim_cfg, seed=self.cfg["seed"],
                            force_cpu=self.device.type != "cuda")

        self.pol_cfg = PolicyConfig(**self.cfg.get("policy", {}))
        self.H = self.pol_cfg.history
        self.policy = JudasPolicy(self.pol_cfg).to(self.device)
        self.ppo = PPO(self.policy, PPOConfig(**self.cfg.get("ppo", {})), self.device)
        self.league = League()

        # ----- population (PBT) : K policies co-entraÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â®nÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©es sur des slices
        self.members: list[Member] = []
        K = int(self.cfg["pbt"]["population"])
        if K > 1:
            self._pbt_rng = random.Random(self.cfg["seed"] + 1303)
            base_ppo = PPOConfig(**self.cfg.get("ppo", {}))
            base_hypers = {k: getattr(base_ppo, k)
                           for k in self.cfg["pbt"]["explore"]}
            for i, (lo, hi) in enumerate(slice_envs(self.N, K)):
                pol = self.policy if i == 0 else \
                    JudasPolicy(self.pol_cfg).to(self.device)
                ppo_cfg = PPOConfig(**self.cfg.get("ppo", {}))
                ppo_cfg.anneal = False        # le PBT pilote lr/entropie
                ppo = self.ppo if i == 0 else PPO(pol, ppo_cfg, self.device)
                if i == 0:
                    self.ppo.cfg.anneal = False
                hypers = dict(base_hypers) if i == 0 else perturb_hypers(
                    base_hypers, self.cfg["pbt"]["explore"],
                    self.cfg["pbt"]["perturb_low"],
                    self.cfg["pbt"]["perturb_high"], self._pbt_rng)
                apply_hypers(ppo, hypers)
                self.members.append(Member(idx=i, policy=pol, ppo=ppo,
                                           hypers=hypers, env_lo=lo, env_hi=hi))
            # membre -> env : pour crÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©diter les ELO des matchs league/cross
            self._env_member = torch.zeros(self.N, dtype=torch.long)
            for m in self.members:
                self._env_member[m.env_lo:m.env_hi] = m.idx
        # infÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©rence de rollout en fp16 (l'update PPO garde sa propre AMP)
        self._use_amp = self.ppo.use_amp
        self._buf = RolloutBuffer(self.T, self.B, OBS_DIM, self.H, self.device)

        self.hist = torch.zeros(self.B, self.H, OBS_DIM, device=self.device)
        # -1 : le premier _push_obs (reset) donne l'ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ge 0 ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â  l'obs de spawn
        self.age = torch.full((self.B,), -1, dtype=torch.long, device=self.device)
        self.iter = 0
        self._combo_eval_origin = 0
        self._fresh_combo_eval_count = 0
        self.total_steps = 0          # agent-steps cumulÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©s (tous agents)

        # automatisations
        self._shaping_base = self.sim_cfg.reward_dist
        self._ramp_on = False        # rampe dÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©clenchÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©e (combat rÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©gulier atteint)
        self._ramp_pos = 0.0         # position 0..1 ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â adaptative, peut reculer
        self._hit_streak = 0
        self._entropy_hist: list[float] = []
        self._eval_sim = None
        self._eval_opp: JudasPolicy | None = None
        self._snapshot_skips = 0
        self._best_bot = -1.0
        self._full_gap = min(min(self.sim_cfg.arena_size_x,
                                 self.sim_cfg.arena_size_z) / 3.0, 8.0)
        from .scripted import (
            ChaseBot,
            ComboChaseBot,
            ComboCounterBot,
            ComboPadBot,
            ComboPressureBot,
            ComboRehitBot,
            ComboSparBot,
        )
        self._bot = ChaseBot()
        self._pad_bot = ComboPadBot()
        self._spar_bot = ComboSparBot()
        self._rehit_bot = ComboRehitBot()
        self._pressure_bot = ComboPressureBot()
        self._combo_chase_bot = ComboChaseBot()
        self._counter_bot = ComboCounterBot()

        self.run_dir = Path("runs") / self.cfg["name"]
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._tb = None

        obs = self._to_torch(self.sim.reset())          # [N, 2, D]
        self._push_obs(obs.reshape(self.B, OBS_DIM),
                       torch.zeros(self.N, device=self.device))

        # adversaires actifs pour le rollout courant
        self._opp_models: list[JudasPolicy] = []
        self._opp_pool_idx: list[int] = []
        self._opp_kind: list[tuple] = []
        self._opp_rows: list[torch.Tensor] = []
        self._bot_rows: torch.Tensor = torch.empty(0, dtype=torch.long,
                                                   device=self.device)
        self._pad_rows: torch.Tensor = torch.empty(0, dtype=torch.long,
                                                   device=self.device)
        self._spar_rows: torch.Tensor = torch.empty(0, dtype=torch.long,
                                                    device=self.device)
        self._rehit_rows: torch.Tensor = torch.empty(0, dtype=torch.long,
                                                     device=self.device)
        self._pressure_rows: torch.Tensor = torch.empty(0, dtype=torch.long,
                                                        device=self.device)
        self._combo_chase_rows: torch.Tensor = torch.empty(
            0, dtype=torch.long, device=self.device)
        self._counter_rows: torch.Tensor = torch.empty(
            0, dtype=torch.long, device=self.device)
        self._env_opp: torch.Tensor = torch.full((self.N,), -1, dtype=torch.long,
                                                 device=self.device)
        self._learner_mask: torch.Tensor | None = None
        # CUDA graphs des forwards de rollout (capturÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©s au 1er rollout,
        # fallback eager) : un graph batch-complet en mode single, un graph
        # par membre en mode population (les slices sont fixes pour le run)
        self._act_graph: "torch.cuda.CUDAGraph | None" = None
        self._act_graph_out: dict | None = None
        self._member_graphs: list | None = None
        self._graphs_tried = False

    # ------------------------------------------------------------------ utils
    def _to_torch(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        return torch.as_tensor(np.asarray(x)).to(self.device)

    def _sim_step(self, actions: torch.Tensor, sim=None):
        sim = sim or self.sim
        if hasattr(sim, "ext"):                          # backend CUDA
            obs, rew, done, info = sim.step(actions)
            info_t = {k: v.clone() for k, v in info.items()
                      if k in ("winner", "combo", "dealt")}
            return obs, rew.clone(), done.bool().clone(), info_t
        a = actions.cpu().numpy()
        obs, rew, done, info = sim.step(a)
        info_t = {k: self._to_torch(v) for k, v in info.items()
                  if k in ("winner", "combo", "dealt")}
        return self._to_torch(obs), self._to_torch(rew), self._to_torch(done).bool(), info_t

    def _push_obs(self, obs_flat: torch.Tensor, done_env: torch.Tensor) -> None:
        """Avance l'historique d'un tick. done_env [N] bool/float.
        self.hist garde un STOCKAGE FIXE (copy_ au lieu de rÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©assignation) :
        c'est l'entrÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©e statique du CUDA graph de _learner_act."""
        done_b = done_env.bool().repeat_interleave(2)
        self.hist.copy_(torch.roll(self.hist, shifts=-1, dims=1))
        self.hist[done_b] = 0.0
        self.hist[:, -1] = obs_flat
        self.age = torch.where(done_b, torch.zeros_like(self.age), self.age + 1)
        # NB: ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ge de l'obs courante = 0 si nouvel ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©pisode

    # -------------------------------------------------------------- opponents
    @staticmethod
    def _full_scripted_curriculum(cfg: dict) -> bool:
        scripted = (
            float(cfg.get("league_bot_frac", 0.0) or 0.0)
            + float(cfg.get("league_pad_bot_frac", 0.0) or 0.0)
            + float(cfg.get("league_spar_bot_frac", 0.0) or 0.0)
            + float(cfg.get("league_rehit_bot_frac", 0.0) or 0.0)
            + float(cfg.get("league_pressure_bot_frac", 0.0) or 0.0)
            + float(cfg.get("league_combo_chase_bot_frac", 0.0) or 0.0)
            + float(cfg.get("league_counter_bot_frac", 0.0) or 0.0)
        )
        return float(cfg.get("league_frac", 0.0) or 0.0) <= 0.0 and scripted >= 0.999

    @staticmethod
    def _scripted_fill_kind(cfg: dict) -> int:
        kinds = [
            (-6, float(cfg.get("league_pressure_bot_frac", 0.0) or 0.0)),
            (-7, float(cfg.get("league_combo_chase_bot_frac", 0.0) or 0.0)),
            (-8, float(cfg.get("league_counter_bot_frac", 0.0) or 0.0)),
            (-4, float(cfg.get("league_spar_bot_frac", 0.0) or 0.0)),
            (-5, float(cfg.get("league_rehit_bot_frac", 0.0) or 0.0)),
            (-3, float(cfg.get("league_pad_bot_frac", 0.0) or 0.0)),
            (-2, float(cfg.get("league_bot_frac", 0.0) or 0.0)),
        ]
        kind, frac = max(kinds, key=lambda item: item[1])
        return kind if frac > 0.0 else -1

    def _assign_opponents(self) -> torch.Tensor:
        """Choisit les adversaires du rollout."""
        if self.members:
            return self._assign_opponents_pbt()
        learner_mask = torch.ones(self.B, dtype=torch.bool, device=self.device)
        self._opp_models, self._opp_pool_idx, self._opp_rows = [], [], []
        self._opp_kind = []
        self._bot_rows = torch.empty(0, dtype=torch.long, device=self.device)
        self._pad_rows = torch.empty(0, dtype=torch.long, device=self.device)
        self._spar_rows = torch.empty(0, dtype=torch.long, device=self.device)
        self._rehit_rows = torch.empty(0, dtype=torch.long, device=self.device)
        self._pressure_rows = torch.empty(0, dtype=torch.long, device=self.device)
        self._combo_chase_rows = torch.empty(0, dtype=torch.long, device=self.device)
        self._counter_rows = torch.empty(0, dtype=torch.long, device=self.device)
        self._env_opp.fill_(-1)
        perm = torch.randperm(self.N, device=self.device)

        cursor = 0
        pf = float(self.cfg.get("league_pad_bot_frac", 0.0))
        n_pad = int(self.N * max(0.0, pf * (1.0 - self._ramp_frac())))
        if n_pad > 0:
            pad_envs = perm[:n_pad]
            cursor = n_pad
            self._env_opp[pad_envs] = -3
            self._pad_rows = pad_envs * 2 + 1
            learner_mask[self._pad_rows] = False

        sf = float(self.cfg.get("league_spar_bot_frac", 0.0))
        n_spar = int(self.N * max(0.0, sf * (1.0 - self._ramp_frac())))
        if n_spar > 0:
            spar_envs = perm[cursor:cursor + n_spar]
            cursor += n_spar
            self._env_opp[spar_envs] = -4
            self._spar_rows = spar_envs * 2 + 1
            learner_mask[self._spar_rows] = False

        rf = float(self.cfg.get("league_rehit_bot_frac", 0.0))
        n_rehit = int(self.N * max(0.0, rf * (1.0 - self._ramp_frac())))
        if n_rehit > 0:
            rehit_envs = perm[cursor:cursor + n_rehit]
            cursor += n_rehit
            self._env_opp[rehit_envs] = -5
            self._rehit_rows = rehit_envs * 2 + 1
            learner_mask[self._rehit_rows] = False

        qf = float(self.cfg.get("league_pressure_bot_frac", 0.0))
        n_pressure = int(self.N * max(0.0, qf * (1.0 - self._ramp_frac())))
        if n_pressure > 0:
            pressure_envs = perm[cursor:cursor + n_pressure]
            cursor += n_pressure
            self._env_opp[pressure_envs] = -6
            self._pressure_rows = pressure_envs * 2 + 1
            learner_mask[self._pressure_rows] = False

        cf = float(self.cfg.get("league_combo_chase_bot_frac", 0.0))
        n_combo_chase = int(self.N * max(0.0, cf * (1.0 - self._ramp_frac())))
        if n_combo_chase > 0:
            combo_chase_envs = perm[cursor:cursor + n_combo_chase]
            cursor += n_combo_chase
            self._env_opp[combo_chase_envs] = -7
            self._combo_chase_rows = combo_chase_envs * 2 + 1
            learner_mask[self._combo_chase_rows] = False

        kf = float(self.cfg.get("league_counter_bot_frac", 0.0))
        n_counter = int(self.N * max(0.0, kf * (1.0 - self._ramp_frac())))
        if n_counter > 0:
            counter_envs = perm[cursor:cursor + n_counter]
            cursor += n_counter
            self._env_opp[counter_envs] = -8
            self._counter_rows = counter_envs * 2 + 1
            learner_mask[self._counter_rows] = False

        bf = self.cfg["league_bot_frac"]
        n_bot = int(self.N * max(0.05, bf * (1.0 - self._ramp_frac()))) if bf > 0 else 0
        if n_bot > 0:
            bot_envs = perm[cursor:cursor + n_bot]
            cursor += n_bot
            self._env_opp[bot_envs] = -2
            self._bot_rows = bot_envs * 2 + 1
            learner_mask[self._bot_rows] = False

        if self._full_scripted_curriculum(self.cfg) and cursor < self.N:
            fill_kind = self._scripted_fill_kind(self.cfg)
            fill_envs = perm[cursor:self.N]
            fill_rows = fill_envs * 2 + 1
            self._env_opp[fill_envs] = fill_kind
            learner_mask[fill_rows] = False
            if fill_kind == -2:
                self._bot_rows = torch.cat((self._bot_rows, fill_rows))
            elif fill_kind == -3:
                self._pad_rows = torch.cat((self._pad_rows, fill_rows))
            elif fill_kind == -4:
                self._spar_rows = torch.cat((self._spar_rows, fill_rows))
            elif fill_kind == -5:
                self._rehit_rows = torch.cat((self._rehit_rows, fill_rows))
            elif fill_kind == -6:
                self._pressure_rows = torch.cat((self._pressure_rows, fill_rows))
            elif fill_kind == -7:
                self._combo_chase_rows = torch.cat((self._combo_chase_rows, fill_rows))
            elif fill_kind == -8:
                self._counter_rows = torch.cat((self._counter_rows, fill_rows))
            cursor = self.N

        frac = self.cfg["league_frac"]
        if not self.league.pool or frac <= 0:
            return learner_mask
        n_league = min(int(self.N * frac), self.N - cursor)
        if n_league <= 0:
            return learner_mask
        n_groups = min(4, len(self.league.pool))
        idxs = self.league.sample(n_groups)
        env_ids = perm[cursor:cursor + n_league]
        groups = env_ids.chunk(n_groups)
        for gi, (pool_idx, envs) in enumerate(zip(idxs, groups)):
            if envs.numel() == 0:
                continue
            m = JudasPolicy(self.pol_cfg).to(self.device)
            m.load_state_dict(self.league.pool[pool_idx]["state_dict"], strict=False)
            m.eval()
            self._opp_models.append(m)
            self._opp_pool_idx.append(pool_idx)
            self._opp_kind.append(("league", pool_idx))
            self._opp_rows.append(envs * 2 + 1)
            self._env_opp[envs] = gi
            learner_mask[envs * 2 + 1] = False
        return learner_mask

    def _assign_opponents_pbt(self) -> torch.Tensor:
        learner_mask = torch.ones(self.B, dtype=torch.bool, device=self.device)
        self._opp_models, self._opp_pool_idx, self._opp_rows = [], [], []
        self._opp_kind = []
        self._env_opp.fill_(-1)
        pbt = self.cfg["pbt"]
        K = len(self.members)
        bot_rows_all = []
        pad_rows_all = []
        spar_rows_all = []
        rehit_rows_all = []
        pressure_rows_all = []
        combo_chase_rows_all = []
        counter_rows_all = []
        bf = self.cfg["league_bot_frac"]
        pf = float(self.cfg.get("league_pad_bot_frac", 0.0))
        sf = float(self.cfg.get("league_spar_bot_frac", 0.0))
        rf = float(self.cfg.get("league_rehit_bot_frac", 0.0))
        qf = float(self.cfg.get("league_pressure_bot_frac", 0.0))
        cf = float(self.cfg.get("league_combo_chase_bot_frac", 0.0))
        kf = float(self.cfg.get("league_counter_bot_frac", 0.0))
        pad_frac = max(0.0, pf * (1.0 - self._ramp_frac()))
        spar_frac = max(0.0, sf * (1.0 - self._ramp_frac()))
        rehit_frac = max(0.0, rf * (1.0 - self._ramp_frac()))
        pressure_frac = max(0.0, qf * (1.0 - self._ramp_frac()))
        combo_chase_frac = max(0.0, cf * (1.0 - self._ramp_frac()))
        counter_frac = max(0.0, kf * (1.0 - self._ramp_frac()))
        bot_frac = max(0.05, bf * (1.0 - self._ramp_frac())) if bf > 0 else 0.0
        rot = (self.iter // max(self.cfg["pool_every"], 1)) % max(K - 1, 1)
        for mb in self.members:
            n_m = mb.env_hi - mb.env_lo
            perm = torch.randperm(n_m, device=self.device) + mb.env_lo
            cursor = 0
            n_pad = int(n_m * pad_frac)
            if n_pad > 0:
                envs = perm[:n_pad]
                cursor = n_pad
                self._env_opp[envs] = -3
                pad_rows_all.append(envs * 2 + 1)
                learner_mask[envs * 2 + 1] = False
            n_spar = int(n_m * spar_frac)
            if n_spar > 0:
                envs = perm[cursor:cursor + n_spar]
                cursor += n_spar
                self._env_opp[envs] = -4
                spar_rows_all.append(envs * 2 + 1)
                learner_mask[envs * 2 + 1] = False
            n_rehit = int(n_m * rehit_frac)
            if n_rehit > 0:
                envs = perm[cursor:cursor + n_rehit]
                cursor += n_rehit
                self._env_opp[envs] = -5
                rehit_rows_all.append(envs * 2 + 1)
                learner_mask[envs * 2 + 1] = False
            n_pressure = int(n_m * pressure_frac)
            if n_pressure > 0:
                envs = perm[cursor:cursor + n_pressure]
                cursor += n_pressure
                self._env_opp[envs] = -6
                pressure_rows_all.append(envs * 2 + 1)
                learner_mask[envs * 2 + 1] = False
            n_combo_chase = int(n_m * combo_chase_frac)
            if n_combo_chase > 0:
                envs = perm[cursor:cursor + n_combo_chase]
                cursor += n_combo_chase
                self._env_opp[envs] = -7
                combo_chase_rows_all.append(envs * 2 + 1)
                learner_mask[envs * 2 + 1] = False
            n_counter = int(n_m * counter_frac)
            if n_counter > 0:
                envs = perm[cursor:cursor + n_counter]
                cursor += n_counter
                self._env_opp[envs] = -8
                counter_rows_all.append(envs * 2 + 1)
                learner_mask[envs * 2 + 1] = False
            n_bot = int(n_m * bot_frac)
            if n_bot > 0:
                envs = perm[cursor:cursor + n_bot]
                cursor += n_bot
                self._env_opp[envs] = -2
                bot_rows_all.append(envs * 2 + 1)
                learner_mask[envs * 2 + 1] = False
            if self._full_scripted_curriculum(self.cfg) and cursor < n_m:
                fill_kind = self._scripted_fill_kind(self.cfg)
                envs = perm[cursor:n_m]
                rows = envs * 2 + 1
                cursor = n_m
                self._env_opp[envs] = fill_kind
                learner_mask[rows] = False
                if fill_kind == -2:
                    bot_rows_all.append(rows)
                elif fill_kind == -3:
                    pad_rows_all.append(rows)
                elif fill_kind == -4:
                    spar_rows_all.append(rows)
                elif fill_kind == -5:
                    rehit_rows_all.append(rows)
                elif fill_kind == -6:
                    pressure_rows_all.append(rows)
                elif fill_kind == -7:
                    combo_chase_rows_all.append(rows)
                elif fill_kind == -8:
                    counter_rows_all.append(rows)
            n_cross = int(n_m * pbt["cross_frac"]) if K > 1 else 0
            if n_cross > 0:
                j = (mb.idx + 1 + rot) % K
                if j == mb.idx:
                    j = (j + 1) % K
                envs = perm[cursor:cursor + n_cross]
                cursor += n_cross
                gi = len(self._opp_models)
                self._opp_models.append(self.members[j].policy)
                self._opp_kind.append(("cross", j))
                self._opp_rows.append(envs * 2 + 1)
                self._env_opp[envs] = gi
                learner_mask[envs * 2 + 1] = False
            if self.league.pool and self.cfg["league_frac"] > 0:
                n_league = min(int(n_m * self.cfg["league_frac"]), n_m - cursor)
                n_groups = min(2, len(self.league.pool))
                if n_league > 0 and n_groups > 0:
                    envs = perm[cursor:cursor + n_league]
                    cursor += n_league
                    for pool_idx, chunk in zip(self.league.sample(n_groups), envs.chunk(n_groups)):
                        if chunk.numel() == 0:
                            continue
                        model = JudasPolicy(self.pol_cfg).to(self.device)
                        model.load_state_dict(self.league.pool[pool_idx]["state_dict"], strict=False)
                        model.eval()
                        gi = len(self._opp_models)
                        self._opp_models.append(model)
                        self._opp_kind.append(("league", pool_idx))
                        self._opp_rows.append(chunk * 2 + 1)
                        self._env_opp[chunk] = gi
                        learner_mask[chunk * 2 + 1] = False
        self._bot_rows = (torch.cat(bot_rows_all) if bot_rows_all
                          else torch.empty(0, dtype=torch.long, device=self.device))
        self._pad_rows = (torch.cat(pad_rows_all) if pad_rows_all
                          else torch.empty(0, dtype=torch.long, device=self.device))
        self._spar_rows = (torch.cat(spar_rows_all) if spar_rows_all
                           else torch.empty(0, dtype=torch.long, device=self.device))
        self._rehit_rows = (torch.cat(rehit_rows_all) if rehit_rows_all
                            else torch.empty(0, dtype=torch.long, device=self.device))
        self._pressure_rows = (torch.cat(pressure_rows_all) if pressure_rows_all
                               else torch.empty(0, dtype=torch.long, device=self.device))
        self._combo_chase_rows = (
            torch.cat(combo_chase_rows_all) if combo_chase_rows_all
            else torch.empty(0, dtype=torch.long, device=self.device))
        self._counter_rows = (torch.cat(counter_rows_all) if counter_rows_all
                              else torch.empty(0, dtype=torch.long, device=self.device))
        return learner_mask

    # ---------------------------------------------------------------- rollout
    @torch.no_grad()
    def _act_eager(self) -> dict:
        with torch.autocast("cuda", dtype=torch.float16, enabled=self._use_amp):
            return self.policy.act(self.hist)

    @torch.no_grad()
    def _capture_act_graph(self) -> None:
        """Capture le forward learner (batch complet, forme fixe) dans un
        CUDA graph : ~30-60 lancements de kernels par tick remplacÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©s par un
        seul replay. Les poids restent ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â  jour (updates Adam in-place) et
        self.hist garde un stockage fixe (_push_obs). Fallback eager si la
        capture ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©choue (op non capturable, driver, etc.)."""
        self._graphs_tried = True
        if not (self.cfg["cuda_graphs"] and self.device.type == "cuda"):
            return
        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):                      # warmup hors capture
                    self._act_eager()
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._act_graph_out = self._act_eager()
            self._act_graph = graph
            print("[trainer] CUDA graph actif sur le forward de rollout")
        except Exception as exc:                        # noqa: BLE001
            self._act_graph = None
            self._act_graph_out = None
            print(f"[trainer] CUDA graph indisponible ({exc}) ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â fallback eager")

    @torch.no_grad()
    def _learner_act(self) -> dict:
        """Forward du learner sur le batch COMPLET (forme fixe, graph-friendly).
        Les lignes des adversaires gelÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©s sont ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©crasÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©es ensuite ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â leurs
        logp/value parasites ne sont jamais ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©chantillonnÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©s (b_keep)."""
        if not self._graphs_tried:
            self._capture_act_graph()
        if self._act_graph is not None:
            self._act_graph.replay()
            # clones : les tenseurs de sortie du graph sont rÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©crits au
            # prochain replay, l'appelant doit possÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©der ses donnÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©es
            return {k: v.clone() for k, v in self._act_graph_out.items()}
        return self._act_eager()

    @torch.no_grad()
    def _member_act_eager(self, mb) -> dict:
        with torch.autocast("cuda", dtype=torch.float16, enabled=self._use_amp):
            return mb.policy.act(self.hist[mb.row_lo:mb.row_hi])

    @torch.no_grad()
    def _capture_member_graphs(self) -> None:
        """Mode population : capture le forward de CHAQUE membre dans son
        propre CUDA graph. Les tranches d'envs sont fixes pour tout le run et
        exploit_explore copie les poids IN-PLACE (load_state_dict) ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â les
        graphs restent valides aprÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¨s les copies de population."""
        self._graphs_tried = True
        if not (self.cfg["cuda_graphs"] and self.device.type == "cuda"):
            return
        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for mb in self.members:
                    for _ in range(3):                  # warmup hors capture
                        self._member_act_eager(mb)
            torch.cuda.current_stream().wait_stream(side)
            graphs = []
            for mb in self.members:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = self._member_act_eager(mb)
                graphs.append((graph, out))
            self._member_graphs = graphs
            print(f"[trainer] CUDA graphs actifs sur les {len(graphs)} membres")
        except Exception as exc:                        # noqa: BLE001
            self._member_graphs = None
            print(f"[trainer] CUDA graphs population indisponibles ({exc}) "
                  f"ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â fallback eager")

    @torch.no_grad()
    def _policy_actions(self) -> dict:
        """act de tous les contrÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â´leurs apprenants : la policy unique (batch
        complet, CUDA graph) ou les K membres de la population (un graph par
        tranche contiguÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â« ; torch.cat matÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©rialise des copies ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â les buffers de
        sortie des graphs sont rÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©crits au replay suivant)."""
        if not self.members:
            return self._learner_act()
        if not self._graphs_tried:
            self._capture_member_graphs()
        if self._member_graphs is not None:
            outs = []
            for graph, out in self._member_graphs:
                graph.replay()
                outs.append(out)
        else:
            outs = [self._member_act_eager(mb) for mb in self.members]
        return {k: torch.cat([o[k] for o in outs], dim=0) for k in outs[0]}

    @torch.no_grad()
    def _collect(self, buf: RolloutBuffer, learner_mask: torch.Tensor) -> dict:
        ep_stats = {"wins": 0, "losses": 0, "draws": 0, "matches": 0,
                    "mirror_matches": 0}
        # rÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©sultats accumulÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©s sur GPU, traitÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©s en une seule synchro aprÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¨s le rollout
        ep_done = torch.zeros(self.T, self.N, dtype=torch.bool, device=self.device)
        ep_winner = torch.zeros(self.T, self.N, dtype=torch.int32, device=self.device)
        ep_combo = torch.zeros(self.T, self.N, 2, dtype=torch.int32, device=self.device)
        row_ids = torch.arange(self.B, device=self.device)
        row_env = torch.div(row_ids, 2, rounding_mode="floor")
        combo_rechain_learner_mask = learner_mask & (
            (self._env_opp[row_env] == -7) | (self._env_opp[row_env] == -8))
        combo_recovery_counter_learner_mask = learner_mask & (
            (self._env_opp[row_env] == -7)
            | (self._env_opp[row_env] == -8)
        )
        combo_spar_counter_learner_mask = learner_mask & (self._env_opp[row_env] == -4)
        buf.set_prefix(self.hist)
        for t in range(self.T):
            obs_now = self.hist[:, -1].clone()

            out = self._policy_actions()
            raw = {"pre": out["pre"].float(), "bins": out["bins"].float(),
                   "fwd": out["fwd"].long(), "strafe": out["strafe"].long()}
            logp = out["logp"].float()
            value = out["value"].float()

            for rows, m in zip(self._opp_rows, self._opp_models):
                with torch.autocast("cuda", dtype=torch.float16,
                                    enabled=self._use_amp):
                    o = m.act(self.hist[rows])
                raw["pre"][rows] = o["pre"].float()
                raw["bins"][rows] = o["bins"].float()
                raw["fwd"][rows] = o["fwd"].long()
                raw["strafe"][rows] = o["strafe"].long()

            sim_actions = to_sim_actions(raw)
            if self._pad_rows.numel() > 0:
                sim_actions[self._pad_rows] = self._pad_bot.act7(self.hist[self._pad_rows])
            if self._spar_rows.numel() > 0:
                sim_actions[self._spar_rows] = self._spar_bot.act7(self.hist[self._spar_rows])
            if self._rehit_rows.numel() > 0:
                sim_actions[self._rehit_rows] = self._rehit_bot.act7(self.hist[self._rehit_rows])
            if self._pressure_rows.numel() > 0:
                sim_actions[self._pressure_rows] = self._pressure_bot.act7(
                    self.hist[self._pressure_rows])
            if self._combo_chase_rows.numel() > 0:
                sim_actions[self._combo_chase_rows] = self._combo_chase_bot.act7(
                    self.hist[self._combo_chase_rows])
            if self._counter_rows.numel() > 0:
                sim_actions[self._counter_rows] = self._counter_bot.act7(
                    self.hist[self._counter_rows])
            if self._bot_rows.numel() > 0:
                sim_actions[self._bot_rows] = self._bot.act7(self.hist[self._bot_rows])
            obs, rew, done, info = self._sim_step(sim_actions.view(self.N, 2, 7))
            winner = info["winner"].int()
            obs_next = obs.reshape(self.B, OBS_DIM).float()
            done_flat = done.float().repeat_interleave(2)
            prev_hits = (obs_now[:, 31] * 100.0).round()
            next_hits = (obs_next[:, 31] * 100.0).round()
            dealt_now = ((next_hits - prev_hits).clamp(min=0.0)
                         * (1.0 - done_flat)) > 0.5
            taken_now = dealt_now.view(self.N, 2).flip(-1).reshape(self.B)
            if self._behavior_reward_enabled:
                combo_info = info.get("combo")
                combo_lengths = (combo_info.reshape(self.B)
                                 if combo_info is not None else None)
                behavior_bonus = _behavior_reward_bonus(
                    obs_now, sim_actions, self._behavior_reward,
                    combo_lengths,
                )
                behavior_bonus = behavior_bonus + _hit_event_reward_bonus(
                    obs_now, sim_actions, dealt_now, taken_now,
                    self._behavior_reward,
                )
                behavior_bonus = behavior_bonus + _post_hit_wtap_reward_bonus(
                    self.hist, sim_actions, self._behavior_reward)
                behavior_bonus = behavior_bonus + _opener_boxing_reward_bonus(
                    self.age, obs_now, sim_actions, self._behavior_reward,
                    int(self.cfg.get("safety_opener_ticks", 20) or 20))
                behavior_bonus = behavior_bonus + _chase_transfer_reward_bonus(
                    obs_now, sim_actions, dealt_now, taken_now,
                    combo_rechain_learner_mask,
                    combo_recovery_counter_learner_mask,
                    combo_spar_counter_learner_mask,
                    self._behavior_reward,
                )
                rew = rew + behavior_bonus.view(self.N, 2)

            buf.add(obs_now, self.age.clamp(min=0), raw, logp, value,
                    rew.reshape(self.B),
                    done.float().repeat_interleave(2))

            self._push_obs(obs.reshape(self.B, OBS_DIM), done)
            ep_done[t] = done
            ep_winner[t] = winner
            if "combo" in info:
                ep_combo[t] = info["combo"].int()

        # rÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©sultats de matchs -> ELO (une seule synchro GPU -> CPU)
        done_cpu = ep_done.cpu().numpy()
        winner_cpu = ep_winner.cpu().numpy()
        env_opp_cpu = self._env_opp.cpu().numpy()
        for t, e in zip(*done_cpu.nonzero()):
            w = int(winner_cpu[t, e])
            ep_stats["matches"] += 1
            gi = int(env_opp_cpu[e])
            if gi in (-2, -3, -4, -5, -6, -7, -8):   # matchs vs bots scriptes : pas d'ELO
                continue
            if gi == -1:
                ep_stats["mirror_matches"] += 1
                continue
            score = 1.0 if w == 0 else (0.5 if w == -1 else 0.0)
            kind, ref = self._opp_kind[gi]
            if self.members:
                me = self.members[int(self._env_member[e])]
                if kind == "league":
                    entry = self.league.pool[ref]
                    d = elo_delta(me.elo, entry["elo"], score)
                    me.elo += d
                    entry["elo"] -= d
                    entry["games"] += 1
                else:                    # cross-play : ELO zÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©ro-somme membres
                    other = self.members[ref]
                    d = elo_delta(me.elo, other.elo, score)
                    me.elo += d
                    other.elo -= d
                    other.games += 1
                me.games += 1
            else:
                self.league.report(ref, score)
            ep_stats["wins" if w == 0 else ("draws" if w == -1 else "losses")] += 1
        ep_stats["sim_combo"] = ep_combo.reshape(self.T, self.B)
        return ep_stats

    # ------------------------------------------------------------------- main
    def train_iter(self, force_combo_eval: bool = False) -> dict:
        t0 = time.perf_counter()
        # Adversaires COLLANTS : rÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©assignÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©s seulement toutes les pool_every
        # itÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©rations. Les ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©pisodes (>1000 ticks) s'ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©talent sur ~10-30 rollouts
        # de 128 ticks : rÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©assigner ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â  chaque itÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©ration attribuait ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â  l'ELO d'un
        # snapshot des matchs majoritairement jouÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©s contre d'autres contrÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â´leurs.
        if self._learner_mask is None or self.iter % self.cfg["pool_every"] == 0:
            self._learner_mask = self._assign_opponents()
        learner_mask = self._learner_mask
        buf = self._buf
        buf.reset()
        ep = self._collect(buf, learner_mask)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_collect = time.perf_counter() - t0

        with torch.no_grad():
            # bootstrap de troncature : value de TOUTES les lignes (les lignes
            # gelÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©es portent des valeurs parasites jamais ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©chantillonnÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©es)
            last_value = self._policy_actions()["value"].float()

        progress = self.iter / max(self.cfg["total_iters"], 1)
        if self.members:
            stats_all = []
            for mb in self.members:
                mask = learner_mask.clone()
                mask[:mb.row_lo] = False
                mask[mb.row_hi:] = False
                stats_all.append(mb.ppo.update(buf, mask, last_value, progress))
            stats = {k: float(np.mean([s[k] for s in stats_all]))
                     for k in stats_all[0]}
        else:
            stats = self.ppo.update(buf, learner_mask, last_value, progress)

        self.iter += 1
        if self.members:
            # le ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â« meilleur membre ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â» porte les chemins single-policy :
            # ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©val, snapshots league, best.pt, export
            best = max(self.members, key=lambda m: m.elo)
            self.policy = best.policy
            self.ppo = best.ppo
            if self.iter % self.cfg["pbt"]["interval"] == 0:
                events = exploit_explore(self.members, self.cfg["pbt"],
                                         self._pbt_rng)
                for loser, winner in events:
                    print(f"[pbt] membre {loser} <- copie du membre {winner} "
                          f"(hypers perturbÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©s)")
        if self.iter % self.cfg["pool_every"] == 0 or len(self.league.pool) == 0:
            self._maybe_snapshot()
        if self.iter % self.cfg["save_every"] == 0:
            self.save()

        lm_all = learner_mask
        row_ids = torch.arange(self.B, device=self.device)
        row_env = torch.div(row_ids, 2, rounding_mode="floor")
        opp_kind = self._env_opp[row_env]
        mirror_lm = lm_all & (opp_kind == -1)
        bot_lm = lm_all & (opp_kind == -2)
        pad_lm = lm_all & (opp_kind == -3)
        spar_lm = lm_all & (opp_kind == -4)
        rehit_lm = lm_all & (opp_kind == -5)
        pressure_lm = lm_all & (opp_kind == -6)
        combo_chase_lm = lm_all & (opp_kind == -7)
        counter_lm = lm_all & (opp_kind == -8)
        nonpad_lm = lm_all & (opp_kind != -3)
        lm = mirror_lm if bool(mirror_lm.any().item()) else nonpad_lm
        if not bool(lm.any().item()):
            lm = lm_all

        # Hits EXACTS depuis l'obs (o[31] = hits/100) plutot que par seuil sur
        # le reward : les trades (hit + hurt le meme tick, reward ~ 0) sont
        # comptes correctement. Le tick de done est exclu (obs = nouveau match).
        hits_obs = (buf.obs[:, :, 31] * 100.0).round()                 # [T, B]
        final_hits = (self.hist[:, -1, 31] * 100.0).round().unsqueeze(0)
        next_hits = torch.cat([hits_obs[1:], final_hits], dim=0)
        dealt = ((next_hits - hits_obs).clamp(min=0.0)
                 * (1.0 - buf.done)) > 0.5                             # [T, B]
        taken = dealt.view(self.T, self.N, 2).flip(-1).reshape(self.T, self.B)

        def combo_stats_for(mask: torch.Tensor) -> dict:
            if not bool(mask.any().item()):
                return {
                    "combo_max": 0, "combo_mean": 0.0,
                    "combo5_hits": 0.0, "combo8_hits": 0.0,
                    "combo12_hits": 0.0, "combo2_hits": 0.0,
                }
            return _combo_rollout_stats(
                dealt, taken, buf.done.bool(), mask, self.sim_cfg.combo_window,
                threshold=2,
            )

        sim_combo = ep.get("sim_combo")
        def sim_combo_stats_for(mask: torch.Tensor) -> tuple[int, float]:
            if sim_combo is None or not bool(mask.any().item()):
                return 0, 0.0
            values = sim_combo[:, mask]
            return int(values.max().item()), float((values >= 12).float().mean())

        def hit_rate_for(mask: torch.Tensor) -> float:
            if not bool(mask.any().item()):
                return 0.0
            return float(dealt[:, mask].float().mean()) * 1200.0

        def engage_for(mask: torch.Tensor) -> float:
            if not bool(mask.any().item()):
                return 0.0
            return float((buf.obs[:, mask, 45] * 8.0 < 3.5).float().mean())

        hits_mask = dealt[:, lm]
        taken_mask = taken[:, lm]
        clean_hits_mask = hits_mask & (~taken_mask)
        hit_rate = hit_rate_for(lm)
        clean_hit_rate = float(clean_hits_mask.float().mean()) * 1200.0
        trade_hit_frac = float((hits_mask & taken_mask).float().sum()
                               / hits_mask.float().sum().clamp(min=1.0))
        clean_dealt = dealt & (~taken)
        clean_taken = taken & (~dealt)

        def followup_stats_for(mask: torch.Tensor, start: torch.Tensor) -> dict:
            if not bool(mask.any().item()):
                return {
                    "opps": 0, "hit_frac": 0.0,
                    "taken_frac": 0.0, "miss_frac": 0.0,
                }
            return _chain_followup_stats(
                start, clean_dealt, clean_taken, buf.done.bool(), mask,
                self.sim_cfg.combo_window,
            )

        rechain_stats = followup_stats_for(lm, clean_dealt)
        spar_rechain_stats = followup_stats_for(spar_lm, clean_dealt)
        rehit_rechain_stats = followup_stats_for(rehit_lm, clean_dealt)
        pressure_rechain_stats = followup_stats_for(pressure_lm, clean_dealt)
        combo_chase_rechain_stats = followup_stats_for(combo_chase_lm, clean_dealt)
        counter_rechain_stats = followup_stats_for(counter_lm, clean_dealt)
        counter_break_stats = followup_stats_for(lm, clean_taken)
        spar_counter_break_stats = followup_stats_for(spar_lm, clean_taken)
        rehit_counter_break_stats = followup_stats_for(rehit_lm, clean_taken)
        pressure_counter_break_stats = followup_stats_for(pressure_lm, clean_taken)
        combo_chase_counter_break_stats = followup_stats_for(
            combo_chase_lm, clean_taken)
        counter_counter_break_stats = followup_stats_for(counter_lm, clean_taken)
        engage = engage_for(lm)
        mirror_hit_rate = hit_rate_for(mirror_lm)
        mirror_engage = engage_for(mirror_lm)
        bot_hit_rate = hit_rate_for(bot_lm)
        bot_engage = engage_for(bot_lm)
        pad_hit_rate = hit_rate_for(pad_lm)
        pad_engage = engage_for(pad_lm)
        spar_hit_rate = hit_rate_for(spar_lm)
        spar_engage = engage_for(spar_lm)
        rehit_hit_rate = hit_rate_for(rehit_lm)
        rehit_engage = engage_for(rehit_lm)
        pressure_hit_rate = hit_rate_for(pressure_lm)
        pressure_engage = engage_for(pressure_lm)
        combo_chase_hit_rate = hit_rate_for(combo_chase_lm)
        combo_chase_engage = engage_for(combo_chase_lm)
        counter_hit_rate = hit_rate_for(counter_lm)
        counter_engage = engage_for(counter_lm)
        pitch_abs = (buf.obs[:, lm, 10] * 90.0).abs()
        sky_frac = float((pitch_abs > 60.0).float().mean())
        yaw_err_deg = torch.rad2deg(
            torch.atan2(buf.obs[:, lm, 11], buf.obs[:, lm, 12]).abs())
        pitch_err_deg = (buf.obs[:, lm, 13] * 90.0).abs()
        aim_body = float(((1.0 - yaw_err_deg / 90.0).clamp(0.0, 1.0)
                          * (1.0 - pitch_err_deg / 45.0).clamp(0.0, 1.0)).mean())
        sprint_act = (buf.bins[:, lm, 1] > 0.5) & (buf.fwd[:, lm] == 2)
        sprint_hit = float((hits_mask & sprint_act).float().sum()
                           / hits_mask.float().sum().clamp(min=1.0))
        fwd_dir_metric = buf.fwd[:, lm].float() - 1.0
        strafe_dir_metric = buf.strafe[:, lm].float() - 1.0
        sprint_on = buf.bins[:, lm, 1] > 0.5
        attack_on = buf.bins[:, lm, 2] > 0.5
        dist_metric = buf.obs[:, lm, 45] * 8.0
        forward_frac = float((fwd_dir_metric > 0.5).float().mean())
        back_mask = fwd_dir_metric < -0.5
        escape_back = back_mask & sprint_on
        back_frac = float(back_mask.float().mean())
        escape_back_frac = float(escape_back.float().mean())
        strafe_frac = float((strafe_dir_metric.abs() > 0.5).float().mean())
        opener_ticks = int(self.cfg.get("safety_opener_ticks", 20) or 0)
        opener_mask = ((buf.age[:, lm] < opener_ticks)
                       & (buf.done[:, lm] < 0.5)) if opener_ticks > 0 else None
        if opener_mask is None:
            opener_samples = float(strafe_dir_metric.numel())
            opener_strafe_frac = strafe_frac
            opener_count = torch.full(
                (strafe_dir_metric.shape[1],), strafe_dir_metric.shape[0],
                device=strafe_dir_metric.device, dtype=torch.float32)
            opener_pos = (strafe_dir_metric > 0.5).float().sum(dim=0)
            opener_neg = (strafe_dir_metric < -0.5).float().sum(dim=0)
            opener_strafe_hold_frac = float(
                (torch.maximum(opener_pos, opener_neg)
                 / opener_count.clamp(min=1.0)).mean())
            opener_pressure_frac = float(
                ((fwd_dir_metric > 0.5)
                 & (strafe_dir_metric.abs() > 0.5)).float().mean())
        else:
            opener_samples = float(opener_mask.float().sum())
            opener_strafe_frac = float(
                (((strafe_dir_metric.abs() > 0.5) & opener_mask).float().sum()
                 / opener_mask.float().sum().clamp(min=1.0)))
            opener_count = opener_mask.float().sum(dim=0)
            opener_pos = ((strafe_dir_metric > 0.5) & opener_mask).float().sum(dim=0)
            opener_neg = ((strafe_dir_metric < -0.5) & opener_mask).float().sum(dim=0)
            valid_openers = opener_count > 0.0
            if bool(valid_openers.any()):
                opener_strafe_hold_frac = float(
                    (torch.maximum(opener_pos, opener_neg)[valid_openers]
                     / opener_count[valid_openers].clamp(min=1.0)).mean())
            else:
                opener_strafe_hold_frac = 0.0
            opener_pressure_mask = (
                opener_mask
                & (dist_metric >= 2.15)
                & (dist_metric <= 7.85)
                & ~(buf.obs[:, lm, 22] > buf.obs[:, lm, 21] + 0.05)
                & ~(buf.obs[:, lm, 21] > buf.obs[:, lm, 22] + 0.05)
            )
            opener_pressure_frac = float(
                (((fwd_dir_metric > 0.5)
                  & (strafe_dir_metric.abs() > 0.5)
                  & opener_pressure_mask).float().sum()
                 / opener_pressure_mask.float().sum().clamp(min=1.0)))
        sprint_frac = float(sprint_on.float().mean())
        attack_frac = float(attack_on.float().mean())
        combo_adv = (buf.obs[:, lm, 22] > buf.obs[:, lm, 21] + 0.05)
        close_combo_band = dist_metric < 3.05
        s_tap = (fwd_dir_metric < -0.5) & (~sprint_on)
        z_tap = (fwd_dir_metric.abs() <= 0.5) & (~sprint_on)
        combo_s_tap = s_tap & combo_adv & close_combo_band
        combo_z_tap = z_tap & combo_adv & close_combo_band
        combo_tap = combo_s_tap | combo_z_tap
        combo_tap_count = (combo_adv & close_combo_band).float().sum().clamp(min=1.0)
        s_tap_frac = float(s_tap.float().mean())
        z_tap_frac = float(z_tap.float().mean())
        combo_tap_frac = float(combo_tap.float().sum() / combo_tap_count)
        combo_s_tap_frac = float(combo_s_tap.float().sum() / combo_tap_count)
        combo_z_tap_frac = float(combo_z_tap.float().sum() / combo_tap_count)
        under_combo = (buf.obs[:, lm, 21] > buf.obs[:, lm, 22] + 0.05)
        hit_wtap_denom = hits_mask & (~under_combo)
        hit_wtap = hit_wtap_denom & z_tap & (strafe_dir_metric.abs() > 0.5)
        hit_wtap_frac = float(
            hit_wtap.float().sum()
            / hit_wtap_denom.float().sum().clamp(min=1.0))
        own_hurt_metric = buf.obs[:, lm, 21].clamp(0.0, 1.0)
        opp_cooldown_metric = buf.obs[:, lm, 37].clamp(0.0, 1.0)
        under_combo_count = under_combo.float().sum().clamp(min=1.0)
        under_combo_frac = float(under_combo.float().mean())
        under_combo_attack_frac = float(
            ((attack_on & under_combo).float().sum() / under_combo_count))
        under_combo_counter_hits = hits_mask & under_combo
        under_combo_counter_hit_rate = float(
            under_combo_counter_hits.float().sum() / under_combo_count) * 1200.0
        under_combo_counter_hit_frac = float(
            under_combo_counter_hits.float().sum()
            / hits_mask.float().sum().clamp(min=1.0))
        hit_select_window = (
            under_combo
            & (dist_metric >= COUNTER_HIT_SELECT_CLEAN_MIN_REACH)
            & (dist_metric <= COUNTER_HIT_SELECT_CLEAN_MAX_REACH)
            & (own_hurt_metric >= COUNTER_HIT_SELECT_MIN_OWN_HURT)
            & (own_hurt_metric <= COUNTER_HIT_SELECT_CLEAN_HURT)
            & (opp_cooldown_metric >= COUNTER_HIT_SELECT_OPP_COOLDOWN)
        )
        hit_select_count = hit_select_window.float().sum().clamp(min=1.0)
        under_combo_hit_select_attack_frac = float(
            ((attack_on & hit_select_window).float().sum() / hit_select_count))
        under_combo_hit_select_clean_frac = float(
            ((clean_hits_mask & hit_select_window).float().sum()
             / hit_select_count))
        under_combo_hit_select_trade_frac = float(
            (((hits_mask & taken_mask) & hit_select_window).float().sum()
             / hit_select_count))
        under_combo_escape_frac = float(
            ((back_mask & under_combo).float().sum() / under_combo_count))
        pitch_action_abs = float(torch.tanh(buf.pre[:, lm, 1]).abs().mean())
        yaw_action_abs = float(torch.tanh(buf.pre[:, lm, 0]).abs().mean())
        combo_chain = combo_stats_for(lm)
        all_combo_chain = combo_stats_for(lm_all)
        mirror_combo_chain = combo_stats_for(mirror_lm)
        bot_combo_chain = combo_stats_for(bot_lm)
        pad_combo_chain = combo_stats_for(pad_lm)
        spar_combo_chain = combo_stats_for(spar_lm)
        rehit_combo_chain = combo_stats_for(rehit_lm)
        pressure_combo_chain = combo_stats_for(pressure_lm)
        combo_chase_combo_chain = combo_stats_for(combo_chase_lm)
        counter_combo_chain = combo_stats_for(counter_lm)
        sim_combo_max, sim_combo12_state = sim_combo_stats_for(lm)
        mirror_sim_combo_max, mirror_sim_combo12_state = sim_combo_stats_for(mirror_lm)
        bot_sim_combo_max, bot_sim_combo12_state = sim_combo_stats_for(bot_lm)
        pad_sim_combo_max, pad_sim_combo12_state = sim_combo_stats_for(pad_lm)
        spar_sim_combo_max, spar_sim_combo12_state = sim_combo_stats_for(spar_lm)
        rehit_sim_combo_max, rehit_sim_combo12_state = sim_combo_stats_for(rehit_lm)
        pressure_sim_combo_max, pressure_sim_combo12_state = sim_combo_stats_for(pressure_lm)
        combo_chase_sim_combo_max, combo_chase_sim_combo12_state = (
            sim_combo_stats_for(combo_chase_lm))
        counter_sim_combo_max, counter_sim_combo12_state = (
            sim_combo_stats_for(counter_lm))
        combo_hit = combo_chain.get("combo2_hits", 0.0)
        combo12_state = max(
            combo_chain["combo12_hits"],
            mirror_combo_chain["combo12_hits"],
            sim_combo12_state,
            mirror_sim_combo12_state,
        )
        self._update_ramp(
            hit_rate, engage, combo_chain["combo5_hits"], sky_frac,
            combo12_state,
        )
        shaping = self._auto_shaping()
        spawn_gap = self._auto_curriculum()
        warn_entropy = self._entropy_guard(stats["entropy"])

        # ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©val automatique : snapshots passÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©s + bot scriptÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â© (ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©talon absolu)
        evals = {}
        if (self.cfg["eval_every"] > 0 and self.league.pool
                and self.iter % self.cfg["eval_every"] == 0):
            evals["eval_first"] = self._evaluate(self.league.pool[0]["state_dict"])
            if len(self.league.pool) > 1:
                evals["eval_past"] = self._evaluate(
                    self.league.pool[-1]["state_dict"])
            evals["eval_bot"] = self._evaluate("bot")
            if evals["eval_bot"] > self._best_bot:
                self._best_bot = evals["eval_bot"]
                if evals["eval_bot"] > self._disk_best_bot():
                    self._write_best_policy(self._best_bot)

        dt = time.perf_counter() - t0
        self.total_steps += self.T * self.N * 2
        best_elo = (max(m.elo for m in self.members) if self.members
                    else self.league.learner_elo)
        metrics = {
            "iter": self.iter,
            "total_steps": self.total_steps,
            "sps": round(self.T * self.N * 2 / dt),
            "reward_mean": float(buf.reward[:, lm].mean()),
            "elo": round(best_elo, 1),
            "pool_size": len(self.league.pool),
            "league_winrate": (ep["wins"] + 0.5 * ep["draws"])
                              / max(ep["wins"] + ep["losses"] + ep["draws"], 1),
            "matches": ep["matches"],
            "hit_rate": round(hit_rate, 2),
            "clean_hit_rate": round(clean_hit_rate, 2),
            "trade_hit_frac": round(trade_hit_frac, 4),
            "rechain_opps": rechain_stats["opps"],
            "rechain_hit_frac": round(rechain_stats["hit_frac"], 4),
            "rechain_taken_frac": round(rechain_stats["taken_frac"], 4),
            "counter_break_opps": counter_break_stats["opps"],
            "counter_break_hit_frac": round(counter_break_stats["hit_frac"], 4),
            "counter_break_taken_frac": round(counter_break_stats["taken_frac"], 4),
            "sprint_hits": round(sprint_hit, 4),
            "combo_hits": round(combo_hit, 4),
            "combo_max": combo_chain["combo_max"],
            "combo_mean": round(combo_chain["combo_mean"], 4),
            "combo5_hits": round(combo_chain["combo5_hits"], 4),
            "combo8_hits": round(combo_chain["combo8_hits"], 4),
            "combo12_hits": round(combo_chain["combo12_hits"], 4),
            "sim_combo_max": sim_combo_max,
            "sim_combo12_state": round(sim_combo12_state, 4),
            "all_combo_max": all_combo_chain["combo_max"],
            "all_combo12_hits": round(all_combo_chain["combo12_hits"], 4),
            "mirror_hit_rate": round(mirror_hit_rate, 2),
            "mirror_engage_rate": round(mirror_engage, 4),
            "mirror_combo_max": mirror_combo_chain["combo_max"],
            "mirror_combo12_hits": round(mirror_combo_chain["combo12_hits"], 4),
            "mirror_combo8_hits": round(mirror_combo_chain["combo8_hits"], 4),
            "mirror_sim_combo_max": mirror_sim_combo_max,
            "mirror_sim_combo12_state": round(mirror_sim_combo12_state, 4),
            "bot_hit_rate": round(bot_hit_rate, 2),
            "bot_engage_rate": round(bot_engage, 4),
            "bot_combo_max": bot_combo_chain["combo_max"],
            "bot_combo12_hits": round(bot_combo_chain["combo12_hits"], 4),
            "bot_sim_combo_max": bot_sim_combo_max,
            "bot_sim_combo12_state": round(bot_sim_combo12_state, 4),
            "spar_hit_rate": round(spar_hit_rate, 2),
            "spar_engage_rate": round(spar_engage, 4),
            "spar_combo_max": spar_combo_chain["combo_max"],
            "spar_combo12_hits": round(spar_combo_chain["combo12_hits"], 4),
            "spar_rechain_hit_frac": round(spar_rechain_stats["hit_frac"], 4),
            "spar_rechain_taken_frac": round(spar_rechain_stats["taken_frac"], 4),
            "spar_counter_break_hit_frac": round(spar_counter_break_stats["hit_frac"], 4),
            "spar_counter_break_taken_frac": round(spar_counter_break_stats["taken_frac"], 4),
            "spar_sim_combo_max": spar_sim_combo_max,
            "spar_sim_combo12_state": round(spar_sim_combo12_state, 4),
            "rehit_hit_rate": round(rehit_hit_rate, 2),
            "rehit_engage_rate": round(rehit_engage, 4),
            "rehit_combo_max": rehit_combo_chain["combo_max"],
            "rehit_combo12_hits": round(rehit_combo_chain["combo12_hits"], 4),
            "rehit_rechain_hit_frac": round(rehit_rechain_stats["hit_frac"], 4),
            "rehit_rechain_taken_frac": round(rehit_rechain_stats["taken_frac"], 4),
            "rehit_counter_break_hit_frac": round(rehit_counter_break_stats["hit_frac"], 4),
            "rehit_counter_break_taken_frac": round(rehit_counter_break_stats["taken_frac"], 4),
            "rehit_sim_combo_max": rehit_sim_combo_max,
            "rehit_sim_combo12_state": round(rehit_sim_combo12_state, 4),
            "pressure_hit_rate": round(pressure_hit_rate, 2),
            "pressure_engage_rate": round(pressure_engage, 4),
            "pressure_combo_max": pressure_combo_chain["combo_max"],
            "pressure_combo12_hits": round(pressure_combo_chain["combo12_hits"], 4),
            "pressure_rechain_hit_frac": round(pressure_rechain_stats["hit_frac"], 4),
            "pressure_rechain_taken_frac": round(pressure_rechain_stats["taken_frac"], 4),
            "pressure_counter_break_hit_frac": round(pressure_counter_break_stats["hit_frac"], 4),
            "pressure_counter_break_taken_frac": round(pressure_counter_break_stats["taken_frac"], 4),
            "pressure_sim_combo_max": pressure_sim_combo_max,
            "pressure_sim_combo12_state": round(pressure_sim_combo12_state, 4),
            "combo_chase_hit_rate": round(combo_chase_hit_rate, 2),
            "combo_chase_engage_rate": round(combo_chase_engage, 4),
            "combo_chase_combo_max": combo_chase_combo_chain["combo_max"],
            "combo_chase_combo12_hits": round(
                combo_chase_combo_chain["combo12_hits"], 4),
            "combo_chase_rechain_hit_frac": round(
                combo_chase_rechain_stats["hit_frac"], 4),
            "combo_chase_rechain_taken_frac": round(
                combo_chase_rechain_stats["taken_frac"], 4),
            "combo_chase_counter_break_hit_frac": round(
                combo_chase_counter_break_stats["hit_frac"], 4),
            "combo_chase_counter_break_taken_frac": round(
                combo_chase_counter_break_stats["taken_frac"], 4),
            "combo_chase_sim_combo_max": combo_chase_sim_combo_max,
            "combo_chase_sim_combo12_state": round(
                combo_chase_sim_combo12_state, 4),
            "counter_lane_hit_rate": round(counter_hit_rate, 2),
            "counter_lane_engage_rate": round(counter_engage, 4),
            "counter_lane_combo_max": counter_combo_chain["combo_max"],
            "counter_lane_combo12_hits": round(
                counter_combo_chain["combo12_hits"], 4),
            "counter_lane_rechain_hit_frac": round(
                counter_rechain_stats["hit_frac"], 4),
            "counter_lane_rechain_taken_frac": round(
                counter_rechain_stats["taken_frac"], 4),
            "counter_lane_break_hit_frac": round(
                counter_counter_break_stats["hit_frac"], 4),
            "counter_lane_break_taken_frac": round(
                counter_counter_break_stats["taken_frac"], 4),
            "counter_lane_sim_combo_max": counter_sim_combo_max,
            "counter_lane_sim_combo12_state": round(counter_sim_combo12_state, 4),
            "pad_hit_rate": round(pad_hit_rate, 2),
            "pad_engage_rate": round(pad_engage, 4),
            "pad_combo_max": pad_combo_chain["combo_max"],
            "pad_combo12_hits": round(pad_combo_chain["combo12_hits"], 4),
            "pad_sim_combo_max": pad_sim_combo_max,
            "pad_sim_combo12_state": round(pad_sim_combo12_state, 4),
            "engage_rate": round(engage, 4),
            "sky_frac": round(sky_frac, 4),
            "aim_body": round(aim_body, 4),
            "attack_frac": round(attack_frac, 4),
            "under_combo_frac": round(under_combo_frac, 4),
            "under_combo_attack_frac": round(under_combo_attack_frac, 4),
            "under_combo_counter_hit_rate": round(under_combo_counter_hit_rate, 2),
            "under_combo_counter_hit_frac": round(under_combo_counter_hit_frac, 4),
            "under_combo_hit_select_attack_frac": round(
                under_combo_hit_select_attack_frac, 4),
            "under_combo_hit_select_clean_frac": round(
                under_combo_hit_select_clean_frac, 4),
            "under_combo_hit_select_trade_frac": round(
                under_combo_hit_select_trade_frac, 4),
            "under_combo_escape_frac": round(under_combo_escape_frac, 4),
            "s_tap_frac": round(s_tap_frac, 4),
            "z_tap_frac": round(z_tap_frac, 4),
            "combo_tap_frac": round(combo_tap_frac, 4),
            "combo_s_tap_frac": round(combo_s_tap_frac, 4),
            "combo_z_tap_frac": round(combo_z_tap_frac, 4),
            "hit_wtap_frac": round(hit_wtap_frac, 4),
            "sprint_frac": round(sprint_frac, 4),
            "forward_frac": round(forward_frac, 4),
            "back_frac": round(back_frac, 4),
            "escape_back_frac": round(escape_back_frac, 4),
            "strafe_frac": round(strafe_frac, 4),
            "opener_samples": round(opener_samples, 4),
            "opener_strafe_frac": round(opener_strafe_frac, 4),
            "opener_strafe_hold_frac": round(opener_strafe_hold_frac, 4),
            "opener_pressure_frac": round(opener_pressure_frac, 4),
            "pitch_action_abs": round(pitch_action_abs, 4),
            "yaw_action_abs": round(yaw_action_abs, 4),
            "shaping": round(shaping, 6),
            "spawn_gap": round(spawn_gap, 2),
            "ramp": round(self._ramp_pos, 3),
            "warn_entropy": warn_entropy,
            **{k: round(v, 4) for k, v in evals.items()},
            **{k: _round_train_stat(k, v) for k, v in stats.items()},
            "time": round(dt, 2),
            "time_collect": round(t_collect, 2),
            "time_update": round(dt - t_collect, 2),
        }
        if self.members:
            metrics["pbt_best"] = int(max(range(len(self.members)),
                                          key=lambda i: self.members[i].elo))
            metrics["pbt_elo"] = [round(m.elo, 1) for m in self.members]
            metrics["pbt_lr"] = [round(m.hypers.get("lr", 0.0), 6)
                                 for m in self.members]
            metrics["pbt_ent"] = [round(m.hypers.get("ent_coef", 0.0), 5)
                                  for m in self.members]
        if force_combo_eval or self._should_combo_eval_now():
            metrics.update(self._evaluate_combo_fresh())
            self._fresh_combo_eval_count += 1
        safety_stop = self._checkpoint_safety_guard(metrics)
        self._log(metrics)
        if safety_stop:
            raise SystemExit(2)
        return metrics

    def _should_combo_eval_now(self) -> bool:
        combo_eval_every = int(self.cfg.get("combo_eval_every", 0) or 0)
        if combo_eval_every <= 0:
            return False
        elapsed = self.iter - self._combo_eval_origin
        return elapsed > 0 and elapsed % combo_eval_every == 0

    def _checkpoint_safety_guard(self, metrics: dict) -> bool:
        if not self.cfg.get("safety_stop_on_regression", False):
            return False
        reasons: list[str] = []
        promotion_reasons: list[str] = []
        promotion_blockers: set[str] = set()

        def add_promotion_reason(reason: str, checkpoint: str) -> None:
            promotion_reasons.append(reason)
            promotion_blockers.add(checkpoint)

        def add_counter_recovery_reason(metric_key: str, reason: str) -> None:
            if self._counter_recovery_failure_blocks_promotion(metric_key):
                add_promotion_reason(reason, "await_counter_recovery")
            else:
                reasons.append(reason)

        def add_hit_wtap_reason(metric_key: str, reason: str) -> None:
            if self._hit_wtap_failure_blocks_promotion(metric_key):
                add_promotion_reason(reason, "await_hit_wtap")
            else:
                reasons.append(reason)

        def max_check(metric_key: str, cfg_key: str) -> None:
            limit = float(self.cfg.get(cfg_key, 1.0e9))
            value = float(metrics.get(metric_key, 0.0))
            if value > limit:
                reasons.append(f"{metric_key}={value:.4g}>{limit:.4g}")

        def min_check(metric_key: str, cfg_key: str) -> None:
            limit = float(self.cfg.get(cfg_key, -1.0))
            value = float(metrics.get(metric_key, 0.0))
            if limit >= 0.0 and value < limit:
                reasons.append(f"{metric_key}={value:.4g}<{limit:.4g}")

        max_check("under_combo_escape_frac", "safety_under_combo_escape")
        max_check("back_frac", "safety_back_frac")
        if "escape_back_frac" in metrics:
            max_check("escape_back_frac", "safety_back_frac")
        min_check("strafe_frac", "safety_min_strafe_frac")
        if float(metrics.get("opener_samples", 1.0) or 0.0) > 0.0:
            min_check("opener_strafe_frac", "safety_min_opener_strafe_frac")
            min_check("opener_strafe_hold_frac", "safety_min_opener_strafe_hold_frac")
            min_check("opener_pressure_frac", "safety_min_opener_pressure_frac")
        min_combo_tap = float(self.cfg.get("safety_min_combo_tap_frac", -1.0))
        if min_combo_tap >= 0.0:
            for key, value in self._safety_combo_tap_values(metrics):
                if value < min_combo_tap:
                    reasons.append(f"{key}={value:.4g}<{min_combo_tap:.4g}")
        min_combo_z_tap = float(
            self.cfg.get("safety_min_combo_z_tap_frac", -1.0))
        if min_combo_z_tap >= 0.0:
            for key, value in self._safety_combo_component_values(metrics, "z"):
                if value < min_combo_z_tap:
                    reasons.append(f"{key}={value:.4g}<{min_combo_z_tap:.4g}")
        max_combo_s_tap = float(
            self.cfg.get("safety_max_combo_s_tap_frac", -1.0))
        if max_combo_s_tap >= 0.0:
            for key, value in self._safety_combo_component_values(metrics, "s"):
                if value > max_combo_s_tap:
                    reasons.append(f"{key}={value:.4g}>{max_combo_s_tap:.4g}")
        min_hit_wtap = float(self.cfg.get("safety_min_hit_wtap_frac", -1.0))
        rollout_hit_wtap_slack = float(
            self.cfg.get("safety_rollout_hit_wtap_slack", 0.0) or 0.0)
        if min_hit_wtap >= 0.0:
            for key, value in self._safety_hit_wtap_values(metrics):
                limit = min_hit_wtap
                if key == "fresh_chase_hit_wtap_frac":
                    limit = float(self.cfg.get(
                        "safety_min_chase_hit_wtap_frac", limit))
                if (
                    key == "hit_wtap_frac"
                    and rollout_hit_wtap_slack > 0.0
                    and float(metrics.get("opener_samples", 0.0) or 0.0) > 0.0
                ):
                    limit = max(0.0, min_hit_wtap - rollout_hit_wtap_slack)
                if value < limit:
                    add_hit_wtap_reason(key, f"{key}={value:.4g}<{limit:.4g}")
        min_under_counter = float(
            self.cfg.get("safety_min_under_combo_counter_hit_frac", -1.0))
        if min_under_counter >= 0.0:
            for key, value in self._safety_under_combo_counter_values(metrics):
                if (
                    value < min_under_counter
                    and not self._under_combo_avoidance_gate_passes(
                        metrics, key)
                ):
                    add_counter_recovery_reason(
                        key, f"{key}={value:.4g}<{min_under_counter:.4g}")
        min_hit_select_clean = float(
            self.cfg.get("safety_min_under_combo_hit_select_clean_frac", -1.0))
        if min_hit_select_clean >= 0.0:
            for key, value in self._safety_under_combo_hit_select_values(
                metrics, "clean"):
                if (
                    value < min_hit_select_clean
                    and not self._under_combo_hit_select_exposure_gate_passes(
                        metrics, key)
                ):
                    add_counter_recovery_reason(
                        key, f"{key}={value:.4g}<{min_hit_select_clean:.4g}")
        max_hit_select_trade = float(
            self.cfg.get("safety_max_under_combo_hit_select_trade_frac", -1.0))
        if max_hit_select_trade >= 0.0:
            for key, value in self._safety_under_combo_hit_select_values(
                metrics, "trade"):
                if (
                    value > max_hit_select_trade
                    and not self._under_combo_hit_select_exposure_gate_passes(
                        metrics, key)
                ):
                    add_counter_recovery_reason(
                        key, f"{key}={value:.4g}>{max_hit_select_trade:.4g}")
        max_check("strafe_frac", "safety_strafe_frac")
        sky_metric = "fresh_sky_frac" if "fresh_sky_frac" in metrics else "sky_frac"
        max_check(sky_metric, "safety_sky_frac")
        min_hit_rate = float(self.cfg.get("safety_min_hit_rate", -1.0))
        hit_rate = float(metrics.get("hit_rate", 0.0))
        if min_hit_rate >= 0.0 and hit_rate < min_hit_rate:
            reasons.append(f"hit_rate={hit_rate:.4g}<{min_hit_rate:.4g}")
        fresh_min_hit_rate = float(self.cfg.get("safety_fresh_min_hit_rate", -1.0))
        has_active_fresh = any(k in metrics for k in (
            "fresh_spar_hit_rate", "fresh_chase_hit_rate"))
        if ("fresh_hit_rate" in metrics and fresh_min_hit_rate >= 0.0
                and not has_active_fresh):
            fresh_hit_rate = float(metrics.get("fresh_hit_rate", 0.0))
            if fresh_hit_rate < fresh_min_hit_rate:
                reasons.append(
                    f"fresh_hit_rate={fresh_hit_rate:.4g}<{fresh_min_hit_rate:.4g}")
        for active_key in ("fresh_spar_hit_rate", "fresh_chase_hit_rate"):
            if active_key not in metrics or fresh_min_hit_rate < 0.0:
                continue
            active_hit_rate = float(metrics.get(active_key, 0.0))
            if active_hit_rate < fresh_min_hit_rate:
                reasons.append(
                    f"{active_key}={active_hit_rate:.4g}"
                    f"<{fresh_min_hit_rate:.4g}")
        if not reasons:
            if promotion_reasons:
                metrics["safety_state"] = "training"
                if "await_counter_recovery" in promotion_blockers:
                    checkpoint = "await_counter_recovery"
                elif "await_hit_wtap" in promotion_blockers:
                    checkpoint = "await_hit_wtap"
                else:
                    checkpoint = "await_promotion"
                metrics["safety_checkpoint"] = checkpoint
                metrics["safety_promotion_reason"] = ";".join(promotion_reasons)
                return False
            metrics["safety_state"] = "safe"
            combo_eval_every = int(self.cfg.get("combo_eval_every", 0) or 0)
            if combo_eval_every > 0 and "fresh_combo12_state" not in metrics:
                metrics["safety_checkpoint"] = "await_fresh_eval"
                return False
            self._mark_safe_checkpoint(metrics)
            return False

        metrics["safety_state"] = "stop"
        metrics["safety_reason"] = ";".join(reasons)
        restored = None
        previous = self._load_safe_combo_score()
        promote_min_combo = float(
            self.cfg.get("safety_promote_min_combo_max", -1.0) or -1.0)
        if previous is not None and (
                promote_min_combo < 0.0 or previous[3] >= promote_min_combo):
            restored = self._restore_safe_checkpoint()
            metrics["safety_restored"] = restored or "missing"
        else:
            metrics["safety_restored"] = "skipped_invalid_safe"
        bad = self.run_dir / f"ckpt_{int(metrics.get('iter', self.iter)):06d}.pt"
        if bad.exists() and (restored is None or bad.name != restored):
            if restored is not None:
                bad.unlink(missing_ok=True)
        print(f"[safety] stop regression: {metrics['safety_reason']} restored={metrics['safety_restored']}")
        return True

    @staticmethod
    def _metrics_max(metrics: dict, keys: tuple[str, ...]) -> float:
        values: list[float] = []
        for key in keys:
            value = metrics.get(key)
            if value is None:
                continue
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue
        return max(values) if values else 0.0

    @staticmethod
    def _metric_float(metrics: dict, key: str) -> float | None:
        value = metrics.get(key)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _combo_tap_metric(cls, metrics: dict, prefix: str) -> float:
        stem = f"{prefix}_" if prefix else ""
        total = cls._metric_float(metrics, f"{stem}combo_tap_frac")
        if total is not None:
            return total
        s_tap = cls._metric_float(metrics, f"{stem}combo_s_tap_frac")
        z_tap = cls._metric_float(metrics, f"{stem}combo_z_tap_frac")
        if s_tap is not None and z_tap is not None:
            return min(1.0, max(0.0, s_tap) + max(0.0, z_tap))
        if z_tap is not None:
            return z_tap
        if s_tap is not None:
            return s_tap
        return 0.0

    @classmethod
    def _combo_tap_component_metric(
        cls, metrics: dict, prefix: str, component: str
    ) -> float:
        stem = f"{prefix}_" if prefix else ""
        value = cls._metric_float(metrics, f"{stem}combo_{component}_tap_frac")
        return value if value is not None else 0.0

    @classmethod
    def _has_combo_tap_metric(cls, metrics: dict, prefix: str) -> bool:
        stem = f"{prefix}_" if prefix else ""
        return any(key in metrics for key in (
            f"{stem}combo_tap_frac",
            f"{stem}combo_s_tap_frac",
            f"{stem}combo_z_tap_frac",
        ))

    @classmethod
    def _has_hit_wtap_metric(cls, metrics: dict, prefix: str) -> bool:
        stem = f"{prefix}_" if prefix else ""
        return f"{stem}hit_wtap_frac" in metrics

    @classmethod
    def _hit_wtap_metric(cls, metrics: dict, prefix: str) -> float:
        stem = f"{prefix}_" if prefix else ""
        value = cls._metric_float(metrics, f"{stem}hit_wtap_frac")
        return value if value is not None else 0.0

    def _safety_combo_tap_values(self, metrics: dict) -> list[tuple[str, float]]:
        active = [
            prefix for prefix in ("fresh_spar", "fresh_chase")
            if self._has_combo_tap_metric(metrics, prefix)
        ]
        if active:
            return [
                (f"{prefix}_combo_tap_frac", self._combo_tap_metric(metrics, prefix))
                for prefix in active
            ]
        if self._has_combo_tap_metric(metrics, "fresh"):
            return [("fresh_combo_tap_frac", self._combo_tap_metric(metrics, "fresh"))]
        combo_eval_every = int(self.cfg.get("combo_eval_every", 0) or 0)
        if combo_eval_every <= 0 or self._has_combo_tap_metric(metrics, ""):
            return [("combo_tap_frac", self._combo_tap_metric(metrics, ""))]
        return []

    def _safety_combo_component_values(
        self, metrics: dict, component: str
    ) -> list[tuple[str, float]]:
        active = [
            prefix for prefix in ("fresh_spar", "fresh_chase")
            if self._has_combo_tap_metric(metrics, prefix)
        ]
        if active:
            return [
                (
                    f"{prefix}_combo_{component}_tap_frac",
                    self._combo_tap_component_metric(metrics, prefix, component),
                )
                for prefix in active
            ]
        if self._has_combo_tap_metric(metrics, "fresh"):
            return [
                (
                    f"fresh_combo_{component}_tap_frac",
                    self._combo_tap_component_metric(metrics, "fresh", component),
                )
            ]
        combo_eval_every = int(self.cfg.get("combo_eval_every", 0) or 0)
        if combo_eval_every <= 0 or self._has_combo_tap_metric(metrics, ""):
            return [
                (
                    f"combo_{component}_tap_frac",
                    self._combo_tap_component_metric(metrics, "", component),
                )
            ]
        return []

    def _safety_hit_wtap_values(self, metrics: dict) -> list[tuple[str, float]]:
        active = [
            prefix for prefix in ("fresh_spar", "fresh_chase")
            if self._has_hit_wtap_metric(metrics, prefix)
        ]
        if active:
            return [
                (f"{prefix}_hit_wtap_frac", self._hit_wtap_metric(metrics, prefix))
                for prefix in active
            ]
        if self._has_hit_wtap_metric(metrics, "fresh"):
            return [("fresh_hit_wtap_frac", self._hit_wtap_metric(metrics, "fresh"))]
        combo_eval_every = int(self.cfg.get("combo_eval_every", 0) or 0)
        has_rollout = self._has_hit_wtap_metric(metrics, "")
        if combo_eval_every <= 0 and has_rollout:
            return [("hit_wtap_frac", self._hit_wtap_metric(metrics, ""))]
        if (
            has_rollout
            and float(metrics.get("opener_samples", 0.0) or 0.0) > 0.0
        ):
            return [("hit_wtap_frac", self._hit_wtap_metric(metrics, ""))]
        return []

    @classmethod
    def _has_under_combo_counter_metric(cls, metrics: dict, prefix: str) -> bool:
        stem = f"{prefix}_" if prefix else ""
        return f"{stem}under_combo_counter_hit_frac" in metrics

    @classmethod
    def _has_under_combo_hit_select_metric(
        cls, metrics: dict, prefix: str, kind: str
    ) -> bool:
        stem = f"{prefix}_" if prefix else ""
        return f"{stem}under_combo_hit_select_{kind}_frac" in metrics

    @classmethod
    def _under_combo_counter_metric(cls, metrics: dict, prefix: str) -> float:
        stem = f"{prefix}_" if prefix else ""
        value = cls._metric_float(metrics, f"{stem}under_combo_counter_hit_frac")
        return value if value is not None else 0.0

    @classmethod
    def _under_combo_hit_select_metric(
        cls, metrics: dict, prefix: str, kind: str
    ) -> float:
        stem = f"{prefix}_" if prefix else ""
        value = cls._metric_float(
            metrics, f"{stem}under_combo_hit_select_{kind}_frac")
        return value if value is not None else 0.0

    def _fresh_under_combo_prefixes(self, metrics: dict) -> tuple[str, ...]:
        prefixes = ["fresh_spar", "fresh_chase"]
        if self.cfg.get("safety_require_counter_recovery", False):
            prefixes.append("fresh_counter")
        return tuple(prefix for prefix in prefixes if any(
            key.startswith(f"{prefix}_under_combo_") for key in metrics))

    def _safety_under_combo_counter_values(
        self, metrics: dict
    ) -> list[tuple[str, float]]:
        active = [
            prefix for prefix in self._fresh_under_combo_prefixes(metrics)
            if self._has_under_combo_counter_metric(metrics, prefix)
        ]
        if active:
            return [
                (
                    f"{prefix}_under_combo_counter_hit_frac",
                    self._under_combo_counter_metric(metrics, prefix),
                )
                for prefix in active
            ]
        if self._has_under_combo_counter_metric(metrics, "fresh"):
            return [
                (
                    "fresh_under_combo_counter_hit_frac",
                    self._under_combo_counter_metric(metrics, "fresh"),
                )
            ]
        if self._has_under_combo_counter_metric(metrics, ""):
            return [
                (
                    "under_combo_counter_hit_frac",
                    self._under_combo_counter_metric(metrics, ""),
                )
            ]
        return []

    def _safety_under_combo_hit_select_values(
        self, metrics: dict, kind: str
    ) -> list[tuple[str, float]]:
        # Hit-select cleanliness is only stable on bounded fresh eval lanes.
        # Rollout values are policy-update samples and can be noisy at iter 1,
        # which would stop a resumed safe model before it gets a real spar/chase
        # recovery evaluation.
        active = [
            prefix for prefix in self._fresh_under_combo_prefixes(metrics)
            if self._has_under_combo_hit_select_metric(metrics, prefix, kind)
        ]
        if active:
            return [
                (
                    f"{prefix}_under_combo_hit_select_{kind}_frac",
                    self._under_combo_hit_select_metric(metrics, prefix, kind),
                )
                for prefix in active
            ]
        if self._has_under_combo_hit_select_metric(metrics, "fresh", kind):
            return [
                (
                    f"fresh_under_combo_hit_select_{kind}_frac",
                    self._under_combo_hit_select_metric(metrics, "fresh", kind),
                )
            ]
        return []

    @staticmethod
    def _counter_metric_prefix(key: str) -> str:
        suffix = "_under_combo_counter_hit_frac"
        if key.endswith(suffix):
            return key[:-len(suffix)]
        if key == "under_combo_counter_hit_frac":
            return ""
        return ""

    @staticmethod
    def _hit_select_metric_prefix(key: str) -> str:
        marker = "_under_combo_hit_select_"
        if marker in key:
            return key.split(marker, 1)[0]
        return ""

    def _counter_recovery_failure_blocks_promotion(self, metric_key: str) -> bool:
        if not self.cfg.get("safety_require_counter_recovery", False):
            return False
        if not metric_key.startswith("fresh_"):
            return False
        return (
            metric_key.endswith("_under_combo_counter_hit_frac")
            or "_under_combo_hit_select_" in metric_key
        )

    def _hit_wtap_failure_blocks_promotion(self, metric_key: str) -> bool:
        if not self.cfg.get("safety_hit_wtap_blocks_promotion", False):
            return False
        return metric_key == "hit_wtap_frac" or (
            metric_key.startswith("fresh_")
            and metric_key.endswith("_hit_wtap_frac")
        )

    @staticmethod
    def _prefix_metric_key(prefix: str, metric: str) -> str:
        return f"{prefix}_{metric}" if prefix else metric

    def _under_combo_avoidance_gate_passes(self, metrics: dict, key: str) -> bool:
        prefix = self._counter_metric_prefix(key)
        return self._under_combo_exposure_gate_passes(
            metrics, prefix, "under_combo_counter_avoidance_gate")

    def _under_combo_hit_select_exposure_gate_passes(
        self, metrics: dict, key: str
    ) -> bool:
        prefix = self._hit_select_metric_prefix(key)
        return self._under_combo_exposure_gate_passes(
            metrics, prefix, "under_combo_hit_select_exposure_gate")

    def _under_combo_exposure_gate_passes(
        self, metrics: dict, prefix: str, marker_metric: str
    ) -> bool:
        allowed_prefixes = ("fresh_spar", "fresh_chase")
        if marker_metric in (
            "under_combo_counter_avoidance_gate",
            "under_combo_hit_select_exposure_gate",
        ):
            allowed_prefixes = (*allowed_prefixes, "fresh_counter")
        if prefix not in allowed_prefixes:
            return False
        avoid_limit = float(self.cfg.get("safety_under_combo_avoid_frac", -1.0))
        if avoid_limit < 0.0:
            return False
        under_frac = self._metric_float(
            metrics, self._prefix_metric_key(prefix, "under_combo_frac"))
        if under_frac is None or under_frac > avoid_limit:
            return False
        min_combo12 = float(
            self.cfg.get("safety_under_combo_avoid_min_combo12", -1.0))
        if min_combo12 >= 0.0:
            combo12 = self._metric_float(
                metrics, self._prefix_metric_key(prefix, "combo12_state"))
            if combo12 is None or combo12 < min_combo12:
                return False
        min_hit_rate = float(
            self.cfg.get("safety_under_combo_avoid_min_hit_rate", -1.0))
        if min_hit_rate >= 0.0:
            hit_rate = self._metric_float(
                metrics, self._prefix_metric_key(prefix, "hit_rate"))
            if hit_rate is None or hit_rate < min_hit_rate:
                return False
        metrics[self._prefix_metric_key(
            prefix, marker_metric)] = 1.0
        return True

    def _under_combo_avoidance_score_bonus(
        self, under_combo_frac: float | None, combo_max: float
    ) -> float:
        if under_combo_frac is None or combo_max < 8.0:
            return 0.0
        target = float(self.cfg.get("score_under_combo_avoid_target", -1.0))
        weight = float(self.cfg.get("score_under_combo_avoid_weight", 0.0))
        cap = float(self.cfg.get("score_under_combo_avoid_cap", 0.0))
        if target <= 0.0 or weight <= 0.0 or cap <= 0.0:
            return 0.0
        return min(cap, max(0.0, target - under_combo_frac) * weight)

    def _safe_score_under_combo_frac(self, metrics: dict) -> float | None:
        if "fresh_spar_under_combo_frac" in metrics:
            values = [float(metrics.get("fresh_spar_under_combo_frac", 1.0) or 0.0)]
            if self.cfg.get("safety_require_chase_combo", False):
                values.append(float(
                    metrics.get("fresh_chase_under_combo_frac", 1.0) or 0.0))
            if self.cfg.get("safety_require_counter_recovery", False):
                values.append(float(
                    metrics.get("fresh_counter_under_combo_frac", 1.0) or 0.0))
            return max(values)
        if "fresh_under_combo_frac" in metrics:
            values = [float(metrics.get("fresh_under_combo_frac", 1.0) or 0.0)]
            if (self.cfg.get("safety_require_chase_combo", False)
                    and "fresh_chase_under_combo_frac" in metrics):
                values.append(float(
                    metrics.get("fresh_chase_under_combo_frac", 1.0) or 0.0))
            if (self.cfg.get("safety_require_counter_recovery", False)
                    and "fresh_counter_under_combo_frac" in metrics):
                values.append(float(
                    metrics.get("fresh_counter_under_combo_frac", 1.0) or 0.0))
            return max(values)
        if "under_combo_frac" in metrics:
            return float(metrics.get("under_combo_frac", 1.0) or 0.0)
        return None

    def _safe_combo_score(self, metrics: dict) -> tuple[float, ...]:
        if "fresh_spar_combo_max" in metrics:
            combo12 = float(metrics.get("fresh_spar_combo12_state", 0.0) or 0.0)
            combo_max = float(metrics.get("fresh_spar_combo_max", 0.0) or 0.0)
            combo8 = float(metrics.get("fresh_spar_combo8_state", 0.0) or 0.0)
            hit_rate = float(metrics.get("fresh_spar_hit_rate", 0.0) or 0.0)
            sky_frac = float(metrics.get("fresh_spar_sky_frac", 0.0) or 0.0)
            combo_tap = self._combo_tap_metric(metrics, "fresh_spar")
            counter_hit = float(metrics.get(
                "fresh_spar_under_combo_counter_hit_frac", 0.0) or 0.0)
            if self.cfg.get("safety_require_chase_combo", False):
                combo12 = min(combo12, float(
                    metrics.get("fresh_chase_combo12_state", 0.0) or 0.0))
                combo_max = min(combo_max, float(
                    metrics.get("fresh_chase_combo_max", 0.0) or 0.0))
                combo8 = min(combo8, float(
                    metrics.get("fresh_chase_combo8_state", 0.0) or 0.0))
                hit_rate = min(hit_rate, float(
                    metrics.get("fresh_chase_hit_rate", 0.0) or 0.0))
                sky_frac = max(sky_frac, float(
                    metrics.get("fresh_chase_sky_frac", 0.0) or 0.0))
                combo_tap = min(
                    combo_tap, self._combo_tap_metric(metrics, "fresh_chase"))
                counter_hit = min(counter_hit, float(metrics.get(
                    "fresh_chase_under_combo_counter_hit_frac", 0.0) or 0.0))
            if self.cfg.get("safety_require_counter_recovery", False):
                counter_hit = min(counter_hit, float(metrics.get(
                    "fresh_counter_under_combo_counter_hit_frac", 0.0) or 0.0))
        elif "fresh_combo12_state" in metrics:
            combo12 = float(metrics.get("fresh_combo12_state", 0.0) or 0.0)
            combo_max = float(metrics.get("fresh_combo_max", 0.0) or 0.0)
            combo8 = float(metrics.get("fresh_combo8_state", 0.0) or 0.0)
            mirror_combo12 = self._metrics_max(metrics, (
                "mirror_combo12_hits", "mirror_sim_combo12_state"))
            mirror_combo_max = self._metrics_max(metrics, (
                "mirror_combo_max", "mirror_sim_combo_max"))
            mirror_combo8 = self._metrics_max(metrics, (
                "mirror_combo8_hits",))
            has_mirror_combo = (
                float(metrics.get("mirror_hit_rate", 0.0) or 0.0) > 0.0
                or float(metrics.get("mirror_engage_rate", 0.0) or 0.0) > 0.0
                or mirror_combo12 > 0.0
                or mirror_combo_max > 0.0
                or mirror_combo8 > 0.0
            )
            if has_mirror_combo:
                combo12 = min(combo12, mirror_combo12)
                combo_max = min(combo_max, mirror_combo_max)
                combo8 = min(combo8, mirror_combo8)
            hit_rate = float(metrics.get("fresh_hit_rate", 0.0) or 0.0)
            sky_frac = float(metrics.get("fresh_sky_frac", 0.0) or 0.0)
            combo_tap = self._combo_tap_metric(metrics, "fresh")
            counter_hit = float(metrics.get(
                "fresh_under_combo_counter_hit_frac", 0.0) or 0.0)
            if (self.cfg.get("safety_require_chase_combo", False)
                    and "fresh_chase_combo_max" in metrics):
                combo12 = min(combo12, float(
                    metrics.get("fresh_chase_combo12_state", 0.0) or 0.0))
                combo_max = min(combo_max, float(
                    metrics.get("fresh_chase_combo_max", 0.0) or 0.0))
                combo8 = min(combo8, float(
                    metrics.get("fresh_chase_combo8_state", 0.0) or 0.0))
                hit_rate = min(hit_rate, float(
                    metrics.get("fresh_chase_hit_rate", 0.0) or 0.0))
                sky_frac = max(sky_frac, float(
                    metrics.get("fresh_chase_sky_frac", 0.0) or 0.0))
                combo_tap = min(
                    combo_tap, self._combo_tap_metric(metrics, "fresh_chase"))
                counter_hit = min(counter_hit, float(metrics.get(
                    "fresh_chase_under_combo_counter_hit_frac", 0.0) or 0.0))
            if self.cfg.get("safety_require_counter_recovery", False):
                counter_hit = min(counter_hit, float(metrics.get(
                    "fresh_counter_under_combo_counter_hit_frac", 0.0) or 0.0))
        else:
            combo12 = self._metrics_max(metrics, (
                "combo12_hits", "sim_combo12_state",
                "mirror_combo12_hits", "mirror_sim_combo12_state"))
            combo_max = self._metrics_max(metrics, (
                "combo_max", "sim_combo_max",
                "mirror_combo_max", "mirror_sim_combo_max"))
            combo8 = self._metrics_max(metrics, ("combo8_hits", "mirror_combo8_hits"))
            hit_rate = float(metrics.get("hit_rate", 0.0) or 0.0)
            sky_frac = float(metrics.get("sky_frac", 0.0) or 0.0)
            combo_tap = self._combo_tap_metric(metrics, "")
            counter_hit = float(metrics.get("under_combo_counter_hit_frac", 0.0) or 0.0)
        style_bonus = 0.0
        if combo_max >= 8.0:
            style_bonus = 0.02 * combo_tap + 0.01 * counter_hit
            clean_values = [
                value for _key, value in self._safety_under_combo_hit_select_values(
                    metrics, "clean")
            ]
            trade_values = [
                value for _key, value in self._safety_under_combo_hit_select_values(
                    metrics, "trade")
            ]
            if clean_values:
                style_bonus += 0.015 * min(clean_values)
            if trade_values:
                style_bonus -= 0.015 * max(trade_values)
        avoid_bonus = self._under_combo_avoidance_score_bonus(
            self._safe_score_under_combo_frac(metrics), combo_max)
        combo_progress = combo12 + 0.25 * combo8 + 0.01 * min(combo_max, 20.0)
        style_score = combo_progress + style_bonus + avoid_bonus
        return (round(style_score, 6), round(combo12, 6), round(combo8, 6),
                round(combo_max, 6), round(combo_tap, 6),
                round(counter_hit, 6), round(hit_rate, 6),
                round(-sky_frac, 6))

    def _safe_meta_path(self) -> Path:
        return self.run_dir / "safe_latest.meta.json"

    def _safe_record_counter_contract_violated(
        self, data: dict, score: list | tuple | None
    ) -> bool:
        if not self.cfg.get("safety_require_counter_recovery", False):
            return False

        def value_from_meta(key: str, fallback_index: int | None = None) -> float | None:
            value = self._metric_float(data, key)
            if value is not None:
                return value
            if (
                fallback_index is not None
                and isinstance(score, (list, tuple))
                and len(score) > fallback_index
            ):
                try:
                    return float(score[fallback_index])
                except (TypeError, ValueError):
                    return None
            return None

        min_counter = float(
            self.cfg.get("safety_min_under_combo_counter_hit_frac", -1.0))
        if min_counter >= 0.0:
            value = value_from_meta("under_combo_counter_hit_frac", 5)
            avoidance_bonus = value_from_meta("under_combo_avoidance_score_bonus")
            if (
                value is None
                or (
                    value < min_counter
                    and float(avoidance_bonus or 0.0) <= 0.0
                )
            ):
                return True

        min_clean = float(
            self.cfg.get("safety_min_under_combo_hit_select_clean_frac", -1.0))
        if min_clean >= 0.0:
            value = value_from_meta("under_combo_hit_select_clean_frac")
            if value is None or value < min_clean:
                return True

        max_trade = float(
            self.cfg.get("safety_max_under_combo_hit_select_trade_frac", -1.0))
        if max_trade >= 0.0:
            value = value_from_meta("under_combo_hit_select_trade_frac")
            if value is None or value > max_trade:
                return True

        min_hit_wtap = float(self.cfg.get("safety_min_hit_wtap_frac", -1.0))
        if min_hit_wtap >= 0.0:
            value = value_from_meta("hit_wtap_frac")
            if value is None or value < min_hit_wtap:
                return True

        return False

    def _load_safe_combo_record(self) -> tuple[tuple[float, ...], bool, bool] | None:
        path = self._safe_meta_path()
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            requires_chase = bool(data.get("requires_chase_combo", False))
            requires_counter = bool(data.get("requires_counter_recovery", False))
            score = data.get("score")
            schema = int(data.get("score_schema", 1) or 1)
            if isinstance(score, list) and len(score) >= 8 and schema >= 8:
                if self._safe_record_counter_contract_violated(data, score):
                    hit_rate = float(score[6])
                    neg_sky = float(score[7])
                    return (
                        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, hit_rate, neg_sky),
                        False,
                        False,
                    )
                return tuple(float(v) for v in score), requires_chase, requires_counter
            if isinstance(score, list) and len(score) >= 8 and schema >= 7:
                # Schema 7 was scored before ComboChaseBot respected the
                # lane-specific re-hit window. Its chase component is not
                # comparable with the current curriculum, so keep only a weak
                # hit-rate/sky baseline instead of blocking new chase safes.
                hit_rate = float(score[6])
                neg_sky = float(score[7])
                return (
                    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, hit_rate, neg_sky),
                    False,
                    False,
                )
            if isinstance(score, list) and len(score) >= 8 and schema >= 2:
                # Older schemas predated hit W-tap and active-opponent combo
                # and under-combo counter gating. Treat them as weak baselines so style-blind
                # safes cannot block promotion.
                hit_rate = float(score[6])
                neg_sky = float(score[7])
                return (
                    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, hit_rate, neg_sky),
                    False,
                    False,
                )
            if isinstance(score, list) and len(score) >= 8:
                _old_style, combo12, combo_tap, counter_hit, combo_max, combo8, hit_rate, neg_sky = (
                    float(v) for v in score[:8])
                combo12 = max(combo12, 0.0)
                combo8 = max(combo8, 0.0)
                combo_max = max(combo_max, 0.0)
                style_bonus = 0.0
                if combo_max >= 8.0:
                    style_bonus = 0.02 * combo_tap + 0.01 * counter_hit
                combo_progress = combo12 + 0.25 * combo8 + 0.01 * min(combo_max, 20.0)
                return (
                    (round(combo_progress + style_bonus, 6),
                     round(combo12, 6), round(combo8, 6),
                     round(combo_max, 6), round(combo_tap, 6),
                     round(counter_hit, 6), round(hit_rate, 6),
                     round(neg_sky, 6)),
                    False,
                    False,
                )
            if isinstance(score, list) and len(score) == 5:
                combo12, combo_max, combo8, hit_rate, neg_sky = (
                    float(v) for v in score)
                # Legacy 5-field metadata came from fresh/pad combo eval before
                # mirror combo was part of the safety contract. Do not let it
                # block mirror-combo candidates as a fake best score.
                return (
                    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, hit_rate, neg_sky),
                    False,
                    False,
                )
        except (OSError, ValueError, TypeError):
            return None
        return None

    def _load_safe_combo_score(self) -> tuple[float, ...] | None:
        record = self._load_safe_combo_record()
        if record is None:
            return None
        return record[0]

    def _safe_meta_combo_component(
        self, metrics: dict, component: str, reducer
    ) -> float:
        values = [
            value
            for _key, value in self._safety_combo_component_values(
                metrics, component)
        ]
        return float(reducer(values)) if values else 0.0

    def _safe_meta_hit_wtap(self, metrics: dict) -> float:
        values = [value for _key, value in self._safety_hit_wtap_values(metrics)]
        return float(min(values)) if values else 0.0

    def _safe_meta_under_combo_counter(
        self, metrics: dict, score: tuple[float, ...]
    ) -> tuple[float, str]:
        values = self._safety_under_combo_counter_values(metrics)
        if values:
            key, value = min(values, key=lambda item: item[1])
            return float(value), key
        return float(score[5]), "score"

    def _safe_meta_under_combo_frac(self, metrics: dict) -> float:
        value = self._safe_score_under_combo_frac(metrics)
        return float(value) if value is not None else 0.0

    def _safe_meta_under_combo_hit_select(self, metrics: dict, kind: str) -> float:
        values = [
            (key, value) for key, value in self._safety_under_combo_hit_select_values(
                metrics, kind)
        ]
        if not values:
            return 0.0
        gated_values = [
            value for key, value in values
            if not self._under_combo_hit_select_exposure_gate_passes(metrics, key)
        ]
        if not gated_values:
            return 1.0 if kind == "clean" else 0.0
        return float(min(gated_values) if kind == "clean" else max(gated_values))

    def _write_safe_meta(self, metrics: dict, source: Path,
                         score: tuple[float, ...]) -> None:
        if len(score) == 5:
            combo12, combo_max, combo8, hit_rate, neg_sky = score
            score = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     hit_rate, neg_sky)
        under_counter_value, under_counter_source = (
            self._safe_meta_under_combo_counter(metrics, score))
        payload = {
            "score_schema": 9,
            "requires_chase_combo": bool(
                self.cfg.get("safety_require_chase_combo", False)),
            "requires_counter_recovery": bool(
                self.cfg.get("safety_require_counter_recovery", False)),
            "iter": int(metrics.get("iter", self.iter)),
            "checkpoint": "safe_latest.pt",
            "source_checkpoint": source.name,
            "score": list(score),
            "style_score": score[0],
            "combo12_state": score[1],
            "combo8_hits": score[2],
            "combo_max": score[3],
            "combo_tap_frac": score[4],
            "combo_s_tap_frac": self._safe_meta_combo_component(metrics, "s", max),
            "combo_z_tap_frac": self._safe_meta_combo_component(metrics, "z", min),
            "hit_wtap_frac": self._safe_meta_hit_wtap(metrics),
            "under_combo_counter_hit_frac": under_counter_value,
            "under_combo_counter_score_frac": score[5],
            "under_combo_counter_source": under_counter_source,
            "under_combo_frac": self._safe_meta_under_combo_frac(metrics),
            "under_combo_hit_select_clean_frac": (
                self._safe_meta_under_combo_hit_select(metrics, "clean")),
            "under_combo_hit_select_trade_frac": (
                self._safe_meta_under_combo_hit_select(metrics, "trade")),
            "under_combo_avoidance_score_bonus": (
                self._under_combo_avoidance_score_bonus(
                    self._safe_score_under_combo_frac(metrics), score[3])),
            "hit_rate": score[6],
            "sky_frac": -score[7],
            "back_frac": float(metrics.get("back_frac", 0.0) or 0.0),
            "strafe_frac": float(metrics.get("strafe_frac", 0.0) or 0.0),
            "opener_strafe_frac": float(
                metrics.get("opener_strafe_frac", 0.0) or 0.0),
            "opener_strafe_hold_frac": float(
                metrics.get("opener_strafe_hold_frac", 0.0) or 0.0),
            "opener_pressure_frac": float(
                metrics.get("opener_pressure_frac", 0.0) or 0.0),
            "opener_samples": float(metrics.get("opener_samples", 0.0) or 0.0),
            "safety_back_frac": float(
                self.cfg.get("safety_back_frac", 0.002) or 0.002),
            "safety_min_strafe_frac": float(
                self.cfg.get("safety_min_strafe_frac", 0.50) or 0.50),
            "safety_min_opener_strafe_frac": float(
                self.cfg.get("safety_min_opener_strafe_frac", 0.75) or 0.75),
            "safety_min_opener_strafe_hold_frac": float(
                self.cfg.get("safety_min_opener_strafe_hold_frac", -1.0)),
            "safety_min_opener_pressure_frac": float(
                self.cfg.get("safety_min_opener_pressure_frac", -1.0)),
            "safety_min_combo_tap_frac": float(
                self.cfg.get("safety_min_combo_tap_frac", -1.0)),
            "safety_min_combo_z_tap_frac": float(
                self.cfg.get("safety_min_combo_z_tap_frac", -1.0)),
            "safety_max_combo_s_tap_frac": float(
                self.cfg.get("safety_max_combo_s_tap_frac", -1.0)),
            "safety_min_hit_wtap_frac": float(
                self.cfg.get("safety_min_hit_wtap_frac", -1.0)),
            "safety_min_under_combo_counter_hit_frac": float(
                self.cfg.get("safety_min_under_combo_counter_hit_frac", -1.0)),
            "safety_min_under_combo_hit_select_clean_frac": float(
                self.cfg.get("safety_min_under_combo_hit_select_clean_frac", -1.0)),
            "safety_max_under_combo_hit_select_trade_frac": float(
                self.cfg.get("safety_max_under_combo_hit_select_trade_frac", -1.0)),
            "safety_opener_ticks": int(
                self.cfg.get("safety_opener_ticks", 20) or 20),
        }
        target = self._safe_meta_path()
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        tmp.replace(target)

    def _keep_safe_checkpoint(self, metrics: dict, score: tuple[float, ...],
                              reason: str | None = None,
                              restore: bool = True) -> None:
        metrics["safety_checkpoint"] = "kept_best"
        metrics["safety_combo_score"] = list(score)
        if reason:
            metrics["safety_promotion_reason"] = reason
        if not restore:
            return
        restored = self._restore_safe_checkpoint()
        metrics["safety_restored"] = restored or "missing"
        memory_restored = self._restore_safe_training_state()
        metrics["safety_memory_restored"] = memory_restored or "missing"

    def _mark_safe_checkpoint(self, metrics: dict) -> None:
        ckpt = self.run_dir / f"ckpt_{int(metrics.get('iter', self.iter)):06d}.pt"
        source = ckpt if ckpt.exists() else self.run_dir / "latest.pt"
        if not source.exists():
            return
        score = self._safe_combo_score(metrics)
        previous_record = self._load_safe_combo_record()
        previous = previous_record[0] if previous_record is not None else None
        previous_requires_chase = (
            previous_record[1] if previous_record is not None else False)
        previous_requires_counter = (
            previous_record[2] if previous_record is not None else False)
        if (self.cfg.get("safety_require_chase_combo", False)
                and not previous_requires_chase):
            previous = None
        if (self.cfg.get("safety_require_counter_recovery", False)
                and not previous_requires_counter):
            previous = None
        promote_min_combo = float(
            self.cfg.get("safety_promote_min_combo_max", -1.0) or -1.0)
        if promote_min_combo >= 0.0 and score[3] < promote_min_combo:
            restore = (
                bool(self.cfg.get("safety_restore_on_low_combo", False))
                and previous is not None
                and previous[3] >= promote_min_combo
            )
            self._keep_safe_checkpoint(
                metrics,
                score,
                f"combo_max={score[3]:.4g}<{promote_min_combo:.4g}",
                restore=restore,
            )
            return
        if previous is not None and score < previous:
            metrics["safety_best_score"] = list(previous)
            self._keep_safe_checkpoint(metrics, score)
            return
        for name in ("safe_latest.pt", "recover_noescape.pt"):
            target = self.run_dir / name
            tmp = target.with_name(target.name + ".tmp")
            shutil.copyfile(source, tmp)
            tmp.replace(target)
        self._write_safe_meta(metrics, source, score)
        metrics["safety_checkpoint"] = source.name
        metrics["safety_combo_score"] = list(score)
    def _restore_safe_checkpoint(self) -> str | None:
        latest = self.run_dir / "latest.pt"
        for name in ("safe_latest.pt", "recover_noescape.pt"):
            source = self.run_dir / name
            if source.exists():
                tmp = latest.with_name(latest.name + ".tmp")
                shutil.copyfile(source, tmp)
                tmp.replace(latest)
                return source.name
        return None

    def _restore_safe_training_state(self) -> str | None:
        for name in ("safe_latest.pt", "recover_noescape.pt"):
            source = self.run_dir / name
            if not source.exists():
                continue
            try:
                ckpt = torch.load(source, map_location=self.device,
                                  weights_only=False)
                self._seed_state_dict(self.policy, ckpt["policy"])
                if ckpt.get("optimizer"):
                    self.ppo.opt.load_state_dict(ckpt["optimizer"])
                    self._fix_fused_optimizer_state(self.ppo.opt, self.device)
                    for group in self.ppo.opt.param_groups:
                        group["lr"] = self.ppo.cfg.lr
                if ckpt.get("scaler"):
                    try:
                        self.ppo.scaler.load_state_dict(ckpt["scaler"])
                    except RuntimeError:
                        pass
                if self.members:
                    self._load_population(ckpt)
                return source.name
            except Exception as exc:  # noqa: BLE001
                print(f"[safety] memory restore failed from {source.name}: {exc}")
        return None
    # -------------------------------------------------------- automatisations
    def _update_ramp(self, hit_rate: float, engage_rate: float | None = None,
                     combo5_rate: float | None = None,
                     sky_frac: float | None = None,
                     combo12_state: float | None = None) -> None:
        """Rampe adaptative: avance seulement si le combat reste sain."""
        thresh = self.cfg["shaping_hit_rate"]
        engage_thresh = float(self.cfg.get("shaping_engage_rate", 0.0))
        combo5_thresh = float(self.cfg.get("shaping_combo5_rate", 0.0))
        combo12_thresh = float(self.cfg.get("shaping_combo12_state", 0.0))
        sky_thresh = float(self.cfg.get("shaping_sky_frac", 1.0))
        combo_ok = (combo5_thresh <= 0.0 or combo5_rate is None
                    or combo5_rate >= combo5_thresh)
        combo12_ok = (combo12_thresh <= 0.0 or combo12_state is None
                      or combo12_state >= combo12_thresh)
        engage_ok = (engage_thresh <= 0.0 or engage_rate is None
                     or engage_rate >= engage_thresh)
        sky_ok = sky_thresh >= 1.0 or sky_frac is None or sky_frac <= sky_thresh
        healthy = (hit_rate >= thresh and engage_ok and combo_ok
                   and combo12_ok and sky_ok)
        passive = (hit_rate < 0.5 * thresh
                   or (engage_thresh > 0.0 and engage_rate is not None
                       and engage_rate < 0.5 * engage_thresh)
                   or (combo5_thresh > 0.0 and combo5_rate is not None
                       and combo5_rate < 0.5 * combo5_thresh)
                   or (combo12_thresh > 0.0 and combo12_state is not None
                       and combo12_state < combo12_thresh)
                   or (sky_thresh < 1.0 and sky_frac is not None
                       and sky_frac > sky_thresh))
        if not self._ramp_on:
            if healthy:
                self._hit_streak += 1
            else:
                self._hit_streak = 0
            if self._hit_streak >= 10:
                self._ramp_on = True
            return
        step = 1.0 / max(self.cfg["shaping_decay_iters"], 1)
        if healthy:
            self._ramp_pos = min(self._ramp_pos + step, 1.0)
        elif passive:
            self._ramp_pos = max(self._ramp_pos - 2.0 * step, 0.0)

    def _ramp_frac(self) -> float:
        return self._ramp_pos

    def _auto_shaping(self) -> float:
        """Phase 2 de la rampe : shaping plein jusqu'ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â  pos 0.5, puis dÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©croÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â®t
        vers le plancher shaping_floor_frac (0 = extinction complÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¨te ;
        > 0 = pression de rapprochement permanente, anti-passivitÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©)."""
        if self._shaping_base <= 0:
            return 0.0
        pos = self._ramp_pos
        floor = min(max(self.cfg["shaping_floor_frac"], 0.0), 1.0)
        factor = 1.0 if pos <= 0.5 else max(1.0 - (pos - 0.5) * 2.0, floor)
        value = self._shaping_base * factor
        if self._ramp_on:
            self.sim.set_reward_dist(value)
        return value

    def _auto_curriculum(self) -> float:
        """Phase 1 de la rampe : spawn proche -> standard sur pos 0 -> 0.5."""
        cg = self.cfg["curriculum_gap"]
        if cg <= 0:
            return self.sim_cfg.spawn_gap or self._full_gap
        gphase = min(self._ramp_pos * 2.0, 1.0)
        gap = cg + (self._full_gap - cg) * gphase
        if self._ramp_on:
            # gphase = 1 -> 0 = mode auto (arÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¨ne/3), valeur standard exacte
            self.sim.set_spawn_gap(0.0 if gphase >= 1.0 else gap)
        return gap

    def _maybe_snapshot(self) -> None:
        """Snapshot league, gatÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â© : seulement si le learner bat le prÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©cÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©dent
        (winrate >= 0.52). ForcÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â© aprÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¨s 2 refus pour ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©viter la stagnation."""
        if (not self.cfg["snapshot_gate"] or not self.league.pool
                or self.cfg["eval_every"] <= 0):
            self.league.add_snapshot(self.policy)
            return
        wr = self._evaluate(self.league.pool[-1]["state_dict"])
        if wr >= 0.52 or self._snapshot_skips >= 2:
            self.league.add_snapshot(self.policy)
            self._snapshot_skips = 0
        else:
            self._snapshot_skips += 1

    def _entropy_guard(self, entropy: float) -> int:
        """1 si l'entropie vient de s'effondrer (< 50% de la mÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©diane rÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©cente)."""
        warn = 0
        if len(self._entropy_hist) >= 20:
            med = sorted(self._entropy_hist)[len(self._entropy_hist) // 2]
            if entropy < 0.5 * med:
                warn = 1
        self._entropy_hist.append(entropy)
        if len(self._entropy_hist) > 50:
            self._entropy_hist.pop(0)
        return warn

    @torch.no_grad()
    def _evaluate_combo_fresh(self) -> dict:
        """Combo metrics from a fresh reset, not inherited rollout state."""
        n = int(self.cfg.get("combo_eval_envs", 128) or 128)
        ticks = int(self.cfg.get("combo_eval_ticks", 1200) or 1200)
        eval_cfg = replace(self.sim_cfg)
        eval_cfg.randomize = False
        eval_cfg.max_ticks = max(ticks, 1)
        eval_cfg.target_hits = max(int(eval_cfg.target_hits), 140)
        gap = float(self.cfg.get("combo_eval_spawn_gap",
                                 self.cfg.get("curriculum_gap", 0.0)) or 0.0)
        if gap > 0.0:
            eval_cfg.spawn_gap = gap
        sim = make_sim(n, eval_cfg, seed=self.cfg["seed"] + 9109,
                       force_cpu=self.device.type != "cuda")
        obs = self._to_torch(sim.reset()).float()
        hist = torch.zeros(n, 2, self.H, OBS_DIM, device=self.device)
        hist[:, :, -1] = obs
        combo_max = 0
        combo12_sum = 0.0
        combo8_sum = 0.0
        dealt_sum = 0.0
        sky_sum = 0.0
        aim_sum = 0.0
        s_tap_sum = 0.0
        z_tap_sum = 0.0
        combo_tap_sum = 0.0
        combo_s_tap_sum = 0.0
        combo_z_tap_sum = 0.0
        combo_s_tap_denom = 0.0
        hit_wtap_sum = 0.0
        hit_wtap_denom_sum = 0.0
        under_combo_sum = 0.0
        under_combo_counter_hits = 0.0
        hit_select_window_sum = 0.0
        hit_select_attack_sum = 0.0
        hit_select_clean_sum = 0.0
        hit_select_trade_sum = 0.0
        dealt_hist: list[torch.Tensor] = []
        taken_hist: list[torch.Tensor] = []
        done_hist: list[torch.Tensor] = []
        samples = 0
        for _ in range(ticks):
            flat = hist.reshape(n * 2, self.H, OBS_DIM)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self._use_amp):
                out = self.policy.act(flat, deterministic=True)
            actions = to_sim_actions(
                {k: out[k] for k in ("pre", "fwd", "strafe", "bins")}
            ).view(n, 2, 7).float()
            obs_now = hist[:, :, -1].reshape(n * 2, OBS_DIM)
            prev_hits = (obs_now[:, 31] * 100.0).round()
            pitch_abs = (obs_now[:, 10] * 90.0).abs()
            sky_sum += float((pitch_abs > 60.0).float().sum())
            actions_flat = actions.reshape(n * 2, 7)
            fwd_dir = actions_flat[:, 2]
            strafe_dir = actions_flat[:, 3]
            sprint_on = actions_flat[:, 5] > 0.5
            attack_on = actions_flat[:, 6] > 0.5
            dist = obs_now[:, 45] * 8.0
            combo_adv = obs_now[:, 22] > obs_now[:, 21] + 0.05
            close_combo = combo_adv & (dist <= COMBO_TAP_REACH)
            s_tap = (fwd_dir < -0.5) & (~sprint_on)
            z_tap = (fwd_dir.abs() <= 0.5) & (~sprint_on)
            hit_wtap_ready = z_tap & (strafe_dir.abs() > 0.5)
            s_tap_sum += float(s_tap.float().sum())
            z_tap_sum += float(z_tap.float().sum())
            combo_s_tap = s_tap & close_combo
            combo_z_tap = z_tap & close_combo
            combo_tap_sum += float((combo_s_tap | combo_z_tap).float().sum())
            combo_s_tap_sum += float(combo_s_tap.float().sum())
            combo_z_tap_sum += float(combo_z_tap.float().sum())
            combo_s_tap_denom += float(close_combo.float().sum())
            under_combo = obs_now[:, 21] > obs_now[:, 22] + 0.05
            under_combo_sum += float(under_combo.float().sum())
            yaw_err_deg = torch.rad2deg(torch.atan2(
                obs_now[:, 11], obs_now[:, 12]).abs())
            pitch_err_deg = (obs_now[:, 13] * 90.0).abs()
            aim = ((1.0 - yaw_err_deg / 90.0).clamp(0.0, 1.0)
                   * (1.0 - pitch_err_deg / 45.0).clamp(0.0, 1.0))
            aim_sum += float(aim.sum())
            obs, _, done, info = self._sim_step(actions, sim=sim)
            combo = info.get("combo")
            if combo is not None:
                combo_f = combo.float()
                combo_max = max(combo_max, int(combo_f.max().item()))
                combo12_sum += float((combo_f >= 12).float().sum())
                combo8_sum += float((combo_f >= 8).float().sum())
            new_hits = (obs.reshape(n * 2, OBS_DIM)[:, 31] * 100.0).round()
            done_b = done.bool().repeat_interleave(2)
            dealt_obs = ((new_hits - prev_hits).clamp(min=0.0)
                         * (~done_b).float()) > 0.5
            taken_obs = dealt_obs.view(n, 2).flip(-1).reshape(n * 2)
            hit_select_window = (
                under_combo
                & (dist >= COUNTER_HIT_SELECT_CLEAN_MIN_REACH)
                & (dist <= COUNTER_HIT_SELECT_CLEAN_MAX_REACH)
                & (obs_now[:, 21].clamp(0.0, 1.0) >= COUNTER_HIT_SELECT_MIN_OWN_HURT)
                & (obs_now[:, 21].clamp(0.0, 1.0) <= COUNTER_HIT_SELECT_CLEAN_HURT)
                & (obs_now[:, 37].clamp(0.0, 1.0) >= COUNTER_HIT_SELECT_OPP_COOLDOWN)
            )
            dealt_hist.append(dealt_obs.detach())
            taken_hist.append(taken_obs.detach())
            done_hist.append(done_b.detach())
            under_combo_counter_hits += float(
                (dealt_obs & under_combo & attack_on).float().sum())
            hit_select_window_sum += float(hit_select_window.float().sum())
            hit_select_attack_sum += float(
                (attack_on & hit_select_window).float().sum())
            hit_select_clean_sum += float(
                (dealt_obs & (~taken_obs) & hit_select_window).float().sum())
            hit_select_trade_sum += float(
                (dealt_obs & taken_obs & hit_select_window).float().sum())
            non_counter_dealt = dealt_obs & (~under_combo)
            hit_wtap_sum += float(
                (non_counter_dealt & hit_wtap_ready).float().sum())
            hit_wtap_denom_sum += float(non_counter_dealt.float().sum())
            dealt_sum += float(dealt_obs.float().sum())
            samples += n * 2
            hist = torch.roll(hist, shifts=-1, dims=2)
            hist[done.bool()] = 0.0
            hist[:, :, -1] = obs.float()
        denom = max(samples, 1)
        under_denom = max(under_combo_sum, 1.0)
        hit_select_denom = max(hit_select_window_sum, 1.0)
        if dealt_hist:
            dealt_stack = torch.stack(dealt_hist)
            taken_stack = torch.stack(taken_hist)
            done_stack = torch.stack(done_hist)
            learner = torch.ones(n * 2, dtype=torch.bool,
                                 device=dealt_stack.device)
            clean_dealt = dealt_stack & (~taken_stack)
            clean_taken = taken_stack & (~dealt_stack)
            rechain = _chain_followup_stats(
                clean_dealt, clean_dealt, clean_taken, done_stack, learner,
                eval_cfg.combo_window,
            )
            counter_break = _chain_followup_stats(
                clean_taken, clean_dealt, clean_taken, done_stack, learner,
                eval_cfg.combo_window,
            )
        else:
            rechain = counter_break = {
                "opps": 0, "hit_frac": 0.0,
                "taken_frac": 0.0, "miss_frac": 0.0,
            }
        metrics = {
            "fresh_combo_max": combo_max,
            "fresh_combo12_state": round(combo12_sum / denom, 4),
            "fresh_combo8_state": round(combo8_sum / denom, 4),
            "fresh_hit_rate": round(dealt_sum / denom * 1200.0, 2),
            "fresh_rechain_hit_frac": round(rechain["hit_frac"], 4),
            "fresh_rechain_taken_frac": round(rechain["taken_frac"], 4),
            "fresh_counter_break_hit_frac": round(counter_break["hit_frac"], 4),
            "fresh_counter_break_taken_frac": round(counter_break["taken_frac"], 4),
            "fresh_sky_frac": round(sky_sum / denom, 4),
            "fresh_aim_body": round(aim_sum / denom, 4),
            "fresh_s_tap_frac": round(s_tap_sum / denom, 4),
            "fresh_z_tap_frac": round(z_tap_sum / denom, 4),
            "fresh_combo_tap_frac": round(
                combo_tap_sum / max(combo_s_tap_denom, 1.0), 4),
            "fresh_combo_s_tap_frac": round(
                combo_s_tap_sum / max(combo_s_tap_denom, 1.0), 4),
            "fresh_combo_z_tap_frac": round(
                combo_z_tap_sum / max(combo_s_tap_denom, 1.0), 4),
            "fresh_hit_wtap_frac": round(
                hit_wtap_sum / max(hit_wtap_denom_sum, 1.0), 4),
            "fresh_under_combo_frac": round(under_combo_sum / denom, 4),
            "fresh_under_combo_counter_hit_rate": round(
                under_combo_counter_hits / under_denom * 1200.0, 2),
            "fresh_under_combo_counter_hit_frac": round(
                under_combo_counter_hits / under_denom, 4),
            "fresh_under_combo_hit_select_attack_frac": round(
                hit_select_attack_sum / hit_select_denom, 4),
            "fresh_under_combo_hit_select_clean_frac": round(
                hit_select_clean_sum / hit_select_denom, 4),
            "fresh_under_combo_hit_select_trade_frac": round(
                hit_select_trade_sum / hit_select_denom, 4),
        }
        if self.cfg.get("combo_eval_spar", True):
            metrics.update(self._evaluate_combo_fresh_vs_spar())
        if self.cfg.get("combo_eval_rehit", True):
            metrics.update(self._evaluate_combo_fresh_vs_rehit())
        if self.cfg.get("combo_eval_pressure", True):
            metrics.update(self._evaluate_combo_fresh_vs_pressure())
        if self.cfg.get("combo_eval_chase", True):
            metrics.update(self._evaluate_combo_fresh_vs_chase())
        if self.cfg.get("combo_eval_counter", False):
            metrics.update(self._evaluate_combo_fresh_vs_counter())
        return metrics

    @torch.no_grad()
    def _evaluate_combo_fresh_vs_chase(self) -> dict:
        from .scripted import ComboChaseBot
        return self._evaluate_combo_fresh_vs_scripted(
            ComboChaseBot(),
            "fresh_chase",
            "combo_eval_chase_envs",
            "combo_eval_chase_ticks",
            9209,
        )

    @torch.no_grad()
    def _evaluate_combo_fresh_vs_spar(self) -> dict:
        from .scripted import ComboSparBot
        return self._evaluate_combo_fresh_vs_scripted(
            ComboSparBot(),
            "fresh_spar",
            "combo_eval_spar_envs",
            "combo_eval_spar_ticks",
            9309,
        )

    @torch.no_grad()
    def _evaluate_combo_fresh_vs_rehit(self) -> dict:
        from .scripted import ComboRehitBot
        return self._evaluate_combo_fresh_vs_scripted(
            ComboRehitBot(),
            "fresh_rehit",
            "combo_eval_rehit_envs",
            "combo_eval_rehit_ticks",
            9409,
        )

    @torch.no_grad()
    def _evaluate_combo_fresh_vs_pressure(self) -> dict:
        from .scripted import ComboPressureBot
        return self._evaluate_combo_fresh_vs_scripted(
            ComboPressureBot(),
            "fresh_pressure",
            "combo_eval_pressure_envs",
            "combo_eval_pressure_ticks",
            9509,
        )

    @torch.no_grad()
    def _evaluate_combo_fresh_vs_counter(self) -> dict:
        from .scripted import ComboCounterBot
        return self._evaluate_combo_fresh_vs_scripted(
            ComboCounterBot(),
            "fresh_counter",
            "combo_eval_counter_envs",
            "combo_eval_counter_ticks",
            9609,
        )

    @torch.no_grad()
    def _evaluate_combo_fresh_vs_scripted(self, bot, prefix: str,
                                          env_key: str, ticks_key: str,
                                          seed_offset: int) -> dict:
        """Fresh learner-vs-scripted active-opponent combo metrics."""
        n = int(self.cfg.get(
            env_key,
            max(1, int(self.cfg.get("combo_eval_envs", 64) or 64) // 2),
        ) or 64)
        ticks = int(self.cfg.get(
            ticks_key,
            max(1, int(self.cfg.get("combo_eval_ticks", 900) or 900)),
        ) or 900)
        eval_cfg = replace(self.sim_cfg)
        eval_cfg.randomize = False
        eval_cfg.max_ticks = max(ticks, 1)
        eval_cfg.target_hits = max(int(eval_cfg.target_hits), 140)
        gap = float(self.cfg.get("combo_eval_spawn_gap",
                                 self.cfg.get("curriculum_gap", 0.0)) or 0.0)
        if gap > 0.0:
            eval_cfg.spawn_gap = gap
        sim = make_sim(n, eval_cfg, seed=self.cfg["seed"] + seed_offset,
                       force_cpu=self.device.type != "cuda")
        obs = self._to_torch(sim.reset()).float()
        hist = torch.zeros(n, 2, self.H, OBS_DIM, device=self.device)
        hist[:, :, -1] = obs

        combo_max = 0
        combo12_sum = 0.0
        combo8_sum = 0.0
        dealt_sum = 0.0
        taken_sum = 0.0
        trade_sum = 0.0
        sky_sum = 0.0
        aim_sum = 0.0
        close_sum = 0.0
        combo_tap_sum = 0.0
        combo_s_tap_sum = 0.0
        combo_z_tap_sum = 0.0
        combo_s_tap_denom = 0.0
        hit_wtap_sum = 0.0
        hit_wtap_denom_sum = 0.0
        under_combo_sum = 0.0
        under_combo_counter_hits = 0.0
        hit_select_window_sum = 0.0
        hit_select_attack_sum = 0.0
        hit_select_clean_sum = 0.0
        hit_select_trade_sum = 0.0
        dealt_hist: list[torch.Tensor] = []
        taken_hist: list[torch.Tensor] = []
        done_hist: list[torch.Tensor] = []
        samples = 0
        for _ in range(ticks):
            with torch.autocast("cuda", dtype=torch.float16, enabled=self._use_amp):
                out = self.policy.act(hist[:, 0], deterministic=True)
            a0 = to_sim_actions(
                {k: out[k] for k in ("pre", "fwd", "strafe", "bins")}
            ).float()
            a1 = bot.act7(hist[:, 1]).float()
            actions = torch.stack([a0, a1], dim=1)

            obs0 = hist[:, 0, -1]
            obs1 = hist[:, 1, -1]
            prev_hits0 = (obs0[:, 31] * 100.0).round()
            prev_hits1 = (obs1[:, 31] * 100.0).round()
            pitch_abs = (obs0[:, 10] * 90.0).abs()
            sky_sum += float((pitch_abs > 60.0).float().sum())
            fwd_dir = a0[:, 2]
            strafe_dir = a0[:, 3]
            sprint_on = a0[:, 5] > 0.5
            attack_on = a0[:, 6] > 0.5
            dist = obs0[:, 45] * 8.0
            close_sum += float((dist < 3.5).float().sum())
            combo_adv = obs0[:, 22] > obs0[:, 21] + 0.05
            close_combo = combo_adv & (dist <= COMBO_TAP_REACH)
            s_tap = (fwd_dir < -0.5) & (~sprint_on)
            z_tap = (fwd_dir.abs() <= 0.5) & (~sprint_on)
            hit_wtap_ready = z_tap & (strafe_dir.abs() > 0.5)
            combo_s_tap = s_tap & close_combo
            combo_z_tap = z_tap & close_combo
            combo_tap_sum += float((combo_s_tap | combo_z_tap).float().sum())
            combo_s_tap_sum += float(combo_s_tap.float().sum())
            combo_z_tap_sum += float(combo_z_tap.float().sum())
            combo_s_tap_denom += float(close_combo.float().sum())
            under_combo = obs0[:, 21] > obs0[:, 22] + 0.05
            under_combo_sum += float(under_combo.float().sum())
            yaw_err_deg = torch.rad2deg(torch.atan2(obs0[:, 11], obs0[:, 12]).abs())
            pitch_err_deg = (obs0[:, 13] * 90.0).abs()
            aim = ((1.0 - yaw_err_deg / 90.0).clamp(0.0, 1.0)
                   * (1.0 - pitch_err_deg / 45.0).clamp(0.0, 1.0))
            aim_sum += float(aim.sum())

            obs, _, done, info = self._sim_step(actions, sim=sim)
            combo = info.get("combo")
            if combo is not None:
                combo0 = combo.float()[:, 0]
                combo_max = max(combo_max, int(combo0.max().item()))
                combo12_sum += float((combo0 >= 12).float().sum())
                combo8_sum += float((combo0 >= 8).float().sum())
            done_f = (~done.bool()).float()
            new_hits0 = (obs[:, 0, 31] * 100.0).round()
            new_hits1 = (obs[:, 1, 31] * 100.0).round()
            dealt = ((new_hits0 - prev_hits0).clamp(min=0.0) * done_f) > 0.5
            taken = ((new_hits1 - prev_hits1).clamp(min=0.0) * done_f) > 0.5
            hit_select_window = (
                under_combo
                & (dist >= COUNTER_HIT_SELECT_CLEAN_MIN_REACH)
                & (dist <= COUNTER_HIT_SELECT_CLEAN_MAX_REACH)
                & (obs0[:, 21].clamp(0.0, 1.0) >= COUNTER_HIT_SELECT_MIN_OWN_HURT)
                & (obs0[:, 21].clamp(0.0, 1.0) <= COUNTER_HIT_SELECT_CLEAN_HURT)
                & (obs0[:, 37].clamp(0.0, 1.0) >= COUNTER_HIT_SELECT_OPP_COOLDOWN)
            )
            dealt_hist.append(dealt.detach())
            taken_hist.append(taken.detach())
            done_hist.append(done.bool().detach())
            under_combo_counter_hits += float(
                (dealt & under_combo & attack_on).float().sum())
            hit_select_window_sum += float(hit_select_window.float().sum())
            hit_select_attack_sum += float(
                (attack_on & hit_select_window).float().sum())
            hit_select_clean_sum += float(
                (dealt & (~taken) & hit_select_window).float().sum())
            hit_select_trade_sum += float(
                (dealt & taken & hit_select_window).float().sum())
            non_counter_dealt = dealt & (~under_combo)
            hit_wtap_sum += float(
                (non_counter_dealt & hit_wtap_ready).float().sum())
            hit_wtap_denom_sum += float(non_counter_dealt.float().sum())
            dealt_sum += float(dealt.float().sum())
            taken_sum += float(taken.float().sum())
            trade_sum += float((dealt & taken).float().sum())
            samples += n
            hist = torch.roll(hist, shifts=-1, dims=2)
            hist[done.bool()] = 0.0
            hist[:, :, -1] = obs.float()

        denom = max(samples, 1)
        under_denom = max(under_combo_sum, 1.0)
        hit_denom = max(dealt_sum, 1.0)
        hit_select_denom = max(hit_select_window_sum, 1.0)
        if dealt_hist:
            dealt_stack = torch.stack(dealt_hist)
            taken_stack = torch.stack(taken_hist)
            done_stack = torch.stack(done_hist)
            learner = torch.ones(n, dtype=torch.bool,
                                 device=dealt_stack.device)
            clean_dealt = dealt_stack & (~taken_stack)
            clean_taken = taken_stack & (~dealt_stack)
            rechain = _chain_followup_stats(
                clean_dealt, clean_dealt, clean_taken, done_stack, learner,
                eval_cfg.combo_window,
            )
            counter_break = _chain_followup_stats(
                clean_taken, clean_dealt, clean_taken, done_stack, learner,
                eval_cfg.combo_window,
            )
        else:
            rechain = counter_break = {
                "opps": 0, "hit_frac": 0.0,
                "taken_frac": 0.0, "miss_frac": 0.0,
            }
        return {
            f"{prefix}_combo_max": combo_max,
            f"{prefix}_combo12_state": round(combo12_sum / denom, 4),
            f"{prefix}_combo8_state": round(combo8_sum / denom, 4),
            f"{prefix}_hit_rate": round(dealt_sum / denom * 1200.0, 2),
            f"{prefix}_taken_rate": round(taken_sum / denom * 1200.0, 2),
            f"{prefix}_trade_hit_frac": round(trade_sum / hit_denom, 4),
            f"{prefix}_rechain_hit_frac": round(rechain["hit_frac"], 4),
            f"{prefix}_rechain_taken_frac": round(rechain["taken_frac"], 4),
            f"{prefix}_counter_break_hit_frac": round(counter_break["hit_frac"], 4),
            f"{prefix}_counter_break_taken_frac": round(counter_break["taken_frac"], 4),
            f"{prefix}_close_frac": round(close_sum / denom, 4),
            f"{prefix}_sky_frac": round(sky_sum / denom, 4),
            f"{prefix}_aim_body": round(aim_sum / denom, 4),
            f"{prefix}_combo_tap_frac": round(
                combo_tap_sum / max(combo_s_tap_denom, 1.0), 4),
            f"{prefix}_combo_s_tap_frac": round(
                combo_s_tap_sum / max(combo_s_tap_denom, 1.0), 4),
            f"{prefix}_combo_z_tap_frac": round(
                combo_z_tap_sum / max(combo_s_tap_denom, 1.0), 4),
            f"{prefix}_hit_wtap_frac": round(
                hit_wtap_sum / max(hit_wtap_denom_sum, 1.0), 4),
            f"{prefix}_under_combo_frac": round(under_combo_sum / denom, 4),
            f"{prefix}_under_combo_counter_hit_rate": round(
                under_combo_counter_hits / under_denom * 1200.0, 2),
            f"{prefix}_under_combo_counter_hit_frac": round(
                under_combo_counter_hits / under_denom, 4),
            f"{prefix}_under_combo_hit_select_attack_frac": round(
                hit_select_attack_sum / hit_select_denom, 4),
            f"{prefix}_under_combo_hit_select_clean_frac": round(
                hit_select_clean_sum / hit_select_denom, 4),
            f"{prefix}_under_combo_hit_select_trade_frac": round(
                hit_select_trade_sum / hit_select_denom, 4),
        }

    @torch.no_grad()
    def _evaluate(self, opponent) -> float:
        """Winrate du learner sur des matchs courts.
        opponent : state_dict d'un snapshot, ou "bot" (chase-bot scriptÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©)."""
        n = self.cfg["eval_envs"]
        if self._eval_sim is None:
            s = self.sim_cfg
            eval_cfg = SimConfig(
                arena_size_x=s.arena_size_x, arena_size_z=s.arena_size_z,
                target_hits=self.cfg["eval_target_hits"],
                max_ticks=self.cfg["eval_max_ticks"],
                speed_amplifier=s.speed_amplifier,
                cps_min=(s.cps_min + s.cps_max) / 2,
                cps_max=(s.cps_min + s.cps_max) / 2,
                rot_speed_min=(s.rot_speed_min + s.rot_speed_max) / 2,
                rot_speed_max=(s.rot_speed_min + s.rot_speed_max) / 2,
                randomize=False,
            )
            self._eval_sim = make_sim(n, eval_cfg, seed=self.cfg["seed"] + 1,
                                      force_cpu=self.device.type != "cuda")
            self._eval_opp = JudasPolicy(self.pol_cfg).to(self.device)
        use_bot = opponent == "bot"
        if use_bot:
            from .scripted import ChaseBot
            bot = ChaseBot()
        else:
            self._eval_opp.load_state_dict(opponent, strict=False)
            self._eval_opp.eval()

        obs = self._to_torch(self._eval_sim.reset()).float()    # [n, 2, D]
        hist = torch.zeros(n, 2, self.H, OBS_DIM, device=self.device)
        hist[:, :, -1] = obs
        score = 0.0
        # un seul match comptÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â© par env : les envs auto-reset continuent de
        # tourner mais leurs matchs suivants (les plus courts) sont ignorÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©s
        counted = torch.zeros(n, dtype=torch.bool, device=self.device)
        for _ in range(self.cfg["eval_max_ticks"]):
            with torch.autocast("cuda", dtype=torch.float16, enabled=self._use_amp):
                a0 = self.policy.act(hist[:, 0])
                a0_7 = to_sim_actions(
                    {k: a0[k] for k in ("pre", "fwd", "strafe", "bins")})
                if use_bot:
                    a1_7 = bot.act7(hist[:, 1])
                else:
                    a1 = self._eval_opp.act(hist[:, 1])
                    a1_7 = to_sim_actions(
                        {k: a1[k] for k in ("pre", "fwd", "strafe", "bins")})
            actions = torch.stack([a0_7.float(), a1_7.float()], dim=1)
            obs, _, done, info = self._sim_step(actions, sim=self._eval_sim)
            winner = info["winner"]
            hist = torch.roll(hist, shifts=-1, dims=2)
            hist[done.bool()] = 0.0
            hist[:, :, -1] = obs.float()
            fresh = done.bool() & ~counted
            for w in winner[fresh].tolist():
                score += 1.0 if w == 0 else (0.5 if w == -1 else 0.0)
            counted |= done.bool()
            if bool(counted.all()):
                break
        finished = int(counted.sum())
        return score / finished if finished else 0.5

    def train(self, iters: int | None = None) -> None:
        total = iters if iters is not None else self.cfg["total_iters"]
        combo_eval_every = int(self.cfg.get("combo_eval_every", 0) or 0)
        fresh_eval_count_at_start = self._fresh_combo_eval_count
        for i in range(total):
            force_combo_eval = (
                combo_eval_every > 0
                and i == total - 1
                and self._fresh_combo_eval_count == fresh_eval_count_at_start
            )
            m = self.train_iter(force_combo_eval=force_combo_eval)
            print(f"[{m['iter']:5d}] sps={m['sps']:>9} rew={m['reward_mean']:+.4f} "
                  f"elo={m['elo']:.0f} wr={m['league_winrate']:.2f} "
                  f"kl={m['approx_kl']:.4f}")

    # ------------------------------------------------------------ persistence
    def save(self) -> Path:
        path = self.run_dir / f"ckpt_{self.iter:06d}.pt"
        head_policy, head_ppo = self.policy, self.ppo
        if self.members:
            best_idx = max(range(len(self.members)), key=lambda i: self.members[i].elo)
            head_policy = self.members[best_idx].policy
            head_ppo = self.members[best_idx].ppo
        payload = {
            "iter": self.iter,
            "total_steps": self.total_steps,
            "ramp_on": self._ramp_on,
            "ramp_pos": self._ramp_pos,
            "best_bot": self._best_bot,
            # toujours le MEILLEUR membre en tÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Âªte de checkpoint : export,
            # serve et les vieux loaders fonctionnent sans connaÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â®tre le PBT
            "policy": head_policy.state_dict(),
            "optimizer": head_ppo.opt.state_dict(),
            "scaler": head_ppo.scaler.state_dict(),
            "league": self.league.state_dict(),
            "cfg": self.cfg,
            "policy_cfg": self.pol_cfg.__dict__,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (torch.cuda.get_rng_state_all()
                         if torch.cuda.is_available() else None),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        }
        if self.members:
            payload["pbt"] = {"members": [
                {"policy": m.policy.state_dict(),
                 "optimizer": m.ppo.opt.state_dict(),
                 "scaler": m.ppo.scaler.state_dict(),
                 "hypers": dict(m.hypers),
                 "elo": m.elo, "games": m.games}
                for m in self.members]}
        # ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©criture atomique : un kill pendant le save (stop de l'app,
        # autorestart) ne peut pas laisser un ckpt/latest.pt tronquÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
        latest = self.run_dir / "latest.pt"
        tmp_latest = latest.with_suffix(".tmp")
        shutil.copyfile(path, tmp_latest)
        tmp_latest.replace(latest)
        self._prune_checkpoints()
        return path

    def _prune_checkpoints(self) -> None:
        """Garde les `keep_ckpts` derniers + 1 checkpoint sur 500 (jalons)."""
        keep = self.cfg["keep_ckpts"]
        ckpts = sorted(self.run_dir.glob("ckpt_*.pt"))
        if len(ckpts) <= keep:
            return
        for p in ckpts[:-keep]:
            try:
                it = int(p.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            if it % 500 != 0:
                p.unlink(missing_ok=True)

    def _disk_best_bot(self) -> float:
        path = self.run_dir / "best.pt"
        if not path.exists():
            return -1.0
        try:
            return float(torch.load(path, map_location="cpu",
                                    weights_only=False).get("eval_bot", -1.0))
        except Exception:  # noqa: BLE001
            return -1.0

    def _write_best_policy(self, eval_bot: float,
                           iter_value: int | None = None,
                           total_steps: int | None = None) -> None:
        head_policy = self.policy
        if self.members:
            best_idx = max(range(len(self.members)), key=lambda i: self.members[i].elo)
            head_policy = self.members[best_idx].policy
        tmp = self.run_dir / "best.tmp"
        torch.save({
            "iter": self.iter if iter_value is None else iter_value,
            "total_steps": self.total_steps if total_steps is None else total_steps,
            "eval_bot": float(eval_bot),
            "policy": head_policy.state_dict(),
            "policy_cfg": self.pol_cfg.__dict__,
        }, tmp)
        tmp.replace(self.run_dir / "best.pt")

    @staticmethod
    def _seed_state_dict(policy, state_dict) -> float:
        own = policy.state_dict()
        filtered = {k: v for k, v in state_dict.items()
                    if k in own and own[k].shape == v.shape}
        if len(filtered) < 0.5 * len(own):
            print(f"[trainer] ATTENTION : architecture incompatible avec le "
                  f"checkpoint ({len(filtered)}/{len(own)} tenseurs repris) - "
                  f"la policy repart essentiellement de zero")
        policy.load_state_dict(filtered, strict=False)
        return len(filtered) / max(len(own), 1)
    @staticmethod
    def _fix_fused_optimizer_state(opt, device: torch.device) -> None:
        """Recharger un checkpoint d'Adam NON-fused dans l'Adam fused :
        load_state_dict ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©crase param_groups (fused redevient None) alors que
        l'attribut d'instance _step_supports_amp_scaling survit -> le
        GradScaler passe grad_scale/found_inf au chemin non-fused, qui
        asserte. Restaurer fused=True et remettre les 'step' sur le device."""
        if device.type != "cuda":
            return
        for group in opt.param_groups:
            group["fused"] = True
        for state in opt.state.values():
            step = state.get("step")
            if torch.is_tensor(step) and step.device != device:
                state["step"] = step.to(device)

    @staticmethod
    def _sync_optimizer_config(ppo: PPO) -> None:
        """Keep a resumed optimizer on the current run config.

        torch optimizer checkpoints carry param_groups, including the old lr.
        When a run is resumed after tightening the JSON config, loading the
        optimizer must not silently keep the old training rate.
        """
        for group in ppo.opt.param_groups:
            group["lr"] = ppo.cfg.lr

    def rotate_metrics(self) -> None:
        """Archive le metrics.jsonl d'un run prÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©cÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©dent du mÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Âªme nom (dÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©marrage
        frais sans --resume) : metrics-NNN.jsonl. Sans ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â§a, les courbes de
        l'app concatÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¨nent les runs et affichent des falaises trompeuses."""
        path = self.run_dir / "metrics.jsonl"
        if not path.exists():
            return
        n = 1
        while (self.run_dir / f"metrics-{n:03d}.jsonl").exists():
            n += 1
        path.rename(self.run_dir / f"metrics-{n:03d}.jsonl")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        ckpt_run = ckpt.get("cfg", {}).get("name") if isinstance(ckpt.get("cfg"), dict) else None
        same_run = ckpt_run == self.cfg.get("name")
        same_run_as_seed = bool(self.cfg.get("resume_same_run_as_seed", False))
        resume_as_seed = (
            bool(self.cfg.get("resume_as_seed", False))
            and (not same_run or same_run_as_seed)
        )
        full_resume = "league" in ckpt and not resume_as_seed
        fresh_optimizer = bool(self.cfg.get("fresh_optimizer_on_resume", False)) and resume_as_seed
        seed_match = self._seed_state_dict(self.policy, ckpt["policy"])
        compatible_seed = seed_match >= 0.5
        if compatible_seed and not fresh_optimizer:
            try:
                self.ppo.opt.load_state_dict(ckpt["optimizer"])
                self._fix_fused_optimizer_state(self.ppo.opt, self.device)
                if not (same_run and bool(self.cfg.get("fresh_optimizer_on_resume", False))):
                    self._sync_optimizer_config(self.ppo)
            except (ValueError, KeyError):
                pass
            if ckpt.get("scaler"):
                try:
                    self.ppo.scaler.load_state_dict(ckpt["scaler"])
                except RuntimeError:
                    pass
        if compatible_seed and fresh_optimizer:
            print("[trainer] optimizer frais sur resume (fresh_optimizer_on_resume=true)")
            self._sync_optimizer_config(self.ppo)
        if full_resume:
            self.league.load_state_dict(ckpt["league"])
            self.iter = ckpt["iter"]
            self.total_steps = ckpt.get("total_steps", 0)
        else:
            print(f"[trainer] seed policy-only depuis {path} ; nouveau run")
            self.iter = 0
            self.total_steps = 0
            self.rotate_metrics()
        self._combo_eval_origin = self.iter
        disk_best = self._disk_best_bot()
        ckpt_best = ckpt.get("best_bot", ckpt.get("eval_bot", -1.0))
        self._best_bot = max(ckpt_best, disk_best) if compatible_seed else disk_best
        if (not full_resume and compatible_seed
                and ckpt.get("eval_bot", -1.0) > disk_best):
            self._write_best_policy(float(ckpt["eval_bot"]),
                                    iter_value=0, total_steps=0)
        elif not full_resume and not compatible_seed:
            print("[trainer] score du checkpoint ignore : seed incompatible")
        if full_resume and "ramp_pos" in ckpt:
            self._ramp_on = ckpt["ramp_on"]
            self._ramp_pos = ckpt["ramp_pos"]
        elif full_resume and ckpt.get("ramp_start") is not None:
            self._ramp_on = True
            self._ramp_pos = min((self.iter - ckpt["ramp_start"])
                                 / max(self.cfg["shaping_decay_iters"], 1), 1.0)
        self._auto_shaping()
        self._auto_curriculum()
        if full_resume and ckpt.get("torch_rng") is not None:
            torch.set_rng_state(ckpt["torch_rng"].cpu())
        if full_resume and ckpt.get("cuda_rng") is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["cuda_rng"]])
            except (RuntimeError, IndexError):
                pass
        if full_resume and ckpt.get("numpy_rng") is not None:
            np.random.set_state(ckpt["numpy_rng"])
        if full_resume and ckpt.get("python_rng") is not None:
            random.setstate(ckpt["python_rng"])
        if self.members and not fresh_optimizer:
            self._load_population(ckpt)
        self._truncate_metrics()

    def _load_population(self, ckpt: dict) -> None:
        """Restaure la population, ou la SEED depuis un checkpoint
        single-policy (tous les membres partent des mÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Âªmes poids, hypers
        dÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©jÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â  diversifiÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©s par l'init ; membre 0 = hypers de base)."""
        pbt_state = ckpt.get("pbt")
        if pbt_state and len(pbt_state["members"]) == len(self.members):
            for m, sd in zip(self.members, pbt_state["members"]):
                m.policy.load_state_dict(sd["policy"], strict=False)
                try:
                    m.ppo.opt.load_state_dict(sd["optimizer"])
                    self._fix_fused_optimizer_state(m.ppo.opt, self.device)
                except (ValueError, KeyError):
                    pass
                if sd.get("scaler"):
                    try:
                        m.ppo.scaler.load_state_dict(sd["scaler"])
                    except RuntimeError:
                        pass
                m.hypers = dict(sd["hypers"])
                m.elo = float(sd["elo"])
                m.games = int(sd.get("games", 0))
                apply_hypers(m.ppo, m.hypers)
            print(f"[pbt] population de {len(self.members)} membres restaurÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©e")
        else:
            for m in self.members:
                if m.policy is not self.policy:    # membre 0 dÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©jÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â  chargÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©
                    self._seed_state_dict(m.policy, ckpt["policy"])
            print(f"[pbt] population de {len(self.members)} membres seedÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©e "
                  f"depuis le checkpoint single-policy (iter {self.iter})")

    def _truncate_metrics(self) -> None:
        """Supprime les lignes de mÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©triques POSTÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â°RIEURES ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â  l'itÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©ration
        reprise (progrÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¨s non sauvegardÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â© d'une session interrompue) : sans ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â§a,
        les courbes contiennent des itÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©rations en double aprÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¨s un resume."""
        path = self.run_dir / "metrics.jsonl"
        if not path.exists():
            return
        kept = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                if json.loads(line).get("iter", 0) <= self.iter:
                    kept.append(line)
            except json.JSONDecodeError:
                continue
        path.write_text("\n".join(kept) + ("\n" if kept else ""),
                        encoding="utf-8")

    def _log(self, metrics: dict) -> None:
        with open(self.run_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(metrics) + "\n")
        if self._tb is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb = SummaryWriter(str((self.run_dir / "tb").resolve()))
            except Exception:
                self._tb = False
        if self._tb:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._tb.add_scalar(f"judas/{k}", v, self.iter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--iters", type=int, default=None)
    args = ap.parse_args()

    cfg = {}
    if args.config:
        # utf-8-sig : tolÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¨re le BOM des ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©diteurs/outils Windows
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    trainer = Trainer(cfg)
    if args.resume:
        trainer.load(args.resume)
        print(f"repris ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â  l'itÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©ration {trainer.iter}")
    else:
        trainer.rotate_metrics()    # run frais : ne pas concatÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â©ner les courbes
    print(f"device={trainer.device} envs={trainer.N} rollout={trainer.T} "
          f"backend={'CUDA' if hasattr(trainer.sim, 'ext') else 'sim_ref(CPU)'}")
    trainer.train(args.iters)


if __name__ == "__main__":
    main()
