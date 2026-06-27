"""PPO clipped + GAE, mixed precision, sur les fenÃªtres d'attention."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .buffer import RolloutBuffer
from .model import (
    COMBO_ATTACK_REACH,
    COMBO_PRESS_REACH,
    COMBO_CLOSE_RESET_REACH,
    COMBO_COOLDOWN_COAST_REACH,
    COMBO_HOLD_RESET_REACH,
    COMBO_REHIT_CLICK_HURT,
    COMBO_REHIT_ATTACK_REACH,
    COMBO_REHIT_PRESS_HURT,
    COMBO_REHIT_COAST_REACH,
    COMBO_REHIT_EDGE_BRAKE_REACH,
    COMBO_REHIT_SPRINT_ATTACK_REACH,
    COMBO_S_TAP_REACH,
    COMBO_TAP_REACH,
    COMBO_Z_RELEASE_S_TAP_REACH,
    COUNTER_CLOSE_COUNTER_REACH,
    COUNTER_CLOSE_RECOVERY_CLICK_HURT,
    COUNTER_HIT_REACH,
    COUNTER_HIT_SELECT_CLEAN_HURT,
    COUNTER_HIT_SELECT_CLEAN_MAX_REACH,
    COUNTER_HIT_SELECT_CLEAN_MIN_REACH,
    COUNTER_HIT_SELECT_MIN_REACH,
    COUNTER_HIT_SELECT_MIN_OWN_HURT,
    COUNTER_HIT_SELECT_OPP_COOLDOWN,
    COUNTER_RECOVERY_CLICK_HURT,
    POST_HIT_RESET_REACH,
    JudasPolicy,
)


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.995
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 3
    minibatch_size: int = 16384
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    amp: bool = True
    # stabilitÃ© / qualitÃ© d'apprentissage
    target_kl: float = 0.02      # early-stop des epochs si KL moyen dÃ©passe 1.5x
    # 0 = dÃ©sactivÃ©. ATTENTION : sur notre Ã©chelle de rewards (win Â±10), un
    # clip serrÃ© (ex 0.2) Ã©trangle le critic -> advantages bruitÃ©s -> policy
    # qui n'apprend plus (clip frac ~1%, entropy bloquÃ©e au max).
    value_clip: float = 0.0
    anneal: bool = True          # lr et entropie -> 0 linÃ©airement sur le run
    sample_frac: float = 0.5     # fraction du buffer utilisÃ©e par epoch (vitesse)
    aux_coef: float = 0.05       # loss auxiliaire : prÃ©dire l'adversaire Ã  t+1
    coach_coef: float = 0.0      # imitation legere: aim + forward + sprint/attack
    coach_until: float = 0.08    # fraction du run ou le coaching decroit vers 0



def _coach_loss(mean: torch.Tensor, fwd_l: torch.Tensor, str_l: torch.Tensor,
                bin_l: torch.Tensor, hist: torch.Tensor,
                leaderboard_boxing: bool = False,
                direct_counter_attack_lock: bool = True,
                age: torch.Tensor | None = None,
                opener_ticks: int = 20) -> torch.Tensor:
    """Small behavior-cloning loss toward direct, range-aware boxing.

    The coach keeps direct body aim and forward pressure. For combo training,
    teaching backsteps or neutral orbiting near range is worse than occasional
    over-commitment: PPO otherwise discovers lateral escape loops instead of
    re-hit pressure.
    """
    obs = hist[:, -1].float()
    rot = (obs[:, 36] * 180.0).clamp_min(1.0)
    yaw_err = torch.rad2deg(torch.atan2(obs[:, 11], obs[:, 12]))
    pitch_err = obs[:, 13] * 90.0
    target_turn = torch.stack([
        (yaw_err / rot).clamp(-1.0, 1.0),
        (pitch_err / rot).clamp(-1.0, 1.0),
    ], dim=-1)
    turn_loss = F.smooth_l1_loss(
        torch.tanh(mean.float()), target_turn, reduction="none",
    ).sum(-1).mean()

    n = hist.shape[0]
    dist = (obs[:, 45] * 8.0).clamp_min(1.0e-6)
    own_hurt = obs[:, 21].clamp(0.0, 1.0)
    opp_hurt = obs[:, 22].clamp(0.0, 1.0)
    opp_cooldown = obs[:, 37].clamp(0.0, 1.0)
    combo_adv = (opp_hurt - own_hurt).clamp(0.0, 1.0)
    combo_disadv = (own_hurt - opp_hurt).clamp(0.0, 1.0)
    yaw_score = (1.0 - yaw_err.abs() / 90.0).clamp(0.0, 1.0)
    pitch_score = (1.0 - pitch_err.abs() / 45.0).clamp(0.0, 1.0)
    aim_score = yaw_score * pitch_score

    combo_adv_active = combo_adv > 0.10
    combo_disadv_active = combo_disadv > 0.10
    rehit_press_ready = opp_hurt <= COMBO_REHIT_PRESS_HURT
    rehit_click_ready = opp_hurt <= COMBO_REHIT_CLICK_HURT
    last_fwd = obs[:, 40].clamp(-1.0, 1.0)
    last_strafe = obs[:, 41].clamp(-1.0, 1.0)
    last_sprint = obs[:, 43].clamp(0.0, 1.0)
    last_pressed = (last_fwd > 0.5) & (last_sprint > 0.5)
    last_z_tap = (last_fwd.abs() <= 0.1) & (last_sprint <= 0.5)
    last_release = last_z_tap | ((last_fwd < -0.5) & (last_sprint <= 0.5))
    prev_obs = hist[:, -2].float() if hist.shape[1] > 1 else obs
    landed_hit = obs[:, 31] > prev_obs[:, 31] + 0.004
    point_blank = (dist < 1.45) & ~combo_adv_active & ~combo_disadv_active
    close_combo = combo_adv_active & (dist <= COMBO_TAP_REACH)
    too_close_combo = combo_adv_active & (dist <= COMBO_CLOSE_RESET_REACH)
    close_wait_release = (
        combo_adv_active
        & (~rehit_press_ready)
        & (dist <= COMBO_S_TAP_REACH)
        & (
            last_pressed
            | (last_z_tap & (dist <= COMBO_Z_RELEASE_S_TAP_REACH))
        )
    )
    hold_reset = combo_adv_active & last_release & (dist <= COMBO_HOLD_RESET_REACH)
    cooldown_hold_release = (
        combo_adv_active
        & last_release
        & (~rehit_press_ready)
        & (dist <= COMBO_Z_RELEASE_S_TAP_REACH)
    )
    ready_edge_release = (
        combo_adv_active
        & rehit_press_ready
        & (dist > COMBO_REHIT_ATTACK_REACH)
        & (dist <= COMBO_REHIT_EDGE_BRAKE_REACH)
    )
    ready_edge_attack_release = (
        combo_adv_active
        & rehit_click_ready
        & (dist > COMBO_REHIT_SPRINT_ATTACK_REACH)
        & (dist <= COMBO_REHIT_ATTACK_REACH)
    )
    post_hit_reset = landed_hit & combo_adv_active & (dist <= POST_HIT_RESET_REACH)
    wait_rehit = combo_adv_active & (~rehit_press_ready) & (dist <= COMBO_PRESS_REACH)
    clean_hit_select_release = (
        combo_disadv_active
        & (dist >= COUNTER_HIT_SELECT_CLEAN_MIN_REACH)
        & (dist <= COUNTER_HIT_SELECT_CLEAN_MAX_REACH)
        & (own_hurt >= COUNTER_HIT_SELECT_MIN_OWN_HURT)
        & (own_hurt <= COUNTER_HIT_SELECT_CLEAN_HURT)
        & (opp_cooldown >= COUNTER_HIT_SELECT_OPP_COOLDOWN)
        & (obs[:, 23] <= 0.05)
        & (opp_hurt <= COMBO_REHIT_CLICK_HURT)
    )
    force_release = ((point_blank & last_pressed)
                     | (too_close_combo & last_pressed)
                     | close_wait_release
                     | hold_reset
                     | cooldown_hold_release
                     | ready_edge_release
                     | ready_edge_attack_release
                     | (post_hit_reset & last_pressed & (dist <= COMBO_S_TAP_REACH))
                     | clean_hit_select_release)
    cooldown_coast = (
        combo_adv_active
        & (~rehit_press_ready)
        & (dist <= COMBO_COOLDOWN_COAST_REACH)
        & (~force_release)
    )
    ready_rehit_coast = (
        combo_adv_active
        & rehit_press_ready
        & (dist > COMBO_REHIT_ATTACK_REACH)
        & (dist <= COMBO_REHIT_COAST_REACH)
        & (~force_release)
    )
    force_neutral = (
        (post_hit_reset & last_pressed)
        | wait_rehit
        | cooldown_coast
        | ready_rehit_coast
        | force_release
    )
    repress_after_tap = ((close_combo & rehit_press_ready) | point_blank) & ~last_pressed
    opener_drive = torch.zeros((n,), dtype=torch.bool, device=hist.device)
    if leaderboard_boxing and age is not None and opener_ticks > 0:
        opener_drive = (
            (age.to(device=hist.device) >= 0)
            & (age.to(device=hist.device) < opener_ticks)
            & (~combo_adv_active)
            & (~combo_disadv_active)
            & (dist >= 2.15)
            & (dist <= 7.85)
            & (obs[:, 12] >= 0.25)
        )
        force_neutral = force_neutral & ~opener_drive
    should_press = (
        (~combo_adv_active & (dist > 3.35))
        | (combo_adv_active & (dist > COMBO_PRESS_REACH))
        | (combo_adv_active & rehit_press_ready & (dist > COMBO_CLOSE_RESET_REACH))
        | repress_after_tap
        | (combo_disadv_active & (dist >= 2.05) & (~clean_hit_select_release))
        | (~combo_adv_active & ~combo_disadv_active & (dist > 1.65))
        | opener_drive
    )
    in_hit_band = dist <= 3.60
    counter_window = combo_disadv_active & (dist <= COUNTER_HIT_REACH)
    own_hurt = obs[:, 21]
    counter_spacing_legal = (
        ((dist < COUNTER_CLOSE_COUNTER_REACH)
         & (own_hurt <= COUNTER_CLOSE_RECOVERY_CLICK_HURT))
        | ((dist >= COUNTER_HIT_SELECT_CLEAN_MIN_REACH)
           & (dist <= COUNTER_HIT_SELECT_CLEAN_MAX_REACH)
           & (own_hurt >= COUNTER_HIT_SELECT_MIN_OWN_HURT)
           & (own_hurt <= COUNTER_HIT_SELECT_CLEAN_HURT)
           & (opp_cooldown >= COUNTER_HIT_SELECT_OPP_COOLDOWN))
    )
    counter_click_ready = (
        counter_window
        & counter_spacing_legal
        & (obs[:, 23] <= 0.05)
        & (opp_hurt <= COMBO_REHIT_CLICK_HURT)
        & (aim_score > 0.06)
    )
    close_counter_ready = (
        counter_click_ready
        & (dist < COUNTER_CLOSE_COUNTER_REACH)
        & (own_hurt <= COUNTER_CLOSE_RECOVERY_CLICK_HURT)
    )
    hit_select_ready = (
        counter_click_ready
        & (dist >= COUNTER_HIT_SELECT_CLEAN_MIN_REACH)
        & (dist <= COUNTER_HIT_SELECT_CLEAN_MAX_REACH)
        & (own_hurt >= COUNTER_HIT_SELECT_MIN_OWN_HURT)
        & (own_hurt <= COUNTER_HIT_SELECT_CLEAN_HURT)
        & (opp_cooldown >= COUNTER_HIT_SELECT_OPP_COOLDOWN)
    )
    if leaderboard_boxing:
        should_counter = close_counter_ready
    elif not direct_counter_attack_lock:
        should_counter = counter_click_ready
    else:
        should_counter = counter_window
    should_attack = (aim_score > 0.06) & (
        (combo_adv_active & rehit_click_ready & (dist <= COMBO_REHIT_ATTACK_REACH))
        | should_counter
        | (~combo_adv_active & ~combo_disadv_active & in_hit_band)
    )

    neutral = torch.full((n,), 1, dtype=torch.long, device=hist.device)
    forward = torch.full((n,), 2, dtype=torch.long, device=hist.device)
    fwd_target = torch.where(
        force_neutral,
        neutral,
        torch.where(should_press, forward, neutral),
    )
    should_sprint = fwd_target == 2
    strafe_left = torch.zeros((n,), dtype=torch.long, device=hist.device)
    strafe_right = torch.full((n,), 2, dtype=torch.long, device=hist.device)
    side = obs[:, 1]
    side_target = torch.where(side >= 0.0, strafe_right, strafe_left)
    centered_target = torch.where(last_strafe < 0.0, strafe_left, strafe_right)
    strafe_target = torch.where(side.abs() <= 0.03, centered_target, side_target)
    strafe_active = (
        (dist >= 1.15)
        & (dist <= COMBO_ATTACK_REACH + 0.35)
    )
    if leaderboard_boxing:
        approach_strafe = (
            (dist >= 2.15)
            & (dist <= 7.85)
            & (obs[:, 12] >= 0.25)
            & (~combo_disadv_active | (dist <= COUNTER_HIT_REACH))
        )
        strafe_active = strafe_active | approach_strafe
    strafe_target = torch.where(strafe_active, strafe_target, neutral)
    if leaderboard_boxing and direct_counter_attack_lock:
        counter_lineup = combo_disadv_active & (dist <= COUNTER_HIT_REACH)
        strafe_target = torch.where(counter_lineup, neutral, strafe_target)
    move_loss = (F.cross_entropy(fwd_l.float(), fwd_target)
                 + 1.75 * F.cross_entropy(str_l.float(), strafe_target))

    bin_target = torch.zeros_like(bin_l.float())
    bin_target[:, 1] = should_sprint.float()     # sprint while keeping combo pressure
    bin_target[:, 2] = should_attack.float()     # click whenever aim + reach are plausible
    bin_loss_raw = F.binary_cross_entropy_with_logits(
        bin_l.float(), bin_target, reduction="none",
    )
    bin_weight = torch.empty_like(bin_loss_raw)
    bin_weight[:, 0] = 0.15
    bin_weight[:, 1] = 1.15
    counter_focus = (
        4.50 * close_counter_ready.float()
        + 1.35 * hit_select_ready.float() * bin_target[:, 2]
        + 1.75 * counter_window.float() * (1.0 - counter_click_ready.float())
    )
    combo_focus = 1.25 * (
        combo_adv_active & rehit_click_ready & (dist <= COMBO_REHIT_ATTACK_REACH)
    ).float()
    bin_weight[:, 2] = 2.10 + counter_focus + combo_focus
    bin_loss = (bin_loss_raw * bin_weight).sum(-1).mean()
    return 10.0 * turn_loss + 2.75 * move_loss + 1.30 * bin_loss

class PPO:
    def __init__(self, policy: JudasPolicy, cfg: PPOConfig, device: torch.device):
        self.policy = policy
        self.cfg = cfg
        self.device = device
        # fused Adam : une seule passe kernel sur tous les params (CUDA)
        self.opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr, eps=1e-5,
                                    fused=device.type == "cuda")
        self.use_amp = cfg.amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def update(self, buf: RolloutBuffer, learner_mask: torch.Tensor,
               last_value: torch.Tensor, progress: float = 0.0) -> dict:
        """progress dans [0,1] : avancement du run (pour l'annealing)."""
        cfg = self.cfg
        frac = max(1.0 - progress, 0.05) if cfg.anneal else 1.0
        for g in self.opt.param_groups:
            g["lr"] = cfg.lr * frac
        ent_coef = cfg.ent_coef * frac
        buf.compute_gae(last_value, cfg.gamma, cfg.lam)

        # Ã©chantillons (t, b) des agents contrÃ´lÃ©s par le learner
        b_keep = torch.nonzero(learner_mask, as_tuple=False).squeeze(-1)
        T = buf.T
        t_all = torch.arange(T, device=self.device).repeat_interleave(b_keep.numel())
        b_all = b_keep.repeat(T)
        n = t_all.numel()

        adv_all = buf.adv[t_all, b_all]
        adv_mean, adv_std = adv_all.mean(), adv_all.std().clamp_min(1e-6)

        # stats accumulÃ©es sur GPU : une seule synchro Ã  la fin de l'update
        acc = {k: torch.zeros((), device=self.device)
               for k in ("loss_pi", "loss_v", "loss_aux", "loss_coach",
                          "entropy", "clip_frac", "approx_kl")}
        n_updates = 0

        n_used = max(int(n * min(max(cfg.sample_frac, 0.05), 1.0)),
                     cfg.minibatch_size)
        n_used = min(n_used, n)
        stop = False
        for _ in range(cfg.epochs):
            if stop:
                break
            kl_epoch = torch.zeros((), device=self.device)
            n_mb = 0
            # permutation complÃ¨te -> sous-ensemble frais Ã  chaque epoch
            perm = torch.randperm(n, device=self.device)[:n_used]
            for start in range(0, n_used, cfg.minibatch_size):
                mb = perm[start:start + cfg.minibatch_size]
                ti, bi = t_all[mb], b_all[mb]
                hist = buf.windows(ti, bi)
                raw = buf.raw_at(ti, bi)
                old_logp = buf.logp[ti, bi]
                old_value = buf.value[ti, bi]
                ret = buf.ret[ti, bi]
                adv = (buf.adv[ti, bi] - adv_mean) / adv_std

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    need_coach = (cfg.coach_coef > 0 and (cfg.coach_until <= 0 or progress < cfg.coach_until))
                    if need_coach:
                        logp, entropy, value, aux, coach_heads = self.policy.evaluate(
                            hist, raw, return_heads=True,
                        )
                    else:
                        logp, entropy, value, aux = self.policy.evaluate(hist, raw)
                        coach_heads = None
                    ratio = (logp - old_logp).exp()
                    s1 = ratio * adv
                    s2 = ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv
                    loss_pi = -torch.min(s1, s2).mean()
                    if cfg.value_clip > 0:
                        v_clip = old_value + (value - old_value).clamp(
                            -cfg.value_clip, cfg.value_clip)
                        loss_v = 0.5 * torch.max((value - ret).pow(2),
                                                 (v_clip - ret).pow(2)).mean()
                    else:
                        loss_v = 0.5 * (value - ret).pow(2).mean()
                    loss_ent = -entropy.mean()
                    loss = (loss_pi + cfg.vf_coef * loss_v
                            + ent_coef * loss_ent)
                    loss_coach = torch.zeros((), device=self.device)
                    if cfg.aux_coef > 0:
                        # cible : Ã©tat adverse au tick suivant (obs[0:7]),
                        # invalide en fin de buffer / fin d'Ã©pisode
                        valid = ((ti < buf.T - 1).float()
                                 * (1.0 - buf.done[ti, bi]))
                        nxt = buf.obs[(ti + 1).clamp(max=buf.T - 1), bi][:, :7]
                        loss_aux = (((aux - nxt).pow(2).mean(-1) * valid).sum()
                                    / valid.sum().clamp(min=1.0))
                        loss = loss + cfg.aux_coef * loss_aux
                    if cfg.coach_coef > 0:
                        if cfg.coach_until > 0:
                            coach_scale = cfg.coach_coef * max(
                                0.0, 1.0 - progress / cfg.coach_until,
                            )
                        else:
                            coach_scale = cfg.coach_coef
                        if coach_scale > 0.0:
                            loss_coach = _coach_loss(
                                *coach_heads,
                                hist,
                                bool(getattr(self.policy.cfg, "leaderboard_boxing", False)),
                                bool(getattr(self.policy.cfg, "direct_counter_attack_lock", True)),
                                age=buf.age[ti, bi],
                            )
                            loss = loss + coach_scale * loss_coach

                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(),
                                               cfg.max_grad_norm)
                self.scaler.step(self.opt)
                self.scaler.update()

                with torch.no_grad():
                    kl = (old_logp - logp).mean().detach()
                    acc["loss_pi"] += loss_pi.detach()
                    acc["loss_v"] += loss_v.detach()
                    if cfg.aux_coef > 0:
                        acc["loss_aux"] += loss_aux.detach()
                    acc["loss_coach"] += loss_coach.detach()
                    acc["entropy"] += entropy.mean().detach()
                    acc["clip_frac"] += ((ratio - 1).abs() > cfg.clip).float().mean()
                    acc["approx_kl"] += kl
                    kl_epoch += kl
                n_updates += 1
                n_mb += 1

            # early-stop : une seule synchro GPU par epoch
            if cfg.target_kl > 0 and n_mb > 0:
                if float(kl_epoch.item()) / n_mb > 1.5 * cfg.target_kl:
                    stop = True

        d = max(n_updates, 1)
        out = {k: float(v.item()) / d for k, v in acc.items()}
        out["lr"] = cfg.lr * frac
        return out
