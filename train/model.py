"""JudasPolicy — policy transformer à actions hybrides.

Entrée  : historique d'observations [B, H, OBS_DIM] (H=16 ticks)
Encodage: MLP par tick -> + embedding positionnel -> TransformerEncoder
          (attention multi-têtes sur le temps) -> dernier token
Sorties : - Gaussienne tanh-squashed (Δyaw, Δpitch) normalisée [-1, 1]
          - Catégorielles 3 classes : forward, strafe (-1/0/+1)
          - Bernoulli : jump, sprint, attack
          - Value (critic)

Les actions échantillonnées sont stockées sous forme brute (pré-tanh +
indices) pour le calcul exact des log-probs PPO, et converties au format
sim [.., 7] par to_sim_actions().
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from sim.obs import OBS_DIM

LOG_STD_MIN, LOG_STD_MAX = -4.0, 1.0
TANH_EPS = 1e-6
COUNTER_HIT_REACH = 3.40
COUNTER_FAR_TRADE_REACH = 3.55
COUNTER_CLOSE_COUNTER_REACH = 2.10
COUNTER_HIT_SELECT_MIN_REACH = 2.45
COUNTER_HIT_SELECT_CLEAN_MIN_REACH = 2.70
COUNTER_HIT_SELECT_CLEAN_MAX_REACH = 3.15
COUNTER_HIT_SELECT_MIN_OWN_HURT = 0.86
COUNTER_HIT_SELECT_CLEAN_HURT = 1.01
COUNTER_HIT_SELECT_OPP_COOLDOWN = 0.025
COUNTER_RECOVERY_CLICK_HURT = 0.45
COUNTER_CLOSE_RECOVERY_CLICK_HURT = 0.55
COMBO_ATTACK_REACH = 3.40
COMBO_REHIT_ATTACK_REACH = 3.37
COMBO_REHIT_SPRINT_ATTACK_REACH = 3.35
COMBO_TAP_REACH = 3.40
COMBO_PRESS_REACH = 3.40
COMBO_CLOSE_RESET_REACH = 2.35
COMBO_S_TAP_REACH = 3.45
COMBO_Z_RELEASE_S_TAP_REACH = 2.95
COMBO_COOLDOWN_COAST_REACH = 4.15
COMBO_REHIT_COAST_REACH = 3.90
COMBO_REHIT_EDGE_BRAKE_REACH = 3.78
COMBO_HOLD_RESET_REACH = 1.65
POST_HIT_RESET_REACH = 3.45
COMBO_REHIT_PRESS_HURT = 0.55
COMBO_REHIT_CLICK_HURT = 0.55
LEADERBOARD_APPROACH_STRAFE_REACH = 14.00
LEADERBOARD_STRAFE_HOLD_SIDE = 0.75


def _categorical_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits.float(), dim=-1)
    return -(logp.exp() * logp).sum(-1)


def _gumbel_sample(logits: torch.Tensor) -> torch.Tensor:
    """Échantillonnage catégoriel par gumbel-max — équivalent exact de
    Categorical(logits).sample(), mais en ops tenseur pures : capturable par
    CUDA graph et sans surcoût d'objets distribution. Le bruit est tiré en
    float32 (un clamp 1e-9 sous-déborderait en fp16 sous autocast)."""
    u = torch.rand(logits.shape, device=logits.device,
                   dtype=torch.float32).clamp_(1e-9, 1.0)
    return (logits.float() - torch.log(-torch.log(u))).argmax(-1)


def _bernoulli_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    return (F.softplus(logits) - logits * torch.sigmoid(logits)).sum(-1)


@dataclass
class PolicyConfig:
    obs_dim: int = OBS_DIM
    history: int = 8
    d_model: int = 96           # taille du cerveau
    n_heads: int = 4
    n_layers: int = 2
    ff_mult: int = 4
    dropout: float = 0.0
    attention: bool = True      # False -> MLP sur l'historique aplati (plus rapide)
    aim_residual: float = 0.65    # direct yaw/pitch pull toward target body
    direct_movement_lock: bool = False  # no-back direct combo guard
    leaderboard_boxing: bool = False  # earlier A/D opener strafe for 10 CPS boxing
    direct_counter_attack_lock: bool = True  # force counter clicks in direct lock
    direct_hit_select_attack_lock: bool = True  # false lets policy learn hit-select click timing
    direct_hit_select_attack_bias: float = 0.0  # soft logit nudge in legal hit-select windows
    under_combo_attack_lock: bool = False  # legacy guard; keep false for counter-hits


class JudasPolicy(nn.Module):
    def __init__(self, cfg: PolicyConfig | None = None):
        super().__init__()
        self.cfg = cfg or PolicyConfig()
        c = self.cfg

        if c.attention:
            self.encoder = nn.Sequential(
                nn.Linear(c.obs_dim, c.d_model),
                nn.LayerNorm(c.d_model),
                nn.GELU(),
                nn.Linear(c.d_model, c.d_model),
            )
            self.pos_emb = nn.Parameter(torch.zeros(1, c.history, c.d_model))
            layer = nn.TransformerEncoderLayer(
                d_model=c.d_model, nhead=c.n_heads,
                dim_feedforward=c.d_model * c.ff_mult,
                dropout=c.dropout, activation="gelu",
                batch_first=True, norm_first=True)
            self.transformer = nn.TransformerEncoder(layer, num_layers=c.n_layers)
            self.norm = nn.LayerNorm(c.d_model)
            nn.init.normal_(self.pos_emb, std=0.02)
        else:
            # trunk MLP : historique aplati, n_layers couches cachées
            dims = [c.obs_dim * c.history] + [c.d_model] * max(c.n_layers, 1)
            mlp = []
            for a, b in zip(dims[:-1], dims[1:]):
                mlp += [nn.Linear(a, b), nn.LayerNorm(b), nn.GELU()]
            self.mlp = nn.Sequential(*mlp)

        self.mean_head = nn.Linear(c.d_model, 2)        # Δyaw, Δpitch
        self.log_std = nn.Parameter(torch.full((2,), -0.5))
        self.fwd_head = nn.Linear(c.d_model, 3)
        self.strafe_head = nn.Linear(c.d_model, 3)
        self.bin_head = nn.Linear(c.d_model, 3)         # jump, sprint, attack
        self.value_head = nn.Linear(c.d_model, 1)
        # tête auxiliaire : prédit l'état adverse au tick suivant (obs[0:7] =
        # position relative + distance + vélocité adverse) — force le trunk à
        # modéliser l'adversaire
        self.aux_head = nn.Linear(c.d_model, 7)

        for head in (self.mean_head, self.fwd_head, self.strafe_head, self.bin_head):
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    # ------------------------------------------------------------------ core
    def trunk(self, hist: torch.Tensor) -> torch.Tensor:
        """[B, H, obs] -> [B, d_model]."""
        if self.cfg.attention:
            x = self.encoder(hist) + self.pos_emb
            x = self.transformer(x)
            return self.norm(x[:, -1])
        return self.mlp(hist.flatten(1))

    def heads(self, z: torch.Tensor):
        mean = self.mean_head(z)
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).expand_as(mean)
        fwd_l = self.fwd_head(z)
        str_l = self.strafe_head(z)
        return (mean, log_std, fwd_l, str_l,
                self.bin_head(z), self.value_head(z).squeeze(-1))

    def mask_action_logits(self, hist: torch.Tensor, fwd_l: torch.Tensor,
                           str_l: torch.Tensor, bin_l: torch.Tensor):
        obs = hist[:, -1].float()
        if self.cfg.direct_movement_lock:
            fwd_l = fwd_l.clone()
            str_l = str_l.clone()
            prev_obs = hist[:, -2].float() if hist.shape[1] > 1 else obs
            dist = obs[:, 45] * 8.0
            last_fwd = obs[:, 40]
            last_strafe = obs[:, 41]
            last_sprint = obs[:, 43]
            opp_hurt = obs[:, 22]
            combo_adv = opp_hurt > obs[:, 21] + 0.05
            under_combo = obs[:, 21] > obs[:, 22] + 0.05
            rehit_press_ready = opp_hurt <= COMBO_REHIT_PRESS_HURT
            rehit_click_ready = opp_hurt <= COMBO_REHIT_CLICK_HURT
            landed_hit = obs[:, 31] > prev_obs[:, 31] + 0.004
            last_pressed = (last_fwd > 0.5) & (last_sprint > 0.5)
            last_z_tap = (last_fwd.abs() <= 0.1) & (last_sprint <= 0.5)
            last_release = last_z_tap | ((last_fwd < -0.5) & (last_sprint <= 0.5))
            point_blank = dist < 1.45
            point_blank_mirror = point_blank & ~combo_adv & ~under_combo
            under_counter_window = under_combo & (dist <= COUNTER_HIT_REACH)
            under_counter_legal, under_counter_forced = self.direct_counter_attack_windows(
                obs, dist, under_counter_window, rehit_click_ready)
            hit_select_release = (
                under_counter_window
                & (dist >= COUNTER_HIT_SELECT_CLEAN_MIN_REACH)
                & (dist <= COUNTER_HIT_SELECT_CLEAN_MAX_REACH)
                & (obs[:, 21] >= COUNTER_HIT_SELECT_MIN_OWN_HURT)
                & (obs[:, 21] <= COUNTER_HIT_SELECT_CLEAN_HURT)
                & (obs[:, 37] >= COUNTER_HIT_SELECT_OPP_COOLDOWN)
                & (obs[:, 23] <= 0.05)
                & rehit_click_ready
            )
            close_combo = combo_adv & (dist <= COMBO_TAP_REACH)
            too_close_combo = combo_adv & (dist <= COMBO_CLOSE_RESET_REACH)
            close_wait_release = (
                combo_adv
                & (~rehit_press_ready)
                & (dist <= COMBO_S_TAP_REACH)
                & (
                    last_pressed
                    | (last_z_tap & (dist <= COMBO_Z_RELEASE_S_TAP_REACH))
                )
            )
            hold_reset = combo_adv & last_release & (dist <= COMBO_HOLD_RESET_REACH)
            cooldown_hold_release = (
                combo_adv
                & last_release
                & (~rehit_press_ready)
                & (dist <= COMBO_Z_RELEASE_S_TAP_REACH)
            )
            ready_edge_release = (
                combo_adv
                & rehit_press_ready
                & (dist > COMBO_REHIT_ATTACK_REACH)
                & (dist <= COMBO_REHIT_EDGE_BRAKE_REACH)
            )
            ready_edge_attack_release = (
                combo_adv
                & rehit_click_ready
                & (dist > COMBO_REHIT_SPRINT_ATTACK_REACH)
                & (dist <= COMBO_REHIT_ATTACK_REACH)
            )
            ready_rehit_attack = (
                combo_adv
                & rehit_click_ready
                & (dist <= COMBO_REHIT_ATTACK_REACH)
            )
            ready_rehit_z_tap = (
                ready_rehit_attack
                & last_pressed
                & (dist > COMBO_Z_RELEASE_S_TAP_REACH)
                & (dist <= COMBO_REHIT_SPRINT_ATTACK_REACH)
            )
            ready_rehit_release_attack = torch.zeros_like(ready_rehit_attack)
            if bool(getattr(self.cfg, "leaderboard_boxing", False)):
                ready_rehit_release_attack = (
                    ready_rehit_attack
                    & last_z_tap
                    & (dist > COMBO_CLOSE_RESET_REACH)
                    & (dist <= COMBO_REHIT_ATTACK_REACH)
                )
            post_hit_reset = landed_hit & combo_adv & (dist <= POST_HIT_RESET_REACH)
            wait_rehit = combo_adv & (~rehit_press_ready) & (dist <= COMBO_PRESS_REACH)
            force_release = (
                (point_blank_mirror & last_pressed)
                | (too_close_combo & last_pressed)
                | close_wait_release
                | hold_reset
                | cooldown_hold_release
                | ready_edge_release
                | ready_edge_attack_release
                | (post_hit_reset & last_pressed & (dist <= COMBO_S_TAP_REACH))
                | hit_select_release
            )
            cooldown_coast = (
                combo_adv
                & (~rehit_press_ready)
                & (dist <= COMBO_COOLDOWN_COAST_REACH)
                & (~force_release)
            )
            ready_rehit_coast = (
                combo_adv
                & rehit_press_ready
                & (dist > COMBO_REHIT_ATTACK_REACH)
                & (dist <= COMBO_REHIT_COAST_REACH)
                & (~force_release)
            )
            force_z_tap = (
                (post_hit_reset & last_pressed & ~force_release)
                | (ready_rehit_z_tap & ~force_release)
                | (ready_rehit_release_attack & ~force_release)
                | (wait_rehit & ~too_close_combo & ~force_release)
                | cooldown_coast
                | ready_rehit_coast
            )
            force_neutral = force_release | force_z_tap
            repress_after_tap = (
                ((close_combo & rehit_press_ready) | point_blank_mirror)
                & ~last_pressed
            )
            counter_drive = under_combo & (~hit_select_release)
            combo_drive = combo_adv & (
                (dist > COMBO_PRESS_REACH) | (rehit_press_ready & (dist > COMBO_CLOSE_RESET_REACH))
            )
            neutral_drive = (~combo_adv) & (~under_combo) & (dist > COMBO_HOLD_RESET_REACH)
            far = ((~combo_adv & (dist > 3.35))
                   | (combo_adv & (dist > COMBO_PRESS_REACH)))
            force_forward = far | repress_after_tap | counter_drive | combo_drive | neutral_drive
            locked = torch.full_like(fwd_l[:, 1], torch.finfo(fwd_l.dtype).min)
            forward_target = force_forward & ~force_neutral
            fwd_l[:, 0] = locked
            fwd_l[:, 1] = torch.where(
                forward_target,
                locked,
                fwd_l[:, 1],
            )
            fwd_l[:, 2] = torch.where(
                forward_target,
                torch.full_like(fwd_l[:, 2], 8.0),
                fwd_l[:, 2],
            )
            fwd_l[:, 2] = torch.where(force_neutral, locked, fwd_l[:, 2])
            strafe_neutral = torch.ones_like(fwd_l[:, 1], dtype=torch.long)
            strafe_left = torch.zeros_like(strafe_neutral)
            strafe_right = torch.full_like(strafe_neutral, 2)
            side = obs[:, 1]
            side_target = torch.where(side >= 0.0, strafe_right, strafe_left)
            centered_target = torch.where(last_strafe < 0.0, strafe_left, strafe_right)
            strafe_target = torch.where(side.abs() <= 0.03, centered_target, side_target)
            strafe_active = (
                (dist >= 1.15)
                & (dist <= COMBO_ATTACK_REACH + 0.35)
            )
            if bool(getattr(self.cfg, "leaderboard_boxing", False)):
                approach_strafe = (
                    (dist >= 1.15)
                    & (dist <= LEADERBOARD_APPROACH_STRAFE_REACH)
                    & (obs[:, 12] >= 0.25)
                    & (~under_combo | (dist <= COUNTER_FAR_TRADE_REACH))
                )
                hold_target = torch.where(last_strafe < 0.0, strafe_left, strafe_right)
                hold_opener_side = (
                    approach_strafe
                    & (last_strafe.abs() > 0.5)
                    & (side.abs() <= LEADERBOARD_STRAFE_HOLD_SIDE)
                )
                strafe_target = torch.where(
                    hold_opener_side,
                    hold_target,
                    strafe_target,
                )
                strafe_active = strafe_active | approach_strafe
            if bool(getattr(self.cfg, "direct_counter_attack_lock", True)):
                strafe_active = strafe_active & (~under_counter_window)
            strafe_target = torch.where(strafe_active, strafe_target, strafe_neutral)
            str_l = torch.full_like(str_l, torch.finfo(str_l.dtype).min)
            str_l.scatter_(1, strafe_target.unsqueeze(1), torch.zeros_like(str_l[:, :1]))
            bin_l = bin_l.clone()
            bin_l[:, 0] = torch.full_like(bin_l[:, 0], torch.finfo(bin_l.dtype).min)
            sprint_on = torch.full_like(bin_l[:, 1], 8.0)
            bin_l[:, 1] = torch.where(
                force_forward & ~force_neutral,
                sprint_on,
                bin_l[:, 1],
            )
            neutral_attack = (~combo_adv) & (~under_combo) & (dist <= COMBO_ATTACK_REACH)
            combo_attack = combo_adv & rehit_click_ready & (dist <= COMBO_REHIT_ATTACK_REACH)
            counter_attack = (
                combo_attack | neutral_attack | under_counter_forced
            )
            attack_on = torch.full_like(bin_l[:, 2], 8.0)
            attack_off = torch.full_like(bin_l[:, 2], torch.finfo(bin_l.dtype).min)
            bin_l[:, 2] = torch.where(
                under_combo & (dist > COUNTER_HIT_REACH),
                attack_off,
                bin_l[:, 2],
            )
            bin_l[:, 2] = torch.where(
                under_counter_window & (~under_counter_legal),
                attack_off,
                bin_l[:, 2],
            )
            bin_l[:, 2] = torch.where(
                combo_adv & (~rehit_click_ready),
                attack_off,
                bin_l[:, 2],
            )
            bin_l[:, 2] = torch.where(
                combo_adv & (dist > COMBO_REHIT_ATTACK_REACH),
                attack_off,
                bin_l[:, 2],
            )
            hit_select_bias = float(getattr(self.cfg, "direct_hit_select_attack_bias", 0.0))
            if hit_select_bias > 0.0:
                hit_select_soft = under_counter_legal & (~under_counter_forced)
                bin_l[:, 2] = torch.where(
                    hit_select_soft,
                    bin_l[:, 2] + hit_select_bias,
                    bin_l[:, 2],
                )
            bin_l[:, 2] = torch.where(counter_attack, attack_on, bin_l[:, 2])
        if self.cfg.under_combo_attack_lock:
            bin_l = bin_l.clone()
            under_combo = obs[:, 21] > obs[:, 22] + 0.05
            locked = torch.full_like(bin_l[:, 2], torch.finfo(bin_l.dtype).min)
            bin_l[:, 2] = torch.where(under_combo, locked, bin_l[:, 2])
        return fwd_l, str_l, bin_l

    def direct_counter_attack_windows(self, obs: torch.Tensor,
                                      dist: torch.Tensor,
                                      under_counter_window: torch.Tensor,
                                      rehit_click_ready: torch.Tensor
                                      ) -> tuple[torch.Tensor, torch.Tensor]:
        zeros = torch.zeros_like(under_counter_window)
        if not bool(getattr(self.cfg, "direct_counter_attack_lock", True)):
            return zeros, zeros
        own_hurt = obs[:, 21]
        opp_cooldown = obs[:, 37]
        hit_select_hurt_ready = (
            (own_hurt >= COUNTER_HIT_SELECT_MIN_OWN_HURT)
            & (own_hurt <= COUNTER_HIT_SELECT_CLEAN_HURT)
            & (opp_cooldown >= COUNTER_HIT_SELECT_OPP_COOLDOWN)
        )
        close_counter_legal = (
            under_counter_window
            & (dist < COUNTER_CLOSE_COUNTER_REACH)
            & (own_hurt <= COUNTER_HIT_SELECT_CLEAN_HURT)
            & (obs[:, 23] <= 0.05)
            & rehit_click_ready
        )
        reach_recovery_legal = (
            under_counter_window
            & (dist >= COUNTER_HIT_SELECT_MIN_REACH)
            & (dist <= COUNTER_HIT_REACH)
            & (own_hurt <= COUNTER_RECOVERY_CLICK_HURT)
            & (obs[:, 23] <= 0.05)
            & rehit_click_ready
        )
        hit_select_legal = (
            under_counter_window
            & (dist >= COUNTER_HIT_SELECT_CLEAN_MIN_REACH)
            & (dist <= COUNTER_HIT_SELECT_CLEAN_MAX_REACH)
            & hit_select_hurt_ready
            & (obs[:, 23] <= 0.05)
            & rehit_click_ready
        )
        under_counter_legal = close_counter_legal | reach_recovery_legal | hit_select_legal
        if bool(getattr(self.cfg, "direct_hit_select_attack_lock", True)):
            return under_counter_legal, under_counter_legal
        return under_counter_legal, close_counter_legal | reach_recovery_legal

    def aim_residual(self, hist: torch.Tensor) -> torch.Tensor:
        gain = float(getattr(self.cfg, "aim_residual", 0.0))
        if gain <= 0.0:
            return torch.zeros(hist.shape[0], 2, dtype=hist.dtype, device=hist.device)
        obs = hist[:, -1].float()
        rot = (obs[:, 36] * 180.0).clamp_min(1.0)
        yaw_err = torch.rad2deg(torch.atan2(obs[:, 11], obs[:, 12]))
        pitch_err = obs[:, 13] * 90.0
        target = torch.stack([
            (yaw_err / rot).clamp(-1.0, 1.0),
            (pitch_err / rot).clamp(-1.0, 1.0),
        ], dim=-1)
        return (gain * target).to(dtype=hist.dtype)


    # ------------------------------------------------------------------- act
    @torch.no_grad()
    def act(self, hist: torch.Tensor, deterministic: bool = False) -> dict:
        """Échantillonne une action. Retourne un dict de tenseurs bruts +
        logp + value (tout sur le device de hist)."""
        mean, log_std, fwd_l, str_l, bin_l, value = self.heads(self.trunk(hist))
        mean = mean + self.aim_residual(hist).to(dtype=mean.dtype)
        fwd_l, str_l, bin_l = self.mask_action_logits(hist, fwd_l, str_l, bin_l)
        if deterministic:
            pre = mean
            fwd = fwd_l.argmax(-1)
            strafe = str_l.argmax(-1)
            bins = (bin_l > 0).float()
        else:
            pre = mean + torch.randn_like(mean) * log_std.exp()
            fwd = _gumbel_sample(fwd_l)
            strafe = _gumbel_sample(str_l)
            bins = (torch.rand(bin_l.shape, device=bin_l.device,
                               dtype=torch.float32)
                    < torch.sigmoid(bin_l.float())).to(bin_l.dtype)
        if self.cfg.direct_movement_lock:
            bins = bins.clone()
            bins[:, 0] = torch.zeros_like(bins[:, 0])
            bins[:, 1] = torch.where(
                fwd == 2,
                torch.ones_like(bins[:, 1]),
                torch.zeros_like(bins[:, 1]),
            )
            obs = hist[:, -1].float()
            dist = obs[:, 45] * 8.0
            combo_adv = obs[:, 22] > obs[:, 21] + 0.05
            under_combo = obs[:, 21] > obs[:, 22] + 0.05
            rehit_click_ready = obs[:, 22] <= COMBO_REHIT_CLICK_HURT
            _under_combo_legal, under_combo_attack = self.direct_counter_attack_windows(
                obs, dist, under_combo & (dist <= COUNTER_HIT_REACH), rehit_click_ready)
            in_combo_exchange = (
                (combo_adv & rehit_click_ready & (dist < 3.15))
                | under_combo_attack
            )
            bins[:, 2] = torch.where(
                in_combo_exchange,
                torch.ones_like(bins[:, 2]),
                bins[:, 2],
            )
        raw = {"pre": pre, "fwd": fwd, "strafe": strafe, "bins": bins}
        logp = self.log_prob(mean, log_std, fwd_l, str_l, bin_l, raw)
        return {**raw, "logp": logp, "value": value}

    # ------------------------------------------------------------- log-probs
    @staticmethod
    def log_prob(mean, log_std, fwd_l, str_l, bin_l, raw) -> torch.Tensor:
        std = log_std.exp()
        pre = raw["pre"]
        normal_lp = (-0.5 * (((pre - mean) / std) ** 2)
                     - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        # correction tanh (changement de variable)
        squash = torch.log(1 - torch.tanh(pre) ** 2 + TANH_EPS).sum(-1)
        lp_cont = normal_lp - squash
        lp_fwd = -F.cross_entropy(fwd_l, raw["fwd"].long(), reduction="none")
        lp_str = -F.cross_entropy(str_l, raw["strafe"].long(), reduction="none")
        lp_bin = -F.binary_cross_entropy_with_logits(
            bin_l, raw["bins"], reduction="none").sum(-1)
        return lp_cont + lp_fwd + lp_str + lp_bin

    def evaluate(self, hist: torch.Tensor, raw: dict, return_heads: bool = False):
        """Log-probs / entropie / value / prediction aux pour PPO."""
        z = self.trunk(hist)
        mean, log_std, fwd_l, str_l, bin_l, value = self.heads(z)
        mean = mean + self.aim_residual(hist).to(dtype=mean.dtype)
        fwd_l, str_l, bin_l = self.mask_action_logits(hist, fwd_l, str_l, bin_l)
        logp = self.log_prob(mean, log_std, fwd_l, str_l, bin_l, raw)
        ent_cont = (0.5 * (1.0 + math.log(2 * math.pi)) + log_std.float()).sum(-1)
        ent_fwd = _categorical_entropy_from_logits(fwd_l)
        ent_str = _categorical_entropy_from_logits(str_l)
        ent_bin = _bernoulli_entropy_from_logits(bin_l)
        entropy = ent_cont + ent_fwd + ent_str + ent_bin
        aux = self.aux_head(z)
        if return_heads:
            return logp, entropy, value, aux, (mean, fwd_l, str_l, bin_l)
        return logp, entropy, value, aux

def to_sim_actions(raw: dict) -> torch.Tensor:
    """dict d'actions brutes [B, ...] -> tenseur sim [B, 7] (convention sim/)."""
    pre = raw["pre"]
    out = torch.zeros(pre.shape[0], 7, dtype=torch.float32, device=pre.device)
    out[:, 0:2] = torch.tanh(pre)
    out[:, 2] = raw["fwd"].float() - 1.0      # 0,1,2 -> -1,0,1
    out[:, 3] = raw["strafe"].float() - 1.0
    out[:, 4:7] = raw["bins"]
    out[:, 5] = torch.where(out[:, 2] <= 0.5, torch.zeros_like(out[:, 5]), out[:, 5])
    return out
