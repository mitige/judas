"""LiveSession — inférence temps réel pour le mod Forge.

Charge un modèle TorchScript exporté (train/export.py), maintient
l'historique d'observations, applique l'humanisation (clamp rotation + CPS)
et la latence d'inférence visée est < 2 ms sur GPU.
"""

import json
import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import torch

from sim.config import SimConfig
from sim.obs import OBS_DIM, build_obs
from sim_ref import HumanizationConfig
from train.model import (
    COMBO_ATTACK_REACH,
    COMBO_CLOSE_RESET_REACH,
    COMBO_COOLDOWN_COAST_REACH,
    COMBO_HOLD_RESET_REACH,
    COMBO_PRESS_REACH,
    COMBO_REHIT_ATTACK_REACH,
    COMBO_REHIT_CLICK_HURT,
    COMBO_REHIT_COAST_REACH,
    COMBO_REHIT_EDGE_BRAKE_REACH,
    COMBO_REHIT_PRESS_HURT,
    COMBO_REHIT_SPRINT_ATTACK_REACH,
    COMBO_S_TAP_REACH,
    COMBO_TAP_REACH,
    COMBO_Z_RELEASE_S_TAP_REACH,
    COUNTER_CLOSE_COUNTER_REACH,
    COUNTER_FAR_TRADE_REACH,
    COUNTER_CLOSE_RECOVERY_CLICK_HURT,
    COUNTER_HIT_REACH,
    COUNTER_HIT_SELECT_CLEAN_HURT,
    COUNTER_HIT_SELECT_CLEAN_MAX_REACH,
    COUNTER_HIT_SELECT_CLEAN_MIN_REACH,
    COUNTER_HIT_SELECT_MIN_REACH,
    COUNTER_HIT_SELECT_MIN_OWN_HURT,
    COUNTER_HIT_SELECT_OPP_COOLDOWN,
    POST_HIT_RESET_REACH,
)

from .protocol import ArenaCalib, action_to_msg, player_from_msg


PLAYER_EYE_HEIGHT = 1.62
PLAYER_AIM_MIN_Y = 0.25
PLAYER_AIM_MAX_Y = 1.45
PLAYER_AIM_MIN_HORIZONTAL_DIST = 2.0
COMBO_DIRECT_RUNS = (
    "combo_god_recovery_kb092_combo12",
    "combo_god_leaderboard10_combo12",
    "combo_god_attn96_combo12",
    "combo_god_consistent",
    "combo_god_candidate_freshopt",
    "combo_god_bodyaim96_combo12",
    "combo_god_countertap96_combo12",
    "combo_god_directpad_lock_combo12",
)
LEADERBOARD_BOXING_RUNS = (
    "combo_god_recovery_kb092_combo12",
    "combo_god_leaderboard10_combo12",
    "combo_god_attn96_combo12",
    "combo_god_consistent",
    "combo_god_candidate_freshopt",
    "combo_god_bodyaim96_combo12",
)
LEADERBOARD_OPENER_TICKS = 48
LEADERBOARD_MAX_NEUTRAL_RESET_TICKS = 1
LEADERBOARD_APPROACH_STRAFE_REACH = 14.00
LEADERBOARD_STRAFE_HOLD_SIDE = 0.75
LEADERBOARD_COUNTER_STRAFE_REACQUIRE_SIDE = 0.20
LEADERBOARD_POINT_BLANK_NEUTRAL_REACH = 1.25
LEADERBOARD_LONG_COMBO_MIN = 6
LEADERBOARD_LONG_COMBO_NEUTRAL_REACH = 2.85
LEADERBOARD_LONG_COMBO_REHIT_BRAKE_REACH = 3.45
LEADERBOARD_PRESSURE_ATTACK_REACH = 6.10
LEADERBOARD_COUNTER_ATTACK_REACH = COUNTER_HIT_REACH
LEADERBOARD_OPP_COMBO_BREAK_TICKS = 12
LEADERBOARD_WTAP_RELEASE_TICKS = 1

def _wrap_degrees(v: float) -> float:
    v = math.fmod(v, 360.0)
    if v >= 180.0:
        v -= 360.0
    if v < -180.0:
        v += 360.0
    return v


@dataclass
class HumanizeConfig:
    """Humanisation moteur de l'inférence live.

    `level` est le CURSEUR unique de la visée :
      0.0 = visée BRUTE du modèle (parfaite, pleine vitesse, zéro latence) ;
      1.0 = humain maximal (dodge killaura/aim mais moins précis).
    Monte `level` par petits pas jusqu'à ce que l'anticheat passe tout en gardant
    une visée qui reste bonne. Les champs ci-dessous sont les MAXIMA atteints à
    level=1 (et scalés linéairement par level).

    La cadence de clic (cps_*) est TOUJOURS humanisée, INDÉPENDAMMENT du level
    (elle ne touche pas la visée) -> bat l'Autoclicker sans coût sur l'aim."""
    level: float = 0.0           # 0 = visée modèle parfaite ; 1 = humain max
    enabled: bool = False        # legacy : {"human": true} == level 1.0
    reaction_ticks: int = 3      # latence de réaction à level=1 (~150 ms)
    smooth: float = 0.6          # lissage de vitesse à level=1
    tremor_deg: float = 0.30     # tremblement de main à level=1 (deg)
    max_turn: float = 22.0       # vitesse de rotation min à level=1 (deg/tick)
    drift_deg: float = 0.30      # dérive basse fréquence à level=1 (deg)
    micro_pause_prob: float = 0.12  # proba de micro-pause à level=1
    gain_var: float = 0.5        # variation de gain de suivi à level=1
    cps_jitter: float = 0.9      # CPS (toujours actif) : écart-type cooldown
    cps_pause_prob: float = 0.12 # CPS (toujours actif) : proba clic plus lent


@dataclass
class CounterAssistConfig:
    enabled: bool = False
    max_reach: float = 5.0
    recovery_cps: float = 10.0
    jitter_prob: float = 0.18
    # Minimum aim window. The live rotation cap can widen this because direct
    # mode applies the corrective turn and click in the same action packet.
    yaw_deg: float = 38.0
    pitch_deg: float = 35.0
    min_own_hurt: int = 2


@dataclass
class HitSelectAssistConfig:
    enabled: bool = False
    release_movement: bool = True
    force_click: bool = True
    setup_min_reach: float = COUNTER_HIT_SELECT_MIN_REACH
    setup_max_reach: float = COUNTER_HIT_REACH
    min_reach: float = COUNTER_HIT_SELECT_CLEAN_MIN_REACH
    max_reach: float = COUNTER_HIT_SELECT_CLEAN_MAX_REACH
    min_own_hurt: float = COUNTER_HIT_SELECT_MIN_OWN_HURT
    max_own_hurt: float = COUNTER_HIT_SELECT_CLEAN_HURT
    min_opp_cooldown: float = COUNTER_HIT_SELECT_OPP_COOLDOWN
    yaw_deg: float = 38.0
    pitch_deg: float = 35.0


@dataclass
class AimSmoothingConfig:
    enabled: bool = False
    strength: float = 0.22
    snap_deg: float = 0.55


@dataclass
class AutoGappleConfig:
    enabled: bool = True
    health_threshold: float = 14.0
    critical_health_threshold: float = 8.0
    safe_distance: float = 11.50
    retreat_enabled: bool = True
    retreat_distance: float = 18.0
    absorption_threshold: float = -1.0
    fast_retreat: bool = True
    retreat_hops: bool = True
    sprint_hop_hold: bool = True
    avoid_obstacles: bool = True
    retreat_strafe: bool = True
    wall_slide: bool = True
    retreat_speed_lock: bool = True
    retreat_velocity_assist: bool = True
    retreat_speed_first: bool = True
    retreat_full_speed: bool = True
    retreat_speed_floor: float = 4.50
    retreat_max_speed: float = 4.80
    retreat_accel: float = 5.50
    retreat_sprint_retap: bool = True
    retreat_sprint_retap_ticks: int = 2
    retreat_air_control: bool = True
    retreat_step_assist: bool = True
    retreat_step_height: float = 1.20
    fallback_retreat: bool = True
    retreat_input_lock: bool = True
    force_sprint_retreat: bool = True
    release_retreat_on_hit: bool = True
    critical_rearm_only: bool = True
    critical_trapped_eat: bool = True
    retreat_turn_limit_deg: float = 360.0
    eating_retreat_turn_limit_deg: float = 360.0
    retreat_path_hold_ticks: int = 2
    retreat_stuck_abort_ticks: int = 4
    retreat_min_ticks: int = 0
    retreat_max_ticks: int = 64
    critical_retreat_max_ticks: int = 6
    critical_eat_commit_ticks: int = 12
    combat_recovery_ticks: int = 6
    retreat_strafe_hold_ticks: int = 5
    retreat_obstacle_jump_hold_ticks: int = 60
    retreat_obstacle_escape_ticks: int = 120
    retreat_panic_speed: bool = True
    retreat_obstacle_lookahead: float = 24.00
    critical_trapped_stuck_ticks: int = 2


@dataclass
class AutoJumpConfig:
    enabled: bool = False


@dataclass
class KnockbackDumpConfig:
    enabled: bool = False


@dataclass
class FriendsConfig:
    enabled: bool = False
    names: list[str] = field(default_factory=list)


@dataclass
class LiveParams:
    max_cps: float = 10.0
    max_rot_speed: float = 190.0
    arena: ArenaCalib = field(default_factory=lambda: ArenaCalib(size_x=40.0, size_z=40.0))
    enabled: bool = True
    human: HumanizeConfig = field(default_factory=HumanizeConfig)
    aim_smoothing: AimSmoothingConfig = field(default_factory=AimSmoothingConfig)
    hit_select_assist: HitSelectAssistConfig = field(default_factory=HitSelectAssistConfig)
    counter_assist: CounterAssistConfig = field(default_factory=CounterAssistConfig)
    auto_gapple: AutoGappleConfig = field(default_factory=AutoGappleConfig)
    auto_jump: AutoJumpConfig = field(default_factory=AutoJumpConfig)
    knockback_dump: KnockbackDumpConfig = field(default_factory=KnockbackDumpConfig)
    friends: FriendsConfig = field(default_factory=FriendsConfig)


class LiveSession:
    def __init__(self, device: str | None = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.params = LiveParams()
        # Repère arène AUTO : origine = centre des 2 joueurs, sol = min Y, déduit
        # au début de chaque combat. Garde murs (obs[25..28]) et hauteurs
        # (obs[46,47]) dans la distribution d'entraînement quelles que soient les
        # coords monde du box (sinon obs hors plage -> la visée part au sol).
        self.auto_frame = True
        self._frame: ArenaCalib | None = None
        self.model = None
        self.model_path: str | None = None
        # 8 = PolicyConfig.history par défaut (16 ferait crasher un modèle
        # entraîné aux défauts si le .json d'export manque)
        self.history = 8
        self.max_ticks = SimConfig().max_ticks
        self.aim_smooth = 0.0      # inertie de visée (du meta d'export)
        self.hist = None
        self.last_action = [0.0] * 7
        self.click_cooldown = 0
        self.tick = 0
        self._last_hits: tuple | None = None
        self._combo_reset_ticks = 0
        self._opener_strafe_sign = 0.0
        self._leaderboard_neutral_ticks = 0
        self._own_combo_streak = 0
        self._opp_combo_streak = 0
        self._opp_combo_break_ticks = 0
        self._opp_combo_break_sign = 0.0
        self._leaderboard_wtap_ticks = 0
        self._urgent_counter_click = False
        self._hit_select_fired_in_window = False
        self._aim_dyaw = 0.0       # état du modèle moteur (miroir du sim)
        self._aim_dpitch = 0.0
        self._smooth_aim_yaw = 0.0
        self._smooth_aim_pitch = 0.0
        self.last_latency_ms = 0.0
        # diag : JUDAS_DEBUG_ACTIONS=1 -> log périodique de l'action brute du
        # modèle (avant/après humanisation) pour diagnostiquer training vs deploy
        self._debug = os.environ.get("JUDAS_DEBUG_ACTIONS", "") not in (
            "", "0", "false", "False")
        action_log = os.environ.get("JUDAS_LIVE_ACTION_LOG", "").strip()
        self._action_log_path = Path(action_log) if action_log else None
        # humanisation moteur : RNG dédié + état du filtre 'main humaine'
        self._rng = random.Random()
        self._reset_human_aim()

    def _log_action_line(self, line: str) -> None:
        if self._action_log_path is None:
            return
        try:
            self._action_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._action_log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass
    # ------------------------------------------------------------------ model
    def load(self, path: str) -> dict:
        p = Path(path)
        meta = {}
        meta_path = p.with_suffix(".json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        self.history = int(meta.get("history", 8))
        # contrat d'obs identique à l'entraînement : o[34] dépend de max_ticks
        self.max_ticks = int(meta.get("max_ticks", SimConfig().max_ticks))
        # Contrat de déploiement : reprendre les limites du run exporté. Un
        # modèle combo entraîné à ~190 deg/tick se dégrade fortement si l'app
        # live reste sur l'ancien cap 40 deg/tick.
        if "max_cps" in meta:
            self.params.max_cps = float(meta["max_cps"])
        if "max_rot_speed" in meta:
            self.params.max_rot_speed = float(meta["max_rot_speed"])
        if "arena_size_x" in meta:
            self.params.arena.size_x = float(meta["arena_size_x"])
        if "arena_size_z" in meta:
            self.params.arena.size_z = float(meta["arena_size_z"])
        self._frame = None
        # modèle moteur de visée : même inertie qu'à l'entraînement
        self.aim_smooth = float(meta.get("aim_smooth", 0.0))
        self.model = torch.jit.load(str(p), map_location=self.device).eval()
        self.model_path = str(p)
        self.reset()
        self._log_action_line(
            "event=start model=%s history=%d max_ticks=%d"
            % (self.model_path, self.history, self.max_ticks))
        # warmup (compile les kernels)
        with torch.no_grad():
            for _ in range(3):
                self.model(self.hist)
        return {"model": self.model_path, "history": self.history}

    def reset(self) -> None:
        self.hist = torch.zeros(1, self.history, OBS_DIM, device=self.device)
        self.last_action = [0.0] * 7
        self.click_cooldown = 0
        self.tick = 0
        self._last_hits = None
        self._combo_reset_ticks = 0
        self._opener_strafe_sign = 0.0
        self._leaderboard_neutral_ticks = 0
        self._own_combo_streak = 0
        self._opp_combo_streak = 0
        self._opp_combo_break_ticks = 0
        self._opp_combo_break_sign = 0.0
        self._leaderboard_wtap_ticks = 0
        self._urgent_counter_click = False
        self._hit_select_fired_in_window = False
        self._aim_dyaw = 0.0
        self._aim_dpitch = 0.0
        self._frame = None      # recalculé au 1er état du prochain combat
        self._reset_human_aim()

    def _reset_human_aim(self) -> None:
        n = max(1, int(self.params.human.reaction_ticks) + 1)
        self._raim_yaw = deque([0.0], maxlen=n)
        self._raim_pitch = deque([0.0], maxlen=n)
        self._aim_vy = 0.0
        self._aim_vp = 0.0
        self._drift_y = 0.0
        self._drift_p = 0.0

    def _human_aim(self, cmd_yaw: float, cmd_pitch: float,
                   level: float) -> tuple[float, float]:
        """Filtre 'main humaine' SCALÉ par `level` (0->1). À level proche de 0 la
        sortie ≈ commande du modèle (visée quasi intacte) ; à level=1, pleine
        humanisation (réaction + gain variable + micro-pauses + dérive +
        tremblement) contre Aim (Linear/Constant) et Killaura (Pattern). Le
        modèle dans la boucle recadre la dérive sur la cible."""
        hc = self.params.human
        rng = self._rng
        L = 0.0 if level < 0.0 else (1.0 if level > 1.0 else level)
        # 1. latence de réaction PROPORTIONNELLE au level (0 tick à level 0)
        self._raim_yaw.append(cmd_yaw)
        self._raim_pitch.append(cmd_pitch)
        delay = int(round(L * hc.reaction_ticks))
        idx = max(0, len(self._raim_yaw) - 1 - delay)
        ry, rp = self._raim_yaw[idx], self._raim_pitch[idx]
        # 2. suivi à gain variable (lissage et variation scalés par level)
        k0 = 1.0 - max(0.0, min(0.97, L * hc.smooth))
        gv = L * hc.gain_var
        ky = k0 * (1.0 + gv * rng.uniform(-1.0, 1.0))
        kp = k0 * (1.0 + gv * rng.uniform(-1.0, 1.0))
        if rng.random() < L * hc.micro_pause_prob:
            ky *= 0.15
            kp *= 0.15
        self._aim_vy += (ry - self._aim_vy) * ky
        self._aim_vp += (rp - self._aim_vp) * kp
        # 3. dérive basse fréquence + tremblement, scalés par level
        dd = L * hc.drift_deg
        self._drift_y += -0.15 * self._drift_y + rng.gauss(0.0, dd)
        self._drift_p += -0.15 * self._drift_p + rng.gauss(0.0, dd)
        td = L * hc.tremor_deg
        act = min(1.0, (abs(self._aim_vy) + abs(self._aim_vp)) / 8.0)
        oy = self._aim_vy + self._drift_y + rng.gauss(0.0, td * (0.5 + act))
        op = self._aim_vp + self._drift_p + rng.gauss(0.0, td * (0.4 + act))
        # 4. butée de vitesse : max_rot_speed (level 0) -> max_turn (level 1)
        full = self.params.max_rot_speed
        m = max(1.0, full - L * (full - min(hc.max_turn, full)))
        return max(-m, min(m, oy)), max(-m, min(m, op))

    @staticmethod
    def _stabilize_axis(cmd: float, err: float, limit: float,
                        deadband: float) -> float:
        """Locks live/direct aim to the current target error.

        State collection and action application are one game tick apart. Using
        the current target error here removes the visible follow delay when the
        target moves laterally, especially in DIRECT mode.
        """
        limit = max(1.0, float(limit))
        if abs(err) <= deadband:
            return 0.0
        return max(-limit, min(limit, err))

    @staticmethod
    def _aim_errors_deg(own, opp) -> tuple[float, float]:
        dx = opp.x - own.x
        dz = opp.z - own.z
        dist_h = math.sqrt(dx * dx + dz * dz)
        if dist_h <= 1.0e-9:
            return 0.0, 0.0
        yaw_to = math.degrees(math.atan2(-dx, dz))
        yaw_err = _wrap_degrees(yaw_to - own.yaw)

        eye_y = own.y + PLAYER_EYE_HEIGHT
        lo = opp.y + PLAYER_AIM_MIN_Y
        hi = opp.y + PLAYER_AIM_MAX_Y
        aim_y = min(max(eye_y, lo), hi)
        # At point blank the horizontal vector can be only a few centimeters.
        # Using that raw distance makes body aim explode toward +/-90 degrees
        # when players overlap, which is exactly the visible "aims sky/floor"
        # failure in direct mode. Keep yaw reactive, but solve pitch against a
        # stable body-distance floor so the crosshair stays on the hitbox.
        pitch_dist_h = max(dist_h, PLAYER_AIM_MIN_HORIZONTAL_DIST)
        pitch_to = -math.degrees(math.atan2(aim_y - eye_y, pitch_dist_h))
        return yaw_err, pitch_to - own.pitch

    def _stabilize_live_aim(self, action: list, own, opp) -> None:
        yaw_err, pitch_err = self._aim_errors_deg(own, opp)
        limit = self.params.max_rot_speed
        if self.params.aim_smoothing.enabled:
            smoothing = max(0.0, min(0.92, float(self.params.aim_smoothing.strength)))
            snap = max(0.0, float(self.params.aim_smoothing.snap_deg))
            if abs(yaw_err) <= snap:
                self._smooth_aim_yaw = yaw_err
            else:
                self._smooth_aim_yaw = (
                    self._smooth_aim_yaw * smoothing
                    + yaw_err * (1.0 - smoothing)
                )
            if abs(pitch_err) <= snap:
                self._smooth_aim_pitch = pitch_err
            else:
                self._smooth_aim_pitch = (
                    self._smooth_aim_pitch * smoothing
                    + pitch_err * (1.0 - smoothing)
                )
            yaw_err = self._smooth_aim_yaw
            pitch_err = self._smooth_aim_pitch
        else:
            self._smooth_aim_yaw = yaw_err
            self._smooth_aim_pitch = pitch_err
        action[0] = self._stabilize_axis(action[0], yaw_err, limit, 0.05)
        action[1] = self._stabilize_axis(action[1], pitch_err, limit, 0.05)

    def _hit_select_assist_window(self, own, opp, dist_h: float | None = None) -> tuple[bool, bool]:
        assist = self.params.hit_select_assist
        if not assist.enabled:
            return False, False
        if dist_h is None:
            dx = opp.x - own.x
            dz = opp.z - own.z
            dist_h = math.sqrt(dx * dx + dz * dz)
        combo_disadv = own.hurt_resistant_time > opp.hurt_resistant_time + 1
        own_hurt_norm = own.hurt_resistant_time / 20.0
        opp_hurt_norm = opp.hurt_resistant_time / 20.0
        opp_click_norm = opp.click_cooldown / 20.0
        setup = (
            combo_disadv
            and float(assist.setup_min_reach) <= dist_h <= float(assist.setup_max_reach)
            and float(assist.min_own_hurt) <= own_hurt_norm <= float(assist.max_own_hurt)
        )
        ready = (
            setup
            and float(assist.min_reach) <= dist_h <= float(assist.max_reach)
            and opp_click_norm >= float(assist.min_opp_cooldown)
            and opp_hurt_norm <= COMBO_REHIT_CLICK_HURT
            and self.click_cooldown <= 0
        )
        return setup, ready

    def _apply_hit_select_assist(self, action: list, own, opp) -> tuple[bool, bool]:
        assist = self.params.hit_select_assist
        if not assist.enabled:
            self._hit_select_fired_in_window = False
            return False, False
        dx = opp.x - own.x
        dz = opp.z - own.z
        dist_h = math.sqrt(dx * dx + dz * dz)
        setup, ready = self._hit_select_assist_window(own, opp, dist_h)
        if not setup:
            self._hit_select_fired_in_window = False
            return False, False
        if assist.release_movement:
            action[2] = 0.0
            action[5] = 0.0
        action[6] = 0.0
        target_not_damageable = (
            (opp.hurt_resistant_time / 20.0) > COMBO_REHIT_CLICK_HURT
            or (opp.click_cooldown / 20.0) < float(assist.min_opp_cooldown)
        )
        if target_not_damageable:
            self._hit_select_fired_in_window = False
        if not (assist.force_click and ready):
            return True, False
        if self._hit_select_fired_in_window:
            return True, False
        yaw_err, pitch_err = self._aim_errors_deg(own, opp)
        yaw_limit = max(float(assist.yaw_deg), float(self.params.max_rot_speed))
        pitch_limit = max(float(assist.pitch_deg), float(self.params.max_rot_speed))
        if abs(yaw_err) > yaw_limit or abs(pitch_err) > pitch_limit:
            return True, False
        action[6] = 1.0
        self._hit_select_fired_in_window = True
        self._urgent_counter_click = True
        return True, True

    def _uses_combo_direct_pad(self) -> bool:
        stem = Path(self.model_path).stem if self.model_path else ""
        return any(run in stem for run in COMBO_DIRECT_RUNS)

    def _uses_leaderboard_boxing(self) -> bool:
        stem = Path(self.model_path).stem if self.model_path else ""
        return any(run in stem for run in LEADERBOARD_BOXING_RUNS)

    def _enforce_combo_direct_pad(self, action: list, own, opp) -> None:
        self._urgent_counter_click = False
        if not self._uses_combo_direct_pad():
            return
        action[4] = 0.0   # no jump escape/desync

        dx = opp.x - own.x
        dz = opp.z - own.z
        dist_h = math.sqrt(dx * dx + dz * dz)
        sy = math.sin(math.radians(own.yaw))
        cy = math.cos(math.radians(own.yaw))
        side = dx * cy + dz * sy
        combo_adv = opp.hurt_resistant_time > own.hurt_resistant_time + 1
        combo_disadv = own.hurt_resistant_time > opp.hurt_resistant_time + 1
        leaderboard_profile = self._uses_leaderboard_boxing()
        hit_deficit = max(0, int(opp.hits) - int(own.hits))
        leaderboard_opening = (
            leaderboard_profile
            and self.tick < LEADERBOARD_OPENER_TICKS
            and 1.15 <= dist_h <= LEADERBOARD_APPROACH_STRAFE_REACH
        )
        opp_hurt_norm = opp.hurt_resistant_time / 20.0
        own_hurt_norm = own.hurt_resistant_time / 20.0
        rehit_press_ready = opp_hurt_norm <= COMBO_REHIT_PRESS_HURT
        rehit_click_ready = opp_hurt_norm <= COMBO_REHIT_CLICK_HURT
        was_forward_sprint = self.last_action[2] > 0.5 and self.last_action[5] > 0.5
        last_z_tap = abs(self.last_action[2]) <= 0.1 and self.last_action[5] <= 0.5
        last_release = last_z_tap or (self.last_action[2] < -0.5 and self.last_action[5] <= 0.5)

        post_hit_reset = False
        if self._combo_reset_ticks > 0:
            post_hit_reset = combo_adv and dist_h <= POST_HIT_RESET_REACH
            if post_hit_reset:
                self._combo_reset_ticks -= 1
            else:
                self._combo_reset_ticks = 0

        point_blank_mirror = dist_h < 1.45 and not combo_adv and not combo_disadv
        close_combo = combo_adv and dist_h <= COMBO_TAP_REACH
        too_close_combo = combo_adv and dist_h <= COMBO_CLOSE_RESET_REACH
        close_wait_release = (
            combo_adv
            and not rehit_press_ready
            and dist_h <= COMBO_S_TAP_REACH
            and (
                was_forward_sprint
                or (last_z_tap and dist_h <= COMBO_Z_RELEASE_S_TAP_REACH)
            )
        )
        hold_reset = combo_adv and last_release and dist_h <= COMBO_HOLD_RESET_REACH
        cooldown_hold_release = (
            combo_adv
            and last_release
            and not rehit_press_ready
            and dist_h <= COMBO_Z_RELEASE_S_TAP_REACH
        )
        ready_edge_release = (
            combo_adv
            and rehit_press_ready
            and COMBO_REHIT_ATTACK_REACH < dist_h <= COMBO_REHIT_EDGE_BRAKE_REACH
        )
        ready_edge_attack_release = (
            combo_adv
            and rehit_click_ready
            and COMBO_REHIT_SPRINT_ATTACK_REACH < dist_h <= COMBO_REHIT_ATTACK_REACH
        )
        ready_rehit_attack = (
            combo_adv
            and rehit_click_ready
            and dist_h <= COMBO_REHIT_ATTACK_REACH
        )
        ready_rehit_z_tap = (
            ready_rehit_attack
            and was_forward_sprint
            and COMBO_Z_RELEASE_S_TAP_REACH < dist_h <= COMBO_REHIT_SPRINT_ATTACK_REACH
        )
        force_release = (
            (point_blank_mirror and was_forward_sprint)
            or (too_close_combo and was_forward_sprint)
            or close_wait_release
            or hold_reset
            or cooldown_hold_release
            or ready_edge_release
            or ready_edge_attack_release
            or (post_hit_reset and was_forward_sprint and dist_h <= COMBO_S_TAP_REACH)
        )
        cooldown_coast = (
            combo_adv
            and not rehit_press_ready
            and dist_h <= COMBO_COOLDOWN_COAST_REACH
            and not force_release
        )
        ready_rehit_coast = (
            combo_adv
            and rehit_press_ready
            and COMBO_REHIT_ATTACK_REACH < dist_h <= COMBO_REHIT_COAST_REACH
            and not force_release
        )
        wait_rehit = combo_adv and not rehit_press_ready and dist_h <= COMBO_PRESS_REACH
        force_z_tap = (
            (post_hit_reset and was_forward_sprint and not force_release)
            or ready_rehit_z_tap
            or (wait_rehit and not too_close_combo and not force_release)
            or cooldown_coast
            or ready_rehit_coast
        )
        force_neutral = force_release or force_z_tap
        repress_after_tap = ((close_combo and rehit_press_ready)
                             or point_blank_mirror) and not was_forward_sprint
        leaderboard_hit_select_release = (
            leaderboard_profile
            and combo_disadv
            and COUNTER_HIT_SELECT_CLEAN_MIN_REACH <= dist_h <= COUNTER_HIT_SELECT_CLEAN_MAX_REACH
            and own_hurt_norm >= COUNTER_HIT_SELECT_MIN_OWN_HURT
            and own_hurt_norm <= COUNTER_HIT_SELECT_CLEAN_HURT
            and (opp.click_cooldown / 20.0) >= COUNTER_HIT_SELECT_OPP_COOLDOWN
            and self.click_cooldown <= 0
            and rehit_click_ready
        )
        leaderboard_lateral_reacquire = (
            leaderboard_profile
            and combo_disadv
            and dist_h <= LEADERBOARD_COUNTER_ATTACK_REACH
            and abs(side) > LEADERBOARD_COUNTER_STRAFE_REACQUIRE_SIDE
            and own_hurt_norm < COUNTER_HIT_SELECT_MIN_OWN_HURT
        )
        leaderboard_counter_drive = (
            leaderboard_profile
            and combo_disadv
            and dist_h >= 1.15
            and not leaderboard_hit_select_release
            and not leaderboard_lateral_reacquire
        )
        leaderboard_score_rescue = (
            leaderboard_profile
            and not combo_adv
            and hit_deficit >= 3
            and 1.15 <= dist_h <= COUNTER_HIT_REACH
        )
        counter_drive = combo_disadv and (dist_h >= 2.05 or leaderboard_counter_drive)
        combo_drive = combo_adv and (
            dist_h > COMBO_PRESS_REACH
            or (rehit_press_ready and dist_h > COMBO_CLOSE_RESET_REACH)
        )
        neutral_drive = (not combo_adv and not combo_disadv
                         and dist_h > COMBO_HOLD_RESET_REACH)
        far = ((not combo_adv and dist_h > 3.35)
               or (combo_adv and dist_h > COMBO_PRESS_REACH))
        force_forward = far or repress_after_tap or counter_drive or combo_drive or neutral_drive

        if leaderboard_counter_drive or leaderboard_score_rescue:
            force_neutral = False
            force_forward = True
        if leaderboard_hit_select_release:
            force_neutral = True
            force_forward = False
        if leaderboard_lateral_reacquire:
            force_neutral = True
            force_forward = False

        if (leaderboard_profile and not combo_adv and hit_deficit >= 2
                and dist_h <= LEADERBOARD_PRESSURE_ATTACK_REACH):
            force_neutral = False
            force_forward = True

        if leaderboard_profile and force_neutral:
            if self._leaderboard_neutral_ticks >= LEADERBOARD_MAX_NEUTRAL_RESET_TICKS:
                force_neutral = False
                force_forward = True
            else:
                self._leaderboard_neutral_ticks += 1
        else:
            self._leaderboard_neutral_ticks = 0
        if (leaderboard_profile and combo_adv
                and dist_h <= LEADERBOARD_POINT_BLANK_NEUTRAL_REACH):
            force_neutral = True
            force_forward = False
        if leaderboard_profile and self._own_combo_streak >= LEADERBOARD_LONG_COMBO_MIN:
            if combo_adv and dist_h <= LEADERBOARD_LONG_COMBO_NEUTRAL_REACH:
                force_neutral = True
                force_forward = False
            if (combo_adv and rehit_click_ready
                    and dist_h <= LEADERBOARD_LONG_COMBO_REHIT_BRAKE_REACH):
                force_neutral = True
                force_forward = False
        scheduled_wtap = (
            leaderboard_profile
            and self._leaderboard_wtap_ticks > 0
            and not combo_disadv
            and 1.15 <= dist_h <= COMBO_PRESS_REACH
        )
        if scheduled_wtap:
            force_neutral = True
            force_forward = False
            self._leaderboard_wtap_ticks -= 1
        elif self._leaderboard_wtap_ticks > 0:
            self._leaderboard_wtap_ticks = 0

        if force_neutral:
            action[2] = 0.0
            action[5] = 0.0
        elif force_forward:
            action[2] = 1.0
            action[5] = 1.0
        elif action[2] > 0.5:
            action[5] = 1.0
        else:
            action[2] = 0.0
            action[5] = 0.0
        if action[2] < 0.0:
            action[2] = 0.0
            action[5] = 0.0
        if leaderboard_opening and dist_h > 1.55 and not scheduled_wtap:
            action[2] = 1.0
            action[5] = 1.0

        point_blank_clinch = leaderboard_profile and dist_h < 1.15
        far_counter_reentry = (
            leaderboard_profile
            and combo_disadv
            and dist_h > LEADERBOARD_COUNTER_ATTACK_REACH
        )
        counter_lineup = (
            leaderboard_profile
            and combo_disadv
            and dist_h <= LEADERBOARD_COUNTER_ATTACK_REACH
        )
        strafe_active = (
            dist_h >= 1.15
            and dist_h <= COMBO_ATTACK_REACH + 0.35
        )
        if leaderboard_profile:
            approach_strafe = (
                1.15 <= dist_h <= LEADERBOARD_APPROACH_STRAFE_REACH
            )
            strafe_active = strafe_active or approach_strafe
        counter_break_strafe = (
            leaderboard_profile
            and combo_disadv
            and self._opp_combo_break_ticks > 0
            and dist_h >= 2.80
        )
        if far_counter_reentry:
            action[3] = 0.0
        elif counter_lineup:
            action[3] = 0.0
        elif counter_break_strafe:
            action[3] = self._opp_combo_break_sign
            self._opp_combo_break_ticks -= 1
        elif point_blank_clinch:
            action[3] = self.last_action[3] if abs(self.last_action[3]) > 0.5 else (
                1.0 if side >= 0.0 else -1.0)
        elif leaderboard_opening:
            if abs(self._opener_strafe_sign) <= 0.5:
                self._opener_strafe_sign = 1.0 if side >= 0.0 else -1.0
            action[3] = self._opener_strafe_sign
        elif strafe_active:
            if (leaderboard_profile and (combo_disadv or leaderboard_score_rescue)
                    and abs(side) > LEADERBOARD_COUNTER_STRAFE_REACQUIRE_SIDE):
                action[3] = 1.0 if side >= 0.0 else -1.0
            elif (leaderboard_profile and abs(self.last_action[3]) > 0.5
                    and abs(side) <= LEADERBOARD_STRAFE_HOLD_SIDE):
                action[3] = 1.0 if self.last_action[3] > 0.0 else -1.0
            elif abs(side) <= 0.24:
                action[3] = self.last_action[3] if abs(self.last_action[3]) > 0.5 else 1.0
            else:
                action[3] = 1.0 if side >= 0.0 else -1.0
        else:
            action[3] = 0.0

        yaw_err, pitch_err = self._aim_errors_deg(own, opp)
        can_lock_aim = (
            abs(yaw_err) <= self.params.max_rot_speed + 1.0e-6
            and abs(pitch_err) <= self.params.max_rot_speed + 1.0e-6
        )
        under_counter_reach = (
            LEADERBOARD_COUNTER_ATTACK_REACH if leaderboard_profile
            else COUNTER_FAR_TRADE_REACH
        )
        under_counter_window = combo_disadv and dist_h <= under_counter_reach
        leaderboard_score_counter_window = (
            leaderboard_profile
            and not combo_adv
            and hit_deficit >= 3
            and dist_h <= COUNTER_HIT_REACH
        )
        combo_attack = combo_adv and rehit_click_ready and dist_h <= COMBO_REHIT_ATTACK_REACH
        neutral_attack = (not combo_adv and not combo_disadv
                          and dist_h <= COMBO_ATTACK_REACH)
        opener_attack = leaderboard_opening and dist_h <= COMBO_ATTACK_REACH
        leaderboard_aim_click = abs(yaw_err) <= max(90.0, min(180.0, self.params.max_rot_speed + 20.0))
        leaderboard_pressure_attack = (
            leaderboard_profile
            and not combo_disadv
            and dist_h <= LEADERBOARD_PRESSURE_ATTACK_REACH
            and leaderboard_aim_click
        )
        model_attack = action[6] > 0.5
        close_counter_attack = (
            combo_disadv
            and dist_h < COUNTER_CLOSE_COUNTER_REACH
            and own_hurt_norm <= COUNTER_CLOSE_RECOVERY_CLICK_HURT
            and self.click_cooldown <= 0
            and rehit_click_ready
        )
        leaderboard_counter_attack = (
            leaderboard_profile
            and close_counter_attack
        )
        action[6] = 0.0
        if leaderboard_counter_attack:
            action[6] = 1.0
            self._urgent_counter_click = True
        elif leaderboard_hit_select_release:
            action[6] = 1.0 if model_attack and can_lock_aim else 0.0
            self._urgent_counter_click = action[6] > 0.5
        elif leaderboard_pressure_attack:
            action[6] = 1.0
        elif combo_attack and can_lock_aim:
            action[6] = 1.0
        elif ((under_counter_window and not leaderboard_profile)
              or leaderboard_score_counter_window):
            action[6] = 1.0
            self._urgent_counter_click = True
        elif opener_attack and can_lock_aim:
            action[6] = 1.0
        elif neutral_attack and can_lock_aim:
            action[6] = 1.0
        far_counter_limit = (
            LEADERBOARD_COUNTER_ATTACK_REACH if leaderboard_profile
            else COUNTER_FAR_TRADE_REACH
        )
        if combo_disadv and dist_h > far_counter_limit:
            action[6] = 0.0
            self._urgent_counter_click = False

    def _apply_counter_assist(self, action: list, own, opp) -> bool:
        assist = self.params.counter_assist
        if not assist.enabled or action[6] > 0.5:
            return False

        combo_disadv = own.hurt_resistant_time > opp.hurt_resistant_time + 1
        if not combo_disadv or own.hurt_resistant_time < int(assist.min_own_hurt):
            return False

        dx = opp.x - own.x
        dz = opp.z - own.z
        dist_h = math.sqrt(dx * dx + dz * dz)
        if dist_h < 0.65 or dist_h > float(assist.max_reach):
            return False

        yaw_err, pitch_err = self._aim_errors_deg(own, opp)
        yaw_limit = max(float(assist.yaw_deg), float(self.params.max_rot_speed))
        pitch_limit = max(float(assist.pitch_deg), float(self.params.max_rot_speed))
        if abs(yaw_err) > yaw_limit:
            return False
        if abs(pitch_err) > pitch_limit:
            return False

        action[6] = 1.0
        self._urgent_counter_click = True
        return True

    def _record_applied_action(self, action: list) -> None:
        applied = list(action)
        rot = max(1.0, self.params.max_rot_speed)
        applied[0] = max(-1.0, min(1.0, applied[0] / rot))
        applied[1] = max(-1.0, min(1.0, applied[1] / rot))
        self.last_action = applied

    def _human_click_cooldown(self, max_cps: float) -> int:
        """Cooldown de clic humain à forte variance (bat Autoclicker
        Consistency/Rounded/Range/Deviation) : CPS tirée dans une plage
        [max-4, max] + jitter, pauses occasionnelles, et ARRONDI STOCHASTIQUE
        (pas toujours le même entier de ticks). Jamais > max_cps (cd >= 20/max)."""
        hc = self.params.human
        rng = self._rng
        hard_min = max(1, math.ceil(20.0 / max(1.0, max_cps)))   # plancher = cap CPS
        if max_cps <= 10.0:
            r = rng.random()
            if r < 0.04:
                return hard_min + 2
            if r < 0.22:
                return hard_min + 1
            return hard_min
        lo = max(1.0, max_cps - 4.0)
        cps = rng.uniform(lo, max_cps) * (1.0 + rng.gauss(0.0, 0.08))
        cps = min(max_cps, max(1.0, cps))
        cd = 20.0 / cps
        if rng.random() < hc.cps_pause_prob:
            cd += rng.uniform(1.0, 4.0)            # hésitation / pause humaine
        # arrondi stochastique : casse le clustering sur un entier de ticks fixe
        base = int(cd)
        cd_ticks = base + (1 if rng.random() < (cd - base) else 0)
        return max(hard_min, cd_ticks)

    def _counter_assist_click_cooldown(self, max_cps: float) -> int:
        assist = self.params.counter_assist
        target_cps = min(max(1.0, float(max_cps)),
                         max(1.0, float(assist.recovery_cps)))
        hard_min = max(1, math.ceil(20.0 / target_cps))
        jitter = max(0.0, min(0.75, float(assist.jitter_prob)))
        r = self._rng.random()
        if r < jitter * 0.10:
            return hard_min + 2
        if r < jitter:
            return hard_min + 1
        return hard_min

    def _arena_from_msg(self, msg: dict) -> ArenaCalib | None:
        raw = msg.get("arena")
        if not isinstance(raw, dict):
            return None
        try:
            origin_x = float(raw["origin_x"])
            origin_z = float(raw["origin_z"])
            size_x = float(raw["size_x"])
            size_z = float(raw["size_z"])
            floor_y = float(raw["floor_y"])
        except (KeyError, TypeError, ValueError):
            return None
        vals = (origin_x, origin_z, size_x, size_z, floor_y)
        if not all(math.isfinite(v) for v in vals):
            return None
        if size_x < 4.0 or size_z < 4.0 or size_x > 128.0 or size_z > 128.0:
            return None
        return ArenaCalib(
            origin_x=origin_x, origin_z=origin_z,
            size_x=size_x, size_z=size_z, floor_y=floor_y,
        )

    def _resolve_arena(self, msg: dict) -> ArenaCalib:
        """Repère arène : auto (centre joueurs / sol min) ou calibré fixe.

        En auto, l'origine recentre le milieu des 2 joueurs dans l'arène et le
        sol = min des Y -> murs (obs[25..28]) et hauteurs (obs[46,47]) restent
        dans la plage d'entraînement, peu importe les coords monde du box.
        """
        pr = self.params
        if not self.auto_frame:
            return pr.arena
        detected = self._arena_from_msg(msg)
        if detected is not None:
            self._frame = detected
            return detected
        if self._frame is None:
            s, t = msg["self"], msg["target"]
            sx, sz, sy = float(s["x"]), float(s["z"]), float(s["y"])
            tx, tz, ty = float(t["x"]), float(t["z"]), float(t["y"])
            self._frame = ArenaCalib(
                origin_x=(sx + tx) / 2.0 - pr.arena.size_x / 2.0,
                origin_z=(sz + tz) / 2.0 - pr.arena.size_z / 2.0,
                size_x=pr.arena.size_x, size_z=pr.arena.size_z,
                floor_y=min(sy, ty),
            )
        return self._frame

    # ------------------------------------------------------------------ state
    def on_state(self, msg: dict) -> dict | None:
        """Message 'state' du mod -> message 'action', ou None si inactif."""
        if self.model is None or not self.params.enabled:
            return None
        t0 = time.perf_counter()
        pr = self.params

        arena = self._resolve_arena(msg)
        own = player_from_msg(msg["self"], arena)
        opp = player_from_msg(msg["target"], arena)

        # frontière de match : les compteurs de hits repartent à zéro ->
        # purge historique/tick/cooldown (sinon o[34] dérive en négatif et
        # l'historique traverse les matchs : entrées hors distribution). Le
        # repère auto est recalculé pour le nouveau combat.
        hits_now = (own.hits, opp.hits)
        prev_hits = self._last_hits
        if prev_hits is not None and (hits_now[0] < prev_hits[0]
                                      or hits_now[1] < prev_hits[1]):
            self.reset()
            prev_hits = None
            arena = self._resolve_arena(msg)
            own = player_from_msg(msg["self"], arena)
            opp = player_from_msg(msg["target"], arena)
        if prev_hits is not None and hits_now[0] > prev_hits[0]:
            self._own_combo_streak += int(hits_now[0] - prev_hits[0])
            self._opp_combo_streak = 0
            dx = opp.x - own.x
            dz = opp.z - own.z
            if math.sqrt(dx * dx + dz * dz) <= POST_HIT_RESET_REACH:
                self._combo_reset_ticks = max(self._combo_reset_ticks, 2)
            if self._uses_leaderboard_boxing():
                self._leaderboard_wtap_ticks = max(
                    self._leaderboard_wtap_ticks,
                    LEADERBOARD_WTAP_RELEASE_TICKS,
                )
        if prev_hits is not None and hits_now[1] > prev_hits[1]:
            self._opp_combo_streak += int(hits_now[1] - prev_hits[1])
            self._own_combo_streak = 0
            if self._uses_leaderboard_boxing() and self._opp_combo_streak >= 3:
                if self._opp_combo_break_ticks <= 0 or abs(self._opp_combo_break_sign) <= 0.5:
                    if abs(self.last_action[3]) > 0.5:
                        self._opp_combo_break_sign = -1.0 if self.last_action[3] > 0.0 else 1.0
                    else:
                        self._opp_combo_break_sign = -1.0 if (self._opp_combo_streak % 2) else 1.0
                self._opp_combo_break_ticks = max(
                    self._opp_combo_break_ticks,
                    LEADERBOARD_OPP_COMBO_BREAK_TICKS,
                )
        self._last_hits = hits_now
        own.click_cooldown = self.click_cooldown

        cfg = SimConfig(arena_size_x=arena.size_x, arena_size_z=arena.size_z,
                        max_ticks=self.max_ticks)
        h = HumanizationConfig(max_cps=pr.max_cps, max_rot_speed=pr.max_rot_speed)
        obs = build_obs(own, opp, cfg, h, self.last_action,
                        min(self.tick, self.max_ticks))

        self.hist = torch.roll(self.hist, shifts=-1, dims=1)
        self.hist[0, -1] = torch.tensor(obs, dtype=torch.float32,
                                        device=self.device)
        with torch.no_grad():
            a = self.model(self.hist)[0].tolist()
        raw_action = list(a)

        # humanisation moteur : le modèle donne la visée voulue (deg) ; un filtre
        # 'main humaine' (réaction + lissage + tremblement + butée) rend le
        # mouvement humain. Sinon repli sur le modèle moteur EMA du sim.
        cmd_yaw = max(-1.0, min(1.0, a[0])) * pr.max_rot_speed
        cmd_pitch = max(-1.0, min(1.0, a[1])) * pr.max_rot_speed
        # curseur visée : 0 = modèle brut (parfait), ->1 = humain ({"human":true}=1)
        aim_level = pr.human.level if pr.human.level > 0.0 else (
            1.0 if pr.human.enabled else 0.0)
        if aim_level > 1e-6:
            a[0], a[1] = self._human_aim(cmd_yaw, cmd_pitch, aim_level)
        else:
            self._aim_dyaw = cmd_yaw
            self._aim_dpitch = cmd_pitch
            a[0] = cmd_yaw
            a[1] = cmd_pitch
        self._stabilize_live_aim(a, own, opp)
        self._enforce_combo_direct_pad(a, own, opp)
        hit_select_active, hit_select_assist_fired = self._apply_hit_select_assist(a, own, opp)
        counter_assist_fired = False
        if not hit_select_active:
            counter_assist_fired = self._apply_counter_assist(a, own, opp)
        # cadence de clic TOUJOURS humanisée (variable, jamais > cap) :
        # INDÉPENDANTE de la visée -> bat l'Autoclicker sans toucher à l'aim
        # (le modèle garde sa visée parfaite, seuls les intervalles de clic
        # deviennent humains).
        if self.click_cooldown > 0:
            self.click_cooldown -= 1
        if a[6] > 0.5:
            if self.click_cooldown > 0:
                a[6] = 0.0
            else:
                if counter_assist_fired:
                    self.click_cooldown = self._counter_assist_click_cooldown(pr.max_cps)
                else:
                    self.click_cooldown = self._human_click_cooldown(pr.max_cps)
        self._record_applied_action(a)
        self.tick += 1
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        action_msg = action_to_msg(a)
        action_msg["aim_smoothing"] = dict(self.params.aim_smoothing.__dict__)
        action_msg["hit_select_assist"] = dict(self.params.hit_select_assist.__dict__)
        action_msg["auto_gapple"] = dict(self.params.auto_gapple.__dict__)
        action_msg["auto_jump"] = dict(self.params.auto_jump.__dict__)
        action_msg["knockback_dump"] = dict(self.params.knockback_dump.__dict__)
        action_msg["friends"] = {
            "enabled": bool(self.params.friends.enabled),
            "names": list(self.params.friends.names),
        }
        yaw_err, pitch_err = self._aim_errors_deg(own, opp)
        self._log_action_line(
            "tick=%d model=%s forward=%d strafe=%d jump=%s sprint=%s attack=%s "
            "dyaw=%.6f dpitch=%.6f ownPitch=%.6f yawErr=%.6f pitchErr=%.6f "
            "dist=%.6f ownHurt=%d oppHurt=%d ownHits=%d oppHits=%d hitSelect=%s counterAssist=%s"
            % (
                self.tick,
                self.model_path or "",
                int(action_msg["forward"]),
                int(action_msg["strafe"]),
                str(bool(action_msg["jump"])).lower(),
                str(bool(action_msg["sprint"])).lower(),
                str(bool(action_msg["attack"])).lower(),
                float(action_msg["dyaw"]),
                float(action_msg["dpitch"]),
                float(own.pitch),
                float(yaw_err),
                float(pitch_err),
                float(math.sqrt((opp.x - own.x) * (opp.x - own.x)
                                + (opp.z - own.z) * (opp.z - own.z))),
                int(own.hurt_resistant_time),
                int(opp.hurt_resistant_time),
                int(own.hits),
                int(opp.hits),
                str(bool(hit_select_assist_fired)).lower(),
                str(bool(counter_assist_fired)).lower(),
            ))
        if self._debug and self.tick % 20 == 0:
            # last_action = sortie brute du modèle (fwd/str/jmp/spr aux indices
            # 2..5) ; a[0],a[1],a[6] = visée/clic après humanisation
            print("[live] t=%d raw=[%s] -> dyaw=%.1f dpitch=%.1f atk=%d"
                  % (self.tick, " ".join("%.2f" % x for x in raw_action),
                     a[0], a[1], int(a[6] > 0.5)), flush=True)
        return action_msg

    # ----------------------------------------------------------------- status
    def status(self) -> dict:
        return {
            "model": self.model_path,
            "enabled": self.params.enabled,
            "max_cps": self.params.max_cps,
            "max_rot_speed": self.params.max_rot_speed,
            "aim_smooth": self.aim_smooth,
            "arena": self.params.arena.__dict__,
            "auto_frame": self.auto_frame,
            "human": self.params.human.enabled,
            "aim_smoothing": self.params.aim_smoothing.__dict__,
            "hit_select_assist": self.params.hit_select_assist.__dict__,
            "counter_assist": self.params.counter_assist.__dict__,
            "auto_gapple": self.params.auto_gapple.__dict__,
            "auto_jump": self.params.auto_jump.__dict__,
            "knockback_dump": self.params.knockback_dump.__dict__,
            "friends": {
                "enabled": self.params.friends.enabled,
                "names": list(self.params.friends.names),
            },
            "tick": self.tick,
            "latency_ms": round(self.last_latency_ms, 3),
            "device": str(self.device),
        }
