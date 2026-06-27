"""ArenaSession â€” matchs IA vs IA dans le simulateur, sans Minecraft.

Charge deux modÃ¨les (checkpoint .pt -> JudasPolicy Ã©chantillonnable, ou export
TorchScript .pts -> dÃ©terministe), les fait s'affronter dans JudasSimRef et
produit un Ã©tat JSON par tick pour le visualiseur 3D (app viz/).

Le pipeline d'observation est EXACTEMENT celui de l'entraÃ®nement (les obs
sortent du simulateur lui-mÃªme), donc ce qu'on voit ici est reprÃ©sentatif.
"""

import math
import time
from pathlib import Path

import numpy as np
import torch

from sim import OBS_DIM, SimConfig
from sim.ref_backend import JudasSimRef


PLAYER_EYE_HEIGHT = 1.62
PLAYER_AIM_MIN_Y = 0.25
PLAYER_AIM_MAX_Y = 1.45


def _wrap_degrees(v: float) -> float:
    v = math.fmod(v, 360.0)
    if v >= 180.0:
        v -= 360.0
    if v < -180.0:
        v += 360.0
    return v


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
    pitch_to = -math.degrees(math.atan2(aim_y - eye_y, dist_h))
    return yaw_err, pitch_to - own.pitch


def _stabilize_axis_norm(cmd_norm: float, err_deg: float, rot_speed: float,
                         deadband: float = 0.05) -> float:
    rot = max(1.0, float(rot_speed))
    if abs(err_deg) <= deadband:
        return 0.0
    return max(-rot, min(rot, float(err_deg))) / rot


class _Agent:
    """Un modÃ¨le chargÃ© : .pt (policy, sample possible) ou .pts (dÃ©terministe)."""

    def __init__(self, path: str, device: torch.device,
                 prefer_policy_sample: bool = False):
        self.path = str(path)
        self.name = Path(path).name
        self.device = device
        self.sample_policy = None
        self.sample_history = 0
        self.script_history = 0
        if self.path in {"__chase_bot__", "chase_bot", "bot:chase"}:
            from train.scripted import ChaseBot
            self.kind = "scripted_bot"
            self.name = "chase-bot"
            self.model = ChaseBot()
            self.history = 8
        elif self.path in {"__combo_pad__", "combo_pad", "bot:combo_pad"}:
            from train.scripted import ComboPadBot
            self.kind = "scripted_bot"
            self.name = "combo-pad"
            self.model = ComboPadBot()
            self.history = 8
        elif self.path in {"__combo_spar__", "combo_spar", "bot:combo_spar"}:
            from train.scripted import ComboSparBot
            self.kind = "scripted_bot"
            self.name = "combo-spar"
            self.model = ComboSparBot()
            self.history = 8
        elif self.path.endswith(".pts"):
            import json
            self.kind = "script"
            self.model = torch.jit.load(self.path, map_location=device).eval()
            meta = Path(self.path).with_suffix(".json")
            raw_meta = json.loads(meta.read_text()) if meta.exists() else {}
            # 8 = PolicyConfig.history par dÃ©faut (un fallback plus grand
            # ferait crasher la trace si le .json d'export manque)
            self.script_history = int(raw_meta.get("history", 8))
            self.history = self.script_history
            source = raw_meta.get("source")
            if prefer_policy_sample and source:
                source_path = Path(source)
                if not source_path.is_absolute():
                    source_path = Path.cwd() / source_path
                if source_path.exists():
                    self.sample_policy, self.sample_history = _load_policy(
                        source_path, device)
                    self.history = max(self.history, self.sample_history)
        else:
            self.kind = "policy"
            self.model, self.history = _load_policy(Path(self.path), device)

    @torch.no_grad()
    def act(self, hist_np: np.ndarray, sample: bool) -> np.ndarray:
        """hist_np [H_max, OBS_DIM] -> action sim [7] float32."""
        if sample and self.sample_policy is not None:
            hist = torch.from_numpy(
                hist_np[-self.sample_history:][None]).float().to(self.device)
            from train.model import to_sim_actions
            out = self.sample_policy.act(hist, deterministic=False)
            raw = {k: out[k] for k in ("pre", "fwd", "strafe", "bins")}
            return to_sim_actions(raw)[0].cpu().numpy()
        hist_len = self.script_history if self.kind == "script" else self.history
        hist = torch.from_numpy(hist_np[-hist_len:][None]).float().to(self.device)
        if self.kind == "scripted_bot":
            return self.model.act7(hist)[0].cpu().numpy()
        if self.kind == "script":
            return self.model(hist)[0].cpu().numpy()
        from train.model import to_sim_actions
        out = self.model.act(hist, deterministic=not sample)
        raw = {k: out[k] for k in ("pre", "fwd", "strafe", "bins")}
        return to_sim_actions(raw)[0].cpu().numpy()


def _load_policy(path: Path, device: torch.device):
    from train.model import JudasPolicy, PolicyConfig

    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = PolicyConfig(**ckpt.get("policy_cfg", {}))
    model = JudasPolicy(cfg).to(device).eval()
    model.load_state_dict(ckpt["policy"], strict=False)
    return model, cfg.history


class ArenaSession:
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.agents: list[_Agent | None] = [None, None]
        self.sim: JudasSimRef | None = None
        self.running = False
        self.speed = 1.0
        self.sample = True
        self.mirror_sample_forced = False
        self.mirror_sample = False
        self.mirror_desync = False
        self._mirror_opened = False
        self.tick = 0
        self.wins = [0, 0]
        self.draws = 0
        self.matches = 0
        self.clicks = [0, 0]      # clics dÃ©cidÃ©s (diagnostic attaque)
        self.combo = [0, 0]
        self.max_combo = [0, 0]
        self.cfg = SimConfig(randomize=False)
        self._hist = None
        self._last_obs = None
        self._last_attack = [False, False]
        self.step_ms = 0.0

    @property
    def ready(self) -> bool:
        return self.sim is not None and all(self.agents)

    # ------------------------------------------------------------------ setup
    def load(self, model_a: str, model_b: str, *, cps: float = 10.0,
             rot_speed: float = 190.0, arena_size: float = 40.0,
             target_hits: int = 50, sample: bool = True,
             spawn_gap: float = 8.0,
             kb_h: float = 0.92, kb_v: float = 0.90,
             kb_idle: float = 0.6, aim_smooth: float = 0.02,
             post_sprint_hit_stop: bool = True) -> dict:
        same_model = str(model_a) == str(model_b)
        self.mirror_sample_forced = False
        if same_model and not sample:
            sample = True
            self.mirror_sample_forced = True
        prefer_policy_sample = same_model and sample
        self.mirror_sample = prefer_policy_sample
        self.mirror_desync = same_model
        self.agents = [
            _Agent(model_a, self.device, prefer_policy_sample=prefer_policy_sample),
            _Agent(model_b, self.device, prefer_policy_sample=prefer_policy_sample),
        ]
        self.cfg = SimConfig(
            arena_size_x=arena_size, arena_size_z=arena_size,
            target_hits=target_hits, max_ticks=20 * 60 * 5,
            spawn_gap=spawn_gap,
            cps_min=cps, cps_max=cps,
            rot_speed_min=rot_speed, rot_speed_max=rot_speed,
            kb_h_mult=kb_h, kb_v_mult=kb_v, kb_idle_mult=kb_idle,
            post_sprint_hit_stop=post_sprint_hit_stop,
            aim_smooth_min=aim_smooth, aim_smooth_max=aim_smooth,
            randomize=False,
        )
        self.sample = sample
        self.sim = JudasSimRef(1, self.cfg)
        self.wins = [0, 0]
        self.draws = 0
        self.matches = 0
        self.clicks = [0, 0]
        self.combo = [0, 0]
        self.max_combo = [0, 0]
        self.reset()
        return self.status()

    def reset(self) -> None:
        if self.sim is None:
            return
        obs = self.sim.reset()                       # [1, 2, OBS_DIM]
        h_max = max(a.history for a in self.agents if a) if any(self.agents) else 8
        self._hist = np.zeros((2, h_max, OBS_DIM), dtype=np.float32)
        self._hist[:, -1] = obs[0]
        self.tick = 0
        self._last_attack = [False, False]
        self._last_actions = np.zeros((2, 7), dtype=np.float32)
        self.combo = [0, 0]
        self._mirror_opened = False

    # ------------------------------------------------------------------- step
    def step(self) -> dict:
        """Avance d'un tick et retourne l'Ã©tat pour le visualiseur."""
        if not self.ready:
            return {"t": "tick", "ready": False}
        t0 = time.perf_counter()

        actions = np.zeros((1, 2, 7), dtype=np.float32)
        for i, agent in enumerate(self.agents):
            if self.mirror_sample:
                torch.manual_seed(20260623 + self.matches * 100000
                                  + self.tick * 2 + i)
            actions[0, i] = agent.act(self._hist[i], self.sample)
        if self.mirror_desync and not self._mirror_opened:
            self._apply_mirror_opening_desync(actions[0])
        match = self.sim._matches[0]
        for i in range(2):
            own = match.players[i]
            opp = match.players[1 - i]
            yaw_err, pitch_err = _aim_errors_deg(own, opp)
            actions[0, i, 0] = _stabilize_axis_norm(
                actions[0, i, 0], yaw_err, self.cfg.rot_speed_max)
            actions[0, i, 1] = _stabilize_axis_norm(
                actions[0, i, 1], pitch_err, self.cfg.rot_speed_max)
        self._last_actions = actions[0].copy()
        self._last_attack = [bool(actions[0, i, 6] > 0.5) for i in range(2)]
        for i in range(2):
            if self._last_attack[i]:
                self.clicks[i] += 1

        hits_before = [p.hits for p in self.sim._matches[0].players]
        obs, reward, done, info = self.sim.step(actions)
        self.tick += 1

        dealt = [int(v) for v in info.get("dealt", [[0, 0]])[0]]
        combo_after = [int(v) for v in info.get("combo", [[0, 0]])[0]]
        winner = int(info["winner"][0])
        is_done = bool(done[0])
        if is_done:
            self.matches += 1
            if winner >= 0:
                self.wins[winner] += 1
            else:
                self.draws += 1
            # le backend a auto-reset : on repart d'un historique vierge
            self._hist[:] = 0.0
            self.tick = 0
            self._mirror_opened = False
            hits_after = [hits_before[i] + dealt[i] for i in range(2)]
        else:
            hits_after = [p.hits for p in self.sim._matches[0].players]

        landed = [dealt[i] > 0 for i in range(2)]
        display_combo = combo_after
        for i in range(2):
            self.max_combo[i] = max(self.max_combo[i], combo_after[i])
        self.combo = [0, 0] if is_done else combo_after
        self._hist = np.roll(self._hist, -1, axis=1)
        self._hist[:, -1] = obs[0]

        players = []
        match = self.sim._matches[0]
        for i, p in enumerate(match.players):
            players.append({
                "x": round(p.x, 4), "y": round(p.y, 4), "z": round(p.z, 4),
                "vx": round(p.vx, 4), "vy": round(p.vy, 4), "vz": round(p.vz, 4),
                "yaw": round(p.yaw, 2), "pitch": round(p.pitch, 2),
                "sprint": p.sprinting, "onGround": p.on_ground,
                "hurt": p.hurt_resistant_time,
                "hits": hits_after[i],
                "combo": display_combo[i],
                "max_combo": self.max_combo[i],
                "forward": round(float(self._last_actions[i, 2]), 4),
                "strafe": round(float(self._last_actions[i, 3]), 4),
                "jump": bool(self._last_actions[i, 4] > 0.5),
                "swing": self._last_attack[i],
                "landed": landed[i],
            })

        self.step_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "t": "tick",
            "ready": True,
            "tick": self.tick,
            "players": players,
            "done": is_done,
            "winner": winner if is_done else -2,
            "wins": self.wins,
            "draws": self.draws,
            "matches": self.matches,
            "clicks": self.clicks,
            "combo": display_combo,
            "max_combo": self.max_combo,
            "arena": {"sx": self.cfg.arena_size_x, "sz": self.cfg.arena_size_z},
            "spawn_gap": self.cfg.spawn_gap,
            "target_hits": self.cfg.target_hits,
            "post_sprint_hit_stop": self.cfg.post_sprint_hit_stop,
            "speed": self.speed,
            "running": self.running,
            "step_ms": round(self.step_ms, 2),
        }

    # ----------------------------------------------------------------- status
    def status(self) -> dict:
        return {
            "ready": self.ready,
            "running": self.running,
            "speed": self.speed,
            "sample": self.sample,
            "mirror_sample_forced": self.mirror_sample_forced,
            "mirror_sample": self.mirror_sample,
            "mirror_desync": self.mirror_desync,
            "models": [a.name if a else None for a in self.agents],
            "wins": self.wins,
            "draws": self.draws,
            "matches": self.matches,
            "clicks": self.clicks,
            "combo": self.combo,
            "max_combo": self.max_combo,
            "tick": self.tick,
            "arena": {"sx": self.cfg.arena_size_x, "sz": self.cfg.arena_size_z},
            "spawn_gap": self.cfg.spawn_gap,
            "target_hits": self.cfg.target_hits,
            "post_sprint_hit_stop": self.cfg.post_sprint_hit_stop,
            "step_ms": round(self.step_ms, 2),
        }

    def _apply_mirror_opening_desync(self, actions: np.ndarray) -> None:
        """Break exact same-model symmetry only for the first neutral exchange.

        Same-policy mirror is otherwise a deterministic timing artifact: both
        sides click on the same tick, every hit trades, and combo stays at 0.
        Alternating the opener keeps the visualizer useful without changing the
        exported model or non-mirror arena behavior.
        """
        if self.sim is None:
            return
        match = self.sim._matches[0]
        players = match.players
        neutral = (
            players[0].hits == 0
            and players[1].hits == 0
            and players[0].hurt_resistant_time == 0
            and players[1].hurt_resistant_time == 0
        )
        if not neutral:
            self._mirror_opened = True
            return
        both_attack = actions[0, 6] > 0.5 and actions[1, 6] > 0.5
        if not both_attack:
            return
        dx = players[0].x - players[1].x
        dz = players[0].z - players[1].z
        dist = math.sqrt(dx * dx + dz * dz)
        if dist > 3.45:
            return
        receiver = 1 - (self.matches % 2)
        actions[receiver, 6] = 0.0
        self._mirror_opened = True
