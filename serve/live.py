"""LiveSession — inférence temps réel pour le mod Forge.

Charge un modèle TorchScript exporté (train/export.py), maintient
l'historique d'observations, applique l'humanisation (clamp rotation + CPS)
et la latence d'inférence visée est < 2 ms sur GPU.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from sim.config import SimConfig
from sim.obs import OBS_DIM, build_obs
from sim_ref import HumanizationConfig

from .protocol import ArenaCalib, action_to_msg, player_from_msg


@dataclass
class LiveParams:
    max_cps: float = 12.0
    max_rot_speed: float = 40.0
    arena: ArenaCalib = field(default_factory=ArenaCalib)
    enabled: bool = True


class LiveSession:
    def __init__(self, device: str | None = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.params = LiveParams()
        self.model = None
        self.model_path: str | None = None
        self.history = 16
        self.hist = None
        self.last_action = [0.0] * 7
        self.click_cooldown = 0
        self.tick = 0
        self.last_latency_ms = 0.0

    # ------------------------------------------------------------------ model
    def load(self, path: str) -> dict:
        p = Path(path)
        meta = {}
        meta_path = p.with_suffix(".json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        self.history = int(meta.get("history", 16))
        self.model = torch.jit.load(str(p), map_location=self.device).eval()
        self.model_path = str(p)
        self.reset()
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

    # ------------------------------------------------------------------ state
    def on_state(self, msg: dict) -> dict | None:
        """Message 'state' du mod -> message 'action', ou None si inactif."""
        if self.model is None or not self.params.enabled:
            return None
        t0 = time.perf_counter()
        pr = self.params

        own = player_from_msg(msg["self"], pr.arena)
        opp = player_from_msg(msg["target"], pr.arena)
        own.click_cooldown = self.click_cooldown

        cfg = SimConfig(arena_size_x=pr.arena.size_x, arena_size_z=pr.arena.size_z)
        h = HumanizationConfig(max_cps=pr.max_cps, max_rot_speed=pr.max_rot_speed)
        obs = build_obs(own, opp, cfg, h, self.last_action, self.tick)

        self.hist = torch.roll(self.hist, shifts=-1, dims=1)
        self.hist[0, -1] = torch.tensor(obs, dtype=torch.float32,
                                        device=self.device)
        with torch.no_grad():
            a = self.model(self.hist)[0].tolist()
        self.last_action = list(a)

        # humanisation : rotations en degrés + limite CPS
        a[0] = max(-1.0, min(1.0, a[0])) * pr.max_rot_speed
        a[1] = max(-1.0, min(1.0, a[1])) * pr.max_rot_speed
        if self.click_cooldown > 0:
            self.click_cooldown -= 1
        if a[6] > 0.5:
            if self.click_cooldown > 0:
                a[6] = 0.0
            else:
                self.click_cooldown = h.click_cooldown_ticks
        self.tick += 1
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        return action_to_msg(a)

    # ----------------------------------------------------------------- status
    def status(self) -> dict:
        return {
            "model": self.model_path,
            "enabled": self.params.enabled,
            "max_cps": self.params.max_cps,
            "max_rot_speed": self.params.max_rot_speed,
            "arena": self.params.arena.__dict__,
            "tick": self.tick,
            "latency_ms": round(self.last_latency_ms, 3),
            "device": str(self.device),
        }
