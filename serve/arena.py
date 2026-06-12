"""ArenaSession — matchs IA vs IA dans le simulateur, sans Minecraft.

Charge deux modèles (checkpoint .pt -> JudasPolicy échantillonnable, ou export
TorchScript .pts -> déterministe), les fait s'affronter dans JudasSimRef et
produit un état JSON par tick pour le visualiseur 3D (app viz/).

Le pipeline d'observation est EXACTEMENT celui de l'entraînement (les obs
sortent du simulateur lui-même), donc ce qu'on voit ici est représentatif.
"""

import time
from pathlib import Path

import numpy as np
import torch

from sim import OBS_DIM, SimConfig
from sim.ref_backend import JudasSimRef


class _Agent:
    """Un modèle chargé : .pt (policy, sample possible) ou .pts (déterministe)."""

    def __init__(self, path: str, device: torch.device):
        self.path = str(path)
        self.name = Path(path).name
        self.device = device
        if self.path.endswith(".pts"):
            import json
            self.kind = "script"
            self.model = torch.jit.load(self.path, map_location=device).eval()
            meta = Path(self.path).with_suffix(".json")
            # 8 = PolicyConfig.history par défaut (un fallback plus grand
            # ferait crasher la trace si le .json d'export manque)
            self.history = int(json.loads(meta.read_text())["history"]) \
                if meta.exists() else 8
        else:
            from train.model import JudasPolicy, PolicyConfig
            self.kind = "policy"
            ckpt = torch.load(self.path, map_location="cpu", weights_only=False)
            cfg = PolicyConfig(**ckpt.get("policy_cfg", {}))
            self.model = JudasPolicy(cfg).to(device).eval()
            self.model.load_state_dict(ckpt["policy"], strict=False)
            self.history = cfg.history

    @torch.no_grad()
    def act(self, hist_np: np.ndarray, sample: bool) -> np.ndarray:
        """hist_np [H_max, OBS_DIM] -> action sim [7] float32."""
        hist = torch.from_numpy(hist_np[-self.history:][None]).float().to(self.device)
        if self.kind == "script":
            return self.model(hist)[0].cpu().numpy()
        from train.model import to_sim_actions
        out = self.model.act(hist, deterministic=not sample)
        raw = {k: out[k] for k in ("pre", "fwd", "strafe", "bins")}
        return to_sim_actions(raw)[0].cpu().numpy()


class ArenaSession:
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.agents: list[_Agent | None] = [None, None]
        self.sim: JudasSimRef | None = None
        self.running = False
        self.speed = 1.0
        self.sample = True
        self.tick = 0
        self.wins = [0, 0]
        self.draws = 0
        self.matches = 0
        self.clicks = [0, 0]      # clics décidés (diagnostic attaque)
        self.cfg = SimConfig(randomize=False)
        self._hist = None
        self._last_obs = None
        self._last_attack = [False, False]
        self.step_ms = 0.0

    @property
    def ready(self) -> bool:
        return self.sim is not None and all(self.agents)

    # ------------------------------------------------------------------ setup
    def load(self, model_a: str, model_b: str, *, cps: float = 12.0,
             rot_speed: float = 40.0, arena_size: float = 18.0,
             target_hits: int = 100, sample: bool = True,
             kb_h: float = 1.0, kb_v: float = 1.0,
             kb_idle: float = 1.0, aim_smooth: float = 0.0) -> dict:
        self.agents = [_Agent(model_a, self.device), _Agent(model_b, self.device)]
        self.cfg = SimConfig(
            arena_size_x=arena_size, arena_size_z=arena_size,
            target_hits=target_hits, max_ticks=20 * 60 * 5,
            cps_min=cps, cps_max=cps,
            rot_speed_min=rot_speed, rot_speed_max=rot_speed,
            kb_h_mult=kb_h, kb_v_mult=kb_v, kb_idle_mult=kb_idle,
            aim_smooth_min=aim_smooth, aim_smooth_max=aim_smooth,
            randomize=False,
        )
        self.sample = sample
        self.sim = JudasSimRef(1, self.cfg)
        self.wins = [0, 0]
        self.draws = 0
        self.matches = 0
        self.clicks = [0, 0]
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

    # ------------------------------------------------------------------- step
    def step(self) -> dict:
        """Avance d'un tick et retourne l'état pour le visualiseur."""
        if not self.ready:
            return {"t": "tick", "ready": False}
        t0 = time.perf_counter()

        actions = np.zeros((1, 2, 7), dtype=np.float32)
        for i, agent in enumerate(self.agents):
            actions[0, i] = agent.act(self._hist[i], self.sample)
        self._last_attack = [bool(actions[0, i, 6] > 0.5) for i in range(2)]
        for i in range(2):
            if self._last_attack[i]:
                self.clicks[i] += 1

        hits_before = [p.hits for p in self.sim._matches[0].players]
        obs, reward, done, info = self.sim.step(actions)
        self.tick += 1

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
            hits_after = hits_before          # état final masqué par l'autoreset
        else:
            hits_after = [p.hits for p in self.sim._matches[0].players]

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
                "hits": p.hits,
                "swing": self._last_attack[i],
                "landed": (hits_after[i] - hits_before[i]) > 0 if not is_done else False,
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
            "arena": {"sx": self.cfg.arena_size_x, "sz": self.cfg.arena_size_z},
            "target_hits": self.cfg.target_hits,
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
            "models": [a.name if a else None for a in self.agents],
            "wins": self.wins,
            "draws": self.draws,
            "matches": self.matches,
            "clicks": self.clicks,
            "tick": self.tick,
            "arena": {"sx": self.cfg.arena_size_x, "sz": self.cfg.arena_size_z},
            "target_hits": self.cfg.target_hits,
            "step_ms": round(self.step_ms, 2),
        }
