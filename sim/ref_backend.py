"""Backend de référence : API vectorisée identique à JudasSim, exécutée par
sim_ref en Python pur (lent — debug, CI sans GPU, équivalence CUDA).

Conventions d'actions (float32 [N, 2, 7]) — identiques au kernel CUDA :
  a[0] dyaw  normalisé [-1, 1]  (multiplié par max_rot_speed)
  a[1] dpitch normalisé [-1, 1]
  a[2] forward brut : > 0.5 -> +1 ; < -0.5 -> -1 ; sinon 0
  a[3] strafe  brut : idem
  a[4] jump   > 0.5
  a[5] sprint > 0.5
  a[6] attack > 0.5
"""

import numpy as np

from sim_ref import Action, BoxingConfig, BoxingMatch, HumanizationConfig

from .config import ACTION_DIM, SimConfig
from .obs import OBS_DIM, build_obs


def _mid(a: float, b: float) -> float:
    return (a + b) / 2.0


class JudasSimRef:
    """Même contrat que sim.JudasSim, sur CPU via sim_ref."""

    def __init__(self, n_envs: int, cfg: SimConfig | None = None, seed: int = 0):
        self.n_envs = n_envs
        self.cfg = cfg or SimConfig()
        if self.cfg.randomize:
            raise NotImplementedError(
                "JudasSimRef ne supporte que randomize=False (valeurs médianes fixes)")
        self._h = HumanizationConfig(
            max_cps=_mid(self.cfg.cps_min, self.cfg.cps_max),
            max_rot_speed=_mid(self.cfg.rot_speed_min, self.cfg.rot_speed_max),
            action_delay=round(_mid(self.cfg.delay_min, self.cfg.delay_max)),
        )
        self._matches: list[BoxingMatch] = []
        self._last_actions = np.zeros((n_envs, 2, ACTION_DIM), dtype=np.float32)

    # ----------------------------------------------------------------- utils
    def _new_match(self) -> BoxingMatch:
        c = self.cfg
        return BoxingMatch(BoxingConfig(
            arena_size_x=c.arena_size_x,
            arena_size_z=c.arena_size_z,
            target_hits=c.target_hits,
            max_ticks=c.max_ticks,
            speed_amplifier=c.speed_amplifier,
            humanization=(self._h, self._h),
        ))

    def _obs_one(self, n: int) -> np.ndarray:
        m = self._matches[n]
        out = np.empty((2, OBS_DIM), dtype=np.float32)
        for i in range(2):
            out[i] = build_obs(m.players[i], m.players[1 - i], self.cfg, self._h,
                               self._last_actions[n, i], m.tick_count)
        return out

    def set_reward_dist(self, v: float) -> None:
        """Shaping de distance modifiable à chaud (decay automatique)."""
        self.cfg.reward_dist = float(v)

    # ------------------------------------------------------------------- API
    def reset(self) -> np.ndarray:
        self._matches = [self._new_match() for _ in range(self.n_envs)]
        self._last_actions[:] = 0.0
        obs = np.empty((self.n_envs, 2, OBS_DIM), dtype=np.float32)
        for n in range(self.n_envs):
            obs[n] = self._obs_one(n)
        return obs

    def step(self, actions: np.ndarray):
        """-> (obs [N,2,OBS_DIM], reward [N,2], done [N], info dict)"""
        actions = np.asarray(actions, dtype=np.float32)
        obs = np.empty((self.n_envs, 2, OBS_DIM), dtype=np.float32)
        reward = np.zeros((self.n_envs, 2), dtype=np.float32)
        done = np.zeros(self.n_envs, dtype=bool)
        wins = np.full(self.n_envs, -2, dtype=np.int32)
        c = self.cfg

        for n in range(self.n_envs):
            m = self._matches[n]
            acts = []
            for i in range(2):
                a = actions[n, i]
                acts.append(Action(
                    dyaw=float(np.clip(a[0], -1.0, 1.0)) * self._h.max_rot_speed,
                    dpitch=float(np.clip(a[1], -1.0, 1.0)) * self._h.max_rot_speed,
                    forward=1 if a[2] > 0.5 else (-1 if a[2] < -0.5 else 0),
                    strafe=1 if a[3] > 0.5 else (-1 if a[3] < -0.5 else 0),
                    jump=bool(a[4] > 0.5),
                    sprint=bool(a[5] > 0.5),
                    attack=bool(a[6] > 0.5),
                ))
            hits_before = [m.players[0].hits, m.players[1].hits]
            m.step((acts[0], acts[1]))

            for i in range(2):
                dealt = m.players[i].hits - hits_before[i]
                taken = m.players[1 - i].hits - hits_before[1 - i]
                reward[n, i] = c.reward_hit * dealt + c.reward_hurt * taken
                if c.reward_dist != 0.0:
                    p, q = m.players[i], m.players[1 - i]
                    d = ((p.x - q.x) ** 2 + (p.y - q.y) ** 2 + (p.z - q.z) ** 2) ** 0.5
                    reward[n, i] -= c.reward_dist * d

            if m.done:
                done[n] = True
                wins[n] = m.winner
                if m.winner >= 0:
                    reward[n, m.winner] += c.reward_win
                    reward[n, 1 - m.winner] -= c.reward_win
                self._matches[n] = self._new_match()
                self._last_actions[n] = 0.0
            else:
                self._last_actions[n] = actions[n]

            obs[n] = self._obs_one(n)

        return obs, reward, done, {"winner": wins}
