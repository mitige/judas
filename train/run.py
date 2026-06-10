"""Boucle d'entraînement Judas : PPO self-play league sur le simulateur.

    python -m train.run --config train/configs/boxing.json [--resume runs/x/ckpt.pt]

Sorties dans runs/<name>/ :
  metrics.jsonl   (1 ligne JSON / itération — consommé par serve/ et l'app)
  tb/             (TensorBoard)
  ckpt_*.pt       (checkpoints complets)
  latest.pt       (lien logique vers le dernier)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from sim import OBS_DIM, SimConfig, make_sim

from .buffer import RolloutBuffer
from .league import League
from .model import JudasPolicy, PolicyConfig, to_sim_actions
from .ppo import PPO, PPOConfig

DEFAULT_CFG = {
    "name": "boxing",
    "total_iters": 10000,
    "n_envs": 4096,
    "rollout_ticks": 128,
    "league_frac": 0.3,
    "pool_every": 50,
    "save_every": 25,
    "seed": 0,
    "sim": {},
    "policy": {},
    "ppo": {},
}


class Trainer:
    def __init__(self, cfg: dict, device: str | None = None):
        self.cfg = {**DEFAULT_CFG, **cfg}
        self.cfg["sim"] = {**DEFAULT_CFG["sim"], **cfg.get("sim", {})}
        torch.manual_seed(self.cfg["seed"])
        np.random.seed(self.cfg["seed"])

        # Ampere+ : TF32 pour les matmuls (gros gain, précision suffisante en RL)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.N = self.cfg["n_envs"]
        self.B = self.N * 2
        self.T = self.cfg["rollout_ticks"]

        self.sim_cfg = SimConfig(**self.cfg["sim"])
        self.sim = make_sim(self.N, self.sim_cfg, seed=self.cfg["seed"],
                            force_cpu=self.device.type != "cuda")

        self.pol_cfg = PolicyConfig(**self.cfg.get("policy", {}))
        self.H = self.pol_cfg.history
        self.policy = JudasPolicy(self.pol_cfg).to(self.device)
        self.ppo = PPO(self.policy, PPOConfig(**self.cfg.get("ppo", {})), self.device)
        self.league = League()
        # inférence de rollout en fp16 (l'update PPO garde sa propre AMP)
        self._use_amp = self.ppo.use_amp
        self._buf = RolloutBuffer(self.T, self.B, OBS_DIM, self.H, self.device)

        self.hist = torch.zeros(self.B, self.H, OBS_DIM, device=self.device)
        # -1 : le premier _push_obs (reset) donne l'âge 0 à l'obs de spawn
        self.age = torch.full((self.B,), -1, dtype=torch.long, device=self.device)
        self.iter = 0
        self.total_steps = 0          # agent-steps cumulés (tous agents)

        self.run_dir = Path("runs") / self.cfg["name"]
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._tb = None

        obs = self._to_torch(self.sim.reset())          # [N, 2, D]
        self._push_obs(obs.reshape(self.B, OBS_DIM),
                       torch.zeros(self.N, device=self.device))

        # adversaires actifs pour le rollout courant
        self._opp_models: list[JudasPolicy] = []
        self._opp_pool_idx: list[int] = []
        self._env_opp: torch.Tensor = torch.full((self.N,), -1, dtype=torch.long,
                                                 device=self.device)

    # ------------------------------------------------------------------ utils
    def _to_torch(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        return torch.as_tensor(np.asarray(x)).to(self.device)

    def _sim_step(self, actions: torch.Tensor):
        if hasattr(self.sim, "ext"):                     # backend CUDA
            obs, rew, done, info = self.sim.step(actions)
            return (obs, rew.clone(), done.bool().clone(),
                    info["winner"].clone())
        a = actions.cpu().numpy()
        obs, rew, done, info = self.sim.step(a)
        return (self._to_torch(obs), self._to_torch(rew),
                self._to_torch(done).bool(), self._to_torch(info["winner"]))

    def _push_obs(self, obs_flat: torch.Tensor, done_env: torch.Tensor) -> None:
        """Avance l'historique d'un tick. done_env [N] bool/float."""
        done_b = done_env.bool().repeat_interleave(2)
        self.hist = torch.roll(self.hist, shifts=-1, dims=1)
        self.hist[done_b] = 0.0
        self.hist[:, -1] = obs_flat
        self.age = torch.where(done_b, torch.zeros_like(self.age), self.age + 1)
        # NB: âge de l'obs courante = 0 si nouvel épisode

    # -------------------------------------------------------------- opponents
    def _assign_opponents(self) -> torch.Tensor:
        """Choisit les adversaires du rollout. Retourne learner_mask [B]."""
        learner_mask = torch.ones(self.B, dtype=torch.bool, device=self.device)
        self._opp_models, self._opp_pool_idx = [], []
        self._env_opp.fill_(-1)
        frac = self.cfg["league_frac"]
        if not self.league.pool or frac <= 0:
            return learner_mask
        n_league = int(self.N * frac)
        if n_league == 0:
            return learner_mask
        n_groups = min(4, len(self.league.pool))
        idxs = self.league.sample(n_groups)
        env_ids = torch.randperm(self.N, device=self.device)[:n_league]
        groups = env_ids.chunk(n_groups)
        for gi, (pool_idx, envs) in enumerate(zip(idxs, groups)):
            if envs.numel() == 0:
                continue
            m = JudasPolicy(self.pol_cfg).to(self.device)
            m.load_state_dict(self.league.pool[pool_idx]["state_dict"])
            m.eval()
            self._opp_models.append(m)
            self._opp_pool_idx.append(pool_idx)
            self._env_opp[envs] = gi
            learner_mask[envs * 2 + 1] = False           # agent 1 = adversaire gelé
        return learner_mask

    # ---------------------------------------------------------------- rollout
    @torch.no_grad()
    def _collect(self, buf: RolloutBuffer, learner_mask: torch.Tensor) -> dict:
        ep_stats = {"wins": 0, "losses": 0, "draws": 0, "matches": 0,
                    "mirror_matches": 0}
        zero_raw = {
            "pre": torch.zeros(self.B, 2, device=self.device),
            "fwd": torch.ones(self.B, dtype=torch.long, device=self.device),
            "strafe": torch.ones(self.B, dtype=torch.long, device=self.device),
            "bins": torch.zeros(self.B, 3, device=self.device),
        }
        # résultats accumulés sur GPU, traités en une seule synchro après le rollout
        ep_done = torch.zeros(self.T, self.N, dtype=torch.bool, device=self.device)
        ep_winner = torch.zeros(self.T, self.N, dtype=torch.int32, device=self.device)
        buf.set_prefix(self.hist)
        for t in range(self.T):
            obs_now = self.hist[:, -1].clone()
            raw = {k: v.clone() for k, v in zero_raw.items()}
            logp = torch.zeros(self.B, device=self.device)
            value = torch.zeros(self.B, device=self.device)

            with torch.autocast("cuda", dtype=torch.float16, enabled=self._use_amp):
                out = self.policy.act(self.hist[learner_mask])
            raw["pre"][learner_mask] = out["pre"].float()
            raw["bins"][learner_mask] = out["bins"].float()
            raw["fwd"][learner_mask] = out["fwd"].long()
            raw["strafe"][learner_mask] = out["strafe"].long()
            logp[learner_mask] = out["logp"].float()
            value[learner_mask] = out["value"].float()

            for gi, m in enumerate(self._opp_models):
                envs = torch.nonzero(self._env_opp == gi, as_tuple=False).squeeze(-1)
                if envs.numel() == 0:
                    continue
                agents = envs * 2 + 1
                with torch.autocast("cuda", dtype=torch.float16,
                                    enabled=self._use_amp):
                    o = m.act(self.hist[agents])
                raw["pre"][agents] = o["pre"].float()
                raw["bins"][agents] = o["bins"].float()
                raw["fwd"][agents] = o["fwd"].long()
                raw["strafe"][agents] = o["strafe"].long()

            sim_actions = to_sim_actions(raw).view(self.N, 2, 7)
            obs, rew, done, winner = self._sim_step(sim_actions)

            buf.add(obs_now, self.age.clamp(min=0), raw, logp, value,
                    rew.reshape(self.B),
                    done.float().repeat_interleave(2))

            self._push_obs(obs.reshape(self.B, OBS_DIM), done)
            ep_done[t] = done
            ep_winner[t] = winner

        # résultats de matchs -> ELO (une seule synchro GPU -> CPU)
        done_cpu = ep_done.cpu().numpy()
        winner_cpu = ep_winner.cpu().numpy()
        env_opp_cpu = self._env_opp.cpu().numpy()
        for t, e in zip(*done_cpu.nonzero()):
            w = int(winner_cpu[t, e])
            ep_stats["matches"] += 1
            gi = int(env_opp_cpu[e])
            if gi < 0:
                ep_stats["mirror_matches"] += 1
                continue
            score = 1.0 if w == 0 else (0.5 if w == -1 else 0.0)
            self.league.report(self._opp_pool_idx[gi], score)
            ep_stats["wins" if w == 0 else ("draws" if w == -1 else "losses")] += 1
        return ep_stats

    # ------------------------------------------------------------------- main
    def train_iter(self) -> dict:
        t0 = time.perf_counter()
        learner_mask = self._assign_opponents()
        buf = self._buf
        buf.reset()
        ep = self._collect(buf, learner_mask)

        with torch.no_grad():
            last_value = torch.zeros(self.B, device=self.device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self._use_amp):
                out = self.policy.act(self.hist[learner_mask])
            last_value[learner_mask] = out["value"].float()

        progress = self.iter / max(self.cfg["total_iters"], 1)
        stats = self.ppo.update(buf, learner_mask, last_value, progress)

        self.iter += 1
        if self.iter % self.cfg["pool_every"] == 0 or len(self.league.pool) == 0:
            self.league.add_snapshot(self.policy)
        if self.iter % self.cfg["save_every"] == 0:
            self.save()

        dt = time.perf_counter() - t0
        self.total_steps += self.T * self.N * 2
        lm = learner_mask
        metrics = {
            "iter": self.iter,
            "total_steps": self.total_steps,
            "sps": round(self.T * self.N * 2 / dt),
            "reward_mean": float(buf.reward[:, lm].mean()),
            "elo": round(self.league.learner_elo, 1),
            "pool_size": len(self.league.pool),
            "league_winrate": (ep["wins"] + 0.5 * ep["draws"])
                              / max(ep["wins"] + ep["losses"] + ep["draws"], 1),
            "matches": ep["matches"],
            **{k: round(v, 5) for k, v in stats.items()},
            "time": round(dt, 2),
        }
        self._log(metrics)
        return metrics

    def train(self, iters: int | None = None) -> None:
        total = iters if iters is not None else self.cfg["total_iters"]
        for _ in range(total):
            m = self.train_iter()
            print(f"[{m['iter']:5d}] sps={m['sps']:>9} rew={m['reward_mean']:+.4f} "
                  f"elo={m['elo']:.0f} wr={m['league_winrate']:.2f} "
                  f"kl={m['approx_kl']:.4f}")

    # ------------------------------------------------------------ persistence
    def save(self) -> Path:
        path = self.run_dir / f"ckpt_{self.iter:06d}.pt"
        torch.save({
            "iter": self.iter,
            "total_steps": self.total_steps,
            "policy": self.policy.state_dict(),
            "optimizer": self.ppo.opt.state_dict(),
            "league": self.league.state_dict(),
            "cfg": self.cfg,
            "policy_cfg": self.pol_cfg.__dict__,
        }, path)
        torch.save(torch.load(path, map_location="cpu", weights_only=False),
                   self.run_dir / "latest.pt")
        return path

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["policy"])
        self.ppo.opt.load_state_dict(ckpt["optimizer"])
        self.league.load_state_dict(ckpt["league"])
        self.iter = ckpt["iter"]
        self.total_steps = ckpt.get("total_steps", 0)

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
        cfg = json.loads(Path(args.config).read_text())
    trainer = Trainer(cfg)
    if args.resume:
        trainer.load(args.resume)
        print(f"repris à l'itération {trainer.iter}")
    print(f"device={trainer.device} envs={trainer.N} rollout={trainer.T} "
          f"backend={'CUDA' if hasattr(trainer.sim, 'ext') else 'sim_ref(CPU)'}")
    trainer.train(args.iters)


if __name__ == "__main__":
    main()
