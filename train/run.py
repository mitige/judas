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
    "keep_ckpts": 10,            # rétention : derniers N + 1 sur 500
    "eval_every": 50,            # matchs d'éval auto vs anciens snapshots
    "eval_envs": 128,
    "eval_target_hits": 20,
    "eval_max_ticks": 1500,
    "shaping_hit_rate": 5.0,     # hits/min déclenchant la rampe (shaping + spawn)
    "shaping_decay_iters": 100,
    "curriculum_gap": 2.0,       # spawn proche au début (0 = désactivé)
    "snapshot_gate": True,       # snapshot league seulement s'il bat le précédent
    "league_bot_frac": 0.2,      # part d'envs vs chase-bot (annealée -> 0.05)
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
        # inférence de rollout en fp16 (l'update PPO garde sa propre AMP)
        self._use_amp = self.ppo.use_amp
        self._buf = RolloutBuffer(self.T, self.B, OBS_DIM, self.H, self.device)

        self.hist = torch.zeros(self.B, self.H, OBS_DIM, device=self.device)
        # -1 : le premier _push_obs (reset) donne l'âge 0 à l'obs de spawn
        self.age = torch.full((self.B,), -1, dtype=torch.long, device=self.device)
        self.iter = 0
        self.total_steps = 0          # agent-steps cumulés (tous agents)

        # automatisations
        self._shaping_base = self.sim_cfg.reward_dist
        self._ramp_start: int | None = None       # début de la rampe combat
        self._hit_streak = 0
        self._entropy_hist: list[float] = []
        self._eval_sim = None
        self._eval_opp: JudasPolicy | None = None
        self._snapshot_skips = 0
        self._best_bot = -1.0
        self._full_gap = min(self.sim_cfg.arena_size_x,
                             self.sim_cfg.arena_size_z) / 3.0
        from .scripted import ChaseBot
        self._bot = ChaseBot()

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

    def _sim_step(self, actions: torch.Tensor, sim=None):
        sim = sim or self.sim
        if hasattr(sim, "ext"):                          # backend CUDA
            obs, rew, done, info = sim.step(actions)
            return (obs, rew.clone(), done.bool().clone(),
                    info["winner"].clone())
        a = actions.cpu().numpy()
        obs, rew, done, info = sim.step(a)
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
        """Choisit les adversaires du rollout (snapshots league + chase-bot).
        Retourne learner_mask [B]. _env_opp : -1 miroir, -2 bot, >=0 groupe."""
        learner_mask = torch.ones(self.B, dtype=torch.bool, device=self.device)
        self._opp_models, self._opp_pool_idx = [], []
        self._env_opp.fill_(-1)
        perm = torch.randperm(self.N, device=self.device)

        # part de matchs contre le chase-bot (forte au début, annealée à 5%)
        bf = self.cfg["league_bot_frac"]
        n_bot = int(self.N * max(0.05, bf * (1.0 - self._ramp_frac()))) if bf > 0 else 0
        if n_bot > 0:
            bot_envs = perm[:n_bot]
            self._env_opp[bot_envs] = -2
            learner_mask[bot_envs * 2 + 1] = False

        frac = self.cfg["league_frac"]
        if not self.league.pool or frac <= 0:
            return learner_mask
        n_league = min(int(self.N * frac), self.N - n_bot)
        if n_league <= 0:
            return learner_mask
        n_groups = min(4, len(self.league.pool))
        idxs = self.league.sample(n_groups)
        env_ids = perm[n_bot:n_bot + n_league]
        groups = env_ids.chunk(n_groups)
        for gi, (pool_idx, envs) in enumerate(zip(idxs, groups)):
            if envs.numel() == 0:
                continue
            m = JudasPolicy(self.pol_cfg).to(self.device)
            m.load_state_dict(self.league.pool[pool_idx]["state_dict"], strict=False)
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

            sim_actions = to_sim_actions(raw)
            bot_envs = torch.nonzero(self._env_opp == -2, as_tuple=False).squeeze(-1)
            if bot_envs.numel() > 0:
                agents = bot_envs * 2 + 1
                sim_actions[agents] = self._bot.act7(self.hist[agents])
            obs, rew, done, winner = self._sim_step(sim_actions.view(self.N, 2, 7))

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
            if gi == -2:                 # match vs chase-bot : pas d'ELO
                continue
            if gi == -1:
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
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_collect = time.perf_counter() - t0

        with torch.no_grad():
            last_value = torch.zeros(self.B, device=self.device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self._use_amp):
                out = self.policy.act(self.hist[learner_mask])
            last_value[learner_mask] = out["value"].float()

        progress = self.iter / max(self.cfg["total_iters"], 1)
        stats = self.ppo.update(buf, learner_mask, last_value, progress)

        self.iter += 1
        if self.iter % self.cfg["pool_every"] == 0 or len(self.league.pool) == 0:
            self._maybe_snapshot()
        if self.iter % self.cfg["save_every"] == 0:
            self.save()

        lm = learner_mask
        # hits/minute par agent learner (proxy : reward d'un hit ~ +1)
        hit_rate = float((buf.reward[:, lm] > 0.9).float().mean()) * 1200.0
        self._update_ramp(hit_rate)
        shaping = self._auto_shaping()
        spawn_gap = self._auto_curriculum()
        warn_entropy = self._entropy_guard(stats["entropy"])

        # éval automatique : snapshots passés + bot scripté (étalon absolu)
        evals = {}
        if (self.cfg["eval_every"] > 0 and self.league.pool
                and self.iter % self.cfg["eval_every"] == 0):
            evals["eval_first"] = self._evaluate(self.league.pool[0]["state_dict"])
            if len(self.league.pool) > 1:
                evals["eval_past"] = self._evaluate(
                    self.league.pool[-1]["state_dict"])
            evals["eval_bot"] = self._evaluate("bot")
            # meilleur checkpoint absolu -> best.pt (celui à déployer)
            if evals["eval_bot"] > self._best_bot:
                self._best_bot = evals["eval_bot"]
                torch.save({
                    "iter": self.iter,
                    "total_steps": self.total_steps,
                    "eval_bot": self._best_bot,
                    "policy": self.policy.state_dict(),
                    "policy_cfg": self.pol_cfg.__dict__,
                }, self.run_dir / "best.pt")

        dt = time.perf_counter() - t0
        self.total_steps += self.T * self.N * 2
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
            "hit_rate": round(hit_rate, 2),
            "shaping": round(shaping, 6),
            "spawn_gap": round(spawn_gap, 2),
            "warn_entropy": warn_entropy,
            **{k: round(v, 4) for k, v in evals.items()},
            **{k: round(v, 5) for k, v in stats.items()},
            "time": round(dt, 2),
            "time_collect": round(t_collect, 2),
            "time_update": round(dt - t_collect, 2),
        }
        self._log(metrics)
        return metrics

    # -------------------------------------------------------- automatisations
    def _update_ramp(self, hit_rate: float) -> None:
        """Déclenche la rampe (decay shaping + élargissement spawn) quand les
        agents frappent régulièrement, de façon soutenue."""
        if self._ramp_start is not None:
            return
        if hit_rate >= self.cfg["shaping_hit_rate"]:
            self._hit_streak += 1
        else:
            self._hit_streak = 0
        if self._hit_streak >= 10:
            self._ramp_start = self.iter

    def _ramp_frac(self) -> float:
        if self._ramp_start is None:
            return 0.0
        return min((self.iter - self._ramp_start)
                   / max(self.cfg["shaping_decay_iters"], 1), 1.0)

    def _auto_shaping(self) -> float:
        """Shaping de distance -> 0 le long de la rampe."""
        if self._shaping_base <= 0:
            return 0.0
        value = self._shaping_base * (1.0 - self._ramp_frac())
        if self._ramp_start is not None:
            self.sim.set_reward_dist(value)
        return value

    def _auto_curriculum(self) -> float:
        """Spawn proche -> distance standard le long de la rampe."""
        cg = self.cfg["curriculum_gap"]
        if cg <= 0:
            return self.sim_cfg.spawn_gap or self._full_gap
        frac = self._ramp_frac()
        gap = cg + (self._full_gap - cg) * frac
        if self._ramp_start is not None:
            # frac = 1 -> 0 = mode auto (arène/3), valeur standard exacte
            self.sim.set_spawn_gap(0.0 if frac >= 1.0 else gap)
        return gap

    def _maybe_snapshot(self) -> None:
        """Snapshot league, gaté : seulement si le learner bat le précédent
        (winrate >= 0.52). Forcé après 2 refus pour éviter la stagnation."""
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
        """1 si l'entropie vient de s'effondrer (< 50% de la médiane récente)."""
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
    def _evaluate(self, opponent) -> float:
        """Winrate du learner sur des matchs courts.
        opponent : state_dict d'un snapshot, ou "bot" (chase-bot scripté)."""
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
        score, finished = 0.0, 0
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
            obs, _, done, winner = self._sim_step(actions, sim=self._eval_sim)
            hist = torch.roll(hist, shifts=-1, dims=2)
            hist[done.bool()] = 0.0
            hist[:, :, -1] = obs.float()
            for w in winner[done.bool()].tolist():
                finished += 1
                score += 1.0 if w == 0 else (0.5 if w == -1 else 0.0)
            if finished >= n:
                break
        return score / finished if finished else 0.5

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
            "ramp_start": self._ramp_start,
            "policy": self.policy.state_dict(),
            "optimizer": self.ppo.opt.state_dict(),
            "league": self.league.state_dict(),
            "cfg": self.cfg,
            "policy_cfg": self.pol_cfg.__dict__,
        }, path)
        torch.save(torch.load(path, map_location="cpu", weights_only=False),
                   self.run_dir / "latest.pt")
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

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        # strict=False : compatibilité avec les checkpoints d'avant la tête aux
        self.policy.load_state_dict(ckpt["policy"], strict=False)
        try:
            self.ppo.opt.load_state_dict(ckpt["optimizer"])
        except (ValueError, KeyError):
            pass    # architecture modifiée -> optimizer frais
        self.league.load_state_dict(ckpt["league"])
        self.iter = ckpt["iter"]
        self.total_steps = ckpt.get("total_steps", 0)
        self._ramp_start = ckpt.get("ramp_start")

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
