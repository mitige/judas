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
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from sim import OBS_DIM, SimConfig, make_sim

from .buffer import RolloutBuffer
from .league import League
from .model import JudasPolicy, PolicyConfig, to_sim_actions
from .pbt import (DEFAULT_PBT, Member, apply_hypers, elo_delta,
                  exploit_explore, perturb_hypers, slice_envs)
from .ppo import PPO, PPOConfig

DEFAULT_CFG = {
    "name": "boxing",
    "total_iters": 300,
    "n_envs": 4096,
    "rollout_ticks": 128,
    "league_frac": 0.3,
    "pool_every": 25,
    "save_every": 25,
    "keep_ckpts": 10,            # rétention : derniers N + 1 sur 500
    "eval_every": 25,            # matchs d'éval auto vs anciens snapshots
    "eval_envs": 128,
    "eval_target_hits": 15,      # 15 hits / 900 ticks : mêmes signaux, taxe
    "eval_max_ticks": 900,       # d'éval ~9% au lieu de ~15% du temps total
    "shaping_hit_rate": 5.0,     # hits/min déclenchant la rampe (shaping + spawn)
    "shaping_decay_iters": 100,
    "shaping_floor_frac": 0.0,   # plancher du shaping distance en fin de rampe
                                 # (0 = s'éteint ; 0.25 = pression permanente)
    "curriculum_gap": 2.0,       # spawn proche au début (0 = désactivé)
    "snapshot_gate": True,       # snapshot league seulement s'il bat le précédent
    "league_bot_frac": 0.25,     # part d'envs vs chase-bot (annealée -> 0.05)
    "cuda_graphs": True,         # capture le forward de rollout (fallback eager)
    "seed": 0,
    "sim": {"target_hits": 50},
    "policy": {"history": 8, "d_model": 96, "n_heads": 4, "n_layers": 2},
    "ppo": {},
    "pbt": {},                   # voir train/pbt.py::DEFAULT_PBT
}


class Trainer:
    def __init__(self, cfg: dict, device: str | None = None):
        self.cfg = {**DEFAULT_CFG, **cfg}
        self.cfg["sim"] = {**DEFAULT_CFG["sim"], **cfg.get("sim", {})}
        self.cfg["pbt"] = {**DEFAULT_PBT, **cfg.get("pbt", {})}
        torch.manual_seed(self.cfg["seed"])
        np.random.seed(self.cfg["seed"])
        random.seed(self.cfg["seed"])          # league.sample (random.choices)

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

        # ----- population (PBT) : K policies co-entraînées sur des slices
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
            # membre -> env : pour créditer les ELO des matchs league/cross
            self._env_member = torch.zeros(self.N, dtype=torch.long)
            for m in self.members:
                self._env_member[m.env_lo:m.env_hi] = m.idx
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
        self._ramp_on = False        # rampe déclenchée (combat régulier atteint)
        self._ramp_pos = 0.0         # position 0..1 — adaptative, peut reculer
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
        self._opp_kind: list[tuple] = []
        self._opp_rows: list[torch.Tensor] = []
        self._bot_rows: torch.Tensor = torch.empty(0, dtype=torch.long,
                                                   device=self.device)
        self._env_opp: torch.Tensor = torch.full((self.N,), -1, dtype=torch.long,
                                                 device=self.device)
        self._learner_mask: torch.Tensor | None = None
        # CUDA graphs des forwards de rollout (capturés au 1er rollout,
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
            return (obs, rew.clone(), done.bool().clone(),
                    info["winner"].clone())
        a = actions.cpu().numpy()
        obs, rew, done, info = sim.step(a)
        return (self._to_torch(obs), self._to_torch(rew),
                self._to_torch(done).bool(), self._to_torch(info["winner"]))

    def _push_obs(self, obs_flat: torch.Tensor, done_env: torch.Tensor) -> None:
        """Avance l'historique d'un tick. done_env [N] bool/float.
        self.hist garde un STOCKAGE FIXE (copy_ au lieu de réassignation) :
        c'est l'entrée statique du CUDA graph de _learner_act."""
        done_b = done_env.bool().repeat_interleave(2)
        self.hist.copy_(torch.roll(self.hist, shifts=-1, dims=1))
        self.hist[done_b] = 0.0
        self.hist[:, -1] = obs_flat
        self.age = torch.where(done_b, torch.zeros_like(self.age), self.age + 1)
        # NB: âge de l'obs courante = 0 si nouvel épisode

    # -------------------------------------------------------------- opponents
    def _assign_opponents(self) -> torch.Tensor:
        """Choisit les adversaires du rollout (snapshots league + chase-bot,
        + autres membres en mode population). Retourne learner_mask [B].
        _env_opp : -1 miroir, -2 bot, >=0 indice de contrôleur. Précalcule
        les lignes par contrôleur (_opp_rows/_bot_rows) une fois par fenêtre
        d'assignation — plus de nonzero par tick. _opp_kind décrit chaque
        contrôleur : ("league", pool_idx) ou ("cross", member_idx)."""
        if self.members:
            return self._assign_opponents_pbt()
        learner_mask = torch.ones(self.B, dtype=torch.bool, device=self.device)
        self._opp_models, self._opp_pool_idx, self._opp_rows = [], [], []
        self._opp_kind = []
        self._bot_rows = torch.empty(0, dtype=torch.long, device=self.device)
        self._env_opp.fill_(-1)
        perm = torch.randperm(self.N, device=self.device)

        # part de matchs contre le chase-bot (forte au début, annealée à 5%)
        bf = self.cfg["league_bot_frac"]
        n_bot = int(self.N * max(0.05, bf * (1.0 - self._ramp_frac()))) if bf > 0 else 0
        if n_bot > 0:
            bot_envs = perm[:n_bot]
            self._env_opp[bot_envs] = -2
            self._bot_rows = bot_envs * 2 + 1
            learner_mask[self._bot_rows] = False

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
            self._opp_kind.append(("league", pool_idx))
            self._opp_rows.append(envs * 2 + 1)          # agent 1 = adversaire gelé
            self._env_opp[envs] = gi
            learner_mask[envs * 2 + 1] = False
        return learner_mask

    def _assign_opponents_pbt(self) -> torch.Tensor:
        """Assignation en mode population : pour chaque membre, sa tranche
        d'envs est répartie entre chase-bot (annealé), CROSS-PLAY contre un
        autre membre (rotation à chaque fenêtre), snapshots league partagés,
        et miroir intra-membre (le reste)."""
        learner_mask = torch.ones(self.B, dtype=torch.bool, device=self.device)
        self._opp_models, self._opp_pool_idx, self._opp_rows = [], [], []
        self._opp_kind = []
        self._env_opp.fill_(-1)
        pbt = self.cfg["pbt"]
        K = len(self.members)
        bot_rows_all = []
        bf = self.cfg["league_bot_frac"]
        bot_frac = max(0.05, bf * (1.0 - self._ramp_frac())) if bf > 0 else 0.0
        rot = (self.iter // max(self.cfg["pool_every"], 1)) % max(K - 1, 1)
        for mb in self.members:
            n_m = mb.env_hi - mb.env_lo
            perm = torch.randperm(n_m, device=self.device) + mb.env_lo
            cursor = 0
            n_bot = int(n_m * bot_frac)
            if n_bot > 0:
                envs = perm[:n_bot]
                cursor = n_bot
                self._env_opp[envs] = -2
                bot_rows_all.append(envs * 2 + 1)
                learner_mask[envs * 2 + 1] = False
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
                n_league = min(int(n_m * self.cfg["league_frac"]),
                               n_m - cursor)
                n_groups = min(2, len(self.league.pool))
                if n_league > 0 and n_groups > 0:
                    envs = perm[cursor:cursor + n_league]
                    cursor += n_league
                    for pool_idx, chunk in zip(self.league.sample(n_groups),
                                               envs.chunk(n_groups)):
                        if chunk.numel() == 0:
                            continue
                        model = JudasPolicy(self.pol_cfg).to(self.device)
                        model.load_state_dict(
                            self.league.pool[pool_idx]["state_dict"],
                            strict=False)
                        model.eval()
                        gi = len(self._opp_models)
                        self._opp_models.append(model)
                        self._opp_kind.append(("league", pool_idx))
                        self._opp_rows.append(chunk * 2 + 1)
                        self._env_opp[chunk] = gi
                        learner_mask[chunk * 2 + 1] = False
            # le reste de la tranche = miroir intra-membre (les deux agents
            # contrôlés par le membre, gradient des deux côtés)
        self._bot_rows = (torch.cat(bot_rows_all) if bot_rows_all
                          else torch.empty(0, dtype=torch.long,
                                           device=self.device))
        return learner_mask

    # ---------------------------------------------------------------- rollout
    @torch.no_grad()
    def _act_eager(self) -> dict:
        with torch.autocast("cuda", dtype=torch.float16, enabled=self._use_amp):
            return self.policy.act(self.hist)

    @torch.no_grad()
    def _capture_act_graph(self) -> None:
        """Capture le forward learner (batch complet, forme fixe) dans un
        CUDA graph : ~30-60 lancements de kernels par tick remplacés par un
        seul replay. Les poids restent à jour (updates Adam in-place) et
        self.hist garde un stockage fixe (_push_obs). Fallback eager si la
        capture échoue (op non capturable, driver, etc.)."""
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
            print(f"[trainer] CUDA graph indisponible ({exc}) — fallback eager")

    @torch.no_grad()
    def _learner_act(self) -> dict:
        """Forward du learner sur le batch COMPLET (forme fixe, graph-friendly).
        Les lignes des adversaires gelés sont écrasées ensuite — leurs
        logp/value parasites ne sont jamais échantillonnés (b_keep)."""
        if not self._graphs_tried:
            self._capture_act_graph()
        if self._act_graph is not None:
            self._act_graph.replay()
            # clones : les tenseurs de sortie du graph sont réécrits au
            # prochain replay, l'appelant doit posséder ses données
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
        exploit_explore copie les poids IN-PLACE (load_state_dict) — les
        graphs restent valides après les copies de population."""
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
                  f"— fallback eager")

    @torch.no_grad()
    def _policy_actions(self) -> dict:
        """act de tous les contrôleurs apprenants : la policy unique (batch
        complet, CUDA graph) ou les K membres de la population (un graph par
        tranche contiguë ; torch.cat matérialise des copies — les buffers de
        sortie des graphs sont réécrits au replay suivant)."""
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
        # résultats accumulés sur GPU, traités en une seule synchro après le rollout
        ep_done = torch.zeros(self.T, self.N, dtype=torch.bool, device=self.device)
        ep_winner = torch.zeros(self.T, self.N, dtype=torch.int32, device=self.device)
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
            if self._bot_rows.numel() > 0:
                sim_actions[self._bot_rows] = self._bot.act7(self.hist[self._bot_rows])
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
            kind, ref = self._opp_kind[gi]
            if self.members:
                me = self.members[int(self._env_member[e])]
                if kind == "league":
                    entry = self.league.pool[ref]
                    d = elo_delta(me.elo, entry["elo"], score)
                    me.elo += d
                    entry["elo"] -= d
                    entry["games"] += 1
                else:                    # cross-play : ELO zéro-somme membres
                    other = self.members[ref]
                    d = elo_delta(me.elo, other.elo, score)
                    me.elo += d
                    other.elo -= d
                    other.games += 1
                me.games += 1
            else:
                self.league.report(ref, score)
            ep_stats["wins" if w == 0 else ("draws" if w == -1 else "losses")] += 1
        return ep_stats

    # ------------------------------------------------------------------- main
    def train_iter(self) -> dict:
        t0 = time.perf_counter()
        # Adversaires COLLANTS : réassignés seulement toutes les pool_every
        # itérations. Les épisodes (>1000 ticks) s'étalent sur ~10-30 rollouts
        # de 128 ticks : réassigner à chaque itération attribuait à l'ELO d'un
        # snapshot des matchs majoritairement joués contre d'autres contrôleurs.
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
            # gelées portent des valeurs parasites jamais échantillonnées)
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
            # le « meilleur membre » porte les chemins single-policy :
            # éval, snapshots league, best.pt, export
            best = max(self.members, key=lambda m: m.elo)
            self.policy = best.policy
            self.ppo = best.ppo
            if self.iter % self.cfg["pbt"]["interval"] == 0:
                events = exploit_explore(self.members, self.cfg["pbt"],
                                         self._pbt_rng)
                for loser, winner in events:
                    print(f"[pbt] membre {loser} <- copie du membre {winner} "
                          f"(hypers perturbés)")
        if self.iter % self.cfg["pool_every"] == 0 or len(self.league.pool) == 0:
            self._maybe_snapshot()
        if self.iter % self.cfg["save_every"] == 0:
            self.save()

        lm = learner_mask
        # Hits EXACTS depuis l'obs (o[31] = hits/100) plutôt que par seuil sur
        # le reward : les trades (hit + hurt le même tick, reward ~ 0) sont
        # comptés correctement. Le tick de done est exclu (obs = nouveau match).
        hits_obs = (buf.obs[:, :, 31] * 100.0).round()                 # [T, B]
        final_hits = (self.hist[:, -1, 31] * 100.0).round().unsqueeze(0)
        next_hits = torch.cat([hits_obs[1:], final_hits], dim=0)
        dealt = ((next_hits - hits_obs).clamp(min=0.0)
                 * (1.0 - buf.done)) > 0.5                             # [T, B]
        taken = dealt.view(self.T, self.N, 2).flip(-1).reshape(self.T, self.B)
        hits_mask = dealt[:, lm]
        hit_rate = float(hits_mask.float().mean()) * 1200.0
        # % de ticks à portée d'engagement (< 3.5 blocs) — thermomètre
        # d'agressivité/pression ; o[45] = dist_h / 8
        engage = float((buf.obs[:, lm, 45] * 8.0 < 3.5).float().mean())
        # % de hits portés en sprint+forward (le "Z-tap" mesurable : un bon
        # joueur converge vers ~1.0 — chaque hit profite du KB bonus sprint)
        sprint_act = (buf.bins[:, lm, 1] > 0.5) & (buf.fwd[:, lm] == 2)
        sprint_hit = float((hits_mask & sprint_act).float().sum()
                           / hits_mask.float().sum().clamp(min=1.0))
        # % de hits portés en chaîne (bonus combo > 0) — thermomètre du style
        # combo ; 0 si le shaping est désactivé. Mesuré sur les hits HORS
        # trade : sur un trade le reward (hit + hurt ± combo) est inclassable.
        rc = float(self.sim_cfg.reward_combo)
        if rc > 0.0:
            clean_hits = hits_mask & ~taken[:, lm]
            combo_mask = buf.reward[:, lm] > self.sim_cfg.reward_hit + 0.5 * rc
            combo_hit = float((clean_hits & combo_mask).float().sum()
                              / clean_hits.float().sum().clamp(min=1.0))
        else:
            combo_hit = 0.0
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
            "sprint_hits": round(sprint_hit, 4),
            "combo_hits": round(combo_hit, 4),
            "engage_rate": round(engage, 4),
            "shaping": round(shaping, 6),
            "spawn_gap": round(spawn_gap, 2),
            "ramp": round(self._ramp_pos, 3),
            "warn_entropy": warn_entropy,
            **{k: round(v, 4) for k, v in evals.items()},
            **{k: round(v, 5) for k, v in stats.items()},
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
        self._log(metrics)
        return metrics

    # -------------------------------------------------------- automatisations
    def _update_ramp(self, hit_rate: float) -> None:
        """Rampe ADAPTATIVE et ÉTAGÉE.

        Étagée  : phase 1 (pos 0 -> 0.5) le spawn s'élargit, shaping intact ;
                  phase 2 (pos 0.5 -> 1) le shaping décroît vers 0.
                  (jamais les deux béquilles retirées en même temps)
        Adaptative : la position avance quand le combat est sain et RECULE
                  si le hit rate s'effondre — auto-récupération du signal."""
        thresh = self.cfg["shaping_hit_rate"]
        if not self._ramp_on:
            if hit_rate >= thresh:
                self._hit_streak += 1
            else:
                self._hit_streak = 0
            if self._hit_streak >= 10:
                self._ramp_on = True
            return
        step = 1.0 / max(self.cfg["shaping_decay_iters"], 1)
        if hit_rate >= thresh:
            self._ramp_pos = min(self._ramp_pos + step, 1.0)
        elif hit_rate < 0.5 * thresh:
            self._ramp_pos = max(self._ramp_pos - 2.0 * step, 0.0)

    def _ramp_frac(self) -> float:
        return self._ramp_pos

    def _auto_shaping(self) -> float:
        """Phase 2 de la rampe : shaping plein jusqu'à pos 0.5, puis décroît
        vers le plancher shaping_floor_frac (0 = extinction complète ;
        > 0 = pression de rapprochement permanente, anti-passivité)."""
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
            # gphase = 1 -> 0 = mode auto (arène/3), valeur standard exacte
            self.sim.set_spawn_gap(0.0 if gphase >= 1.0 else gap)
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
        score = 0.0
        # un seul match compté par env : les envs auto-reset continuent de
        # tourner mais leurs matchs suivants (les plus courts) sont ignorés
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
            obs, _, done, winner = self._sim_step(actions, sim=self._eval_sim)
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
        for _ in range(total):
            m = self.train_iter()
            print(f"[{m['iter']:5d}] sps={m['sps']:>9} rew={m['reward_mean']:+.4f} "
                  f"elo={m['elo']:.0f} wr={m['league_winrate']:.2f} "
                  f"kl={m['approx_kl']:.4f}")

    # ------------------------------------------------------------ persistence
    def save(self) -> Path:
        path = self.run_dir / f"ckpt_{self.iter:06d}.pt"
        payload = {
            "iter": self.iter,
            "total_steps": self.total_steps,
            "ramp_on": self._ramp_on,
            "ramp_pos": self._ramp_pos,
            "best_bot": self._best_bot,
            # toujours le MEILLEUR membre en tête de checkpoint : export,
            # serve et les vieux loaders fonctionnent sans connaître le PBT
            "policy": self.policy.state_dict(),
            "optimizer": self.ppo.opt.state_dict(),
            "scaler": self.ppo.scaler.state_dict(),
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
        # écriture atomique : un kill pendant le save (stop de l'app,
        # autorestart) ne peut pas laisser un ckpt/latest.pt tronqué
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

    @staticmethod
    def _fix_fused_optimizer_state(opt, device: torch.device) -> None:
        """Recharger un checkpoint d'Adam NON-fused dans l'Adam fused :
        load_state_dict écrase param_groups (fused redevient None) alors que
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

    def rotate_metrics(self) -> None:
        """Archive le metrics.jsonl d'un run précédent du même nom (démarrage
        frais sans --resume) : metrics-NNN.jsonl. Sans ça, les courbes de
        l'app concatènent les runs et affichent des falaises trompeuses."""
        path = self.run_dir / "metrics.jsonl"
        if not path.exists():
            return
        n = 1
        while (self.run_dir / f"metrics-{n:03d}.jsonl").exists():
            n += 1
        path.rename(self.run_dir / f"metrics-{n:03d}.jsonl")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        # strict=False : compatibilité avec les checkpoints d'avant la tête aux
        self.policy.load_state_dict(ckpt["policy"], strict=False)
        try:
            self.ppo.opt.load_state_dict(ckpt["optimizer"])
            self._fix_fused_optimizer_state(self.ppo.opt, self.device)
        except (ValueError, KeyError):
            pass    # architecture modifiée -> optimizer frais
        if ckpt.get("scaler"):              # {} = scaler désactivé au save
            try:
                self.ppo.scaler.load_state_dict(ckpt["scaler"])
            except RuntimeError:
                pass                        # amp off/on entre les deux runs
        self.league.load_state_dict(ckpt["league"])
        self.iter = ckpt["iter"]
        self.total_steps = ckpt.get("total_steps", 0)
        self._best_bot = ckpt.get("best_bot", -1.0)
        if "ramp_pos" in ckpt:
            self._ramp_on = ckpt["ramp_on"]
            self._ramp_pos = ckpt["ramp_pos"]
        elif ckpt.get("ramp_start") is not None:
            # ancien format (rampe minutée) : reprise adaptative depuis sa position
            self._ramp_on = True
            self._ramp_pos = min((self.iter - ckpt["ramp_start"])
                                 / max(self.cfg["shaping_decay_iters"], 1), 1.0)
        # ré-applique immédiatement le curriculum (sinon la 1re itération
        # post-resume collecte avec le spawn_gap/shaping initiaux périmés)
        self._auto_shaping()
        self._auto_curriculum()
        # reprise reproductible : restaure tous les états RNG si présents
        if ckpt.get("torch_rng") is not None:
            torch.set_rng_state(ckpt["torch_rng"].cpu())
        if ckpt.get("cuda_rng") is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["cuda_rng"]])
            except (RuntimeError, IndexError):
                pass    # nombre de GPU différent -> RNG frais
        if ckpt.get("numpy_rng") is not None:
            np.random.set_state(ckpt["numpy_rng"])
        if ckpt.get("python_rng") is not None:
            random.setstate(ckpt["python_rng"])
        if self.members:
            self._load_population(ckpt)
        self._truncate_metrics()

    def _load_population(self, ckpt: dict) -> None:
        """Restaure la population, ou la SEED depuis un checkpoint
        single-policy (tous les membres partent des mêmes poids, hypers
        déjà diversifiés par l'init ; membre 0 = hypers de base)."""
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
            print(f"[pbt] population de {len(self.members)} membres restaurée")
        else:
            for m in self.members:
                if m.policy is not self.policy:    # membre 0 déjà chargé
                    m.policy.load_state_dict(ckpt["policy"], strict=False)
            print(f"[pbt] population de {len(self.members)} membres seedée "
                  f"depuis le checkpoint single-policy (iter {self.iter})")

    def _truncate_metrics(self) -> None:
        """Supprime les lignes de métriques POSTÉRIEURES à l'itération
        reprise (progrès non sauvegardé d'une session interrompue) : sans ça,
        les courbes contiennent des itérations en double après un resume."""
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
        # utf-8-sig : tolère le BOM des éditeurs/outils Windows
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    trainer = Trainer(cfg)
    if args.resume:
        trainer.load(args.resume)
        print(f"repris à l'itération {trainer.iter}")
    else:
        trainer.rotate_metrics()    # run frais : ne pas concaténer les courbes
    print(f"device={trainer.device} envs={trainer.N} rollout={trainer.T} "
          f"backend={'CUDA' if hasattr(trainer.sim, 'ext') else 'sim_ref(CPU)'}")
    trainer.train(args.iters)


if __name__ == "__main__":
    main()
