"""League self-play : pool de checkpoints adverses + classement ELO.

Le learner affronte en partie son propre miroir (poids courants) et en partie
des snapshots passés, échantillonnés en priorité près de son ELO
(prioritized fictitious self-play simplifié).
"""

import copy
import random


class League:
    def __init__(self, k_elo: float = 16.0, max_pool: int = 64):
        self.k = k_elo
        self.max_pool = max_pool
        self.pool: list[dict] = []      # {"name", "state_dict", "elo", "games"}
        self.learner_elo = 1000.0
        self._gen = 0

    # ------------------------------------------------------------------ pool
    def add_snapshot(self, policy) -> str:
        name = f"gen{self._gen:04d}"
        self._gen += 1
        sd = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
        self.pool.append({"name": name, "state_dict": sd,
                          "elo": self.learner_elo, "games": 0})
        if len(self.pool) > self.max_pool:
            # garde le plus ancien (ancrage) + les plus récents
            self.pool = [self.pool[0]] + self.pool[-(self.max_pool - 1):]
        return name

    def sample(self, n: int) -> list[int]:
        """n indices d'adversaires, pondérés par la proximité d'ELO."""
        if not self.pool:
            return []
        weights = [1.0 / (1.0 + ((e["elo"] - self.learner_elo) / 200.0) ** 2)
                   for e in self.pool]
        return random.choices(range(len(self.pool)), weights=weights, k=n)

    # ------------------------------------------------------------------- elo
    def report(self, idx: int, score: float) -> None:
        """score : 1 victoire learner, 0.5 nul, 0 défaite."""
        opp = self.pool[idx]
        expected = 1.0 / (1.0 + 10.0 ** ((opp["elo"] - self.learner_elo) / 400.0))
        delta = self.k * (score - expected)
        self.learner_elo += delta
        opp["elo"] -= delta
        opp["games"] += 1

    # ----------------------------------------------------------- persistence
    def state_dict(self) -> dict:
        return {"learner_elo": self.learner_elo, "gen": self._gen,
                "pool": copy.deepcopy(self.pool)}

    def load_state_dict(self, sd: dict) -> None:
        self.learner_elo = sd["learner_elo"]
        self._gen = sd["gen"]
        self.pool = sd["pool"]

    def summary(self) -> list[dict]:
        return [{"name": e["name"], "elo": round(e["elo"], 1), "games": e["games"]}
                for e in self.pool]
