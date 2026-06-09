"""sim_ref — simulateur de référence Minecraft 1.8.9 boxing (Python pur).

Vérité terrain de Judas : physique portée du code décompilé MCP, testée
tick par tick. Le kernel CUDA de `sim/` doit produire des trajectoires
équivalentes (voir tests/test_equivalence.py).
"""

from .config import BoxingConfig, HumanizationConfig
from .match import Action, BoxingMatch
from .player import PlayerState

__all__ = ["BoxingConfig", "HumanizationConfig", "Action", "BoxingMatch", "PlayerState"]
