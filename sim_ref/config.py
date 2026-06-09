"""Configuration d'un match boxing et des contraintes d'humanisation."""

from dataclasses import dataclass, field


@dataclass
class HumanizationConfig:
    """Limites "humaines" imposées à un agent (apprises pendant l'entraînement).

    max_cps         : clics par seconde max (le clic est ignoré pendant le cooldown)
    max_rot_speed   : vitesse de rotation max en degrés / tick (yaw et pitch)
    action_delay    : latence simulée en ticks entre décision et application
    """
    max_cps: float = 12.0
    max_rot_speed: float = 40.0
    action_delay: int = 0

    @property
    def click_cooldown_ticks(self) -> int:
        if self.max_cps <= 0:
            return 0
        return max(1, round(20.0 / self.max_cps))


@dataclass
class BoxingConfig:
    """Règles et géométrie d'un match boxing 1.8.9."""
    arena_size_x: float = 18.0      # murs à x=0 et x=arena_size_x
    arena_size_z: float = 18.0
    target_hits: int = 100          # premier à N hits gagne
    max_ticks: int = 20 * 60 * 5    # durée max d'un match (5 min)
    speed_amplifier: int = 1        # Speed II
    humanization: tuple = field(
        default_factory=lambda: (HumanizationConfig(), HumanizationConfig())
    )
