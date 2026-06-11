"""Configuration du simulateur vectorisé (CUDA et backend de référence)."""

from dataclasses import dataclass


@dataclass
class SimConfig:
    # Géométrie / règles
    arena_size_x: float = 18.0
    arena_size_z: float = 18.0
    target_hits: int = 100
    max_ticks: int = 20 * 60 * 5
    speed_amplifier: int = 1            # Speed II (boxing)

    # Humanisation — plages de domain randomization (min == max => fixe)
    cps_min: float = 12.0
    cps_max: float = 12.0
    rot_speed_min: float = 40.0         # degrés / tick
    rot_speed_max: float = 40.0
    delay_min: int = 0                  # ticks de latence action
    delay_max: int = 0

    # Spawn
    spawn_jitter: float = 0.0           # jitter horizontal max (blocs)
    spawn_gap: float = 0.0              # demi-distance de spawn (0 = arène/3)

    # Knockback custom (plugins serveur type CustomKB) — 1.0 = vanilla
    kb_h_mult: float = 1.0              # multiplicateur horizontal
    kb_v_mult: float = 1.0              # multiplicateur vertical
    kb_idle_mult: float = 1.0           # mult. si la victime est immobile (sans input)

    # Reward
    reward_hit: float = 1.0
    reward_hurt: float = -1.0
    reward_win: float = 10.0
    reward_dist: float = 0.0            # shaping optionnel : -d * reward_dist
    reward_combo: float = 0.0           # bonus par maillon de chaîne (0 = off)
    combo_window: int = 25              # ticks max entre 2 hits d'une chaîne
    combo_cap: int = 5                  # plafond du multiplicateur de chaîne

    # Divers
    randomize: bool = False             # active jitter + randomization humanisation

    def as_floats(self) -> list:
        """Sérialise pour le kernel CUDA (ordre = struct SimParams du .cu)."""
        return [
            self.arena_size_x, self.arena_size_z,
            float(self.target_hits), float(self.max_ticks),
            float(self.speed_amplifier),
            self.cps_min, self.cps_max,
            self.rot_speed_min, self.rot_speed_max,
            float(self.delay_min), float(self.delay_max),
            self.spawn_jitter,
            self.reward_hit, self.reward_hurt, self.reward_win, self.reward_dist,
            1.0 if self.randomize else 0.0,
            self.spawn_gap,
            self.kb_h_mult, self.kb_v_mult, self.kb_idle_mult,
            self.reward_combo, float(self.combo_window), float(self.combo_cap),
        ]


MAX_ACTION_DELAY = 8   # taille des files circulaires d'actions (ticks)
ACTION_DIM = 7         # dyaw, dpitch, forward, strafe, jump, sprint, attack
