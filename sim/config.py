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
    aim_smooth_min: float = 0.0         # inertie de visée [0, 1) — EMA de la
    aim_smooth_max: float = 0.0         # commande de rotation (0 = instantané)

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

    def __post_init__(self):
        # Le kernel clampe h_delay à MAX_ACTION_DELAY-1 (file circulaire) ;
        # la référence honorerait n'importe quel delay -> refuser en amont
        # plutôt que diverger silencieusement.
        if not (0 <= self.delay_min <= self.delay_max <= MAX_ACTION_DELAY - 1):
            raise ValueError(
                f"delay_min/delay_max doivent vérifier 0 <= min <= max <= "
                f"{MAX_ACTION_DELAY - 1} (file kernel), reçu "
                f"({self.delay_min}, {self.delay_max})")
        if self.cps_min <= 0 or self.cps_max < self.cps_min:
            raise ValueError(
                f"cps_min/cps_max invalides : ({self.cps_min}, {self.cps_max})")
        if not (0.0 <= self.aim_smooth_min <= self.aim_smooth_max < 1.0):
            raise ValueError(
                f"aim_smooth_min/max doivent vérifier 0 <= min <= max < 1, "
                f"reçu ({self.aim_smooth_min}, {self.aim_smooth_max})")
        # Le kernel caste combo_window/combo_cap en int : exiger des entiers
        # pour qu'aucune divergence kernel <-> ref ne soit possible.
        if self.combo_window != int(self.combo_window) \
                or self.combo_cap != int(self.combo_cap):
            raise ValueError("combo_window et combo_cap doivent être entiers")
        self.combo_window = int(self.combo_window)
        self.combo_cap = int(self.combo_cap)

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
            self.aim_smooth_min, self.aim_smooth_max,
        ]


MAX_ACTION_DELAY = 8   # taille des files circulaires d'actions (ticks)
ACTION_DIM = 7         # dyaw, dpitch, forward, strafe, jump, sprint, attack
