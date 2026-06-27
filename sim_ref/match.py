"""Match boxing complet à 20 TPS — boucle de tick de référence.

Ordre d'un tick Judas (les deux agents sont résolus symétriquement) :
  1. décrément des timers (hurtResistantTime, click_cooldown)
  2. application des rotations (clamp humanisation)
  3. mise à jour du sprint (clé + forward > 0, coupé par collision murale)
  4. résolution SÉQUENTIELLE des attaques (agent 0 puis agent 1, état muté
     entre les deux — comme le serveur vanilla qui traite les paquets dans
     l'ordre d'arrivée), avant le tick de mouvement
  5. tick de mouvement des deux agents (saut + moveEntityWithHeading)
  6. règles boxing : victoire à target_hits (double atteinte = égalité),
     timeout = égalité (anti-stall : fuir avec le score ne gagne jamais)

Note de fidélité : vanilla traite les paquets des deux clients dans un ordre
réseau arbitraire ; Judas fixe l'ordre agent 0 puis agent 1 (déterminisme).
Les trades restent symétriques : un hit ne bloque pas la riposte du même tick
(hurtResistantTime ne gate que les hits REÇUS). Le sprint est de type
"toggle-sprint" (re-engagé tant que la touche est tenue), comportement
standard des clients PvP 1.8.9.
"""

from collections import deque
from dataclasses import dataclass

from .combat import apply_entity_collision, try_attack
from .config import BoxingConfig
from .physics import living_update_movement
from .player import PlayerState


@dataclass
class Action:
    """Action d'un agent pour un tick."""
    dyaw: float = 0.0       # degrés (clampé par humanisation)
    dpitch: float = 0.0
    forward: int = 0        # -1, 0, 1
    strafe: int = 0         # -1 (droite), 0, 1 (gauche) — convention vanilla
    jump: bool = False
    sprint: bool = False    # touche sprint tenue
    attack: bool = False    # clic gauche


NOOP = Action()


class BoxingMatch:
    def __init__(self, cfg: BoxingConfig | None = None):
        self.cfg = cfg or BoxingConfig()
        self.players: list[PlayerState] = []
        self.tick_count = 0
        self.winner: int | None = None
        self._delay_queues: list[deque] = []
        self.last_dealt = [0, 0]
        self.last_sprint_hits = [False, False]
        self.reset()

    # ------------------------------------------------------------------ reset
    def reset(self) -> None:
        cfg = self.cfg
        cx, cz = cfg.arena_size_x / 2.0, cfg.arena_size_z / 2.0
        gap = cfg.spawn_gap if cfg.spawn_gap > 0.0 else \
            min(cfg.arena_size_x, cfg.arena_size_z) / 3.0
        # Face à face le long de l'axe Z, yaw 0 = +Z, yaw 180 = -Z
        self.players = [
            PlayerState(x=cx, y=0.0, z=cz - gap, yaw=0.0),
            PlayerState(x=cx, y=0.0, z=cz + gap, yaw=180.0),
        ]
        self.tick_count = 0
        self.winner = None
        self.last_dealt = [0, 0]
        self.last_sprint_hits = [False, False]
        self._delay_queues = []
        for h in cfg.humanization:
            q: deque = deque()
            for _ in range(h.action_delay):
                q.append(NOOP)
            self._delay_queues.append(q)

    # ------------------------------------------------------------------- step
    def step(self, actions: tuple) -> None:
        """Avance le match d'un tick avec (action_agent0, action_agent1)."""
        if self.winner is not None:
            return
        cfg = self.cfg
        self.last_dealt = [0, 0]
        self.last_sprint_hits = [False, False]

        # Latence simulée : l'action décidée maintenant s'applique plus tard
        applied: list[Action] = []
        for i, act in enumerate(actions):
            q = self._delay_queues[i]
            q.append(act)
            applied.append(q.popleft())

        # 1. timers (Entity.onEntityUpdate, avant le mouvement)
        for p in self.players:
            if p.hurt_resistant_time > 0:
                p.hurt_resistant_time -= 1
            if p.click_cooldown > 0:
                p.click_cooldown -= 1

        # 2. rotations clampées + modèle moteur de visée (EMA — inertie ;
        #    transparent quand aim_smooth = 0 : response 1 -> état = commande)
        for i, p in enumerate(self.players):
            h = cfg.humanization[i]
            m = h.max_rot_speed
            cmd_yaw = _clamp(applied[i].dyaw, -m, m)
            cmd_pitch = _clamp(applied[i].dpitch, -m, m)
            r = 1.0 - h.aim_smooth
            p.aim_dyaw += (cmd_yaw - p.aim_dyaw) * r
            p.aim_dpitch += (cmd_pitch - p.aim_dpitch) * r
            p.yaw += p.aim_dyaw
            p.pitch = _clamp(p.pitch + p.aim_dpitch, -90.0, 90.0)

        # 3. sprint (style toggle-sprint : touche + avancer, coupé par mur)
        for i, p in enumerate(self.players):
            a = applied[i]
            if a.sprint and a.forward > 0 and not p.collided_horizontally:
                p.sprinting = True
            elif not a.sprint or a.forward <= 0 or p.collided_horizontally:
                p.sprinting = False

        # 4. attaques séquentielles (agent 0 puis 1, état muté), pré-mouvement
        for i, p in enumerate(self.players):
            if applied[i].attack:
                tgt = applied[1 - i]
                res = try_attack(p, self.players[1 - i],
                                 cfg.humanization[i].click_cooldown_ticks,
                                 kb_h=cfg.kb_h_mult, kb_v=cfg.kb_v_mult,
                                 kb_idle=cfg.kb_idle_mult,
                                 target_idle=(tgt.forward == 0 and tgt.strafe == 0))
                self.last_dealt[i] = 1 if res.landed else 0
                self.last_sprint_hits[i] = bool(res.sprint_hit)

        # 5. mouvement
        for i, p in enumerate(self.players):
            a = applied[i]
            move_forward = a.forward
            move_strafe = a.strafe
            if cfg.post_sprint_hit_stop and self.last_sprint_hits[i]:
                move_forward = 0
                move_strafe = 0
            living_update_movement(p, float(move_strafe), float(move_forward), a.jump,
                                   cfg.arena_size_x, cfg.arena_size_z,
                                   cfg.speed_amplifier)

        # 5b. poussée entre joueurs (Entity.applyEntityCollision, 2x comme vanilla)
        apply_entity_collision(self.players[0], self.players[1])
        apply_entity_collision(self.players[1], self.players[0])

        # 6. règles boxing
        self.tick_count += 1
        w0 = self.players[0].hits >= cfg.target_hits
        w1 = self.players[1].hits >= cfg.target_hits
        if w0 and w1:
            self.winner = -1     # double atteinte le même tick : égalité
        elif w0:
            self.winner = 0
        elif w1:
            self.winner = 1
        elif self.tick_count >= cfg.max_ticks:
            # timeout = ÉGALITÉ (anti-stall) : mener au score puis fuir ne
            # gagne jamais — la seule victoire est d'atteindre target_hits
            self.winner = -1

    @property
    def done(self) -> bool:
        return self.winner is not None


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v
