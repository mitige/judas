"""Tests de la boucle de match boxing complète."""

from sim_ref import Action, BoxingConfig, HumanizationConfig
from sim_ref.match import BoxingMatch

from helpers import aim_action, make_match


def test_spawn_face_to_face():
    m = make_match()
    p0, p1 = m.players
    assert p0.yaw == 0.0 and p1.yaw == 180.0
    assert p0.x == p1.x
    assert p1.z > p0.z


def test_rotation_clamped_by_humanization():
    cfg = BoxingConfig(humanization=(
        HumanizationConfig(max_rot_speed=10.0),
        HumanizationConfig(max_rot_speed=10.0),
    ))
    m = BoxingMatch(cfg)
    m.step((Action(dyaw=90.0, dpitch=-50.0), Action()))
    assert m.players[0].yaw == 10.0
    assert m.players[0].pitch == -10.0


def test_pitch_clamped_to_90():
    m = make_match()
    for _ in range(2):
        m.step((Action(dpitch=80.0), Action()))
    assert m.players[0].pitch == 90.0


def test_aim_smooth_motor_model():
    """aim_smooth=0.5 : la rotation appliquée est l'EMA de la commande —
    convergence géométrique vers la commande, pas de snap instantané."""
    cfg = BoxingConfig(humanization=(
        HumanizationConfig(max_rot_speed=40.0, aim_smooth=0.5),
        HumanizationConfig(),
    ))
    m = BoxingMatch(cfg)
    yaws = []
    for _ in range(3):
        m.step((Action(dyaw=10.0), Action()))
        yaws.append(m.players[0].yaw)
    # aim: 5.0 puis 7.5 puis 8.75 -> yaw cumulé 5.0, 12.5, 21.25
    assert abs(yaws[0] - 5.0) < 1e-12
    assert abs(yaws[1] - 12.5) < 1e-12
    assert abs(yaws[2] - 21.25) < 1e-12
    # l'inertie persiste après l'arrêt de la commande (pas d'arrêt sec)
    m.step((Action(dyaw=0.0), Action()))
    assert abs(m.players[0].yaw - (21.25 + 8.75 / 2)) < 1e-12


def test_aim_smooth_zero_is_transparent():
    """aim_smooth=0 (défaut) : rotation appliquée = commande, comportement
    historique inchangé."""
    cfg = BoxingConfig(humanization=(
        HumanizationConfig(max_rot_speed=40.0, aim_smooth=0.0),
        HumanizationConfig(),
    ))
    m = BoxingMatch(cfg)
    m.step((Action(dyaw=10.0), Action()))
    assert m.players[0].yaw == 10.0


def test_action_delay_queue():
    """Avec une latence de 2 ticks, l'action n'agit qu'au 3e step."""
    cfg = BoxingConfig(humanization=(
        HumanizationConfig(action_delay=2, max_rot_speed=360.0),
        HumanizationConfig(),
    ))
    m = BoxingMatch(cfg)
    m.step((Action(dyaw=45.0), Action()))
    assert m.players[0].yaw == 0.0
    m.step((Action(), Action()))
    assert m.players[0].yaw == 0.0
    m.step((Action(), Action()))
    assert m.players[0].yaw == 45.0


def test_chaser_beats_idle():
    """Un bot scripté (poursuite + clic) doit atteindre 100 hits contre un idle."""
    m = make_match(target_hits=100)
    while not m.done:
        atk = aim_action(m.players[0], m.players[1],
                         forward=1, sprint=True, attack=True, jump=False)
        m.step((atk, Action()))
    assert m.winner == 0
    assert m.players[0].hits == 100
    assert m.tick_count < m.cfg.max_ticks


def test_hit_rate_capped_by_hurt_time():
    """Même en face à face permanent, max ~2 hits/s (re-hit 10 ticks)."""
    m = make_match(target_hits=10)
    ticks = 0
    while not m.done and ticks < 2000:
        atk0 = aim_action(m.players[0], m.players[1],
                          forward=1, sprint=True, attack=True)
        m.step((atk0, Action()))
        ticks += 1
    assert m.done
    assert ticks >= 10 * 10 - 10  # ~10 ticks minimum entre deux hits


def test_timeout_is_draw_even_for_leader():
    """Anti-stall : mener au score au timeout ne donne PAS la victoire —
    la seule façon de gagner est d'atteindre target_hits."""
    m = make_match(target_hits=100, max_ticks=80)
    while not m.done:
        atk = aim_action(m.players[0], m.players[1],
                         forward=1, sprint=True, attack=True)
        m.step((atk, Action()))
    assert m.players[0].hits > 0          # le leader a bien frappé
    assert m.players[0].hits < 100
    assert m.winner == -1                 # ... mais le timeout reste une égalité


def test_timeout_draw():
    m = make_match(max_ticks=5)
    for _ in range(5):
        m.step((Action(), Action()))
    assert m.done and m.winner == -1


def test_players_push_each_other_apart():
    """applyEntityCollision : deux joueurs qui se chevauchent se repoussent."""
    m = make_match()
    p0, p1 = m.players
    p1.x, p1.z = p0.x + 0.2, p0.z + 0.1
    p1.y = p0.y
    z0, z1 = p0.z, p1.z
    for _ in range(5):
        m.step((Action(), Action()))
    assert m.players[0].z < z0
    assert m.players[1].z > z1


def test_sprint_engages_and_cuts_on_wall():
    m = make_match()
    p0 = m.players[0]
    m.step((Action(forward=1, sprint=True), Action()))
    assert p0.sprinting
    # fonce dans le mur +Z (l'adversaire est écarté du chemin pour le test)
    m.players[1].x = 1.0
    for _ in range(100):
        m.step((Action(forward=1, sprint=True), Action()))
    assert p0.collided_horizontally
    assert not p0.sprinting
