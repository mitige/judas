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


def test_winner_by_timeout_leader():
    m = make_match(target_hits=100, max_ticks=80)
    while not m.done:
        atk = aim_action(m.players[0], m.players[1],
                         forward=1, sprint=True, attack=True)
        m.step((atk, Action()))
    assert m.winner == 0  # a frappé au moins une fois avant le timeout
    assert m.players[0].hits < 100


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
