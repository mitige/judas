"""Tests du combat 1.8.9 : reach, knockback, hurtResistantTime, CPS."""


from sim_ref import PlayerState
from sim_ref.combat import can_hit, try_attack
from sim_ref.constants import KNOCKBACK_Y_CAP


def duo(dist=2.0):
    """Attaquant à l'origine regardant +Z (yaw 0), cible à `dist` blocs."""
    a = PlayerState(x=0.0, y=0.0, z=0.0, yaw=0.0, pitch=0.0)
    t = PlayerState(x=0.0, y=0.0, z=dist, yaw=180.0)
    return a, t


# -------------------------------------------------------------------- reach

def test_reach_hits_at_3_blocks_to_hitbox():
    """Reach 3.0 jusqu'à l'AABB étendue (face avant à z - 0.4)."""
    a, t = duo(dist=3.39)   # face avant à 2.99 du joueur
    assert can_hit(a, t)


def test_reach_misses_past_3_blocks():
    a, t = duo(dist=3.45)   # face avant à 3.05
    assert not can_hit(a, t)


def test_miss_when_looking_away():
    a, t = duo(dist=2.0)
    a.yaw = 90.0
    assert not can_hit(a, t)


def test_hit_requires_pitch_in_range():
    a, t = duo(dist=2.0)
    a.pitch = -89.0  # regarde le ciel
    assert not can_hit(a, t)


# ---------------------------------------------------------------- knockback

def test_knockback_base_values():
    """Cible au sol, au repos : KB horizontal 0.4 et vertical 0.4."""
    a, t = duo(dist=2.0)
    res = try_attack(a, t, click_cooldown_ticks=2)
    assert res.landed
    assert abs(t.vz - 0.4) < 1e-9        # poussée le long de +Z
    assert abs(t.vx) < 1e-9
    assert abs(t.vy - KNOCKBACK_Y_CAP) < 1e-9


def test_knockback_sprint_bonus_and_reset():
    """Sprint : +0.5 horizontal, +0.1 vertical, attaquant x0.6 et sprint coupé."""
    a, t = duo(dist=2.0)
    a.sprinting = True
    a.vz = 0.2
    res = try_attack(a, t, click_cooldown_ticks=2)
    assert res.sprint_hit
    assert abs(t.vz - (0.4 + 0.5)) < 1e-9
    assert abs(t.vy - (KNOCKBACK_Y_CAP + 0.1)) < 1e-9
    assert abs(a.vz - 0.2 * 0.6) < 1e-9
    assert not a.sprinting               # sprint reset vanilla


def test_attack_result_marks_only_landed_sprint_hits():
    a, t = duo(dist=2.0)
    assert not try_attack(a, t, click_cooldown_ticks=2).sprint_hit

    a, t = duo(dist=3.45)
    a.sprinting = True
    res = try_attack(a, t, click_cooldown_ticks=2)
    assert res.swung and not res.landed
    assert not res.sprint_hit


def test_knockback_halves_existing_velocity():
    """motionX/Z sont divisés par 2 avant la poussée."""
    a, t = duo(dist=2.0)
    t.vz = -0.6                          # la cible fonce vers l'attaquant
    try_attack(a, t, click_cooldown_ticks=2)
    assert abs(t.vz - (-0.3 + 0.4)) < 1e-9


def test_knockback_custom_multipliers():
    """KB custom (plugins serveur) : h x0.9055, v x0.8835, idle x0.6."""
    a, t = duo(dist=2.0)
    try_attack(a, t, 0, kb_h=0.9055, kb_v=0.8835)
    assert abs(t.vz - 0.4 * 0.9055) < 1e-9
    assert abs(t.vy - 0.4 * 0.8835) < 1e-9

    a2, t2 = duo(dist=2.0)
    try_attack(a2, t2, 0, kb_h=1.0, kb_v=1.0, kb_idle=0.6, target_idle=True)
    assert abs(t2.vz - 0.4 * 0.6) < 1e-9
    assert abs(t2.vy - 0.4 * 0.6) < 1e-9

    a3, t3 = duo(dist=2.0)
    try_attack(a3, t3, 0, kb_idle=0.6, target_idle=False)
    assert abs(t3.vz - 0.4) < 1e-9


def test_knockback_airborne_keeps_vertical_boost_189():
    """1.8.9 : le +0.4 vertical s'applique AUSSI en l'air (le juggle aérien).
    La garde onGround n'existe qu'en 1.9+."""
    a, t = duo(dist=2.0)
    t.on_ground = False
    t.y = 0.5
    t.vy = -0.2                          # en train de retomber
    try_attack(a, t, click_cooldown_ticks=2)
    assert abs(t.vy - (-0.1 + 0.4)) < 1e-9   # vy/2 puis +0.4 (cap 0.4)


def test_knockback_vertical_cap_applies_in_air():
    """Cible déjà en pleine ascension : le cap 0.4 borne le cumul."""
    a, t = duo(dist=2.0)
    t.on_ground = False
    t.y = 0.5
    t.vy = 0.5
    try_attack(a, t, click_cooldown_ticks=2)
    assert abs(t.vy - 0.4) < 1e-9        # 0.25 + 0.4 = 0.65 -> cap 0.4


# ----------------------------------------------------- hurtResistantTime

def test_rehit_blocked_during_10_ticks():
    a, t = duo(dist=2.0)
    assert try_attack(a, t, 0).landed
    assert t.hurt_resistant_time == 20
    for tick in range(9):                # hurt 19 -> 11 : tout est bloqué
        t.hurt_resistant_time -= 1       # simulation du décrément par tick
        a.click_cooldown = 0
        assert not try_attack(a, t, 0).landed, f"re-hit autorisé (hurt={t.hurt_resistant_time})"
    t.hurt_resistant_time -= 1           # hurt == 10 -> le hit passe
    a.click_cooldown = 0
    assert try_attack(a, t, 0).landed


def test_hits_counter_increments():
    a, t = duo(dist=2.0)
    try_attack(a, t, 0)
    assert a.hits == 1 and t.hits == 0


# --------------------------------------------------------------------- CPS

def test_click_cooldown_consumed_even_on_miss():
    a, t = duo(dist=3.45)                # hors de portée
    res = try_attack(a, t, click_cooldown_ticks=2)
    assert res.swung and not res.landed
    assert a.click_cooldown == 2


def test_click_blocked_during_cooldown():
    a, t = duo(dist=2.0)
    try_attack(a, t, click_cooldown_ticks=2)
    res = try_attack(a, t, click_cooldown_ticks=2)
    assert not res.swung and not res.landed
