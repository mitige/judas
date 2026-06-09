"""Tests de l'API vectorisée (backend de référence CPU)."""

import numpy as np

from sim import OBS_DIM, JudasSimRef, SimConfig


def make(n=4, **kw):
    kw.setdefault("target_hits", 5)
    kw.setdefault("max_ticks", 400)
    return JudasSimRef(n, SimConfig(**kw))


def chase_actions(n_envs):
    """Avancer + sprint + clic, sans toucher à la visée (déjà face à face)."""
    a = np.zeros((n_envs, 2, 7), dtype=np.float32)
    a[:, :, 2] = 1.0   # forward
    a[:, :, 5] = 1.0   # sprint
    a[:, :, 6] = 1.0   # attack
    return a


def test_reset_shapes():
    env = make(n=3)
    obs = env.reset()
    assert obs.shape == (3, 2, OBS_DIM)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()


def test_obs_symmetry_at_spawn():
    """Au spawn, la partie égocentrique de l'obs est identique pour les deux
    agents (situation miroir). Les indices 25-30 (murs + yaw, repère monde)
    sont exclus : ils diffèrent légitimement."""
    obs = make(n=1).reset()
    ego = [i for i in range(obs.shape[-1]) if i not in (25, 26, 27, 28, 29, 30)]
    np.testing.assert_allclose(obs[0, 0, ego], obs[0, 1, ego], atol=1e-6)


def test_step_returns_and_autoreset():
    env = make(n=2)
    env.reset()
    a = chase_actions(2)
    saw_done = False
    for _ in range(400):
        obs, reward, done, info = env.step(a)
        assert obs.shape == (2, 2, OBS_DIM)
        assert reward.shape == (2, 2)
        if done.any():
            saw_done = True
            i = int(np.argmax(done))
            assert info["winner"][i] in (-1, 0, 1)
            # auto-reset : l'obs est celle d'un match neuf (ticks restants = 1)
            assert abs(obs[i, 0, 34] - 1.0) < 1e-6
            break
    assert saw_done, "aucun match terminé : les deux agents face à face doivent se hit"


def test_zero_sum_hit_rewards():
    """Sans shaping, les rewards de hit sont à somme nulle."""
    env = make(n=4)
    env.reset()
    a = chase_actions(4)
    for _ in range(100):
        _, reward, done, _ = env.step(a)
        if not done.any():
            assert np.allclose(reward[:, 0], -reward[:, 1])


def test_humanization_visible_in_obs():
    env = make(n=1, cps_min=10.0, cps_max=10.0,
               rot_speed_min=30.0, rot_speed_max=30.0)
    obs = env.reset()
    assert abs(obs[0, 0, 35] - 10.0 / 20.0) < 1e-6
    assert abs(obs[0, 0, 36] - 30.0 / 180.0) < 1e-6


def test_randomize_unsupported_on_cpu():
    import pytest
    with pytest.raises(NotImplementedError):
        JudasSimRef(1, SimConfig(randomize=True))
