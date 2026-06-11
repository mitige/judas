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


def _mirror_obs(o):
    """Transforme une obs sous la rotation de 180° autour du centre d'arène.

    Tout est égocentrique sauf : distances aux murs (o[25]<->o[26],
    o[27]<->o[28]) et sin/cos du yaw monde (o[29], o[30] négés)."""
    m = o.copy()
    m[..., 25], m[..., 26] = o[..., 26].copy(), o[..., 25].copy()
    m[..., 27], m[..., 28] = o[..., 28].copy(), o[..., 27].copy()
    m[..., 29] = -o[..., 29]
    m[..., 30] = -o[..., 30]
    return m


def test_mirror_trajectory_symmetry():
    """Égocentrisme RÉEL sur trajectoire : échanger les deux agents (mêmes
    séquences d'actions, rôles inversés) produit exactement les mêmes obs
    (modulo la permutation murs/yaw), rewards, done et winner inversé.

    Une asymétrie ici = un agent apprendrait un jeu différent selon son côté.
    Seul l'agent alpha attaque (l'ordre de résolution séquentiel agent 0 puis
    agent 1 rend les trades simultanés légitimement sensibles à l'ordre)."""
    n, ticks = 4, 600
    cfg = SimConfig(randomize=False, spawn_gap=1.5, target_hits=8,
                    max_ticks=200, reward_combo=0.25, combo_window=40,
                    combo_cap=5, reward_dist=0.002)
    env_a = JudasSimRef(n, cfg)
    env_b = JudasSimRef(n, cfg)
    env_a.reset()
    env_b.reset()

    rng = np.random.default_rng(11)
    for t in range(ticks):
        alpha = rng.uniform(-1.0, 1.0, size=(n, 7)).astype(np.float32)
        beta = rng.uniform(-1.0, 1.0, size=(n, 7)).astype(np.float32)
        alpha[:, 6] = 1.0   # alpha attaque toujours
        beta[:, 6] = 0.0    # beta jamais (trades sensibles à l'ordre exclus)

        acts_a = np.stack([alpha, beta], axis=1)   # alpha = agent 0
        acts_b = np.stack([beta, alpha], axis=1)   # alpha = agent 1

        obs_a, rew_a, done_a, info_a = env_a.step(acts_a)
        obs_b, rew_b, done_b, info_b = env_b.step(acts_b)

        np.testing.assert_allclose(obs_b[:, 1], _mirror_obs(obs_a[:, 0]),
                                   atol=2e-6, err_msg=f"obs alpha, tick {t}")
        np.testing.assert_allclose(obs_b[:, 0], _mirror_obs(obs_a[:, 1]),
                                   atol=2e-6, err_msg=f"obs beta, tick {t}")
        np.testing.assert_allclose(rew_b[:, 1], rew_a[:, 0], atol=1e-5,
                                   err_msg=f"reward alpha, tick {t}")
        np.testing.assert_allclose(rew_b[:, 0], rew_a[:, 1], atol=1e-5,
                                   err_msg=f"reward beta, tick {t}")
        assert np.array_equal(done_a, done_b), f"done, tick {t}"
        wa, wb = info_a["winner"], info_b["winner"]
        flipped = np.where(wa >= 0, 1 - wa, wa)
        assert np.array_equal(flipped, wb), f"winner, tick {t}"
