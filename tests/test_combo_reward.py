"""Reward combo : règle pure, intégration backend ref, équivalence kernel CPU.

Spec : docs/specs/2026-06-11-combo-reward-design.md
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from helpers import HAS_CPU_CHECK_COMPILER, build_cpu_check
from sim import SimConfig
from sim.ref_backend import JudasSimRef, combo_step

ROOT = Path(__file__).resolve().parent.parent

W, CAP = 25, 5


# ------------------------------------------------------- règle pure combo_step
def test_premier_hit_sans_bonus():
    combo, last, m0, m1 = combo_step([0, 0], [0, 0], 1, True, False, W, CAP)
    assert combo == [1, 0]
    assert last == [1, 0]
    assert (m0, m1) == (0, 0)


def test_chaine_croissante_puis_cap():
    combo, last = [0, 0], [0, 0]
    mults, t = [], 1
    for _ in range(8):
        combo, last, m0, _ = combo_step(combo, last, t, True, False, W, CAP)
        mults.append(m0)
        t += 10  # toujours dans la fenêtre
    assert mults == [0, 1, 2, 3, 4, 5, 5, 5]   # plafonné à cap


def test_hit_recu_brise_la_chaine():
    combo, last, _, _ = combo_step([0, 0], [0, 0], 1, True, False, W, CAP)
    assert combo == [1, 0]
    # l'agent 1 touche l'agent 0 au tick 5 -> chaîne de 0 brisée
    combo, last, m0, m1 = combo_step(combo, last, 5, False, True, W, CAP)
    assert combo[0] == 0
    assert m1 == 0          # premier hit de l'agent 1 : pas de bonus
    # le prochain hit de l'agent 0 repart de 1, sans bonus
    combo, last, m0, _ = combo_step(combo, last, 9, True, False, W, CAP)
    assert combo[0] == 1
    assert m0 == 0


def test_trade_ne_paie_pas_le_combo_et_brise_les_deux():
    # agent 0 a une chaîne de 2 ; trade simultané au tick 20
    combo, last, m0, m1 = combo_step([2, 0], [10, 0], 20, True, True, W, CAP)
    assert (m0, m1) == (0, 0)
    assert combo == [0, 0]
    """
    assert (m0, m1) == (2, 0)   # 3e hit de l'agent 0 payé, 1er de l'agent 1
    assert combo == [0, 0]      # les deux chaînes brisées par le trade


    """


def test_fenetre_expiree_repart_a_un():
    combo, last, m0, _ = combo_step([3, 0], [10, 0], 10 + W + 1, True, False,
                                    W, CAP)
    assert combo[0] == 1
    assert m0 == 0


def test_fenetre_limite_incluse():
    combo, last, m0, _ = combo_step([3, 0], [10, 0], 10 + W, True, False,
                                    W, CAP)
    assert combo[0] == 4
    assert m0 == 3


def test_chaine_agent_1_symetrique():
    # miroir : la chaîne de l'agent 1 utilise bien SES compteurs
    combo, last, m0, m1 = combo_step([0, 3], [0, 10], 20, False, True, W, CAP)
    assert combo == [0, 4]
    assert last == [0, 20]
    assert (m0, m1) == (0, 3)


def test_tick_sans_hit_ne_change_rien():
    combo, last, m0, m1 = combo_step([3, 1], [10, 12], 50, False, False, W, CAP)
    assert combo == [3, 1]          # pas de decay sans hit
    assert last == [10, 12]
    assert (m0, m1) == (0, 0)


# ------------------------------------------------- intégration JudasSimRef
def test_ref_backend_combo_rewards_et_zero_somme():
    """Agent 0 avance + attaque, agent 1 passif : les hits s'enchaînent et
    les rewards montent de +0.25 par maillon ; la somme des rewards des
    deux agents reste nulle à chaque tick (zéro-somme, reward_dist=0)."""
    cfg = SimConfig(randomize=False, spawn_gap=1.0, target_hits=5,
                    max_ticks=400, reward_combo=0.25, combo_window=60,
                    combo_cap=5)
    env = JudasSimRef(1, cfg)
    env.reset()
    acts = np.zeros((1, 2, 7), dtype=np.float32)
    acts[0, 0, 2] = 1.0   # forward (poursuite du knockback)
    acts[0, 0, 6] = 1.0   # attack à chaque tick

    hit_rewards, sums, winner = [], [], None
    for _ in range(400):
        _, rew, done, info = env.step(acts)
        sums.append(float(rew[0, 0] + rew[0, 1]))
        if rew[0, 0] > 0.5:
            hit_rewards.append(float(rew[0, 0]))
        if done[0]:
            winner = int(info["winner"][0])
            break

    assert winner == 0
    assert len(hit_rewards) == 5
    hit_rewards[-1] -= cfg.reward_win          # le 5e hit porte aussi le +10 de win
    assert hit_rewards == pytest.approx([1.0, 1.25, 1.5, 1.75, 2.0])
    assert sums == pytest.approx([0.0] * len(sums), abs=1e-6)


def test_ref_backend_combo_drop_penalty_resets_expired_chain():
    cfg = SimConfig(randomize=False, spawn_gap=5.0, target_hits=20,
                    max_ticks=400, reward_hit=0.0, reward_hurt=0.0,
                    reward_win=0.0, reward_combo=0.25, combo_window=6,
                    combo_cap=5, reward_combo_drop=0.5,
                    combo_drop_min=3)
    env = JudasSimRef(1, cfg)
    env.reset()
    env._combo[0] = [4, 0]
    env._last_hit[0] = [1, 0]
    env._matches[0].tick_count = 1 + cfg.combo_window
    acts = np.zeros((1, 2, 7), dtype=np.float32)

    _, rew, done, _ = env.step(acts)

    assert not done[0]
    assert rew[0, 0] == pytest.approx(-1.5)
    assert rew[0, 1] == pytest.approx(0.0)
    assert env._combo[0, 0] == 0


def test_ref_backend_combo_pressure_rewards_close_active_chain():
    cfg = SimConfig(randomize=False, spawn_gap=1.0, target_hits=20,
                    max_ticks=400, reward_hit=0.0, reward_hurt=0.0,
                    reward_win=0.0, reward_combo=0.0, combo_window=20,
                    combo_cap=5, reward_combo_pressure=0.1)
    env = JudasSimRef(1, cfg)
    env.reset()
    env._combo[0] = [3, 0]
    env._last_hit[0] = [env._matches[0].tick_count, 0]
    acts = np.zeros((1, 2, 7), dtype=np.float32)

    _, rew, done, _ = env.step(acts)

    assert not done[0]
    assert rew[0, 0] > 0.0
    assert rew[0, 1] == pytest.approx(0.0)


def test_ref_backend_combo_off_par_defaut():
    """reward_combo=0 (défaut) : rewards de hit inchangés (=1.0)."""
    cfg = SimConfig(randomize=False, spawn_gap=1.0, target_hits=3,
                    max_ticks=400)
    env = JudasSimRef(1, cfg)
    env.reset()
    acts = np.zeros((1, 2, 7), dtype=np.float32)
    acts[0, 0, 2] = 1.0
    acts[0, 0, 6] = 1.0
    hit_rewards = []
    for _ in range(400):
        _, rew, done, _ = env.step(acts)
        if rew[0, 0] > 0.5:
            hit_rewards.append(float(rew[0, 0]))
        if done[0]:
            break
    hit_rewards[-1] -= cfg.reward_win
    assert hit_rewards == pytest.approx([1.0, 1.0, 1.0])


def test_ref_backend_sprint_hit_reward_et_zero_somme():
    """reward_sprint_hit paie uniquement les hits portes en sprint+forward."""
    cfg = SimConfig(randomize=False, spawn_gap=1.0, target_hits=3,
                    max_ticks=400, reward_combo=0.0,
                    reward_sprint_hit=0.35)
    env = JudasSimRef(1, cfg)
    env.reset()
    acts = np.zeros((1, 2, 7), dtype=np.float32)
    acts[0, 0, 2] = 1.0   # forward
    acts[0, 0, 5] = 1.0   # sprint key
    acts[0, 0, 6] = 1.0   # attack

    _, rew, _, _ = env.step(acts)
    assert rew[0, 0] == pytest.approx(1.35)
    assert rew[0, 1] == pytest.approx(-1.35)
    assert float(rew[0, 0] + rew[0, 1]) == pytest.approx(0.0)

    cfg_no_sprint = SimConfig(randomize=False, spawn_gap=1.0, target_hits=3,
                              max_ticks=400, reward_combo=0.0,
                              reward_sprint_hit=0.35)
    env_no_sprint = JudasSimRef(1, cfg_no_sprint)
    env_no_sprint.reset()
    acts_no_sprint = np.zeros((1, 2, 7), dtype=np.float32)
    acts_no_sprint[0, 0, 2] = 1.0
    acts_no_sprint[0, 0, 6] = 1.0
    _, rew_no_sprint, _, _ = env_no_sprint.step(acts_no_sprint)
    assert rew_no_sprint[0, 0] == pytest.approx(1.0)
    assert rew_no_sprint[0, 1] == pytest.approx(-1.0)


def test_ref_backend_post_sprint_hit_stop_releases_forward_for_movement():
    base = dict(randomize=False, spawn_gap=1.0, target_hits=3,
                max_ticks=80, cps_min=20.0, cps_max=20.0,
                reward_combo=0.0)
    cfg_move = SimConfig(**base, post_sprint_hit_stop=False)
    cfg_stop = SimConfig(**base, post_sprint_hit_stop=True)
    env_move = JudasSimRef(1, cfg_move)
    env_stop = JudasSimRef(1, cfg_stop)
    env_move.reset()
    env_stop.reset()
    acts = np.zeros((1, 2, 7), dtype=np.float32)
    acts[0, 0, 2] = 1.0
    acts[0, 0, 5] = 1.0
    acts[0, 0, 6] = 1.0

    _, _, _, info_move = env_move.step(acts)
    _, _, _, info_stop = env_stop.step(acts)

    assert int(info_move["dealt"][0, 0]) == 1
    assert int(info_stop["dealt"][0, 0]) == 1
    z_move = env_move._matches[0].players[0].z
    z_stop = env_stop._matches[0].players[0].z
    assert z_stop < z_move


def test_ref_backend_trade_penalty_appliquee_aux_deux_agents():
    """Un trade hit+hit devient mauvais pour les deux agents."""
    cfg = SimConfig(randomize=False, spawn_gap=1.0, target_hits=3,
                    max_ticks=400, reward_trade_penalty=0.4)
    env = JudasSimRef(1, cfg)
    env.reset()
    acts = np.zeros((1, 2, 7), dtype=np.float32)
    acts[0, 0, 6] = 1.0
    acts[0, 1, 6] = 1.0

    _, rew, _, _ = env.step(acts)
    assert rew[0, 0] == pytest.approx(-0.4)
    assert rew[0, 1] == pytest.approx(-0.4)


# ----------------------------------- équivalence kernel (CPU, double) <-> ref
def _build_cpu_check(tmp_path):
    """Compile le harnais CPU (boxing_core.h en double) -> chemin du binaire."""
    return build_cpu_check(tmp_path, "JUDAS_DOUBLE")


def _run_cpu_check(binary, tmp_path, cfg, acts):
    """Exécute le harnais sur les actions [T,N,2,7] -> octets bruts de sortie."""
    n_ticks, n_envs = acts.shape[0], acts.shape[1]
    actions_f = tmp_path / "actions.bin"
    acts.astype(np.float32).tofile(actions_f)
    params_f = tmp_path / "params.txt"
    params_f.write_text("\n".join(repr(float(v)) for v in cfg.as_floats()))
    out_f = tmp_path / "out.bin"
    subprocess.run([str(binary), str(n_envs), str(n_ticks),
                    str(actions_f), str(out_f), str(params_f)], check=True)
    return np.fromfile(out_f, dtype=np.uint8)


@pytest.mark.skipif(not HAS_CPU_CHECK_COMPILER, reason="g++ ou cl (MSVC) requis")
def test_kernel_combo_matches_ref(tmp_path):
    """Le bloc combo du kernel (boxing_core.h) reproduit exactement le
    backend de référence avec reward_combo actif, sur actions aléatoires."""
    from sim.verify import random_actions

    binary = _build_cpu_check(tmp_path)

    n_envs, n_ticks = 8, 1200
    # spawn_gap=1.0 + combo_window=60 : indispensables pour que des chaînes
    # de hits se forment avec des actions aléatoires (sinon test inopérant)
    cfg = SimConfig(randomize=False, spawn_gap=1.0, target_hits=15,
                    max_ticks=300, reward_combo=0.25, combo_window=60,
                    combo_cap=5)

    rng = np.random.default_rng(123)
    acts = np.stack([random_actions(rng, n_envs) for _ in range(n_ticks)])
    # attaque permanente : densifie les hits -> chaines jusqu'au cap
    acts[..., 6] = 1.0
    raw = _run_cpu_check(binary, tmp_path, cfg, acts)
    obs_sz = n_envs * 2 * 48 * 4
    rew_sz = n_envs * 2 * 4
    off = obs_sz  # saute les obs de reset

    ref = JudasSimRef(n_envs, cfg)
    ref.reset()
    combo_ticks = 0
    for t in range(n_ticks):
        obs_c = raw[off:off + obs_sz].view(np.float32)
        off += obs_sz
        rew_c = raw[off:off + rew_sz].view(np.float32).reshape(n_envs, 2)
        off += rew_sz
        done_c = raw[off:off + n_envs].astype(bool)
        off += n_envs
        win_c = raw[off:off + n_envs * 4].view(np.int32)
        off += n_envs * 4

        obs_r, rew_r, done_r, info = ref.step(acts[t])
        np.testing.assert_allclose(obs_c.reshape(n_envs, 2, 48), obs_r,
                                   atol=1e-6, err_msg=f"obs, tick {t}")
        np.testing.assert_allclose(rew_c, rew_r, atol=1e-6,
                                   err_msg=f"reward, tick {t}")
        assert np.array_equal(done_c, done_r), f"done, tick {t}"
        assert np.array_equal(win_c, info["winner"]), f"winner, tick {t}"
        # hors envs done : le reward_win (±10) masquerait l'absence de combo
        combo_ticks += int((np.abs(rew_r[~done_r]) > 1.1).any())

    assert off == raw.nbytes
    assert combo_ticks > 0, "aucun hit en chaîne — test inopérant"


@pytest.mark.skipif(not HAS_CPU_CHECK_COMPILER, reason="g++ ou cl (MSVC) requis")
def test_kernel_combo_cap_matches_ref(tmp_path):
    """Poursuite scriptée (agent 0 avance+attaque, agent 1 passif) à travers
    le harnais CPU : la chaîne dépasse combo_cap -> la branche de clamp du
    kernel est réellement exécutée. Les hits sont aussi portés en sprint pour
    couvrir reward_sprint_hit côté kernel."""
    binary = _build_cpu_check(tmp_path)

    n_envs, n_ticks = 1, 400
    # 12 hits enchaînés : la chaîne dépasse largement le cap de 5
    cfg = SimConfig(randomize=False, spawn_gap=1.0, target_hits=12,
                    max_ticks=400, reward_sprint_hit=0.35,
                    reward_combo=0.25, combo_window=60, combo_cap=5)

    acts = np.zeros((n_ticks, n_envs, 2, 7), dtype=np.float32)
    acts[:, :, 0, 2] = 1.0   # forward (poursuite du knockback)
    acts[:, :, 0, 5] = 1.0   # sprint key -> sprint-hit bonus sur les hits
    acts[:, :, 0, 6] = 1.0   # attack à chaque tick
    raw = _run_cpu_check(binary, tmp_path, cfg, acts)
    obs_sz = n_envs * 2 * 48 * 4
    rew_sz = n_envs * 2 * 4
    off = obs_sz  # saute les obs de reset

    ref = JudasSimRef(n_envs, cfg)
    ref.reset()
    rewards_a0 = []   # rewards de l'agent 0 hors ticks done
    for t in range(n_ticks):
        obs_c = raw[off:off + obs_sz].view(np.float32)
        off += obs_sz
        rew_c = raw[off:off + rew_sz].view(np.float32).reshape(n_envs, 2)
        off += rew_sz
        done_c = raw[off:off + n_envs].astype(bool)
        off += n_envs
        win_c = raw[off:off + n_envs * 4].view(np.int32)
        off += n_envs * 4

        obs_r, rew_r, done_r, info = ref.step(acts[t])
        np.testing.assert_allclose(obs_c.reshape(n_envs, 2, 48), obs_r,
                                   atol=1e-6, err_msg=f"obs, tick {t}")
        np.testing.assert_allclose(rew_c, rew_r, atol=1e-6,
                                   err_msg=f"reward, tick {t}")
        assert np.array_equal(done_c, done_r), f"done, tick {t}"
        assert np.array_equal(win_c, info["winner"]), f"winner, tick {t}"
        if not done_r[0]:
            rewards_a0.append(float(rew_r[0, 0]))

    assert off == raw.nbytes
    # preuve que le clamp est atteint : le hit cappe le bonus combo.
    # Le bonus sprint-hit peut apparaître sur d'autres ticks, mais il n'est pas
    # garanti sur le tick exact du cap avec le stop post-sprint-hit.
    assert max(rewards_a0) == pytest.approx(
        cfg.reward_hit + cfg.reward_combo * cfg.combo_cap)
    assert any(
        r >= cfg.reward_hit + cfg.reward_sprint_hit
        for r in rewards_a0
    )
