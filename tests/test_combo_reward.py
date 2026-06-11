"""Reward combo : règle pure, intégration backend ref, équivalence kernel CPU.

Spec : docs/specs/2026-06-11-combo-reward-design.md
"""

import numpy as np
import pytest

from sim import SimConfig
from sim.ref_backend import JudasSimRef, combo_step

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


def test_trade_paye_au_tarif_courant_puis_brise_les_deux():
    # agent 0 a une chaîne de 2 ; trade simultané au tick 20
    combo, last, m0, m1 = combo_step([2, 0], [10, 0], 20, True, True, W, CAP)
    assert (m0, m1) == (2, 0)   # 3e hit de l'agent 0 payé, 1er de l'agent 1
    assert combo == [0, 0]      # les deux chaînes brisées par le trade


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
