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
