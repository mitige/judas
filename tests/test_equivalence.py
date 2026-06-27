"""Équivalence sim_ref (CPU, double Python) <-> kernel CUDA (double device).

Nécessite un GPU CUDA — exécuté sur le PC RTX 3060, skip ailleurs.
Version longue : python -m sim.verify
"""

import numpy as np
import pytest
import os
import shutil

torch = pytest.importorskip("torch")
from torch.utils.cpp_extension import CUDA_HOME  # noqa: E402


def _has_cuda_build_toolchain() -> bool:
    if not torch.cuda.is_available() or CUDA_HOME is None:
        return False
    if os.name == "nt":
        return shutil.which("cl") is not None
    return shutil.which("c++") is not None or shutil.which("g++") is not None

pytestmark = pytest.mark.skipif(
    not _has_cuda_build_toolchain(),
    reason="CUDA + toolchain C++ requis pour compiler l'extension")

from sim import SimConfig                      # noqa: E402
from sim.ref_backend import JudasSimRef        # noqa: E402
from sim.verify import ATOL, random_actions    # noqa: E402

N_ENVS = 16
N_TICKS = 400


def _run_equivalence(cfg, n_ticks=N_TICKS, force_attack=False, seed=7):
    from sim.judas_sim import JudasSim

    gpu = JudasSim(N_ENVS, cfg, seed=0, precision="double")
    cpu = JudasSimRef(N_ENVS, cfg, seed=0)

    np.testing.assert_allclose(gpu.reset().cpu().numpy(), cpu.reset(), atol=ATOL)

    rng = np.random.default_rng(seed)
    for t in range(n_ticks):
        a = random_actions(rng, N_ENVS)
        if force_attack:
            a[..., 6] = 1.0
        og, rg, dg, _ = gpu.step(torch.from_numpy(a))
        oc, rc, dc, _ = cpu.step(a)
        np.testing.assert_allclose(og.cpu().numpy(), oc, atol=ATOL,
                                   err_msg=f"obs divergent au tick {t}")
        np.testing.assert_allclose(rg.cpu().numpy(), rc, atol=ATOL,
                                   err_msg=f"reward divergent au tick {t}")
        assert np.array_equal(dg.cpu().numpy().astype(bool), dc), \
            f"done divergent au tick {t}"


def test_cuda_matches_reference():
    _run_equivalence(SimConfig(randomize=False, target_hits=15, max_ticks=300))


def test_cuda_matches_reference_full_options():
    """Équivalence GPU avec TOUTES les options actives : reward_combo (exigé
    par la spec combo), kb custom, shaping distance, latence d'action et
    modèle moteur de visée (0.5/0.75 : exacts en float32, médiane 0.625)."""
    cfg = SimConfig(randomize=False, spawn_gap=1.0, target_hits=15,
                    max_ticks=300, reward_sprint_hit=0.35,
                    reward_trade_penalty=0.4,
                    reward_combo=0.25, combo_window=60, combo_cap=5,
                    reward_dist=0.002,
                    kb_h_mult=0.9055, kb_v_mult=0.8835, kb_idle_mult=0.6,
                    delay_min=2, delay_max=2,
                    aim_smooth_min=0.5, aim_smooth_max=0.75)
    _run_equivalence(cfg, force_attack=True)


def test_cuda_determinism_and_seed_sensitivity():
    """Même seed -> trajectoires bit-identiques (randomize=True inclus) ;
    seed différent -> trajectoires différentes."""
    from sim.judas_sim import JudasSim

    cfg = SimConfig(randomize=True, spawn_jitter=2.0, target_hits=10,
                    max_ticks=200, cps_min=8.0, cps_max=16.0,
                    rot_speed_min=20.0, rot_speed_max=60.0,
                    delay_min=0, delay_max=3)
    rng = np.random.default_rng(3)
    acts = [random_actions(rng, 8) for _ in range(150)]

    def run(seed):
        sim = JudasSim(8, cfg, seed=seed)          # build float32 (entraînement)
        frames = [sim.reset().cpu().numpy().copy()]
        for a in acts:
            o, r, _, _ = sim.step(torch.from_numpy(a))
            frames.append(np.concatenate(
                [o.cpu().numpy().reshape(8, -1),
                 r.cpu().numpy().reshape(8, -1)], axis=1))
        return np.concatenate([f.reshape(8, -1) for f in frames], axis=1)

    run_a, run_b, run_c = run(0), run(0), run(1)
    assert np.array_equal(run_a, run_b), "même seed -> doit être bit-identique"
    assert not np.array_equal(run_a, run_c), "seeds différents -> doit différer"
