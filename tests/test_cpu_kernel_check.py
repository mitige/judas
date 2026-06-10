"""Équivalence logique kernel (boxing_core.h, compilé en C++ CPU) <-> sim_ref.

C'est le test clef de portage : il valide la logique EXACTE du kernel CUDA
sans GPU. Sur le PC 3060, tests/test_equivalence.py valide en plus l'exécution
CUDA réelle (indexation, lancement, mémoire).
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from sim import SimConfig
from sim.ref_backend import JudasSimRef
from sim.verify import random_actions

ROOT = Path(__file__).resolve().parent.parent
N_ENVS = 8
N_TICKS = 600

pytestmark = pytest.mark.skipif(shutil.which("g++") is None,
                                reason="g++ requis")


def _build(tmp, define=None):
    out = tmp / ("judas_cpu_check_" + (define or "float"))
    cmd = ["g++", "-O2", "-I", str(ROOT / "sim" / "csrc")]
    if define:
        cmd.append(f"-D{define}")
    cmd += ["-o", str(out), str(ROOT / "tools" / "cpu_check.cpp")]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


@pytest.fixture(scope="module")
def binary(tmp_path_factory):
    """Build double : équivalence stricte avec sim_ref (1e-6)."""
    return _build(tmp_path_factory.mktemp("cpu_check"), "JUDAS_DOUBLE")


@pytest.fixture(scope="module")
def binary_f32(tmp_path_factory):
    """Build float32 (celui de l'entraînement) : test de stabilité."""
    return _build(tmp_path_factory.mktemp("cpu_check_f32"))


def test_kernel_logic_matches_sim_ref(binary, tmp_path):
    cfg = SimConfig(randomize=False, target_hits=15, max_ticks=300)

    rng = np.random.default_rng(123)
    acts = np.stack([random_actions(rng, N_ENVS) for _ in range(N_TICKS)])
    actions_f = tmp_path / "actions.bin"
    acts.astype(np.float32).tofile(actions_f)
    params_f = tmp_path / "params.txt"
    params_f.write_text("\n".join(repr(float(v)) for v in cfg.as_floats()))
    out_f = tmp_path / "out.bin"

    subprocess.run([str(binary), str(N_ENVS), str(N_TICKS),
                    str(actions_f), str(out_f), str(params_f)], check=True)

    # parse la sortie binaire
    raw = np.fromfile(out_f, dtype=np.uint8)
    obs_sz = N_ENVS * 2 * 48 * 4
    rew_sz = N_ENVS * 2 * 4
    done_sz = N_ENVS
    win_sz = N_ENVS * 4
    off = 0

    def take(nbytes):
        nonlocal off
        chunk = raw[off:off + nbytes]
        off += nbytes
        return chunk

    obs0 = take(obs_sz).view(np.float32).reshape(N_ENVS, 2, 48)

    ref = JudasSimRef(N_ENVS, cfg)
    np.testing.assert_allclose(obs0, ref.reset(), atol=1e-6,
                               err_msg="obs de reset divergentes")

    worst = 0.0
    for t in range(N_TICKS):
        obs_c = take(obs_sz).view(np.float32).reshape(N_ENVS, 2, 48)
        rew_c = take(rew_sz).view(np.float32).reshape(N_ENVS, 2)
        done_c = take(done_sz).astype(bool)
        win_c = take(win_sz).view(np.int32)

        obs_r, rew_r, done_r, info = ref.step(acts[t])
        np.testing.assert_allclose(obs_c, obs_r, atol=1e-6,
                                   err_msg=f"obs divergentes au tick {t}")
        np.testing.assert_allclose(rew_c, rew_r, atol=1e-6,
                                   err_msg=f"reward divergent au tick {t}")
        assert np.array_equal(done_c, done_r), f"done divergent au tick {t}"
        assert np.array_equal(win_c, info["winner"]), f"winner divergent au tick {t}"
        worst = max(worst, float(np.max(np.abs(obs_c - obs_r))))

    assert off == raw.nbytes
    print(f"écart max obs sur {N_TICKS} ticks : {worst:.2e}")


def test_kernel_float32_stable(binary_f32, tmp_path):
    """Le build float32 (entraînement) reste fini, borné et fonctionnel
    sur un long horizon avec autoresets."""
    cfg = SimConfig(randomize=False, target_hits=10, max_ticks=200)
    n_ticks = 3000

    rng = np.random.default_rng(99)
    acts = np.stack([random_actions(rng, N_ENVS) for _ in range(n_ticks)])
    actions_f = tmp_path / "actions.bin"
    acts.astype(np.float32).tofile(actions_f)
    params_f = tmp_path / "params.txt"
    params_f.write_text("\n".join(repr(float(v)) for v in cfg.as_floats()))
    out_f = tmp_path / "out.bin"

    subprocess.run([str(binary_f32), str(N_ENVS), str(n_ticks),
                    str(actions_f), str(out_f), str(params_f)], check=True)

    raw = np.fromfile(out_f, dtype=np.uint8)
    obs_sz = N_ENVS * 2 * 48 * 4
    tick_sz = obs_sz + N_ENVS * 2 * 4 + N_ENVS + N_ENVS * 4
    assert raw.nbytes == obs_sz + n_ticks * tick_sz

    dones = 0
    for t in range(n_ticks):
        base = obs_sz + t * tick_sz
        obs = raw[base:base + obs_sz].view(np.float32)
        assert np.isfinite(obs).all(), f"obs non finie au tick {t}"
        dones += int(raw[base + obs_sz + N_ENVS * 2 * 4:
                         base + obs_sz + N_ENVS * 2 * 4 + N_ENVS].sum())
    assert dones >= N_ENVS, "aucun match terminé sur 3000 ticks"
