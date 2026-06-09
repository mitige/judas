"""Équivalence sim_ref (CPU, double Python) <-> kernel CUDA (double device).

Nécessite un GPU CUDA — exécuté sur le PC RTX 3060, skip ailleurs.
Version longue : python -m sim.verify
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA requis (PC RTX 3060)")

from sim import SimConfig                      # noqa: E402
from sim.ref_backend import JudasSimRef        # noqa: E402
from sim.verify import ATOL, random_actions    # noqa: E402

N_ENVS = 16
N_TICKS = 400


def test_cuda_matches_reference():
    from sim.judas_sim import JudasSim

    cfg = SimConfig(randomize=False, target_hits=15, max_ticks=300)
    gpu = JudasSim(N_ENVS, cfg, seed=0)
    cpu = JudasSimRef(N_ENVS, cfg, seed=0)

    np.testing.assert_allclose(gpu.reset().cpu().numpy(), cpu.reset(), atol=ATOL)

    rng = np.random.default_rng(7)
    for t in range(N_TICKS):
        a = random_actions(rng, N_ENVS)
        og, rg, dg, _ = gpu.step(torch.from_numpy(a))
        oc, rc, dc, _ = cpu.step(a)
        np.testing.assert_allclose(og.cpu().numpy(), oc, atol=ATOL,
                                   err_msg=f"obs divergent au tick {t}")
        np.testing.assert_allclose(rg.cpu().numpy(), rc, atol=ATOL,
                                   err_msg=f"reward divergent au tick {t}")
        assert np.array_equal(dg.cpu().numpy().astype(bool), dc), \
            f"done divergent au tick {t}"
