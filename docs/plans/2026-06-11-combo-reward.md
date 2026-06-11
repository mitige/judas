# Reward Combo — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ajouter un reward shaping zéro-somme qui récompense les chaînes de hits (combos) pour casser la politique « poke 1-frame », mesurable via une métrique `combo_hits`.

**Architecture:** Les compteurs `combo`/`last_hit` par agent vivent dans l'état du sim (kernel CUDA `boxing_core.h` + backend de référence Python `sim/ref_backend.py`, qui doivent rester bit-équivalents). La règle est isolée dans une fonction pure `combo_step` côté Python, testée unitairement. La physique 1.8.9 (`sim_ref/`) et `OBS_DIM` ne changent pas.

**Tech Stack:** Python 3.10+ / numpy / pytest ; C++ (header partagé CUDA/CPU compilé via g++ pour les tests sans GPU) ; React (dashboard).

**Spec:** `docs/specs/2026-06-11-combo-reward-design.md`

**Vérification générale :** `pytest tests` depuis la racine. Pas de GPU sur la machine de dev — la logique kernel est validée par le harnais CPU (`tools/cpu_check.cpp`, compilé par g++ dans `tests/test_cpu_kernel_check.py` et le nouveau test). `tests/test_equivalence.py` (CUDA réel) tournera sur le PC RTX 3060.

---

### Task 1: Plomberie config + état (params 21 → 24, ints 8 → 10)

Champs de config et état combo dans toute la chaîne, **sans** la logique de reward (les compteurs existent mais restent à 0 → aucun comportement ne change, tous les tests existants doivent rester verts).

**Files:**
- Modify: `sim/config.py`
- Modify: `sim/csrc/boxing_core.h` (structs, load/store, reset)
- Modify: `sim/csrc/boxing_kernel.cu` (parsing params)
- Modify: `tools/cpu_check.cpp` (parsing params + alloc ints)
- Modify: `sim/judas_sim.py` (alloc ints)

- [ ] **Step 1: `sim/config.py` — 3 champs + `as_floats()`**

Dans `SimConfig`, après la ligne `reward_dist: float = 0.0` :

```python
    reward_combo: float = 0.0           # bonus par maillon de chaîne (0 = off)
    combo_window: int = 25              # ticks max entre 2 hits d'une chaîne
    combo_cap: int = 5                  # plafond du multiplicateur de chaîne
```

Dans `as_floats()`, la dernière ligne de la liste devient :

```python
            self.kb_h_mult, self.kb_v_mult, self.kb_idle_mult,
            self.reward_combo, float(self.combo_window), float(self.combo_cap),
```

- [ ] **Step 2: `sim/csrc/boxing_core.h` — structs et état**

a) `struct SimParams` : après la ligne `float kb_h, kb_v, kb_idle;   // knockback custom (1.0 = vanilla)` ajouter :

```cpp
    float r_combo, combo_window, combo_cap;   // bonus combo (0 = off)
```

b) `struct P` : après la ligne `int og, spr, col;          // on_ground, sprinting, collided_horizontally` ajouter :

```cpp
    int combo, last_hit;       // chaîne de hits portés, tick du dernier hit
```

c) `struct StatePtrs` : le commentaire de `ints` devient :

```cpp
    int *ints;          // [N,2,10] hurt, jt, ccd, hits, og, spr, col, h_delay, combo, last_hit
```

d) `load_agent` : remplacer `const int *q = S.ints + ((long long)n * 2 + i) * 8;` par `* 10` et après `p.og = q[4]; p.spr = q[5]; p.col = q[6]; p.h_delay = q[7];` ajouter :

```cpp
    p.combo = q[8]; p.last_hit = q[9];
```

e) `store_agent` : remplacer `int *q = S.ints + ((long long)n * 2 + i) * 8;` par `* 10` et après `q[4] = p.og; q[5] = p.spr; q[6] = p.col; q[7] = p.h_delay;` ajouter :

```cpp
    q[8] = p.combo; q[9] = p.last_hit;
```

f) `reset_match` : après la ligne `p.og = 1; p.spr = 0; p.col = 0;` ajouter :

```cpp
        p.combo = 0; p.last_hit = 0;
```

- [ ] **Step 3: `sim/csrc/boxing_kernel.cu` — parsing 24 params**

Dans `params_from_vec` : `TORCH_CHECK(v.size() == 24, "SimParams: 24 valeurs attendues");` et après `p.kb_idle = (float)v[20];` :

```cpp
    p.r_combo = (float)v[21];
    p.combo_window = (float)v[22];
    p.combo_cap = (float)v[23];
```

- [ ] **Step 4: `tools/cpu_check.cpp` — parsing 24 params + ints[N,2,10]**

a) Commentaire d'entête : `// params.txt  : 24 floats (SimConfig.as_floats), un par ligne`
b) `double pv[21];` → `double pv[24];` et la boucle `for (int i = 0; i < 21; ++i)` → `< 24`.
c) Après `pr.kb_idle = (float)pv[20];` :

```cpp
    pr.r_combo = (float)pv[21];
    pr.combo_window = (float)pv[22];
    pr.combo_cap = (float)pv[23];
```

d) `std::vector<int> ints((size_t)n_envs * 2 * 8, 0);` → `* 2 * 10`.

- [ ] **Step 5: `sim/judas_sim.py` — alloc ints**

`self._ints = torch.zeros((N, 2, 8), ...)` → `(N, 2, 10)`.

- [ ] **Step 6: Vérifier que rien ne casse**

Run: `pytest tests -x -q`
Expected: tous les tests passent (le harnais CPU recompile boxing_core.h avec le nouveau layout et reste équivalent à sim_ref ; les rewards sont inchangés car la logique combo n'existe pas encore).

- [ ] **Step 7: Commit**

```bash
git add sim/config.py sim/csrc/boxing_core.h sim/csrc/boxing_kernel.cu tools/cpu_check.cpp sim/judas_sim.py
git commit -m "Plomberie reward combo: params 21->24, etat ints 8->10 (combo, last_hit), logique inerte"
```

---

### Task 2: Règle combo pure `combo_step` (TDD)

**Files:**
- Create: `tests/test_combo_reward.py`
- Modify: `sim/ref_backend.py` (fonction module-level)

- [ ] **Step 1: Écrire les tests unitaires (échec attendu)**

Créer `tests/test_combo_reward.py` :

```python
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
```

- [ ] **Step 2: Vérifier l'échec**

Run: `pytest tests/test_combo_reward.py -q`
Expected: FAIL — `ImportError: cannot import name 'combo_step'`

- [ ] **Step 3: Implémenter `combo_step` dans `sim/ref_backend.py`**

Après la fonction `_mid` :

```python
def combo_step(combo, last_hit, tick, dealt0, dealt1, window, cap):
    """Règles combo (docs/specs/2026-06-11-combo-reward-design.md).

    combo/last_hit : séquences de 2 ints (état des deux agents)
    tick           : tick post-incrément du match
    dealt0/dealt1  : l'agent i a-t-il porté un hit ce tick
    -> (combo', last_hit', mult0, mult1) avec mult_i = min(combo'-1, cap) si
       hit, 0 sinon. L'appelant applique bonus_i = reward_combo * mult_i
    (zéro-somme : +bonus_i pour i, -bonus_i pour 1-i). Le kernel CUDA
    (boxing_core.h, bloc 6 de tick_one) implémente exactement ces règles.
    """
    combo = [int(combo[0]), int(combo[1])]
    last_hit = [int(last_hit[0]), int(last_hit[1])]
    dealt = (dealt0, dealt1)
    mult = [0, 0]
    for i in range(2):
        if dealt[i]:
            combo[i] = combo[i] + 1 if tick - last_hit[i] <= window else 1
            last_hit[i] = tick
            mult[i] = min(combo[i] - 1, cap)
    for i in range(2):
        if dealt[1 - i]:
            combo[i] = 0
    return combo, last_hit, mult[0], mult[1]
```

- [ ] **Step 4: Vérifier le succès**

Run: `pytest tests/test_combo_reward.py -q`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add tests/test_combo_reward.py sim/ref_backend.py
git commit -m "Reward combo: regle pure combo_step + tests unitaires"
```

---

### Task 3: Intégration backend de référence (TDD)

**Files:**
- Modify: `tests/test_combo_reward.py` (test d'intégration)
- Modify: `sim/ref_backend.py` (`JudasSimRef`)

- [ ] **Step 1: Écrire le test d'intégration (échec attendu)**

Ajouter à la fin de `tests/test_combo_reward.py` :

```python
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
```

- [ ] **Step 2: Vérifier l'échec**

Run: `pytest tests/test_combo_reward.py -q`
Expected: `test_ref_backend_combo_rewards_et_zero_somme` FAIL (rewards de hit tous à 1.0 — pas encore de bonus). `test_ref_backend_combo_off_par_defaut` peut déjà passer.

- [ ] **Step 3: Intégrer dans `JudasSimRef`**

a) Fin de `__init__` (après `self._last_actions = ...`) :

```python
        self._combo = np.zeros((n_envs, 2), dtype=np.int32)
        self._last_hit = np.zeros((n_envs, 2), dtype=np.int32)
```

b) Dans `reset()`, après `self._last_actions[:] = 0.0` :

```python
        self._combo[:] = 0
        self._last_hit[:] = 0
```

c) Dans `step()`, remplacer le bloc reward existant :

```python
            hits_before = [m.players[0].hits, m.players[1].hits]
            m.step((acts[0], acts[1]))

            for i in range(2):
                dealt = m.players[i].hits - hits_before[i]
                taken = m.players[1 - i].hits - hits_before[1 - i]
                reward[n, i] = c.reward_hit * dealt + c.reward_hurt * taken
                if c.reward_dist != 0.0:
                    p, q = m.players[i], m.players[1 - i]
                    d = ((p.x - q.x) ** 2 + (p.y - q.y) ** 2 + (p.z - q.z) ** 2) ** 0.5
                    reward[n, i] -= c.reward_dist * d
```

par :

```python
            hits_before = [m.players[0].hits, m.players[1].hits]
            m.step((acts[0], acts[1]))
            dealt = [m.players[k].hits - hits_before[k] for k in range(2)]

            # bonus combo (zéro-somme) — mêmes règles que le kernel CUDA
            cb, lh, m0, m1 = combo_step(self._combo[n], self._last_hit[n],
                                        m.tick_count, dealt[0] > 0,
                                        dealt[1] > 0, c.combo_window,
                                        c.combo_cap)
            self._combo[n], self._last_hit[n] = cb, lh
            bonus = (c.reward_combo * m0, c.reward_combo * m1)

            for i in range(2):
                reward[n, i] = (c.reward_hit * dealt[i]
                                + c.reward_hurt * dealt[1 - i]
                                + bonus[i] - bonus[1 - i])
                if c.reward_dist != 0.0:
                    p, q = m.players[i], m.players[1 - i]
                    d = ((p.x - q.x) ** 2 + (p.y - q.y) ** 2 + (p.z - q.z) ** 2) ** 0.5
                    reward[n, i] -= c.reward_dist * d
```

d) Dans le bloc `if m.done:` (reset du match), après `self._last_actions[n] = 0.0` :

```python
                self._combo[n] = 0
                self._last_hit[n] = 0
```

- [ ] **Step 4: Vérifier le succès + non-régression**

Run: `pytest tests/test_combo_reward.py tests/test_ref_backend.py tests/test_cpu_kernel_check.py -q`
Expected: tout passe. Note : l'équivalence kernel↔ref reste verte car la config par défaut a `reward_combo=0.0` → bonus nul des deux côtés (le kernel n'a pas encore la logique, le ref l'a avec bonus 0).

- [ ] **Step 5: Commit**

```bash
git add tests/test_combo_reward.py sim/ref_backend.py
git commit -m "Reward combo: integration backend de reference (compteurs par env, bonus zero-somme)"
```

---

### Task 4: Logique combo dans le kernel + équivalence (TDD)

**Files:**
- Modify: `tests/test_combo_reward.py` (test d'équivalence harnais CPU)
- Modify: `sim/csrc/boxing_core.h` (`tick_one`, bloc 6)

- [ ] **Step 1: Écrire le test d'équivalence (échec attendu)**

Ajouter à la fin de `tests/test_combo_reward.py` (et compléter les imports en tête de fichier) :

```python
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
```

```python
# ----------------------------------- équivalence kernel (CPU, double) <-> ref
@pytest.mark.skipif(shutil.which("g++") is None, reason="g++ requis")
def test_kernel_combo_matches_ref(tmp_path):
    """Le bloc combo du kernel (boxing_core.h) reproduit exactement le
    backend de référence avec reward_combo actif, sur actions aléatoires."""
    from sim.verify import random_actions

    binary = tmp_path / "judas_cpu_check_combo"
    subprocess.run(
        ["g++", "-O2", "-I", str(ROOT / "sim" / "csrc"), "-DJUDAS_DOUBLE",
         "-o", str(binary), str(ROOT / "tools" / "cpu_check.cpp")],
        check=True, capture_output=True)

    n_envs, n_ticks = 8, 600
    cfg = SimConfig(randomize=False, target_hits=15, max_ticks=300,
                    reward_combo=0.25, combo_window=25, combo_cap=5)

    rng = np.random.default_rng(123)
    acts = np.stack([random_actions(rng, n_envs) for _ in range(n_ticks)])
    actions_f = tmp_path / "actions.bin"
    acts.astype(np.float32).tofile(actions_f)
    params_f = tmp_path / "params.txt"
    params_f.write_text("\n".join(repr(float(v)) for v in cfg.as_floats()))
    out_f = tmp_path / "out.bin"
    subprocess.run([str(binary), str(n_envs), str(n_ticks),
                    str(actions_f), str(out_f), str(params_f)], check=True)

    raw = np.fromfile(out_f, dtype=np.uint8)
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
        combo_ticks += int((np.abs(rew_r) > 1.1).any())

    assert off == raw.nbytes
    assert combo_ticks > 0, "aucun hit en chaîne sur 600 ticks — test inopérant"
```

- [ ] **Step 2: Vérifier l'échec**

Run: `pytest tests/test_combo_reward.py::test_kernel_combo_matches_ref -q`
Expected: FAIL « reward, tick N » — le ref applique le bonus, pas encore le kernel. (Si l'échec est `combo_ticks > 0`, les actions aléatoires n'enchaînent pas de hits : élargir `combo_window` à 60 dans le test ET dans rien d'autre — la fenêtre est un paramètre libre du test.)

- [ ] **Step 3: Implémenter le bloc combo dans `tick_one` (`sim/csrc/boxing_core.h`)**

Dans le bloc `// 6. règles boxing + reward`, juste après la boucle qui remplit `rw[i]` (celle qui se termine par le shaping `r_dist`) et avant `int win = -2;`, insérer :

```cpp
    // bonus combo : chaîne de hits sans en recevoir, fenêtre combo_window
    // (zéro-somme ; règles : docs/specs/2026-06-11-combo-reward-design.md,
    //  miroir exact de combo_step dans sim/ref_backend.py)
    for (int i = 0; i < 2; ++i) {
        if (dealt[i]) {
            pl[i].combo = (tick - pl[i].last_hit <= (int)pr.combo_window)
                              ? pl[i].combo + 1 : 1;
            pl[i].last_hit = tick;
            int mc = pl[i].combo - 1;
            if (mc > (int)pr.combo_cap) mc = (int)pr.combo_cap;
            float bonus = pr.r_combo * (float)mc;
            rw[i] += bonus;
            rw[1 - i] -= bonus;
        }
    }
    for (int i = 0; i < 2; ++i)
        if (dealt[1 - i]) pl[i].combo = 0;
```

Note : `tick` est déjà post-incrément à cet endroit (`tick += 1` en tête du bloc 6), identique au `m.tick_count` utilisé côté Python.

- [ ] **Step 4: Vérifier le succès + non-régression complète**

Run: `pytest tests/test_combo_reward.py tests/test_cpu_kernel_check.py -q` puis `pytest tests -q`
Expected: tout passe (équivalence avec et sans combo).

- [ ] **Step 5: Commit**

```bash
git add tests/test_combo_reward.py sim/csrc/boxing_core.h
git commit -m "Reward combo: bloc kernel (boxing_core.h) + test d'equivalence harnais CPU"
```

---

### Task 5: Métrique `combo_hits`, dashboard, config d'entraînement

**Files:**
- Modify: `train/run.py` (~ligne 292, après le calcul de `sprint_hit`)
- Modify: `app/src/pages/Dashboard.jsx:84`
- Modify: `train/configs/boxing.json`
- Modify: `tests/test_train_smoke.py` (assertion métrique)

- [ ] **Step 1: Assertion dans le smoke test (échec attendu)**

Dans `tests/test_train_smoke.py`, fonction `test_trainer_two_iters`, après `assert np.isfinite(m["approx_kl"])` ajouter :

```python
        assert "combo_hits" in m
```

Run: `pytest tests/test_train_smoke.py::test_trainer_two_iters -q`
Expected: FAIL — KeyError/assert (la métrique n'existe pas).

- [ ] **Step 2: Métrique dans `train/run.py`**

Après le bloc `sprint_hit` (qui se termine par `/ hits_mask.float().sum().clamp(min=1.0))`) :

```python
        # % de hits portés en chaîne (bonus combo > 0) — thermomètre du
        # style combo ; 0 si le shaping est désactivé
        rc = float(self.sim_cfg.reward_combo)
        if rc > 0.0:
            hits_bounded = hits_mask & (buf.reward[:, lm] < 5.0)  # exclut le win
            combo_mask = buf.reward[:, lm] > self.sim_cfg.reward_hit + 0.5 * rc
            combo_hit = float((hits_bounded & combo_mask).float().sum()
                              / hits_bounded.float().sum().clamp(min=1.0))
        else:
            combo_hit = 0.0
```

Dans le dict `metrics`, après `"sprint_hits": round(sprint_hit, 4),` :

```python
            "combo_hits": round(combo_hit, 4),
```

Run: `pytest tests/test_train_smoke.py -q`
Expected: PASS

- [ ] **Step 3: Dashboard**

Dans `app/src/pages/Dashboard.jsx`, après la ligne `<KV k="sprint hits" v={pct(last(metrics, "sprint_hits"))} hl />` :

```jsx
          <KV k="combo hits" v={pct(last(metrics, "combo_hits"))} hl />
```

- [ ] **Step 4: Config d'entraînement**

Dans `train/configs/boxing.json`, bloc `"sim"`, après `"reward_dist": 0.002,` :

```json
    "reward_combo": 0.25,
    "combo_window": 25,
    "combo_cap": 5,
```

- [ ] **Step 5: Suite complète + commit**

Run: `pytest tests -q`
Expected: tous verts.

```bash
git add train/run.py app/src/pages/Dashboard.jsx train/configs/boxing.json tests/test_train_smoke.py
git commit -m "Metrique combo_hits (% de hits en chaine) + dashboard + reward_combo actif dans boxing.json"
```

---

### Task 6: Vérification finale

- [ ] **Step 1: Suite complète**

Run: `pytest tests -q`
Expected: 0 failed (les tests CUDA sont skippés sans GPU — normal sur cette machine).

- [ ] **Step 2: Rappel pour le PC RTX 3060** (à faire par l'utilisateur au prochain run GPU)

```bat
python -m sim.verify        :: équivalence CUDA réelle avec les nouveaux params
python -m train.run --config train/configs/boxing.json --resume runs/boxing/latest.pt
```

Surveiller au dashboard : `combo_hits` doit monter depuis ~0 ; `hit_rate` ne doit pas s'effondrer (recalibrage du critic sur quelques itérations). Tuning si style dégénéré : `reward_combo` 0.15 ou `combo_cap` 3.

- [ ] **Step 3: `graphify update .`** (si l'outil est disponible) pour rafraîchir le graphe de code.
