# Spec — Reward combo (faire émerger les combos en self-play)

Date : 2026-06-11
Statut : validé (approche « reward combo seul », physique 1.8.9 intacte)

## Problème

Après ~200 M steps, la politique dominante est le « poke 1-frame » : rester
juste hors reach (3.0) et n'entrer dans la range qu'un seul tick pour frapper.
C'est la politique optimale du MDP actuel :

- trade simultané = EV 0 (`reward_hit=+1`, `reward_hurt=-1`) ;
- après un hit, la victime est invulnérable 10 ticks (`HURT_REHIT`) mais pas
  l'attaquant → rester dans la range après avoir touché est strictement perdant ;
- la victime garde une visée parfaite pendant son knockback (pas de hurt-cam)
  → chasser pour combotter ≈ trades → le combo ne paie pas.

Objectif : rendre l'enchaînement de hits (combo) directement rémunérateur pour
que le self-play converge vers un style combo, sans toucher à la physique.

## Décision

Reward shaping pur, **zéro-somme**, implémenté dans les deux backends de
reward (kernel CUDA + backend de référence Python). `sim_ref` (physique) n'est
pas modifié. `OBS_DIM` inchangé → les checkpoints existants restent
chargeables (seul le critic se recalibre à la nouvelle échelle de reward).

## Sémantique du combo

Nouvel état par agent (persistant dans le match, remis à zéro au reset) :

- `combo`    : longueur de la chaîne courante (int, init 0)
- `last_hit` : tick du dernier hit porté (int, init 0)

Règles, appliquées dans le bloc reward du tick (après l'incrément de tick,
`t` = tick post-incrément, identique CUDA/Python) :

1. **Hit porté** par l'agent `i` au tick `t` :
   - `combo[i] = (t - last_hit[i] <= combo_window) ? combo[i] + 1 : 1`
     (note : si `combo[i]` venait d'être brisé à 0, les deux branches donnent 1)
   - `last_hit[i] = t`
   - `bonus = reward_combo * min(combo[i] - 1, combo_cap)`
   - `rw[i] += bonus` ; `rw[1-i] -= bonus`  (zéro-somme : la victime apprend
     aussi à s'échapper des combos)
2. **Hit reçu** par l'agent `i` : `combo[i] = 0`.
3. **Ordre déterministe** (trade simultané) : d'abord la passe 1 pour i=0 puis
   i=1 (chaque hit est crédité avec le bonus de sa chaîne en cours), ensuite
   la passe 2 (resets). Un trade brise donc les deux chaînes, mais le hit du
   tick courant est payé au tarif de la chaîne qui existait avant le trade.

Le 1er hit d'une chaîne a un bonus de 0 : le poke isolé ne rapporte rien de
plus qu'avant.

## Paramètres

`sim/config.py` (`SimConfig`) — défauts neutres, le shaping est opt-in :

| Champ          | Défaut | boxing.json | Rôle |
|----------------|--------|-------------|------|
| `reward_combo` | 0.0    | 0.25        | bonus par maillon de chaîne (0 = off) |
| `combo_window` | 25     | 25          | ticks max entre deux hits d'une chaîne (re-hit de combo réel : 10-20 ticks ; un cycle de poke complet est plus lent) |
| `combo_cap`    | 5      | 5           | plafond du multiplicateur (`min(combo-1, cap)`) |

Échelle : avec 0.25/5, un hit en chaîne vaut 1.25, 1.50, … 2.25 max —
contenu vs `reward_win=10`. Les trois champs sont ajoutés à `as_floats()`
(fin de liste, ordre = struct C++) et à `train/configs/boxing.json`.

## Changements par fichier

### `sim/csrc/boxing_core.h`
- `struct SimParams` : ajout `float r_combo, combo_window, combo_cap;`
  (même ordre que `as_floats()`).
- `struct P` : ajout `int combo, last_hit;`.
- `StatePtrs.ints` passe de `[N,2,8]` à `[N,2,10]` : strides `*8 → *10` dans
  `load_agent`/`store_agent`, lecture/écriture des indices 8 (`combo`) et
  9 (`last_hit`), commentaire de layout mis à jour.
- `reset_match` : `combo = last_hit = 0`.
- `tick_one`, bloc 6 (reward) : implémentation des règles ci-dessus à partir
  de `dealt[i]` (déjà disponible) ; les compteurs sont mis à jour
  inconditionnellement et le bonus `r_combo * mult` est naturellement nul
  quand `r_combo == 0` — coût négligeable, pas de garde nécessaire.

### `sim/csrc/boxing_kernel.cu`
- Parsing du vecteur de params : 3 floats supplémentaires (suivre le pattern
  existant, vérifier l'éventuelle assertion de taille côté binding).

### `sim/judas_sim.py`
- Allocation `self._ints = torch.zeros((N, 2, 10), ...)` (au lieu de 8).

### `sim/config.py`
- 3 nouveaux champs + extension de `as_floats()`.

### `sim/ref_backend.py`
- Fonction module-level pure, testable unitairement :
  `combo_step(combo, last_hit, tick, dealt0, dealt1, window, cap) ->
  (combo', last_hit', mult0, mult1)` — implémente exactement les règles ;
  `mult_i = min(combo'[i] - 1, cap)` si l'agent i a touché, sinon 0.
  L'appelant applique `bonus_i = reward_combo * mult_i` (le kernel CUDA fait
  l'équivalent inline avec `pr.r_combo`).
- `JudasSimRef` : tableaux `self._combo`, `self._last_hit` (numpy `[N,2]`
  int32), appel de `combo_step` dans `step()` à partir de `dealt`/`taken`
  (déjà calculés), `rw ± bonus`, remise à zéro au reset du match (`m.done`)
  et dans `reset()`.
- Le tick utilisé est `m.tick_count` après `m.step()` (= tick post-incrément,
  identique au kernel).

### `train/run.py`
- Métrique `combo_hits` (même style que `sprint_hits`, train-side) :
  - `hits_bounded = (reward > 0.9) & (reward < 5.0)` (exclut le tick de win) ;
  - `combo_mask = reward > reward_hit + 0.5 * r_combo` ;
  - `combo_hits = (hits_bounded & combo_mask).sum() / hits_bounded.sum().clamp(min=1)` ;
  - vaut 0 si `reward_combo == 0`. Le bruit du shaping distance
    (−0.002·d ≈ −0.006) est négligeable devant les seuils.
- Ajout au dict `metrics`.

### `app/src/pages/Dashboard.jsx`
- `<KV k="combo hits" v={pct(last(metrics, "combo_hits"))} hl />` à côté de
  « sprint hits ».

### `train/configs/boxing.json`
- `"reward_combo": 0.25`, `"combo_window": 25`, `"combo_cap": 5` dans `sim`.

## Tests

`tests/test_combo_reward.py` :

1. **Unitaires sur `combo_step`** (pur, sans sim) :
   - 1er hit → bonus 0, combo 1 ;
   - hits successifs dans la fenêtre → bonus 0.25·1, 0.25·2, … plafonné au cap ;
   - hit reçu → combo brisé (prochain hit porté = bonus 0) ;
   - trade simultané → les deux hits payés au tarif des chaînes courantes,
     puis les deux chaînes à 0 ;
   - expiration de fenêtre (`t - last_hit > window`) → chaîne repart à 1 ;
   - zéro-somme : `bonus0` ajouté à l'un = soustrait à l'autre.
2. **Intégration `JudasSimRef`** (`randomize=False`, `spawn_gap` court,
   `reward_dist=0`) : agent 0 avance + attaque, agent 1 passif → la séquence
   de rewards aux ticks de hit est croissante (1.0 puis > 1.0) et la somme
   des rewards des deux agents reste nulle hors tick de win.
3. **Équivalence** : `python -m sim.verify` (et `tests/test_equivalence.py`)
   doivent passer — si la config d'équivalence n'active pas `reward_combo`,
   ajouter un cas avec `reward_combo > 0` pour exercer le chemin kernel.

## Validation d'entraînement (hors scope du code, à observer)

- `combo_hits` doit monter depuis ~0 après reprise du checkpoint 200M ;
- `hit_rate` ne doit pas s'effondrer (le critic se recalibre en quelques
  itérations) ;
- en cas de style dégénéré (sur-agression suicidaire), baisser
  `reward_combo` à 0.15 ou le cap à 3 — paramètres faits pour être tunés.

## Hors scope

- Hurt-cam / perturbation de visée (option écartée par l'utilisateur).
- Modification d'`OBS_DIM` (le combo n'est pas observé directement ; la
  policy transformer dispose de 8 ticks d'historique — `policy.history`,
  boxing.json. NB : 8 < combo_window=25, l'état de chaîne est donc
  partiellement inobservable ; tension assumée, voir « Validation »).
- Setter hot-reload type `set_reward_dist` (paramètre statique par run).
- GUI de configuration dans l'app (boxing.json suffit).
