# Judas — Design

Date : 2026-06-09 — validé par l'utilisateur.

## Objectif

Framework produisant des IA Minecraft 1.8.9 de niveau "dieu du PvP" en **boxing**
(Speed II pour tous, épée slot 1, pas de dégâts, premier à 100 hits gagne),
avec le simulateur CUDA le plus exact et le plus rapide possible sur une RTX 3060,
et une app Electron de contrôle au design céleste/spatial épuré.

## Décisions

| Sujet | Choix |
|---|---|
| Déploiement in-game | Mod Forge 1.8.9 (lecture d'état mémoire + injection d'inputs) |
| Machines | Tout sur le PC Windows RTX 3060 (sim, train, Minecraft, Electron) |
| Espace d'actions | Hybride : Δyaw/Δpitch continus (Gaussienne tanh) + touches/clic discrets |
| Observations | Vecteur d'état exact égocentrique, historique 16 ticks → transformer |
| Arène | Plate + 4 murs, taille paramétrable |
| Humanisation | CPS max, vitesse de rotation, latence, bruit — configurables, randomisés à l'entraînement |
| Cœur technique | Option A : kernels CUDA C++ fusionnés + extension PyTorch, sim_ref Python comme vérité terrain |

## Composants

### sim_ref/ — référence physique 1.8.9 (Python)

Port exact du code décompilé MCP 1.8.9 :

- Inputs strafe/forward ×0.98 ; `moveFlying` avec normalisation ;
  friction sol `0.6 × 0.91 = 0.546`, drag aérien `0.91`,
  facteur sol `0.16277136 / f4³`, vitesse air `jumpMovementFactor` (0.02, ×1.3 sprint)
- Gravité : `motionY = (motionY − 0.08) × 0.98` après déplacement
- Saut : `motionY = 0.42`, boost sprint-jump directionnel ±0.2 (sin/cos yaw)
- Vitesse : attribut base 0.1, sprint ×1.3, Speed II ×1.4 (amplifier 1 → 1 + 0.2×2)
- Collisions : AABB joueur 0.6×1.8, résolution par axe (Y puis X puis Z),
  sol y=0 + 4 murs ; collision horizontale coupe le sprint et annule motionX/Z
- Attaque : raycast œil (1.62) → vecteur de visée vs AABB adverse étendue de 0.1, reach 3.0
- Knockback : `motionX/Y/Z ÷= 2` puis `motionX/Z −= ratio/dist × 0.4` et
  `motionY += 0.4` (cap 0.4) — **inconditionnel en 1.8.9** (la garde onGround
  n'apparaît qu'en 1.9 ; c'est le juggle aérien des combos) ; bonus sprint :
  `addVelocity(−sin(yaw)×0.5, 0.1, cos(yaw)×0.5)`,
  attaquant : `motionX/Z ×= 0.6` + sprint reset
- `hurtResistantTime = 20` ticks ; re-hit bloqué tant que > 10 (dégâts égaux en boxing)
- Boxing : compteur de hits, victoire à 100, budget CPS, 20 TPS

### sim/ — simulateur CUDA

- SoA en mémoire GPU, `[N_matchs, 2]` agents, 1 thread = 1 agent, un kernel fusionné par tick :
  actions → clamp rotation → sprint → physique → collisions → attaque → knockback → règles →
  reward/done → auto-reset
- Paramètres par match (arène, CPS, vitesse rotation, latence en files circulaires, bruit)
  randomisés au reset (domain randomization)
- API : `JudasSim(n_envs, cfg).step(actions) -> obs, reward, done, info`
- Validation : équivalence sim_ref ↔ CUDA sur 10k ticks aléatoires (tol 1e-6),
  benchmark cible > 1M agent-steps/s sur RTX 3060

### train/ — PPO self-play

- Policy : encodeur MLP par tick → transformer 2 couches / 4 heads / d=128 sur 16 ticks →
  têtes policy (Gaussienne tanh Δyaw/Δpitch + Bernoulli touches/clic) + value
- PPO clipped + GAE, AMP, rollouts entièrement GPU
- League : pool de checkpoints passés, matchmaking priorisé, ELO
- Reward : +1 hit donné, −1 hit reçu, bonus victoire, shaping optionnel
- Export TorchScript ; logs TensorBoard + flux JSON pour le daemon

### mod/ — Forge 1.8.9 (judas-bridge)

- `StateCollector` : chaque ClientTickEvent, sérialise soi + cible (pos/vel/rot/onGround/
  sprint/hurtTime/hits) en binaire compact
- `ActionApplier` : rotations lissées, KeyBindings, attackEntity + swing, respect CPS
- `WsClient` : WebSocket localhost, reconnexion auto
- Mode enregistrement de traces golden (inputs scriptés → états tick par tick) pour
  calibrer sim_ref contre le vrai jeu
- Toggle (K), kill-switch (L), sélection de cible, overlay discret

### serve/ — daemon

- FastAPI + WS : `training/{start,stop,status}`, `models/`, `live/{connect,params,kill}`
- Supervise les process d'entraînement, stream les métriques à l'app
- Boucle d'inférence temps réel TorchScript GPU, état mod → action < 2 ms,
  humanisation modifiable à chaud

### app/ — Electron "Judas"

- Electron + Vite + React ; design céleste épuré : noir bleuté profond, starfield subtil,
  typo fine, accents froids, beaucoup d'air
- Pages : Dashboard (courbes live, GPU), Training (hyperparams, humanisation, arène),
  Models (checkpoints, ELO, export), Live (connexion mod, modèle, sliders, kill-switch)

## Tests

1. Unitaires physique : hauteur de saut 1.2522, distances sprint-jump, friction, KB,
   fenêtres hurtResistantTime, reach
2. Golden : traces du vrai jeu (mod) vs sim_ref, tick par tick
3. Équivalence : sim_ref vs kernels CUDA, tol 1e-6
4. Entraînement sanity : le modèle bat un bot scripté > 95 %
5. Bout en bout : match boxing complet joué par le bot via le mod

## Ordre d'implémentation

0. Squelette repo → 1. sim_ref + tests → 2. CUDA → 3. train → 4. mod (parallèle possible)
→ 5. serve → 6. app → 7. calibration finale.
