# Judas

Framework d'IA Minecraft **1.8.9 PvP boxing** : simulateur CUDA exact de la physique du jeu,
entraînement PPO self-play avec policy transformer (attention), déploiement in-game via mod
Forge, le tout piloté par une app Electron au design céleste épuré.

**Boxing** : tous les joueurs ont Speed II, une épée dans le slot 1, pas de dégâts —
le premier à **100 hits** gagne (1 coup = 1 hit).

## Architecture

```
judas/
├── sim/        Simulateur CUDA C++ (extension PyTorch) - des milliers de matchs en parallèle
├── sim_ref/    Simulateur de référence Python pur (vérité terrain, testé tick par tick)
├── train/      PPO self-play, transformer policy, league ELO, export TorchScript
├── serve/      Daemon FastAPI + WebSocket : orchestre training et inférence temps réel
├── mod/        Mod Forge 1.8.9 (Java 8) : capteurs d'état + injection d'inputs
├── app/        App Electron "Judas" - contrôle total (dashboard, training, live)
└── tests/      Tests unitaires physique + équivalence sim_ref <-> CUDA
```

Flux : `sim` entraîne via `train` → checkpoint TorchScript → `serve` infère en <2 ms →
le mod Forge échange état/actions par WebSocket localhost à chaque tick → `app` pilote tout.

## Prérequis (PC Windows + RTX 3060)

| Composant | Version | Usage |
|---|---|---|
| Python | 3.10+ | sim_ref, train, serve |
| PyTorch | 2.2+ (build cu12x) | entraînement + binding CUDA |
| CUDA Toolkit | 12.x | compilation des kernels |
| MSVC Build Tools | 2019/2022 (C++ workload) | compilation de l'extension |
| JDK | 8 | mod Forge 1.8.9 |
| Node.js | 20+ | app Electron |

## Installation

```bat
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e .[dev]
```

L'extension CUDA est compilée à la volée (JIT) au premier import de `sim` —
lancer depuis un terminal "x64 Native Tools Command Prompt" pour que MSVC soit visible.

## Démarrage rapide

```bat
:: 1. Vérifier la physique de référence
pytest tests

:: 2. Vérifier l'équivalence sim_ref <-> CUDA + benchmark
python -m sim.verify
python -m sim.bench

:: 3. Lancer un entraînement
python -m train.run --config train/configs/boxing.json

:: 4. Lancer le daemon (training + inférence pilotés par l'app)
python -m serve.daemon

:: 5. Lancer l'app Electron
cd app && npm install && npm run dev
```

## Mod Forge

```bat
cd mod
gradlew setupDecompWorkspace
gradlew build
```

Copier `mod/build/libs/judas-bridge-*.jar` dans `.minecraft/mods` (Forge 1.8.9).
In-game : touche `K` = toggle bot, `L` = kill-switch, le mod se connecte à
`ws://127.0.0.1:8765/live`.

## Checklist de mise en route (PC RTX 3060)

1. `pytest tests` — 56 tests : physique 1.8.9, combat, match, backends, PPO, daemon
2. `python -m sim.verify` — équivalence sim_ref ↔ kernel CUDA (tol 1e-6)
3. `python -m sim.bench` — viser > 1 M agent-steps/s
4. Build du mod (`mod/README.md`), lancer Minecraft, toucher `J` sur sol plat
   → `python tools/golden_compare.py <trace.jsonl>` : sim_ref fidèle au vrai client
5. `python -m train.run --config train/configs/boxing.json` — premier entraînement
6. `python -m train.export runs/boxing/latest.pt --out models/judas.pts`
7. `python -m serve.daemon` + app Electron (`cd app && npm run dev`)
8. In-game : page Live → charger le modèle → armer → touche `K`

## Documentation

- Spec design : [`docs/specs/2026-06-09-judas-design.md`](docs/specs/2026-06-09-judas-design.md)
- Mod Forge & protocole : [`mod/README.md`](mod/README.md)
