# Judas

Judas is a Windows-first AI framework for Minecraft 1.8.9 PvP boxing. It
combines an exact CUDA simulator, PPO self-play training, Population Based
Training, an Electron control app, a 3D arena viewer, and a Forge bridge mod
for live inference.

French documentation is available in [README.fr.md](README.fr.md).

> Responsible use: Judas is intended for local research, private testing, and
> servers where automation is explicitly allowed. Do not use it on public
> servers or communities where bots, automation, or client-side assistance are
> prohibited.

## What Judas Does

- Simulates Minecraft 1.8.9 boxing physics at high throughput with a CUDA
  extension built through PyTorch.
- Keeps a pure Python reference simulator in `sim_ref/` and verifies the CUDA
  implementation against it.
- Trains policies with PPO self-play, league snapshots, scripted baselines, and
  Population Based Training.
- Exports trained checkpoints to deterministic TorchScript models for live
  inference.
- Runs a FastAPI and WebSocket daemon that powers the desktop app, arena viewer,
  and Minecraft mod.
- Provides a Forge 1.8.9 bridge mod that reads game state and applies model
  actions.
- Includes Windows scripts for setup, testing, training, model export,
  deployment checks, packet-order checks, and field proof workflows.

In Judas boxing, players have Speed II, a sword in slot 1, and no damage. A hit
adds one point, and the configured hit target decides the match. Timeouts are
treated as draws so running away with a lead does not become optimal.

## Repository Layout

```text
judas/
|-- sim/        CUDA/PyTorch simulator and compiled kernel sources
|-- sim_ref/    Pure Python reference simulator used as the correctness oracle
|-- train/      PPO, model definitions, PBT, league logic, export tools
|-- serve/      FastAPI daemon, live WebSocket protocol, arena orchestration
|-- mod/        Forge 1.8.9 bridge mod
|-- app/        Electron control app
|-- viz/        Electron 3D arena viewer
|-- scripts/    Windows wrappers for common workflows
|-- tools/      Verification, smoke, and log-analysis tools
|-- tests/      Python and Node test coverage
|-- docs/       Design notes, training guide, and operational notes
|-- assets/     Public project assets
```

The core invariant is that `sim_ref`, the CPU test harness compiled from the
same C++ kernel logic, and the real CUDA path must agree. If physics changes,
update the reference first, then the shared kernel, then run the verification
suite.

## Requirements

Recommended development platform:

| Component | Version | Purpose |
|---|---:|---|
| Windows 10/11 | current | Main supported platform |
| Python | 3.10+; 3.11 recommended | Simulator, training, daemon, tests |
| NVIDIA GPU | 8 GB+ VRAM recommended | CUDA simulation and training |
| CUDA Toolkit | 12.x | JIT compilation of CUDA kernels |
| PyTorch | CUDA 12.8/12.9 wheel recommended | Training and extension loading |
| MSVC Build Tools | 2019+ C++ workload | C++/CUDA extension compilation |
| Node.js | 20+ | Electron apps and Node tests |
| JDK 17 + JDK 8 | Zulu paths used by scripts by default | Mod build runtime and toolchain |
| Gradle | 7.5.1 portable or on PATH | Forge mod build |

The helper scripts assume Windows paths and batch/PowerShell. The Python core
is mostly portable, but the live Minecraft and native-input workflows are
Windows-oriented.

## Quick Start

From the repository root:

```bat
setup.bat
run.bat
```

`setup.bat` creates `.venv`, installs PyTorch with CUDA 12.8 wheels, installs
Judas in editable mode with development dependencies, and prints the detected
Torch/CUDA/GPU status.

`run.bat` opens a menu for the daemon, training, Electron app, 3D arena viewer,
tests, verification, combo proof workflows, and stop commands.

Manual setup:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[dev]"
```

If the CUDA extension cannot compile, open an x64 Native Tools Command Prompt
or install the Visual Studio C++ build tools, then run the command again.

## Common Workflows

Run the daemon:

```bat
scripts\daemon.bat
```

The daemon is idempotent. If it is already running, the script reports that
instead of stacking another process. Use `scripts\daemon.bat -Force` to replace
an old Judas daemon.

Start training:

```bat
scripts\train.bat
```

Equivalent direct command:

```bat
python -m train.run --config train/configs/boxing.json
```

Open the control app:

```bat
scripts\app.bat
```

Open the 3D arena viewer:

```bat
scripts\viz.bat
```

Export a checkpoint:

```bat
python -m train.export runs\boxing\latest.pt --out models\judas.pts
```

Run a local combo proof:

```bat
scripts\prove_combo_god.bat
```

Run a field preflight without starting Minecraft workflows:

```bat
scripts\check_field_preflight.bat
```

## Testing and Verification

Run the main test suite:

```bat
scripts\tests.bat
```

This runs `python -m pytest tests` and, when Node is available, the Node tests:

```bat
node --test tools/persistence.test.mjs tools/health.test.mjs
```

Run simulator equivalence and benchmark checks:

```bat
scripts\verify.bat
```

Direct commands:

```bat
python -m sim.verify
python -m sim.bench
```

Build the web surfaces:

```bat
npm --prefix app ci
npm --prefix app run build
npm --prefix viz ci
npm --prefix viz run build
```

Build the Forge mod:

```bat
scripts\build_mod.bat
```

The mod build expects JDK 17 and JDK 8 at the default Zulu paths used by
`scripts/build_mod.ps1`, unless you pass custom paths.

## Training Notes

The default configuration lives in `train/configs/boxing.json`. Additional
profiles live in `train/configs/`.

Important concepts:

- `reward_hit`, `reward_hurt`, and win reward are the game-native reward core.
- Combo, sprint-hit, pressure, and trade penalties shape behavior but should not
  replace actual match quality.
- `eval vs bot` is the strongest practical signal for progress. Self-play ELO
  can be noisy or relative to the current league.
- Safe checkpoints are preferred by live and arena workflows when available.
- Do not compare radically different architectures under the same run name; use
  separate run names to keep lineage clear.

Read the detailed training guide in [docs/GUIDE.md](docs/GUIDE.md).

## Live Minecraft Deployment

1. Build the mod:

   ```bat
   scripts\build_mod.bat
   ```

2. Copy or deploy `mod/build/libs/judas-bridge-0.1.0.jar` into the Forge 1.8.9
   `mods` directory, or use the deployment scripts described in
   [docs/GUIDE.md](docs/GUIDE.md).

3. Start the daemon:

   ```bat
   scripts\daemon.bat
   ```

4. Start the app:

   ```bat
   scripts\app.bat
   ```

5. In the Live page, load a `.pts` export, arm the model, and use the in-game
   toggle configured by the mod.

Default mod keys:

| Key | Action |
|---|---|
| `K` | Toggle bot on/off |
| `L` | Kill switch |
| `J` | Record a golden trace |
| `O` | Toggle native OS mouse mode where supported |

For packet-order, aim, live-action, and field proof details, see
[docs/GUIDE.md](docs/GUIDE.md) and [mod/README.md](mod/README.md).

## Public Repository Hygiene

The repository intentionally excludes:

- virtual environments and Python caches;
- Node modules and Electron build outputs;
- model checkpoints, exported models, and training runs;
- CUDA/Torch/Gradle local build caches;
- local logs, proof outputs, and generated graph-analysis artifacts;
- manual source archives such as root-level `.zip` and `.7z` files.

Large trained models should be published as GitHub Releases or external
artifacts, not committed to Git.

## Troubleshooting

`pip install -e ".[dev]"` fails:

- Check Python is 3.10+.
- Install/upgrade pip.
- Install the CPU or CUDA PyTorch wheel first, then install Judas.

CUDA extension fails to build:

- Install CUDA Toolkit 12.x.
- Install MSVC C++ build tools.
- Run from an x64 Native Tools Command Prompt.
- Remove local Torch extension caches if a previous build is stale.

The daemon does not start:

- Check whether port `8765` is already used.
- Run `scripts\stop_judas_live.bat`, then `scripts\daemon.bat -Force`.

Electron app cannot reach the daemon:

- Start `scripts\daemon.bat`.
- Check `http://127.0.0.1:8765` from the same machine.
- Rebuild the app only after dependency changes.

Minecraft keeps an old mod jar loaded:

- Close Minecraft fully.
- Use `scripts\prepare_aim_os_test.bat -StopMinecraft` or the deployment check
  scripts in `docs/GUIDE.md`.

## License

Judas is released under the MIT License. See [LICENSE](LICENSE).
