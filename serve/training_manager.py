"""Supervision du process d'entraînement (train.run) par le daemon."""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

MAX_AUTORESTARTS = 3


class TrainingManager:
    def __init__(self, repo_root: Path | None = None):
        self.root = repo_root or Path(__file__).resolve().parent.parent
        self.venv_scripts = self._resolve_venv_scripts()
        self.python = self._resolve_training_python()
        self.proc: subprocess.Popen | None = None
        self.run_name: str | None = None
        self.started_at: float | None = None
        self.autorestart = True
        self.restarts = 0
        self._manual_stop = False
        self._last_cfg: dict | None = None
        self.last_exit_code: int | None = None
        self.last_error: str | None = None
        # protège start/stop/watchdog (un stop pendant le backoff de restart
        # doit vraiment stopper ; pas de double start concurrent)
        self._lock = threading.RLock()

    def _resolve_venv_scripts(self) -> Path:
        if os.name == "nt":
            return self.root / ".venv" / "Scripts"
        return self.root / ".venv" / "bin"

    def _resolve_training_python(self) -> Path:
        """Toujours lancer le training avec le venv du repo s'il existe."""
        exe = "python.exe" if os.name == "nt" else "python"
        venv_python = self.venv_scripts / exe
        return venv_python if venv_python.exists() else Path(sys.executable)

    def _training_env(self) -> dict[str, str]:
        env = self._env_from_script() or os.environ.copy()
        path_value = self._env_path(env)
        if self.venv_scripts.exists():
            path_value = str(self.venv_scripts) + os.pathsep + path_value
        self._set_env_path(env, path_value)
        env.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
        return env

    @staticmethod
    def _env_path(env: dict[str, str]) -> str:
        for key, value in env.items():
            if key.upper() == "PATH":
                return value
        return ""

    @staticmethod
    def _set_env_path(env: dict[str, str], value: str) -> None:
        for key in list(env):
            if key.upper() == "PATH":
                del env[key]
        env["PATH"] = value

    def _env_from_script(self) -> dict[str, str] | None:
        if os.name != "nt":
            return None
        env_bat = self.root / "scripts" / "env.bat"
        if not env_bat.exists():
            return None
        cmd = r"call scripts\env.bat >nul && set"
        try:
            proc = subprocess.run(
                ["cmd.exe", "/d", "/c", cmd],
                cwd=str(self.root),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=True,
            )
        except Exception:
            return None
        env: dict[str, str] = {}
        for line in proc.stdout.splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key] = value
        return env or None

    # ------------------------------------------------------------------ ctrl
    def start(self, cfg: dict, resume: str | None = None,
              autorestart: bool = True, _is_restart: bool = False) -> dict:
        with self._lock:
            if self.is_running():
                raise RuntimeError("un entraînement tourne déjà")
            name = cfg.get("name", "boxing")
            run_dir = self.root / "runs" / name
            run_dir.mkdir(parents=True, exist_ok=True)
            cfg_path = run_dir / "config.json"
            cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

            cmd = [str(self.python), "-m", "train.run", "--config", str(cfg_path)]
            if resume:
                cmd += ["--resume", resume]
            self.last_exit_code = None
            self.last_error = None
            log = open(run_dir / "train.log", "a", encoding="utf-8")
            log.write(f"\n[daemon] start cmd: {' '.join(cmd)}\n")
            log.flush()
            try:
                proc = subprocess.Popen(cmd, cwd=str(self.root),
                                        stdout=log, stderr=subprocess.STDOUT,
                                        env=self._training_env())
            except Exception as exc:
                log.close()
                self.last_error = f"start failed: {exc}"
                raise
            log.close()
            self.proc = proc
            self.run_name = name
            self.started_at = time.time()
            self.autorestart = autorestart
            self._manual_stop = False
            self._last_cfg = cfg
            if not _is_restart:
                self.restarts = 0
            threading.Thread(target=self._watchdog, args=(proc,),
                             daemon=True).start()
            return self.status()

    def _watchdog(self, proc: subprocess.Popen) -> None:
        """Relance automatiquement (resume latest) un entraînement crashé."""
        code = proc.wait()
        with self._lock:
            if self._manual_stop or proc is not self.proc:
                return
            self.last_exit_code = code
            if code == 0:
                return
            self.last_error = f"training exited with code {code}"
            if (not self.autorestart or self.restarts >= MAX_AUTORESTARTS
                    or self._last_cfg is None):
                return
        time.sleep(5)
        with self._lock:
            # un stop a pu arriver pendant le backoff : ne pas relancer
            if self._manual_stop or proc is not self.proc:
                return
            self.restarts += 1
            latest = (self.root / "runs" / (self.run_name or "boxing")
                      / "latest.pt")
            self.proc = None
            try:
                self.start(self._last_cfg,
                           resume=str(latest) if latest.exists() else None,
                           autorestart=True, _is_restart=True)
            except RuntimeError:
                pass

    def stop(self) -> dict:
        with self._lock:
            self._manual_stop = True
            proc = self.proc
        if proc is not None:
            # kill dur (Windows) : sans risque pour les checkpoints, les
            # écritures de train.run sont atomiques (tmp + replace)
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
        with self._lock:
            if self.proc is proc:
                self.proc = None
        return self.status()

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    # ---------------------------------------------------------------- metrics
    def metrics(self, name: str | None = None, tail: int = 200) -> list[dict]:
        name = name or self.run_name or "boxing"
        path = self.root / "runs" / name / "metrics.jsonl"
        if not path.exists():
            return []
        lines = path.read_text().strip().splitlines()[-tail:]
        out = []
        for ln in lines:
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
        return out

    def status(self) -> dict:
        return {
            "running": self.is_running(),
            "run": self.run_name,
            "pid": self.proc.pid if self.is_running() else None,
            "uptime_s": round(time.time() - self.started_at)
                        if self.is_running() and self.started_at else 0,
            "autorestart": self.autorestart,
            "restarts": self.restarts,
            "python": str(self.python),
            "last_exit_code": self.last_exit_code,
            "last_error": self.last_error,
        }

    # ----------------------------------------------------------------- models
    def list_runs(self) -> list[dict]:
        runs_dir = self.root / "runs"
        out = []
        if not runs_dir.exists():
            return out
        for d in sorted(runs_dir.iterdir()):
            if not d.is_dir():
                continue
            ckpts = sorted(d.glob("ckpt_*.pt"))
            last = self.metrics(d.name, tail=1)
            out.append({
                "name": d.name,
                "checkpoints": [c.name for c in ckpts],
                "latest": (d / "latest.pt").exists(),
                "last_metrics": last[0] if last else None,
            })
        return out

    def list_exported(self) -> list[dict]:
        models_dir = self.root / "models"
        out = []
        if models_dir.exists():
            for p in sorted(models_dir.glob("*.pts")):
                meta = {}
                if p.with_suffix(".json").exists():
                    meta = json.loads(p.with_suffix(".json").read_text())
                out.append({"path": str(p), "name": p.stem, **meta})
        return out
