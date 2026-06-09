"""Supervision du process d'entraînement (train.run) par le daemon."""

import json
import subprocess
import sys
import time
from pathlib import Path


class TrainingManager:
    def __init__(self, repo_root: Path | None = None):
        self.root = repo_root or Path(__file__).resolve().parent.parent
        self.proc: subprocess.Popen | None = None
        self.run_name: str | None = None
        self.started_at: float | None = None

    # ------------------------------------------------------------------ ctrl
    def start(self, cfg: dict, resume: str | None = None) -> dict:
        if self.is_running():
            raise RuntimeError("un entraînement tourne déjà")
        name = cfg.get("name", "boxing")
        run_dir = self.root / "runs" / name
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = run_dir / "config.json"
        cfg_path.write_text(json.dumps(cfg, indent=2))

        cmd = [sys.executable, "-m", "train.run", "--config", str(cfg_path)]
        if resume:
            cmd += ["--resume", resume]
        log = open(run_dir / "train.log", "a")
        self.proc = subprocess.Popen(cmd, cwd=str(self.root),
                                     stdout=log, stderr=subprocess.STDOUT)
        self.run_name = name
        self.started_at = time.time()
        return self.status()

    def stop(self) -> dict:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
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
