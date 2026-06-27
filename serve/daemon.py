"""Daemon Judas — API de contrôle (app Electron) + WebSocket live (mod Forge).

    python -m serve.daemon [--host 127.0.0.1] [--port 8765]

Endpoints REST (app) :
  GET  /status                    état global (training, live, GPU)
  POST /training/start            {cfg JSON train.run}
  POST /training/stop
  GET  /training/metrics          ?run=&tail=
  GET  /models                    runs + checkpoints + modèles exportés
  POST /models/export             {"ckpt": "...", "out": "models/x.pts"}
  POST /live/load                 {"model": "models/x.pts"}
  POST /live/params               {"max_cps":.., "max_rot_speed":.., "arena":{...}}
  POST /live/kill                 coupe le bot immédiatement
WebSocket :
  /live                           le mod Forge (état -> action, chaque tick)
  /events                         flux JSON pour l'app (metrics, live stats)
"""

import argparse
import asyncio
import hashlib
import json
import logging
import math
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger("judas.daemon")

from .arena import ArenaSession
from .live import LiveSession
from .protocol import ArenaCalib
from .training_manager import TrainingManager

app = FastAPI(title="judas-daemon")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

training = TrainingManager()
live = LiveSession()
arena = ArenaSession()
_event_clients: set = set()
_arena_clients: set = set()
_arena_task = None
COMBO_SAFE_RUN = "combo_god_recovery_kb092_combo12"
COMBO_SAFE_EXPORT = "models/combo_god_recovery_kb092_combo12-safe_latest.pts"
COMBO_LEADERBOARD_SAFE_RUN = "combo_god_leaderboard10_combo12"
COMBO_LEADERBOARD_SAFE_EXPORT = "models/combo_god_leaderboard10_combo12-safe_latest.pts"
COMBO_COUNTER_SAFE_RUN = "combo_god_countertap96_combo12"
COMBO_COUNTER_SAFE_EXPORT = "models/combo_god_countertap96_combo12-safe_latest.pts"
COMBO_LEGACY_SAFE_RUN = "combo_god_directpad_lock_combo12"
COMBO_LEGACY_SAFE_EXPORT = "models/combo_god_directpad_lock_combo12-safe_latest.pts"
COMBO_SAFE_EXPORT_BY_RUN = {
    COMBO_SAFE_RUN: COMBO_SAFE_EXPORT,
    COMBO_LEADERBOARD_SAFE_RUN: COMBO_LEADERBOARD_SAFE_EXPORT,
    COMBO_COUNTER_SAFE_RUN: COMBO_COUNTER_SAFE_EXPORT,
    COMBO_LEGACY_SAFE_RUN: COMBO_LEGACY_SAFE_EXPORT,
}
COMBO_SAFE_RUN_BY_EXPORT = {
    Path(path).name: run for run, path in COMBO_SAFE_EXPORT_BY_RUN.items()
}
COMBO_SAFE_MIN_SCORE_SCHEMA = 8


def _prefer_combo_safe_model_path(model: str, *, live_export: bool = False) -> str:
    """Redirect raw combo checkpoints to the validated safe artifact."""
    if not isinstance(model, str):
        return model
    normalized = model.replace("\\", "/")
    run = None
    for candidate in COMBO_SAFE_EXPORT_BY_RUN:
        if f"runs/{candidate}/" in normalized and normalized.endswith(".pt"):
            run = candidate
            break
    if run is None:
        return model

    if live_export:
        export_name = COMBO_SAFE_EXPORT_BY_RUN[run]
        export_path = training.root / export_name
        if export_path.exists():
            return export_name

    safe_path = training.root / "runs" / run / "safe_latest.pt"
    if not safe_path.exists():
        return model
    return str(safe_path) if Path(model).is_absolute() else f"runs/{run}/safe_latest.pt"


def _repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else training.root / p


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _combo_safe_checkpoint_run(model: str | Path) -> str | None:
    try:
        model_path = _repo_path(model).resolve()
    except (OSError, RuntimeError, TypeError):
        return None
    for run in COMBO_SAFE_EXPORT_BY_RUN:
        safe_path = (training.root / "runs" / run / "safe_latest.pt").resolve()
        if model_path == safe_path:
            return run
    return None


def _assert_combo_safe_checkpoint_contract(model: str | Path) -> None:
    run = _combo_safe_checkpoint_run(model)
    if run is None:
        return
    safe_path = training.root / "runs" / run / "safe_latest.pt"
    meta_path = safe_path.with_name("safe_latest.meta.json")
    if not meta_path.exists():
        raise ValueError(f"missing combo safe metadata: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    schema = int(meta.get("score_schema", 0) or 0)
    if schema < COMBO_SAFE_MIN_SCORE_SCHEMA:
        raise ValueError(
            f"combo safe metadata too old: score_schema={schema}"
            f"<{COMBO_SAFE_MIN_SCORE_SCHEMA}")

    def metric(name: str) -> float:
        if name not in meta:
            raise ValueError(f"combo safe metadata missing {name}")
        return float(meta[name])

    back_frac = metric("back_frac")
    max_back = float(meta.get("safety_back_frac", 0.002) or 0.002)
    if back_frac > max_back:
        raise ValueError(
            f"combo safe back_frac={back_frac:.4g}>{max_back:.4g}")

    strafe_frac = metric("strafe_frac")
    min_strafe = float(meta.get("safety_min_strafe_frac", 0.50) or 0.50)
    if strafe_frac < min_strafe:
        raise ValueError(
            f"combo safe strafe_frac={strafe_frac:.4g}<{min_strafe:.4g}")

    opener_samples = meta.get("opener_samples")
    if opener_samples is not None:
        check_opener = float(opener_samples) > 0.0
    else:
        check_opener = any(
            float(meta.get(name, 0.0) or 0.0) > 0.0
            for name in (
                "opener_strafe_frac",
                "opener_strafe_hold_frac",
                "opener_pressure_frac",
            )
        )
    if check_opener:
        opener_strafe = metric("opener_strafe_frac")
        min_opener = float(
            meta.get("safety_min_opener_strafe_frac", 0.75) or 0.75)
        if opener_strafe < min_opener:
            raise ValueError(
                f"combo safe opener_strafe_frac={opener_strafe:.4g}"
                f"<{min_opener:.4g}")

        opener_hold = metric("opener_strafe_hold_frac")
        if "safety_min_opener_strafe_hold_frac" not in meta:
            raise ValueError(
                "combo safe metadata missing safety_min_opener_strafe_hold_frac")
        min_opener_hold = float(meta["safety_min_opener_strafe_hold_frac"])
        if min_opener_hold >= 0.0 and opener_hold < min_opener_hold:
            raise ValueError(
                f"combo safe opener_strafe_hold_frac={opener_hold:.4g}"
                f"<{min_opener_hold:.4g}")

        opener_pressure = metric("opener_pressure_frac")
        if "safety_min_opener_pressure_frac" not in meta:
            raise ValueError(
                "combo safe metadata missing safety_min_opener_pressure_frac")
        min_opener_pressure = float(meta["safety_min_opener_pressure_frac"])
        if min_opener_pressure >= 0.0 and opener_pressure < min_opener_pressure:
            raise ValueError(
                f"combo safe opener_pressure_frac={opener_pressure:.4g}"
                f"<{min_opener_pressure:.4g}")

    combo_tap = metric("combo_tap_frac")
    if "safety_min_combo_tap_frac" not in meta:
        raise ValueError("combo safe metadata missing safety_min_combo_tap_frac")
    min_combo_tap = float(meta["safety_min_combo_tap_frac"])
    if min_combo_tap >= 0.0 and combo_tap < min_combo_tap:
        raise ValueError(
            f"combo safe combo_tap_frac={combo_tap:.4g}<{min_combo_tap:.4g}")
    combo_z_tap = metric("combo_z_tap_frac")
    if "safety_min_combo_z_tap_frac" not in meta:
        raise ValueError("combo safe metadata missing safety_min_combo_z_tap_frac")
    min_combo_z_tap = float(meta["safety_min_combo_z_tap_frac"])
    if min_combo_z_tap >= 0.0 and combo_z_tap < min_combo_z_tap:
        raise ValueError(
            f"combo safe combo_z_tap_frac={combo_z_tap:.4g}<{min_combo_z_tap:.4g}")
    combo_s_tap = metric("combo_s_tap_frac")
    if "safety_max_combo_s_tap_frac" not in meta:
        raise ValueError("combo safe metadata missing safety_max_combo_s_tap_frac")
    max_combo_s_tap = float(meta["safety_max_combo_s_tap_frac"])
    if max_combo_s_tap >= 0.0 and combo_s_tap > max_combo_s_tap:
        raise ValueError(
            f"combo safe combo_s_tap_frac={combo_s_tap:.4g}>{max_combo_s_tap:.4g}")
    hit_wtap = metric("hit_wtap_frac")
    if "safety_min_hit_wtap_frac" not in meta:
        raise ValueError("combo safe metadata missing safety_min_hit_wtap_frac")
    min_hit_wtap = float(meta["safety_min_hit_wtap_frac"])
    if min_hit_wtap >= 0.0 and hit_wtap < min_hit_wtap:
        raise ValueError(
            f"combo safe hit_wtap_frac={hit_wtap:.4g}<{min_hit_wtap:.4g}")
    counter_hit = metric("under_combo_counter_hit_frac")
    if "safety_min_under_combo_counter_hit_frac" not in meta:
        raise ValueError(
            "combo safe metadata missing safety_min_under_combo_counter_hit_frac")
    min_counter_hit = float(meta["safety_min_under_combo_counter_hit_frac"])
    counter_avoidance_bonus = float(
        meta.get("under_combo_avoidance_score_bonus", 0.0) or 0.0)
    if (min_counter_hit >= 0.0
            and counter_hit < min_counter_hit
            and counter_avoidance_bonus <= 0.0):
        raise ValueError(
            "combo safe under_combo_counter_hit_frac="
            f"{counter_hit:.4g}<{min_counter_hit:.4g}")
    if bool(meta.get("requires_counter_recovery", False)):
        clean = metric("under_combo_hit_select_clean_frac")
        min_clean = float(
            meta.get("safety_min_under_combo_hit_select_clean_frac", -1.0))
        if min_clean >= 0.0 and clean < min_clean:
            raise ValueError(
                "combo safe under_combo_hit_select_clean_frac="
                f"{clean:.4g}<{min_clean:.4g}")
        trade = metric("under_combo_hit_select_trade_frac")
        max_trade = float(
            meta.get("safety_max_under_combo_hit_select_trade_frac", -1.0))
        if max_trade >= 0.0 and trade > max_trade:
            raise ValueError(
                "combo safe under_combo_hit_select_trade_frac="
                f"{trade:.4g}>{max_trade:.4g}")


def _assert_combo_safe_export_fresh(model: str) -> None:
    if not isinstance(model, str):
        return
    run = COMBO_SAFE_RUN_BY_EXPORT.get(Path(model).name)
    if run is None:
        return

    export_path = _repo_path(model)
    meta_path = export_path.with_suffix(".json")
    safe_path = training.root / "runs" / run / "safe_latest.pt"
    if not export_path.exists():
        raise FileNotFoundError(f"missing exported combo model: {model}")
    if not safe_path.exists():
        raise FileNotFoundError(f"missing combo safe checkpoint: {safe_path}")
    if not meta_path.exists():
        raise ValueError(f"missing combo export metadata: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    source = meta.get("source")
    if not source:
        raise ValueError("combo export metadata missing source")
    source_path = _repo_path(str(source))
    if source_path.resolve() != safe_path.resolve():
        raise ValueError(
            f"combo export source mismatch: {source}, expected {safe_path}")

    if "source_size" not in meta:
        raise ValueError("combo export metadata missing source_size")
    expected_size = int(meta["source_size"])
    actual_size = safe_path.stat().st_size
    if expected_size != actual_size:
        raise ValueError(
            f"combo export size mismatch: metadata={expected_size}, actual={actual_size}")

    if "source_sha256" not in meta:
        raise ValueError("combo export metadata missing source_sha256")
    expected_hash = str(meta["source_sha256"]).lower()
    actual_hash = _sha256_file(safe_path)
    if expected_hash != actual_hash:
        raise ValueError(
            f"combo export hash mismatch: metadata={expected_hash}, actual={actual_hash}")
    _assert_combo_safe_checkpoint_contract(safe_path)


def _annotate_exported_model(model: dict) -> dict:
    out = dict(model)
    path = str(out.get("path", ""))
    if Path(path).name not in COMBO_SAFE_RUN_BY_EXPORT:
        return out
    try:
        _assert_combo_safe_export_fresh(path)
        out["export_fresh"] = True
        out["export_status"] = "fresh"
    except (FileNotFoundError, ValueError, OSError) as exc:
        out["export_fresh"] = False
        out["export_status"] = "stale"
        out["export_error"] = str(exc)
    return out


def _annotate_run_combo_safe_contract(run: dict) -> dict:
    out = dict(run)
    name = str(out.get("name", ""))
    if name not in COMBO_SAFE_EXPORT_BY_RUN:
        return out

    out["combo_safe"] = True
    safe_path = training.root / "runs" / name / "safe_latest.pt"
    if not safe_path.exists():
        out["safe_status"] = "missing"
        out["safe_error"] = f"missing combo safe checkpoint: {safe_path}"
        return out

    try:
        _assert_combo_safe_checkpoint_contract(safe_path)
        out["safe_status"] = "fresh"
    except (FileNotFoundError, ValueError, OSError) as exc:
        out["safe_status"] = "stale"
        out["safe_error"] = str(exc)
    return out


def _json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    return value


def _json_text(value) -> str:
    return json.dumps(_json_safe(value), allow_nan=False)


def _normalize_training_cfg(cfg: dict) -> dict:
    """Normalise les champs venant du formulaire avant écriture config.json."""
    sim = cfg.get("sim")
    if isinstance(sim, dict) and "reward_hurt" in sim:
        sim["reward_hurt"] = -abs(float(sim["reward_hurt"]))
    return cfg


def _gpu_status() -> dict:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False}
        free, total = torch.cuda.mem_get_info()
        return {
            "available": True,
            "name": torch.cuda.get_device_name(0),
            "mem_used_gb": round((total - free) / 1e9, 2),
            "mem_total_gb": round(total / 1e9, 2),
        }
    except Exception:
        return {"available": False}


# ----------------------------------------------------------------------- REST
@app.get("/status")
def status():
    return _json_safe({"training": training.status(), "live": live.status(),
                       "gpu": _gpu_status()})


@app.post("/training/start")
def training_start(cfg: dict):
    resume = cfg.pop("resume", None)
    autorestart = bool(cfg.pop("autorestart", True))
    cfg = _normalize_training_cfg(cfg)
    try:
        return _json_safe(training.start(cfg, resume=resume,
                                         autorestart=autorestart))
    except RuntimeError as exc:        # déjà en cours -> conflit, pas un 500
        raise HTTPException(status_code=409, detail=str(exc))


@app.post("/training/stop")
def training_stop():
    return _json_safe(training.stop())


@app.get("/training/metrics")
def training_metrics(run: str | None = None, tail: int = 200):
    return _json_safe(training.metrics(run, tail))


@app.get("/models")
def models():
    exported = [_annotate_exported_model(m) for m in training.list_exported()]
    runs = [_annotate_run_combo_safe_contract(r) for r in training.list_runs()]
    return _json_safe({"runs": runs,
                       "exported": exported})


@app.post("/models/export")
def models_export(body: dict):
    from train.export import export
    if "ckpt" not in body:
        raise HTTPException(status_code=400, detail="champ 'ckpt' manquant")
    try:
        ckpt = _prefer_combo_safe_model_path(body["ckpt"])
        _assert_combo_safe_checkpoint_contract(ckpt)
        out = export(ckpt, body.get("out", "models/judas.pts"))
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"exported": str(out)}


@app.post("/live/load")
def live_load(body: dict):
    if "model" not in body:
        raise HTTPException(status_code=400, detail="champ 'model' manquant")
    try:
        model = _prefer_combo_safe_model_path(body["model"], live_export=True)
        _assert_combo_safe_export_fresh(model)
        return _json_safe(live.load(model))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/live/params")
def live_params(body: dict):
    p = live.params
    if "max_cps" in body:
        p.max_cps = float(body["max_cps"])
    if "max_rot_speed" in body:
        p.max_rot_speed = float(body["max_rot_speed"])
    if "enabled" in body:
        p.enabled = bool(body["enabled"])
    if "arena" in body:
        p.arena = ArenaCalib(**body["arena"])
        live._frame = None
    if "auto_frame" in body:
        live.auto_frame = bool(body["auto_frame"])
        live._frame = None
    if "human" in body:
        hb = body["human"]
        hc = live.params.human
        if isinstance(hb, bool):
            hc.enabled = hb
        elif isinstance(hb, dict):
            for k, v in hb.items():
                if hasattr(hc, k):
                    setattr(hc, k, v)
        live._reset_human_aim()
    if "aim_smoothing" in body:
        ab = body["aim_smoothing"]
        ac = live.params.aim_smoothing
        if isinstance(ab, bool):
            ac.enabled = ab
        elif isinstance(ab, dict):
            for k, v in ab.items():
                if hasattr(ac, k):
                    setattr(ac, k, v)
    if "hit_select_assist" in body:
        hb = body["hit_select_assist"]
        hs = live.params.hit_select_assist
        if isinstance(hb, bool):
            hs.enabled = hb
        elif isinstance(hb, dict):
            for k, v in hb.items():
                if hasattr(hs, k):
                    setattr(hs, k, v)
    if "counter_assist" in body:
        cb = body["counter_assist"]
        ca = live.params.counter_assist
        if isinstance(cb, bool):
            ca.enabled = cb
        elif isinstance(cb, dict):
            for k, v in cb.items():
                if hasattr(ca, k):
                    setattr(ca, k, v)
    if "auto_gapple" in body:
        gb = body["auto_gapple"]
        ga = live.params.auto_gapple
        if isinstance(gb, bool):
            ga.enabled = gb
        elif isinstance(gb, dict):
            for k, v in gb.items():
                if hasattr(ga, k):
                    setattr(ga, k, v)
    if "auto_jump" in body:
        jb = body["auto_jump"]
        aj = live.params.auto_jump
        if isinstance(jb, bool):
            aj.enabled = jb
        elif isinstance(jb, dict):
            for k, v in jb.items():
                if hasattr(aj, k):
                    setattr(aj, k, v)
    if "knockback_dump" in body:
        kb = body["knockback_dump"]
        kd = live.params.knockback_dump
        if isinstance(kb, bool):
            kd.enabled = kb
        elif isinstance(kb, dict):
            for k, v in kb.items():
                if hasattr(kd, k):
                    setattr(kd, k, v)
    if "friends" in body:
        fb = body["friends"]
        fr = live.params.friends
        if isinstance(fb, bool):
            fr.enabled = fb
        elif isinstance(fb, list):
            fr.names = [str(name).strip() for name in fb if str(name).strip()]
        elif isinstance(fb, dict):
            if "enabled" in fb:
                fr.enabled = bool(fb["enabled"])
            if "names" in fb and isinstance(fb["names"], list):
                fr.names = [str(name).strip() for name in fb["names"] if str(name).strip()]
    return _json_safe(live.status())


@app.post("/live/kill")
def live_kill():
    live.params.enabled = False
    live.reset()
    return _json_safe(live.status())


# ------------------------------------------------------------------ arène viz
@app.get("/arena/status")
def arena_status():
    return _json_safe(arena.status())


@app.post("/arena/load")
async def arena_load(body: dict):
    if "model_a" not in body or "model_b" not in body:
        raise HTTPException(status_code=400,
                            detail="champs 'model_a'/'model_b' manquants")
    arena.running = False
    model_a = _prefer_combo_safe_model_path(body["model_a"])
    model_b = _prefer_combo_safe_model_path(body["model_b"])
    return _json_safe(arena.load(
        model_a, model_b,
                cps=float(body.get("cps", 10.0)),
        rot_speed=float(body.get("rot_speed", 190.0)),
        arena_size=float(body.get("arena_size", 40.0)),
        target_hits=int(body.get("target_hits", 50)),
        sample=bool(body.get("sample", True)),
        spawn_gap=float(body.get("spawn_gap", 8.0)),
        kb_h=float(body.get("kb_h", 0.92)),
        kb_v=float(body.get("kb_v", 0.90)),
        kb_idle=float(body.get("kb_idle", 0.6)),
        aim_smooth=float(body.get("aim_smooth", 0.02)),
        post_sprint_hit_stop=bool(body.get("post_sprint_hit_stop", True)),
    ))


@app.post("/arena/control")
async def arena_control(body: dict):
    global _arena_task
    if "speed" in body:
        arena.speed = max(0.25, min(16.0, float(body["speed"])))
    if "sample" in body:
        arena.sample = bool(body["sample"])
    if body.get("reset"):
        arena.reset()
    if "running" in body:
        arena.running = bool(body["running"]) and arena.ready
        if arena.running and (_arena_task is None or _arena_task.done()):
            _arena_task = asyncio.create_task(_arena_loop())
    return _json_safe(arena.status())


async def _arena_loop():
    """Boucle de match : step à 20 TPS x vitesse, broadcast aux clients viz."""
    try:
        while arena.running and arena.ready:
            state = arena.step()
            msg = _json_text(state)
            for ws in list(_arena_clients):
                try:
                    await ws.send_text(msg)
                except Exception:
                    _arena_clients.discard(ws)
            await asyncio.sleep(1.0 / (20.0 * arena.speed))
    except Exception:
        # sans ce filet, la tâche meurt en silence et status reste running=True
        logger.exception("arena loop crashed")
        arena.running = False
        status_msg = _json_text({"t": "status", **arena.status()})
        for ws in list(_arena_clients):
            try:
                await ws.send_text(status_msg)
            except Exception:
                _arena_clients.discard(ws)


@app.websocket("/arena")
async def ws_arena(ws: WebSocket):
    """Flux d'états de match pour le visualiseur 3D."""
    await ws.accept()
    _arena_clients.add(ws)
    try:
        await ws.send_text(_json_text({"t": "status", **arena.status()}))
        while True:
            await ws.receive_text()        # détecte la déconnexion
    except WebSocketDisconnect:
        pass
    finally:
        _arena_clients.discard(ws)


# ----------------------------------------------------------------- WebSockets
_live_ws: WebSocket | None = None


@app.websocket("/live")
async def ws_live(ws: WebSocket):
    """Connexion du mod Forge : état -> action à chaque tick."""
    global _live_ws
    # session unique : une 2e connexion détruirait l'état (historique, tick)
    # de la session active du mod
    if _live_ws is not None:
        await ws.close(code=1008, reason="session live déjà connectée")
        return
    await ws.accept()
    _live_ws = ws
    live.reset()
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if msg.get("t") != "state":
                continue
            try:
                action = live.on_state(msg)
            except (KeyError, TypeError, ValueError):
                # message 'state' malformé : ignorer plutôt que tuer le lien
                logger.warning("message state malformé ignoré", exc_info=True)
                continue
            if action is not None:
                await ws.send_text(_json_text(action))
    except WebSocketDisconnect:
        pass
    finally:
        _live_ws = None


@app.websocket("/events")
async def ws_events(ws: WebSocket):
    """Flux périodique d'état global pour l'app Electron."""
    await ws.accept()
    _event_clients.add(ws)
    try:
        while True:
            payload = {
                "t": "status",
                "training": training.status(),
                "live": live.status(),
                "gpu": _gpu_status(),
                "metrics_tail": training.metrics(tail=1),
            }
            await ws.send_text(_json_text(payload))
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        pass
    finally:
        _event_clients.discard(ws)


def main():
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
