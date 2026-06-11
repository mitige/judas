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
import json
import logging
import math

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
    return _json_safe({"runs": training.list_runs(),
                       "exported": training.list_exported()})


@app.post("/models/export")
def models_export(body: dict):
    from train.export import export
    if "ckpt" not in body:
        raise HTTPException(status_code=400, detail="champ 'ckpt' manquant")
    try:
        out = export(body["ckpt"], body.get("out", "models/judas.pts"))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"exported": str(out)}


@app.post("/live/load")
def live_load(body: dict):
    if "model" not in body:
        raise HTTPException(status_code=400, detail="champ 'model' manquant")
    try:
        return _json_safe(live.load(body["model"]))
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
    return _json_safe(arena.load(
        body["model_a"], body["model_b"],
        cps=float(body.get("cps", 12.0)),
        rot_speed=float(body.get("rot_speed", 40.0)),
        arena_size=float(body.get("arena_size", 18.0)),
        target_hits=int(body.get("target_hits", 100)),
        sample=bool(body.get("sample", True)),
        kb_h=float(body.get("kb_h", 1.0)),
        kb_v=float(body.get("kb_v", 1.0)),
        kb_idle=float(body.get("kb_idle", 1.0)),
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
