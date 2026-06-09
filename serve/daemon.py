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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .live import LiveSession
from .protocol import ArenaCalib
from .training_manager import TrainingManager

app = FastAPI(title="judas-daemon")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

training = TrainingManager()
live = LiveSession()
_event_clients: set = set()


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
    return {"training": training.status(), "live": live.status(),
            "gpu": _gpu_status()}


@app.post("/training/start")
def training_start(cfg: dict):
    resume = cfg.pop("resume", None)
    return training.start(cfg, resume=resume)


@app.post("/training/stop")
def training_stop():
    return training.stop()


@app.get("/training/metrics")
def training_metrics(run: str | None = None, tail: int = 200):
    return training.metrics(run, tail)


@app.get("/models")
def models():
    return {"runs": training.list_runs(), "exported": training.list_exported()}


@app.post("/models/export")
def models_export(body: dict):
    from train.export import export
    out = export(body["ckpt"], body.get("out", "models/judas.pts"))
    return {"exported": str(out)}


@app.post("/live/load")
def live_load(body: dict):
    return live.load(body["model"])


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
    return live.status()


@app.post("/live/kill")
def live_kill():
    live.params.enabled = False
    live.reset()
    return live.status()


# ----------------------------------------------------------------- WebSockets
@app.websocket("/live")
async def ws_live(ws: WebSocket):
    """Connexion du mod Forge : état -> action à chaque tick."""
    await ws.accept()
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
            action = live.on_state(msg)
            if action is not None:
                await ws.send_text(json.dumps(action))
    except WebSocketDisconnect:
        pass


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
            await ws.send_text(json.dumps(payload))
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
