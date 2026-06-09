// Client du daemon Judas (REST + WebSocket /events)

const BASE = "http://127.0.0.1:8765";

async function req(method, path, body) {
  const res = await fetch(BASE + path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`${method} ${path}: ${res.status}`);
  return res.json();
}

export const api = {
  status: () => req("GET", "/status"),
  metrics: (run, tail = 300) =>
    req("GET", `/training/metrics?tail=${tail}` + (run ? `&run=${run}` : "")),
  trainingStart: (cfg) => req("POST", "/training/start", cfg),
  trainingStop: () => req("POST", "/training/stop"),
  models: () => req("GET", "/models"),
  exportModel: (ckpt, out) => req("POST", "/models/export", { ckpt, out }),
  liveLoad: (model) => req("POST", "/live/load", { model }),
  liveParams: (params) => req("POST", "/live/params", params),
  liveKill: () => req("POST", "/live/kill"),
};

/** Flux d'événements du daemon ; onMsg(payload), reconnexion auto. */
export function connectEvents(onMsg, onState) {
  let ws = null;
  let stop = false;

  const open = () => {
    if (stop) return;
    ws = new WebSocket(BASE.replace("http", "ws") + "/events");
    ws.onopen = () => onState?.(true);
    ws.onmessage = (e) => {
      try {
        onMsg(JSON.parse(e.data));
      } catch {}
    };
    ws.onclose = () => {
      onState?.(false);
      if (!stop) setTimeout(open, 2000);
    };
    ws.onerror = () => ws.close();
  };
  open();
  return () => {
    stop = true;
    ws?.close();
  };
}
