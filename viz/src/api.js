// Client du daemon Judas — endpoints arène + modèles

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
  models: () => req("GET", "/models"),
  arenaStatus: () => req("GET", "/arena/status"),
  arenaLoad: (body) => req("POST", "/arena/load", body),
  arenaControl: (body) => req("POST", "/arena/control", body),
};

/** Flux d'états de match ; onMsg(state), reconnexion auto. */
export function connectArena(onMsg, onState) {
  let ws = null;
  let stop = false;

  const open = () => {
    if (stop) return;
    ws = new WebSocket(BASE.replace("http", "ws") + "/arena");
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
