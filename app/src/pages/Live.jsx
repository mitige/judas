import { useEffect, useState } from "react";
import { api } from "../api.js";

export default function Live({ status }) {
  const [models, setModels] = useState([]);
  const [model, setModel] = useState("");
  const [cps, setCps] = useState(12);
  const [rot, setRot] = useState(40);
  const [arena, setArena] = useState({ origin_x: 0, origin_z: 0, size_x: 18, size_z: 18, floor_y: 0 });
  const live = status?.live;

  useEffect(() => {
    api.models().then((d) => {
      setModels(d.exported);
      if (d.exported[0]) setModel(d.exported[0].path);
    }).catch(() => {});
  }, []);

  // humanisation à chaud (débouncée)
  useEffect(() => {
    const id = setTimeout(() => {
      api.liveParams({ max_cps: +cps, max_rot_speed: +rot }).catch(() => {});
    }, 250);
    return () => clearTimeout(id);
  }, [cps, rot]);

  const load = () => model && api.liveLoad(model).catch(() => {});
  const arm = () => api.liveParams({ enabled: true, arena }).catch(() => {});
  const kill = () => api.liveKill().catch(() => {});
  const setA = (k) => (e) => setArena({ ...arena, [k]: +e.target.value });

  return (
    <>
      <div className="eyebrow">déploiement in-game · forge 1.8.9</div>
      <h2 className="title">Mise en <em>orbite</em></h2>
      <p className="subtitle">
        Connexion du mod : ws://127.0.0.1:8765/live — touche K in-game pour
        activer le bot, L pour tout couper.
      </p>

      <div className="grid cols-2">
        <div className="panel">
          <div className="label">modèle</div>
          <div className="field">
            <label>torchscript</label>
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              {models.length === 0 && <option value="">— aucun export —</option>}
              {models.map((m) => (
                <option key={m.path} value={m.path}>{m.name}</option>
              ))}
            </select>
          </div>
          <hr className="sep" />
          <div className="controls-row">
            <button className="btn" onClick={load} disabled={!model}>charger</button>
            <button className="btn" onClick={arm}>armer</button>
            <button className="btn danger" onClick={kill}>KILL</button>
          </div>
          <hr className="sep" />
          <Kv k="chargé" v={live?.model?.split(/[\\/]/).pop() ?? "—"} hl={!!live?.model} />
          <Kv k="armé" v={live?.enabled ? "oui" : "non"} hl={live?.enabled} />
          <Kv k="latence inférence" v={live?.latency_ms != null ? `${live.latency_ms} ms` : "—"} />
          <Kv k="ticks traités" v={live?.tick ?? "—"} />
          <Kv k="device" v={live?.device ?? "—"} />
        </div>

        <div className="panel">
          <div className="label">humanisation (temps réel)</div>
          <Slider label="CPS max" val={cps} set={setCps} min={1} max={20} unit="clics/s" />
          <Slider label="Rotation max" val={rot} set={setRot} min={5} max={180} unit="°/tick" />
          <hr className="sep" />
          <div className="label">calibration arène</div>
          <div className="grid cols-2">
            <Field label="origine X" value={arena.origin_x} onChange={setA("origin_x")} />
            <Field label="origine Z" value={arena.origin_z} onChange={setA("origin_z")} />
            <Field label="taille X" value={arena.size_x} onChange={setA("size_x")} />
            <Field label="taille Z" value={arena.size_z} onChange={setA("size_z")} />
            <Field label="sol Y" value={arena.floor_y} onChange={setA("floor_y")} />
          </div>
        </div>
      </div>
    </>
  );
}

function Slider({ label, val, set, min, max, unit }) {
  return (
    <div className="slider" style={{ marginBottom: 16 }}>
      <div className="row">
        <span className="field"><label>{label}</label></span>
        <span className="val">{val} <small style={{ color: "var(--ink-faint)" }}>{unit}</small></span>
      </div>
      <input type="range" min={min} max={max} value={val}
             onChange={(e) => set(+e.target.value)} />
    </div>
  );
}

function Field({ label, value, onChange }) {
  return (
    <div className="field">
      <label>{label}</label>
      <input type="number" value={value} onChange={onChange} />
    </div>
  );
}

function Kv({ k, v, hl }) {
  return (
    <div className="kv">
      <span className="k">{k}</span>
      <span className={"v" + (hl ? " hl" : "")}>{v}</span>
    </div>
  );
}
