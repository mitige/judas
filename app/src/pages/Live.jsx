import { useEffect, useState } from "react";
import { api } from "../api.js";
import { usePersistentState } from "../persistence.mjs";

const LIVE_DEFAULTS = {
  model: "",
  cps: 12,
  rot: 40,
  arena: { origin_x: 0, origin_z: 0, size_x: 18, size_z: 18, floor_y: 0 },
};

export default function Live({ status }) {
  const [models, setModels] = useState([]);
  const [settings, setSettings] = usePersistentState("judas:app:live", LIVE_DEFAULTS);
  const [err, setErr] = useState(null);
  const { model, cps, rot, arena } = settings;
  const live = status?.live;
  const onErr = (e) => setErr(String(e?.message || e));

  useEffect(() => {
    api.models().then((d) => {
      setModels(d.exported);
      if (d.exported[0]) {
        setSettings((cur) => ({ ...cur, model: cur.model || d.exported[0].path }));
      }
    }).catch(() => {});
  }, [setSettings]);

  useEffect(() => {
    const id = setTimeout(() => {
      api.liveParams({ max_cps: +cps, max_rot_speed: +rot }).catch(() => {});
    }, 250);
    return () => clearTimeout(id);
  }, [cps, rot]);

  const load = () => model && api.liveLoad(model).then(() => setErr(null)).catch(onErr);
  const arm = () => api.liveParams({ enabled: true, arena }).then(() => setErr(null)).catch(onErr);
  const kill = () => api.liveKill().then(() => setErr(null)).catch(onErr);
  const setModel = (value) => setSettings((cur) => ({ ...cur, model: value }));
  const setCps = (value) => setSettings((cur) => ({ ...cur, cps: value }));
  const setRot = (value) => setSettings((cur) => ({ ...cur, rot: value }));
  const setA = (k) => (e) =>
    setSettings((cur) => ({ ...cur, arena: { ...cur.arena, [k]: +e.target.value } }));

  return (
    <>
      <h2 className="title">Live</h2>
      <div style={{ height: 24 }} />

      <div className="grid cols-2">
        <div className="panel">
          <div className="label">model</div>
          <div className="field">
            <label>torchscript</label>
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              {models.length === 0 && <option value="">—</option>}
              {models.map((m) => (
                <option key={m.path} value={m.path}>{m.name}</option>
              ))}
            </select>
          </div>
          <hr className="sep" />
          <div className="controls-row">
            <button className="btn" onClick={load} disabled={!model}>load</button>
            <button className="btn" onClick={arm}>arm</button>
            <button className="btn danger" onClick={kill}>kill</button>
            {err && <span className="tag" style={{ color: "var(--ember)" }}>{err}</span>}
          </div>
          <hr className="sep" />
          <Kv k="loaded" v={live?.model?.split(/[\\/]/).pop() ?? "—"} hl={!!live?.model} />
          <Kv k="armed" v={live?.enabled ? "yes" : "no"} hl={live?.enabled} />
          <Kv k="latency" v={live?.latency_ms != null ? `${live.latency_ms} ms` : "—"} />
          <Kv k="ticks" v={live?.tick ?? "—"} />
          <Kv k="device" v={live?.device ?? "—"} />
        </div>

        <div className="panel">
          <div className="label">humanization · realtime</div>
          <Slider l="cps" v={cps} set={setCps} min={1} max={20} />
          <Slider l="rotation °/t" v={rot} set={setRot} min={5} max={180} />
          <hr className="sep" />
          <div className="label">arena</div>
          <div className="grid cols-2">
            <Field l="origin x" v={arena.origin_x} on={setA("origin_x")} />
            <Field l="origin z" v={arena.origin_z} on={setA("origin_z")} />
            <Field l="size x" v={arena.size_x} on={setA("size_x")} />
            <Field l="size z" v={arena.size_z} on={setA("size_z")} />
            <Field l="floor y" v={arena.floor_y} on={setA("floor_y")} />
          </div>
        </div>
      </div>
    </>
  );
}

function Slider({ l, v, set, min, max }) {
  return (
    <div className="slider" style={{ marginBottom: 16 }}>
      <div className="row">
        <span className="field"><label>{l}</label></span>
        <span className="val">{v}</span>
      </div>
      <input type="range" min={min} max={max} value={v}
             onChange={(e) => set(+e.target.value)} />
    </div>
  );
}

function Field({ l, v, on }) {
  return (
    <div className="field">
      <label>{l}</label>
      <input type="number" value={v} onChange={on} />
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
