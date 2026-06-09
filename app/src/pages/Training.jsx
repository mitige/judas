import { useState } from "react";
import { api } from "../api.js";

const DEFAULTS = {
  name: "boxing",
  n_envs: 4096,
  rollout_ticks: 128,
  league_frac: 0.3,
  cps_min: 8, cps_max: 16,
  rot_min: 20, rot_max: 60,
  delay_min: 0, delay_max: 3,
  arena: 18,
  target_hits: 100,
};

export default function Training({ status }) {
  const [f, setF] = useState(DEFAULTS);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const running = status?.training?.running;

  const set = (k) => (e) => setF({ ...f, [k]: e.target.value });

  const start = async () => {
    setBusy(true); setErr(null);
    try {
      await api.trainingStart({
        name: f.name,
        n_envs: +f.n_envs,
        rollout_ticks: +f.rollout_ticks,
        league_frac: +f.league_frac,
        sim: {
          arena_size_x: +f.arena, arena_size_z: +f.arena,
          target_hits: +f.target_hits,
          cps_min: +f.cps_min, cps_max: +f.cps_max,
          rot_speed_min: +f.rot_min, rot_speed_max: +f.rot_max,
          delay_min: +f.delay_min, delay_max: +f.delay_max,
          spawn_jitter: 2.0, randomize: true,
        },
      });
    } catch (e) { setErr(String(e)); }
    setBusy(false);
  };

  const stop = async () => {
    setBusy(true);
    try { await api.trainingStop(); } catch {}
    setBusy(false);
  };

  return (
    <>
      <div className="eyebrow">simulateur cuda · ppo self-play</div>
      <h2 className="title">Entraîner un <em>champion</em></h2>
      <p className="subtitle">
        Des milliers de matchs boxing simulés en parallèle. L&apos;humanisation est
        apprise pendant l&apos;entraînement (domain randomization).
      </p>

      <div className="panel">
        <div className="label">run</div>
        <div className="grid cols-4">
          <Field label="nom" type="text" value={f.name} onChange={set("name")} />
          <Field label="environnements" value={f.n_envs} onChange={set("n_envs")} />
          <Field label="rollout (ticks)" value={f.rollout_ticks} onChange={set("rollout_ticks")} />
          <Field label="fraction league" value={f.league_frac} onChange={set("league_frac")} step="0.05" />
        </div>
      </div>

      <div className="panel">
        <div className="label">arène &amp; règles</div>
        <div className="grid cols-4">
          <Field label="taille arène (blocs)" value={f.arena} onChange={set("arena")} />
          <Field label="hits pour gagner" value={f.target_hits} onChange={set("target_hits")} />
        </div>
      </div>

      <div className="panel">
        <div className="label">humanisation (plages randomisées par match)</div>
        <div className="grid cols-3">
          <Range label="CPS" lo={f.cps_min} hi={f.cps_max} min={1} max={20}
                 onLo={set("cps_min")} onHi={set("cps_max")} unit="clics/s" />
          <Range label="Rotation" lo={f.rot_min} hi={f.rot_max} min={5} max={180}
                 onLo={set("rot_min")} onHi={set("rot_max")} unit="°/tick" />
          <Range label="Latence" lo={f.delay_min} hi={f.delay_max} min={0} max={7}
                 onLo={set("delay_min")} onHi={set("delay_max")} unit="ticks" />
        </div>
      </div>

      <div className="panel">
        <div className="controls-row">
          <button className="btn" onClick={start} disabled={busy || running}>
            ▶ Lancer
          </button>
          <button className="btn danger" onClick={stop} disabled={busy || !running}>
            ■ Arrêter
          </button>
          {running && <span className="tag green">run « {status.training.run} » actif
            · {status.training.uptime_s}s</span>}
          {err && <span className="tag" style={{ color: "var(--ember)" }}>{err}</span>}
        </div>
      </div>
    </>
  );
}

function Field({ label, value, onChange, type = "number", step }) {
  return (
    <div className="field">
      <label>{label}</label>
      <input type={type} value={value} onChange={onChange} step={step} />
    </div>
  );
}

function Range({ label, lo, hi, min, max, onLo, onHi, unit }) {
  return (
    <div className="slider">
      <div className="row">
        <span className="field"><label>{label}</label></span>
        <span className="val">{lo} – {hi} <small style={{ color: "var(--ink-faint)" }}>{unit}</small></span>
      </div>
      <input type="range" min={min} max={max} value={lo} onChange={onLo} />
      <input type="range" min={min} max={max} value={hi} onChange={onHi} />
    </div>
  );
}
