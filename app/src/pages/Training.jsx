import { useState } from "react";
import { api } from "../api.js";

const D = {
  name: "boxing", n_envs: 4096, rollout_ticks: 128, total_iters: 10000,
  lr: 0.0003, gamma: 0.995, ent_coef: 0.005, clip: 0.2, epochs: 3,
  minibatch_size: 16384,
  league_frac: 0.3, pool_every: 50,
  arena: 18, target_hits: 100, max_ticks: 6000,
  reward_hit: 1.0, reward_win: 10, reward_dist: 0.002,
  cps_min: 8, cps_max: 16, rot_min: 20, rot_max: 60,
  delay_min: 0, delay_max: 3, spawn_jitter: 2,
  resume: false,
};

export default function Training({ status }) {
  const [f, setF] = useState(D);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const running = status?.training?.running;
  const set = (k) => (e) =>
    setF({ ...f, [k]: e.target.type === "checkbox" ? e.target.checked : e.target.value });

  const start = async () => {
    setBusy(true); setErr(null);
    try {
      await api.trainingStart({
        name: f.name, n_envs: +f.n_envs, rollout_ticks: +f.rollout_ticks,
        total_iters: +f.total_iters, league_frac: +f.league_frac,
        pool_every: +f.pool_every,
        ppo: {
          lr: +f.lr, gamma: +f.gamma, ent_coef: +f.ent_coef, clip: +f.clip,
          epochs: +f.epochs, minibatch_size: +f.minibatch_size,
        },
        sim: {
          arena_size_x: +f.arena, arena_size_z: +f.arena,
          target_hits: +f.target_hits, max_ticks: +f.max_ticks,
          reward_hit: +f.reward_hit, reward_hurt: -f.reward_hit,
          reward_win: +f.reward_win, reward_dist: +f.reward_dist,
          cps_min: +f.cps_min, cps_max: +f.cps_max,
          rot_speed_min: +f.rot_min, rot_speed_max: +f.rot_max,
          delay_min: +f.delay_min, delay_max: +f.delay_max,
          spawn_jitter: +f.spawn_jitter, randomize: true,
        },
        ...(f.resume ? { resume: `runs/${f.name}/latest.pt` } : {}),
      });
    } catch (e) { setErr(String(e.message || e)); }
    setBusy(false);
  };
  const stop = async () => {
    setBusy(true);
    try { await api.trainingStop(); } catch {}
    setBusy(false);
  };

  return (
    <>
      <h2 className="title">Training</h2>
      <div style={{ height: 24 }} />

      <div className="panel">
        <div className="label">run</div>
        <div className="grid cols-4">
          <Field l="name" type="text" v={f.name} on={set("name")} />
          <Field l="envs" v={f.n_envs} on={set("n_envs")} />
          <Field l="rollout" v={f.rollout_ticks} on={set("rollout_ticks")} />
          <Field l="iterations" v={f.total_iters} on={set("total_iters")} />
        </div>
      </div>

      <div className="panel">
        <div className="label">ppo</div>
        <div className="grid cols-3">
          <Field l="lr" v={f.lr} on={set("lr")} step="0.0001" />
          <Field l="gamma" v={f.gamma} on={set("gamma")} step="0.001" />
          <Field l="entropy" v={f.ent_coef} on={set("ent_coef")} step="0.001" />
          <Field l="clip" v={f.clip} on={set("clip")} step="0.05" />
          <Field l="epochs" v={f.epochs} on={set("epochs")} />
          <Field l="minibatch" v={f.minibatch_size} on={set("minibatch_size")} />
        </div>
      </div>

      <div className="grid cols-2">
        <div className="panel">
          <div className="label">league</div>
          <div className="grid cols-2">
            <Field l="league %" v={f.league_frac} on={set("league_frac")} step="0.05" />
            <Field l="snapshot every" v={f.pool_every} on={set("pool_every")} />
          </div>
        </div>
        <div className="panel">
          <div className="label">rules</div>
          <div className="grid cols-3">
            <Field l="arena" v={f.arena} on={set("arena")} />
            <Field l="target hits" v={f.target_hits} on={set("target_hits")} />
            <Field l="max ticks" v={f.max_ticks} on={set("max_ticks")} />
          </div>
        </div>
      </div>

      <div className="grid cols-2">
        <div className="panel">
          <div className="label">rewards</div>
          <div className="grid cols-3">
            <Field l="hit" v={f.reward_hit} on={set("reward_hit")} step="0.1" />
            <Field l="win" v={f.reward_win} on={set("reward_win")} step="1" />
            <Field l="distance" v={f.reward_dist} on={set("reward_dist")} step="0.001" />
          </div>
        </div>
        <div className="panel">
          <div className="label">spawn</div>
          <div className="grid cols-3">
            <Field l="jitter" v={f.spawn_jitter} on={set("spawn_jitter")} step="0.5" />
          </div>
        </div>
      </div>

      <div className="panel">
        <div className="label">humanization · randomized per match</div>
        <div className="grid cols-3">
          <Range l="cps" lo={f.cps_min} hi={f.cps_max} min={1} max={20}
                 onLo={set("cps_min")} onHi={set("cps_max")} />
          <Range l="rotation °/t" lo={f.rot_min} hi={f.rot_max} min={5} max={180}
                 onLo={set("rot_min")} onHi={set("rot_max")} />
          <Range l="latency ticks" lo={f.delay_min} hi={f.delay_max} min={0} max={7}
                 onLo={set("delay_min")} onHi={set("delay_max")} />
        </div>
      </div>

      <div className="panel">
        <div className="controls-row">
          <button className="btn" onClick={start} disabled={busy || running}>
            ▶ start
          </button>
          <button className="btn danger" onClick={stop} disabled={busy || !running}>
            ■ stop
          </button>
          <label className="toggle">
            <input type="checkbox" checked={f.resume} onChange={set("resume")} />
            resume latest
          </label>
          {running && (
            <span className="tag green">
              {status.training.run} · {status.training.uptime_s}s
            </span>
          )}
          {err && <span className="tag" style={{ color: "var(--ember)" }}>{err}</span>}
        </div>
      </div>
    </>
  );
}

function Field({ l, v, on, type = "number", step }) {
  return (
    <div className="field">
      <label>{l}</label>
      <input type={type} value={v} onChange={on} step={step} />
    </div>
  );
}

function Range({ l, lo, hi, min, max, onLo, onHi }) {
  return (
    <div className="slider">
      <div className="row">
        <span className="field"><label>{l}</label></span>
        <span className="val">{lo} – {hi}</span>
      </div>
      <input type="range" min={min} max={max} value={lo} onChange={onLo} />
      <input type="range" min={min} max={max} value={hi} onChange={onHi} />
    </div>
  );
}
