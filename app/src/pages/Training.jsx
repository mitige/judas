import { useState } from "react";
import { api } from "../api.js";
import { usePersistentState } from "../persistence.mjs";

// Profil de production RTX 3060 (boxing-mlp) : trunk MLP d128 (~1.7-3.5M sps),
// 32k envs, update grand-batch, PBT-4, rewards agressifs, CustomKB Minemen.
const D = {
  name: "boxing-mlp", n_envs: 32768, rollout_ticks: 128, total_iters: 2000,
  lr: 0.0003, gamma: 0.995, ent_coef: 0.005, clip: 0.2, epochs: 1,
  minibatch_size: 65536, sample_frac: 0.7,
  league_frac: 0.3, league_bot_frac: 0.25, pool_every: 25, save_every: 25,
  arena: 18, target_hits: 50, max_ticks: 6000,
  reward_hit: 1.0, reward_hurt: 0.85, reward_win: 10, reward_dist: 0.002,
  shaping_floor: 0.25,
  reward_combo: 0.25, combo_window: 25, combo_cap: 5,
  cps_min: 8, cps_max: 16, rot_min: 20, rot_max: 60,
  delay_min: 0, delay_max: 3, spawn_jitter: 2,
  aim_smooth_min: 0.4, aim_smooth_max: 0.75,
  attention: false, d_model: 128, n_layers: 2, n_heads: 4, history: 8,
  kb_h: 0.9055, kb_v: 0.8835, kb_idle: 0.6,
  eval_every: 25,
  population: 4, pbt_interval: 25, cross_frac: 0.25,
  resume: true, autorestart: true,
};

export default function Training({ status }) {
  // clé v2 : invalide les anciens formulaires sauvegardés (profil transformer)
  const [f, setF] = usePersistentState("judas:app:training:v2", D);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const running = status?.training?.running;
  const trainingError = status?.training?.last_error;
  const set = (k) => (e) =>
    setF((cur) => ({
      ...cur,
      [k]: e.target.type === "checkbox" ? e.target.checked : e.target.value,
    }));

  const start = async () => {
    setBusy(true); setErr(null);
    try {
      await api.trainingStart({
        name: f.name, n_envs: +f.n_envs, rollout_ticks: +f.rollout_ticks,
        total_iters: +f.total_iters, league_frac: +f.league_frac,
        league_bot_frac: +f.league_bot_frac,
        pool_every: +f.pool_every, save_every: +f.save_every,
        eval_every: +f.eval_every,
        autorestart: f.autorestart,
        policy: {
          attention: f.attention, d_model: +f.d_model,
          n_layers: +f.n_layers, n_heads: +f.n_heads, history: +f.history,
        },
        ppo: {
          lr: +f.lr, gamma: +f.gamma, ent_coef: +f.ent_coef, clip: +f.clip,
          epochs: +f.epochs, minibatch_size: +f.minibatch_size,
          sample_frac: +f.sample_frac,
        },
        shaping_floor_frac: +f.shaping_floor,
        pbt: {
          population: +f.population, interval: +f.pbt_interval,
          cross_frac: +f.cross_frac,
        },
        sim: {
          arena_size_x: +f.arena, arena_size_z: +f.arena,
          target_hits: +f.target_hits, max_ticks: +f.max_ticks,
          reward_hit: +f.reward_hit, reward_hurt: -f.reward_hurt,
          reward_win: +f.reward_win, reward_dist: +f.reward_dist,
          reward_combo: +f.reward_combo, combo_window: +f.combo_window,
          combo_cap: +f.combo_cap,
          cps_min: +f.cps_min, cps_max: +f.cps_max,
          rot_speed_min: +f.rot_min, rot_speed_max: +f.rot_max,
          delay_min: +f.delay_min, delay_max: +f.delay_max,
          aim_smooth_min: +f.aim_smooth_min, aim_smooth_max: +f.aim_smooth_max,
          spawn_jitter: +f.spawn_jitter, randomize: true,
          kb_h_mult: +f.kb_h, kb_v_mult: +f.kb_v, kb_idle_mult: +f.kb_idle,
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
        <div className="label">brain</div>
        <div className="grid cols-4">
          <Field l="size" v={f.d_model} on={set("d_model")} />
          <Field l="layers" v={f.n_layers} on={set("n_layers")} />
          <Field l="heads" v={f.n_heads} on={set("n_heads")} />
          <Field l="history" v={f.history} on={set("history")} />
        </div>
        <hr className="sep" />
        <label className="toggle">
          <input type="checkbox" checked={f.attention} onChange={set("attention")} />
          attention
        </label>
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
          <Field l="sample %" v={f.sample_frac} on={set("sample_frac")} step="0.05" />
        </div>
      </div>

      <div className="grid cols-2">
        <div className="panel">
          <div className="label">league · population (PBT)</div>
          <div className="grid cols-3">
            <Field l="league %" v={f.league_frac} on={set("league_frac")} step="0.05" />
            <Field l="bot league" v={f.league_bot_frac} on={set("league_bot_frac")} step="0.05" />
            <Field l="snapshot every" v={f.pool_every} on={set("pool_every")} />
            <Field l="save every" v={f.save_every} on={set("save_every")} />
            <Field l="eval every" v={f.eval_every} on={set("eval_every")} />
            <Field l="population" v={f.population} on={set("population")} />
            <Field l="pbt every" v={f.pbt_interval} on={set("pbt_interval")} />
            <Field l="cross %" v={f.cross_frac} on={set("cross_frac")} step="0.05" />
          </div>
        </div>
        <div className="panel">
          <div className="label">rules</div>
          <div className="grid cols-4">
            <Field l="arena" v={f.arena} on={set("arena")} />
            <Field l="target hits" v={f.target_hits} on={set("target_hits")} />
            <Field l="max ticks" v={f.max_ticks} on={set("max_ticks")} />
            <Field l="spawn jitter" v={f.spawn_jitter} on={set("spawn_jitter")} step="0.5" />
          </div>
        </div>
      </div>

      <div className="grid cols-2">
        <div className="panel">
          <div className="label">rewards</div>
          <div className="grid cols-3">
            <Field l="hit" v={f.reward_hit} on={set("reward_hit")} step="0.1" />
            <Field l="hurt (−)" v={f.reward_hurt} on={set("reward_hurt")} step="0.05" />
            <Field l="win" v={f.reward_win} on={set("reward_win")} step="1" />
            <Field l="distance" v={f.reward_dist} on={set("reward_dist")} step="0.001" />
            <Field l="combo" v={f.reward_combo} on={set("reward_combo")} step="0.05" />
            <Field l="combo window" v={f.combo_window} on={set("combo_window")} />
            <Field l="combo cap" v={f.combo_cap} on={set("combo_cap")} />
            <Field l="dist floor" v={f.shaping_floor} on={set("shaping_floor")} step="0.05" />
          </div>
        </div>
        <div className="panel">
          <div className="label">knockback · 1.0 = vanilla</div>
          <div className="grid cols-3">
            <Field l="horizontal" v={f.kb_h} on={set("kb_h")} step="0.01" />
            <Field l="vertical" v={f.kb_v} on={set("kb_v")} step="0.01" />
            <Field l="idle mult" v={f.kb_idle} on={set("kb_idle")} step="0.05" />
          </div>
        </div>
      </div>

      <div className="panel">
        <div className="label">humanization · randomized per match</div>
        <div className="grid cols-4">
          <Range l="cps" lo={f.cps_min} hi={f.cps_max} min={1} max={20}
                 onLo={set("cps_min")} onHi={set("cps_max")} />
          <Range l="rotation °/t" lo={f.rot_min} hi={f.rot_max} min={5} max={180}
                 onLo={set("rot_min")} onHi={set("rot_max")} />
          <Range l="latency ticks" lo={f.delay_min} hi={f.delay_max} min={0} max={7}
                 onLo={set("delay_min")} onHi={set("delay_max")} />
          <Range l="aim smooth" lo={f.aim_smooth_min} hi={f.aim_smooth_max}
                 min={0} max={0.95} step={0.05}
                 onLo={set("aim_smooth_min")} onHi={set("aim_smooth_max")} />
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
          <label className="toggle">
            <input type="checkbox" checked={f.autorestart} onChange={set("autorestart")} />
            auto restart
          </label>
          {running && (
            <span className="tag green">
              {status.training.run} · {status.training.uptime_s}s
            </span>
          )}
          {err && <span className="tag" style={{ color: "var(--ember)" }}>{err}</span>}
          {!running && trainingError && (
            <span className="tag" style={{ color: "var(--ember)" }}>{trainingError}</span>
          )}
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

function Range({ l, lo, hi, min, max, onLo, onHi, step = 1 }) {
  return (
    <div className="slider">
      <div className="row">
        <span className="field"><label>{l}</label></span>
        <span className="val">{lo} – {hi}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={lo} onChange={onLo} />
      <input type="range" min={min} max={max} step={step} value={hi} onChange={onHi} />
    </div>
  );
}
