import { useState } from "react";
import { api } from "../api.js";
import { usePersistentState } from "../persistence.mjs";

// Profil combo_god_recovery_kb092_combo12 : successeur du leaderboard avec
// recovery/hit-select plus strict et KB calibre depuis le dump live.
const D = {
  name: "combo_god_recovery_kb092_combo12", n_envs: 4096, rollout_ticks: 96, total_iters: 5000,
  seed: 5,
  lr: 0.0000015, gamma: 0.995, lam: 0.95, ent_coef: 0.0065, clip: 0.04, epochs: 1,
  minibatch_size: 16384, sample_frac: 0.28, vf_coef: 0.5, max_grad_norm: 0.5,
  coach_coef: 0.36, coach_until: 0.90,
  amp: true,
  league_frac: 0.00, league_bot_frac: 0.00, league_spar_bot_frac: 0.12,
  league_rehit_bot_frac: 0.16, league_pressure_bot_frac: 0.06,
  league_combo_chase_bot_frac: 0.24, league_counter_bot_frac: 0.42,
  league_pad_bot_frac: 0.00,
  pool_every: 100, save_every: 1, keep_ckpts: 96,
  resume_as_seed: true,
  resume_same_run_as_seed: true,
  fresh_optimizer_on_resume: true,
  safety_stop_on_regression: true, safety_restore_on_low_combo: false,
  safety_under_combo_escape: 0.015,
  safety_back_frac: 0.001, safety_min_strafe_frac: 0.50,
  safety_min_opener_strafe_frac: 0.75, safety_opener_ticks: 20,
  safety_min_opener_strafe_hold_frac: 0.70,
  safety_min_opener_pressure_frac: 0.45,
  safety_min_combo_tap_frac: 0.18,
  safety_min_combo_z_tap_frac: 0.16, safety_max_combo_s_tap_frac: 0.015,
  safety_min_hit_wtap_frac: 0.055,
  safety_min_chase_hit_wtap_frac: 0.055,
  safety_rollout_hit_wtap_slack: 0.30,
  safety_hit_wtap_blocks_promotion: true,
  safety_min_under_combo_counter_hit_frac: 0.115,
  safety_under_combo_avoid_frac: 0.040,
  safety_under_combo_avoid_min_combo12: 0.10,
  safety_under_combo_avoid_min_hit_rate: 90.0,
  score_under_combo_avoid_target: 0.10,
  score_under_combo_avoid_weight: 0.45,
  score_under_combo_avoid_cap: 0.025,
  safety_min_under_combo_hit_select_clean_frac: 0.30,
  safety_max_under_combo_hit_select_trade_frac: 0.08,
  safety_strafe_frac: 1.0, safety_sky_frac: 0.14,
  safety_min_hit_rate: 42.0, safety_fresh_min_hit_rate: 12.0,
  safety_promote_min_combo_max: 10.0,
  arena: 40, target_hits: 140, max_ticks: 11000, speed_amplifier: 1,
  reward_hit: 1.15, reward_hurt: 2.60, reward_win: 10, reward_dist: 0.12,
  reward_sprint_hit: 3.2,
  reward_trade_penalty: 7.0,
  shaping_floor: 1.0,
  shaping_hit_rate: 24.0, shaping_engage_rate: 0.34,
  shaping_combo5_rate: 0.120, shaping_combo12_state: 0.48, shaping_sky_frac: 0.22, shaping_decay_iters: 620, curriculum_gap: 1.5,
  reward_combo: 13.2, combo_window: 92, combo_cap: 30,
  reward_combo_drop: 0.95, combo_drop_min: 4, reward_combo_pressure: 4.0,
  reward_aim: 0.28, reward_bad_pitch: 2.00, reward_chase: 1.35,
  reward_turn_aim: 0.34, reward_aggression: 1.10, reward_no_escape: 2.20,
  reward_combo_focus: 4.20, reward_combo_tap: 7.20, reward_opener_strafe: 3.00,
  reward_hit_wtap: 18.00,
  reward_counter_hit: 34.00, reward_hit_select: 36.00,
  reward_chase_rechain: 10.80, reward_chase_hit_select: 30.00, reward_chase_close_counter: 22.00,
  reward_chase_counter: 38.00,
  reward_spar_counter: 34.00,
  cps_min: 10, cps_max: 10, rot_min: 150, rot_max: 240,
  delay_min: 0, delay_max: 0, spawn_jitter: 0.4,
  aim_smooth_min: 0.0, aim_smooth_max: 0.04,
  attention: true, d_model: 96, n_layers: 2, n_heads: 4, history: 8, aim_residual: 1.15,
  direct_movement_lock: true, leaderboard_boxing: true,
  direct_counter_attack_lock: true, direct_hit_select_attack_lock: true,
  direct_hit_select_attack_bias: 22.0,
  under_combo_attack_lock: false,
  kb_h: 0.92, kb_v: 0.90, kb_idle: 0.6,
  eval_every: 0, combo_eval_every: 16, combo_eval_envs: 128, combo_eval_ticks: 1200,
  combo_eval_chase: true, combo_eval_chase_envs: 64, combo_eval_chase_ticks: 900,
  combo_eval_spar: true, combo_eval_spar_envs: 64, combo_eval_spar_ticks: 900,
  combo_eval_rehit: true, combo_eval_rehit_envs: 64, combo_eval_rehit_ticks: 900,
  combo_eval_pressure: true, combo_eval_pressure_envs: 64, combo_eval_pressure_ticks: 900,
  safety_require_chase_combo: true,
  population: 1, pbt_interval: 25, cross_frac: 0.25,
  resume: true, autorestart: true,
};

const resumePathFor = (name) =>
  name === "combo_god_recovery_kb092_combo12"
    ? `runs/${name}/safe_latest.pt`
    : name === "combo_god_leaderboard10_combo12"
    ? `runs/${name}/safe_latest.pt`
    : name === "combo_god_countertap96_combo12"
    ? `runs/${name}/latest.pt`
    : name === "combo_god_directpad_lock_combo12"
    ? `runs/${name}/safe_latest.pt`
    : `runs/${name}/latest.pt`;

export default function Training({ status }) {
  // Cle v92 : nouveau profil recovery KB 0.92/0.90 et gates hit-select plus stricts.
  const [f, setF] = usePersistentState("judas:app:training:v92", D);
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
        total_iters: +f.total_iters, seed: +f.seed,
        league_frac: +f.league_frac,
        league_bot_frac: +f.league_bot_frac,
        league_spar_bot_frac: +f.league_spar_bot_frac,
        league_rehit_bot_frac: +f.league_rehit_bot_frac,
        league_pressure_bot_frac: +f.league_pressure_bot_frac,
        league_combo_chase_bot_frac: +f.league_combo_chase_bot_frac,
        league_counter_bot_frac: +f.league_counter_bot_frac,
        league_pad_bot_frac: +f.league_pad_bot_frac,
        pool_every: +f.pool_every, save_every: +f.save_every,
        keep_ckpts: +f.keep_ckpts,
        eval_every: +f.eval_every,
        combo_eval_every: +f.combo_eval_every,
        combo_eval_envs: +f.combo_eval_envs,
        combo_eval_ticks: +f.combo_eval_ticks,
        combo_eval_chase: Boolean(f.combo_eval_chase),
        combo_eval_chase_envs: +f.combo_eval_chase_envs,
        combo_eval_chase_ticks: +f.combo_eval_chase_ticks,
        combo_eval_spar: Boolean(f.combo_eval_spar),
        combo_eval_spar_envs: +f.combo_eval_spar_envs,
        combo_eval_spar_ticks: +f.combo_eval_spar_ticks,
        combo_eval_rehit: Boolean(f.combo_eval_rehit),
        combo_eval_rehit_envs: +f.combo_eval_rehit_envs,
        combo_eval_rehit_ticks: +f.combo_eval_rehit_ticks,
        combo_eval_pressure: Boolean(f.combo_eval_pressure),
        combo_eval_pressure_envs: +f.combo_eval_pressure_envs,
        combo_eval_pressure_ticks: +f.combo_eval_pressure_ticks,
        safety_require_chase_combo: Boolean(f.safety_require_chase_combo),
        resume_as_seed: Boolean(f.resume_as_seed),
        resume_same_run_as_seed: Boolean(f.resume_same_run_as_seed),
        fresh_optimizer_on_resume: Boolean(f.fresh_optimizer_on_resume),
        safety_stop_on_regression: Boolean(f.safety_stop_on_regression),
        safety_restore_on_low_combo: Boolean(f.safety_restore_on_low_combo),
        safety_under_combo_escape: +f.safety_under_combo_escape,
        safety_back_frac: +f.safety_back_frac,
        safety_min_strafe_frac: +f.safety_min_strafe_frac,
        safety_min_opener_strafe_frac: +f.safety_min_opener_strafe_frac,
        safety_opener_ticks: +f.safety_opener_ticks,
        safety_min_opener_strafe_hold_frac: +f.safety_min_opener_strafe_hold_frac,
        safety_min_opener_pressure_frac: +f.safety_min_opener_pressure_frac,
        safety_min_combo_tap_frac: +f.safety_min_combo_tap_frac,
        safety_min_combo_z_tap_frac: +f.safety_min_combo_z_tap_frac,
        safety_max_combo_s_tap_frac: +f.safety_max_combo_s_tap_frac,
        safety_min_hit_wtap_frac: +f.safety_min_hit_wtap_frac,
        safety_min_chase_hit_wtap_frac: +f.safety_min_chase_hit_wtap_frac,
        safety_rollout_hit_wtap_slack: +f.safety_rollout_hit_wtap_slack,
        safety_hit_wtap_blocks_promotion: Boolean(f.safety_hit_wtap_blocks_promotion),
        safety_min_under_combo_counter_hit_frac: +f.safety_min_under_combo_counter_hit_frac,
        safety_under_combo_avoid_frac: +f.safety_under_combo_avoid_frac,
        safety_under_combo_avoid_min_combo12: +f.safety_under_combo_avoid_min_combo12,
        safety_under_combo_avoid_min_hit_rate: +f.safety_under_combo_avoid_min_hit_rate,
        score_under_combo_avoid_target: +f.score_under_combo_avoid_target,
        score_under_combo_avoid_weight: +f.score_under_combo_avoid_weight,
        score_under_combo_avoid_cap: +f.score_under_combo_avoid_cap,
        safety_min_under_combo_hit_select_clean_frac: +f.safety_min_under_combo_hit_select_clean_frac,
        safety_max_under_combo_hit_select_trade_frac: +f.safety_max_under_combo_hit_select_trade_frac,
        safety_strafe_frac: +f.safety_strafe_frac,
        safety_sky_frac: +f.safety_sky_frac,
        safety_min_hit_rate: +f.safety_min_hit_rate,
        safety_fresh_min_hit_rate: +f.safety_fresh_min_hit_rate,
        safety_promote_min_combo_max: +f.safety_promote_min_combo_max,
        autorestart: f.autorestart,
        policy: {
          attention: f.attention, d_model: +f.d_model,
          n_layers: +f.n_layers, n_heads: +f.n_heads, history: +f.history, aim_residual: +f.aim_residual,
          direct_movement_lock: Boolean(f.direct_movement_lock),
          leaderboard_boxing: Boolean(f.leaderboard_boxing),
          direct_counter_attack_lock: Boolean(f.direct_counter_attack_lock),
          direct_hit_select_attack_lock: Boolean(f.direct_hit_select_attack_lock),
          direct_hit_select_attack_bias: +f.direct_hit_select_attack_bias,
          under_combo_attack_lock: Boolean(f.under_combo_attack_lock),
        },
        ppo: {
          lr: +f.lr, gamma: +f.gamma, lam: +f.lam, ent_coef: +f.ent_coef,
          clip: +f.clip, epochs: +f.epochs, minibatch_size: +f.minibatch_size,
          sample_frac: +f.sample_frac, vf_coef: +f.vf_coef,
          max_grad_norm: +f.max_grad_norm, coach_coef: +f.coach_coef,
          coach_until: +f.coach_until, amp: Boolean(f.amp),
        },
        shaping_floor_frac: +f.shaping_floor,
        shaping_hit_rate: +f.shaping_hit_rate,
        shaping_engage_rate: +f.shaping_engage_rate,
        shaping_combo5_rate: +f.shaping_combo5_rate,
        shaping_combo12_state: +f.shaping_combo12_state,
        shaping_sky_frac: +f.shaping_sky_frac,
        shaping_decay_iters: +f.shaping_decay_iters,
        curriculum_gap: +f.curriculum_gap,
        pbt: {
          population: +f.population, interval: +f.pbt_interval,
          cross_frac: +f.cross_frac,
        },
        sim: {
          arena_size_x: +f.arena, arena_size_z: +f.arena,
          target_hits: +f.target_hits, max_ticks: +f.max_ticks,
          speed_amplifier: +f.speed_amplifier,
          reward_hit: +f.reward_hit, reward_hurt: -Math.abs(+f.reward_hurt),
          reward_win: +f.reward_win, reward_dist: +f.reward_dist,
          reward_sprint_hit: +f.reward_sprint_hit,
          reward_trade_penalty: +f.reward_trade_penalty,
          reward_combo: +f.reward_combo, combo_window: +f.combo_window,
          combo_cap: +f.combo_cap, reward_combo_drop: +f.reward_combo_drop,
          combo_drop_min: +f.combo_drop_min,
          reward_combo_pressure: +f.reward_combo_pressure,
          reward_aim: +f.reward_aim,
          reward_bad_pitch: +f.reward_bad_pitch,
          reward_chase: +f.reward_chase,
          reward_turn_aim: +f.reward_turn_aim,
          reward_aggression: +f.reward_aggression,
          reward_no_escape: +f.reward_no_escape,
          reward_combo_focus: +f.reward_combo_focus,
          reward_combo_tap: +f.reward_combo_tap,
          reward_opener_strafe: +f.reward_opener_strafe,
          reward_hit_wtap: +f.reward_hit_wtap,
          reward_counter_hit: +f.reward_counter_hit,
          reward_hit_select: +f.reward_hit_select,
          reward_chase_rechain: +f.reward_chase_rechain,
          reward_chase_hit_select: +f.reward_chase_hit_select,
          reward_chase_close_counter: +f.reward_chase_close_counter,
          reward_chase_counter: +f.reward_chase_counter,
          reward_spar_counter: +f.reward_spar_counter,
          cps_min: +f.cps_min, cps_max: +f.cps_max,
          rot_speed_min: +f.rot_min, rot_speed_max: +f.rot_max,
          delay_min: +f.delay_min, delay_max: +f.delay_max,
          aim_smooth_min: +f.aim_smooth_min, aim_smooth_max: +f.aim_smooth_max,
          spawn_jitter: +f.spawn_jitter, randomize: true,
          kb_h_mult: +f.kb_h, kb_v_mult: +f.kb_v, kb_idle_mult: +f.kb_idle,
        },
        ...(f.resume ? { resume: resumePathFor(f.name) } : {}),
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
          <Field l="seed" v={f.seed} on={set("seed")} />
        </div>
      </div>

      <div className="panel">
        <div className="label">brain</div>
        <div className="grid cols-4">
          <Field l="size" v={f.d_model} on={set("d_model")} />
          <Field l="layers" v={f.n_layers} on={set("n_layers")} />
          <Field l="heads" v={f.n_heads} on={set("n_heads")} />
          <Field l="history" v={f.history} on={set("history")} />
          <Field l="aim residual" v={f.aim_residual} on={set("aim_residual")} step="0.05" />
        </div>
        <hr className="sep" />
        <label className="toggle">
          <input type="checkbox" checked={f.attention} onChange={set("attention")} />
          attention
        </label>
        <label className="toggle">
          <input type="checkbox" checked={f.direct_movement_lock} onChange={set("direct_movement_lock")} />
          direct move lock
        </label>
        <label className="toggle">
          <input type="checkbox" checked={f.leaderboard_boxing} onChange={set("leaderboard_boxing")} />
          leaderboard boxing
        </label>
        <label className="toggle">
          <input type="checkbox" checked={f.direct_counter_attack_lock} onChange={set("direct_counter_attack_lock")} />
          force counter click
        </label>
        <label className="toggle">
          <input type="checkbox" checked={f.direct_hit_select_attack_lock} onChange={set("direct_hit_select_attack_lock")} />
          force hit-select click
        </label>
        <div className="grid cols-4">
          <Field l="hit-select bias" v={f.direct_hit_select_attack_bias} on={set("direct_hit_select_attack_bias")} step="0.25" />
        </div>
        <label className="toggle">
          <input type="checkbox" checked={f.under_combo_attack_lock} onChange={set("under_combo_attack_lock")} />
          block counter-hit
        </label>
      </div>

      <div className="panel">
        <div className="label">ppo</div>
        <div className="grid cols-3">
          <Field l="lr" v={f.lr} on={set("lr")} step="0.0001" />
          <Field l="gamma" v={f.gamma} on={set("gamma")} step="0.001" />
          <Field l="lambda" v={f.lam} on={set("lam")} step="0.01" />
          <Field l="entropy" v={f.ent_coef} on={set("ent_coef")} step="0.001" />
          <Field l="clip" v={f.clip} on={set("clip")} step="0.05" />
          <Field l="epochs" v={f.epochs} on={set("epochs")} />
          <Field l="minibatch" v={f.minibatch_size} on={set("minibatch_size")} />
          <Field l="sample %" v={f.sample_frac} on={set("sample_frac")} step="0.05" />
          <Field l="vf coef" v={f.vf_coef} on={set("vf_coef")} step="0.05" />
          <Field l="grad norm" v={f.max_grad_norm} on={set("max_grad_norm")} step="0.05" />
          <Field l="coach" v={f.coach_coef} on={set("coach_coef")} step="0.01" />
          <Field l="coach until" v={f.coach_until} on={set("coach_until")} step="0.01" />
        </div>
        <hr className="sep" />
        <label className="toggle">
          <input type="checkbox" checked={f.amp} onChange={set("amp")} />
          amp
        </label>
      </div>

      <div className="grid cols-2">
        <div className="panel">
          <div className="label">league / population (PBT)</div>
          <div className="grid cols-3">
            <Field l="league %" v={f.league_frac} on={set("league_frac")} step="0.01" />
            <Field l="bot league" v={f.league_bot_frac} on={set("league_bot_frac")} step="0.05" />
            <Field l="spar bot" v={f.league_spar_bot_frac} on={set("league_spar_bot_frac")} step="0.05" />
            <Field l="rehit bot" v={f.league_rehit_bot_frac} on={set("league_rehit_bot_frac")} step="0.05" />
            <Field l="pressure bot" v={f.league_pressure_bot_frac} on={set("league_pressure_bot_frac")} step="0.05" />
            <Field l="combo chase" v={f.league_combo_chase_bot_frac} on={set("league_combo_chase_bot_frac")} step="0.05" />
            <Field l="counter recovery" v={f.league_counter_bot_frac} on={set("league_counter_bot_frac")} step="0.05" />
            <Field l="pad bot" v={f.league_pad_bot_frac} on={set("league_pad_bot_frac")} step="0.05" />
            <Field l="snapshot every" v={f.pool_every} on={set("pool_every")} />
            <Field l="save every" v={f.save_every} on={set("save_every")} />
            <Field l="keep ckpts" v={f.keep_ckpts} on={set("keep_ckpts")} />
            <Field l="eval every" v={f.eval_every} on={set("eval_every")} />
            <Field l="combo eval" v={f.combo_eval_every} on={set("combo_eval_every")} />
            <Field l="combo envs" v={f.combo_eval_envs} on={set("combo_eval_envs")} />
            <Field l="combo ticks" v={f.combo_eval_ticks} on={set("combo_eval_ticks")} />
            <Field l="chase envs" v={f.combo_eval_chase_envs} on={set("combo_eval_chase_envs")} />
            <Field l="chase ticks" v={f.combo_eval_chase_ticks} on={set("combo_eval_chase_ticks")} />
            <Field l="spar envs" v={f.combo_eval_spar_envs} on={set("combo_eval_spar_envs")} />
            <Field l="spar ticks" v={f.combo_eval_spar_ticks} on={set("combo_eval_spar_ticks")} />
            <Field l="rehit envs" v={f.combo_eval_rehit_envs} on={set("combo_eval_rehit_envs")} />
            <Field l="rehit ticks" v={f.combo_eval_rehit_ticks} on={set("combo_eval_rehit_ticks")} />
            <Field l="pressure envs" v={f.combo_eval_pressure_envs} on={set("combo_eval_pressure_envs")} />
            <Field l="pressure ticks" v={f.combo_eval_pressure_ticks} on={set("combo_eval_pressure_ticks")} />
            <Field l="min hit safe" v={f.safety_min_hit_rate} on={set("safety_min_hit_rate")} step="1" />
            <Field l="fresh min hit" v={f.safety_fresh_min_hit_rate} on={set("safety_fresh_min_hit_rate")} step="1" />
            <Field l="promote combo" v={f.safety_promote_min_combo_max} on={set("safety_promote_min_combo_max")} step="1" />
            <Field l="under esc max" v={f.safety_under_combo_escape} on={set("safety_under_combo_escape")} step="0.001" />
            <Field l="back max" v={f.safety_back_frac} on={set("safety_back_frac")} step="0.001" />
            <Field l="strafe min" v={f.safety_min_strafe_frac} on={set("safety_min_strafe_frac")} step="0.001" />
            <Field l="opener strafe" v={f.safety_min_opener_strafe_frac} on={set("safety_min_opener_strafe_frac")} step="0.001" />
            <Field l="opener ticks" v={f.safety_opener_ticks} on={set("safety_opener_ticks")} step="1" />
            <Field l="opener hold" v={f.safety_min_opener_strafe_hold_frac} on={set("safety_min_opener_strafe_hold_frac")} step="0.001" />
            <Field l="opener pressure" v={f.safety_min_opener_pressure_frac} on={set("safety_min_opener_pressure_frac")} step="0.001" />
            <Field l="combo tap min" v={f.safety_min_combo_tap_frac} on={set("safety_min_combo_tap_frac")} step="0.001" />
            <Field l="z tap min" v={f.safety_min_combo_z_tap_frac} on={set("safety_min_combo_z_tap_frac")} step="0.001" />
            <Field l="s tap max" v={f.safety_max_combo_s_tap_frac} on={set("safety_max_combo_s_tap_frac")} step="0.001" />
            <Field l="hit wtap min" v={f.safety_min_hit_wtap_frac} on={set("safety_min_hit_wtap_frac")} step="0.001" />
            <Field l="chase wtap min" v={f.safety_min_chase_hit_wtap_frac} on={set("safety_min_chase_hit_wtap_frac")} step="0.001" />
            <Field l="rollout hit wtap slack" v={f.safety_rollout_hit_wtap_slack} on={set("safety_rollout_hit_wtap_slack")} step="0.001" />
            <Field l="counter min" v={f.safety_min_under_combo_counter_hit_frac} on={set("safety_min_under_combo_counter_hit_frac")} step="0.001" />
            <Field l="avoid combo max" v={f.safety_under_combo_avoid_frac} on={set("safety_under_combo_avoid_frac")} step="0.001" />
            <Field l="avoid combo12 min" v={f.safety_under_combo_avoid_min_combo12} on={set("safety_under_combo_avoid_min_combo12")} step="0.001" />
            <Field l="avoid hit min" v={f.safety_under_combo_avoid_min_hit_rate} on={set("safety_under_combo_avoid_min_hit_rate")} step="1" />
            <Field l="avoid score target" v={f.score_under_combo_avoid_target} on={set("score_under_combo_avoid_target")} step="0.001" />
            <Field l="avoid score weight" v={f.score_under_combo_avoid_weight} on={set("score_under_combo_avoid_weight")} step="0.01" />
            <Field l="avoid score cap" v={f.score_under_combo_avoid_cap} on={set("score_under_combo_avoid_cap")} step="0.001" />
            <Field l="hit select clean" v={f.safety_min_under_combo_hit_select_clean_frac} on={set("safety_min_under_combo_hit_select_clean_frac")} step="0.001" />
            <Field l="hit select trade max" v={f.safety_max_under_combo_hit_select_trade_frac} on={set("safety_max_under_combo_hit_select_trade_frac")} step="0.001" />
            <Field l="strafe max" v={f.safety_strafe_frac} on={set("safety_strafe_frac")} step="0.01" />
            <Field l="population" v={f.population} on={set("population")} />
            <Field l="pbt every" v={f.pbt_interval} on={set("pbt_interval")} />
            <Field l="cross %" v={f.cross_frac} on={set("cross_frac")} step="0.01" />
          </div>
          <hr className="sep" />
          <label className="toggle">
            <input type="checkbox" checked={f.combo_eval_chase} onChange={set("combo_eval_chase")} />
            chase eval
          </label>
          <label className="toggle">
            <input type="checkbox" checked={f.combo_eval_spar} onChange={set("combo_eval_spar")} />
            spar eval
          </label>
          <label className="toggle">
            <input type="checkbox" checked={f.combo_eval_rehit} onChange={set("combo_eval_rehit")} />
            rehit eval
          </label>
          <label className="toggle">
            <input type="checkbox" checked={f.combo_eval_pressure} onChange={set("combo_eval_pressure")} />
            pressure eval
          </label>
          <label className="toggle">
            <input type="checkbox" checked={f.safety_require_chase_combo} onChange={set("safety_require_chase_combo")} />
            chase safe gate
          </label>
          <label className="toggle">
            <input type="checkbox" checked={f.safety_restore_on_low_combo} onChange={set("safety_restore_on_low_combo")} />
            restore low combo
          </label>
          <label className="toggle">
            <input type="checkbox" checked={f.safety_hit_wtap_blocks_promotion} onChange={set("safety_hit_wtap_blocks_promotion")} />
            wtap waits safe
          </label>
        </div>
        <div className="panel">
          <div className="label">rules</div>
          <div className="grid cols-4">
            <Field l="arena" v={f.arena} on={set("arena")} />
            <Field l="target hits" v={f.target_hits} on={set("target_hits")} />
            <Field l="max ticks" v={f.max_ticks} on={set("max_ticks")} />
            <Field l="speed amp" v={f.speed_amplifier} on={set("speed_amplifier")} />
            <Field l="curriculum gap" v={f.curriculum_gap} on={set("curriculum_gap")} step="0.1" />
            <Field l="spawn jitter" v={f.spawn_jitter} on={set("spawn_jitter")} step="0.1" />
          </div>
        </div>
      </div>

      <div className="grid cols-2">
        <div className="panel">
          <div className="label">rewards</div>
          <div className="grid cols-3">
            <Field l="hit" v={f.reward_hit} on={set("reward_hit")} step="0.1" />
            <Field l="hurt (-)" v={f.reward_hurt} on={set("reward_hurt")} step="0.05" />
            <Field l="win" v={f.reward_win} on={set("reward_win")} step="1" />
            <Field l="distance" v={f.reward_dist} on={set("reward_dist")} step="0.001" />
            <Field l="sprint hit" v={f.reward_sprint_hit} on={set("reward_sprint_hit")} step="0.05" />
            <Field l="trade penalty" v={f.reward_trade_penalty} on={set("reward_trade_penalty")} step="0.05" />
            <Field l="combo" v={f.reward_combo} on={set("reward_combo")} step="0.05" />
            <Field l="combo window" v={f.combo_window} on={set("combo_window")} />
            <Field l="combo cap" v={f.combo_cap} on={set("combo_cap")} />
            <Field l="combo drop" v={f.reward_combo_drop} on={set("reward_combo_drop")} step="0.05" />
            <Field l="drop min" v={f.combo_drop_min} on={set("combo_drop_min")} />
            <Field l="pressure" v={f.reward_combo_pressure} on={set("reward_combo_pressure")} step="0.005" />
            <Field l="aim body" v={f.reward_aim} on={set("reward_aim")} step="0.005" />
            <Field l="bad pitch" v={f.reward_bad_pitch} on={set("reward_bad_pitch")} step="0.005" />
            <Field l="chase" v={f.reward_chase} on={set("reward_chase")} step="0.005" />
            <Field l="turn aim" v={f.reward_turn_aim} on={set("reward_turn_aim")} step="0.005" />
            <Field l="aggression" v={f.reward_aggression} on={set("reward_aggression")} step="0.005" />
            <Field l="no escape" v={f.reward_no_escape} on={set("reward_no_escape")} step="0.005" />
            <Field l="combo focus" v={f.reward_combo_focus} on={set("reward_combo_focus")} step="0.05" />
            <Field l="combo tap" v={f.reward_combo_tap} on={set("reward_combo_tap")} step="0.05" />
            <Field l="opener strafe" v={f.reward_opener_strafe} on={set("reward_opener_strafe")} step="0.05" />
            <Field l="hit wtap" v={f.reward_hit_wtap} on={set("reward_hit_wtap")} step="0.05" />
            <Field l="counter hit" v={f.reward_counter_hit} on={set("reward_counter_hit")} step="0.05" />
            <Field l="hit select" v={f.reward_hit_select} on={set("reward_hit_select")} step="0.05" />
            <Field l="chase rechain" v={f.reward_chase_rechain} on={set("reward_chase_rechain")} step="0.05" />
            <Field l="chase hit select" v={f.reward_chase_hit_select} on={set("reward_chase_hit_select")} step="0.05" />
            <Field l="chase close counter" v={f.reward_chase_close_counter} on={set("reward_chase_close_counter")} step="0.05" />
            <Field l="chase counter" v={f.reward_chase_counter} on={set("reward_chase_counter")} step="0.05" />
            <Field l="spar counter" v={f.reward_spar_counter} on={set("reward_spar_counter")} step="0.05" />
            <Field l="dist floor" v={f.shaping_floor} on={set("shaping_floor")} step="0.05" />
            <Field l="hit rate gate" v={f.shaping_hit_rate} on={set("shaping_hit_rate")} step="0.5" />
            <Field l="engage gate" v={f.shaping_engage_rate} on={set("shaping_engage_rate")} step="0.01" />
            <Field l="combo5 gate" v={f.shaping_combo5_rate} on={set("shaping_combo5_rate")} step="0.01" />
            <Field l="combo12 gate" v={f.shaping_combo12_state} on={set("shaping_combo12_state")} step="0.01" />
            <Field l="sky gate" v={f.shaping_sky_frac} on={set("shaping_sky_frac")} step="0.01" />
            <Field l="ramp iters" v={f.shaping_decay_iters} on={set("shaping_decay_iters")} />
          </div>
        </div>
        <div className="panel">
          <div className="label">knockback / 1.0 = vanilla</div>
          <div className="grid cols-3">
            <Field l="horizontal" v={f.kb_h} on={set("kb_h")} step="0.01" />
            <Field l="vertical" v={f.kb_v} on={set("kb_v")} step="0.01" />
            <Field l="idle mult" v={f.kb_idle} on={set("kb_idle")} step="0.05" />
          </div>
        </div>
      </div>

      <div className="panel">
        <div className="label">humanization / randomized per match</div>
        <div className="grid cols-4">
          <Range l="cps" lo={f.cps_min} hi={f.cps_max} min={1} max={20}
                 onLo={set("cps_min")} onHi={set("cps_max")} />
          <Range l="rotation deg/t" lo={f.rot_min} hi={f.rot_max} min={5} max={260}
                 onLo={set("rot_min")} onHi={set("rot_max")} />
          <Range l="latency ticks" lo={f.delay_min} hi={f.delay_max} min={0} max={7}
                 onLo={set("delay_min")} onHi={set("delay_max")} />
          <Range l="aim smooth" lo={f.aim_smooth_min} hi={f.aim_smooth_max}
                 min={0} max={0.95} step={0.01}
                 onLo={set("aim_smooth_min")} onHi={set("aim_smooth_max")} />
        </div>
      </div>

      <div className="panel">
        <div className="controls-row">
          <button className="btn" onClick={start} disabled={busy || running}>
            start
          </button>
          <button className="btn danger" onClick={stop} disabled={busy || !running}>
            stop
          </button>
          <label className="toggle">
            <input type="checkbox" checked={f.resume} onChange={set("resume")} />
            resume latest
          </label>
          <label className="toggle">
            <input type="checkbox" checked={f.resume_same_run_as_seed} onChange={set("resume_same_run_as_seed")} />
            same run seed
          </label>
          <label className="toggle">
            <input type="checkbox" checked={f.autorestart} onChange={set("autorestart")} />
            auto restart
          </label>
          <label className="toggle">
            <input type="checkbox" checked={f.safety_stop_on_regression} onChange={set("safety_stop_on_regression")} />
            safety stop
          </label>
          {running && (
            <span className="tag green">
              {status.training.run} / {status.training.uptime_s}s
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
        <span className="val">{lo} - {hi}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={lo} onChange={onLo} />
      <input type="range" min={min} max={max} step={step} value={hi} onChange={onHi} />
    </div>
  );
}
