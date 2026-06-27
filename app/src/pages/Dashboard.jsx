import { BigChart, Sparkline } from "../components/Charts.jsx";
import { metricHealth } from "../health.js";

// dernière valeur DÉFINIE (les évals n'existent qu'une itération sur N)
const last = (arr, k) => {
  for (let i = arr.length - 1; i >= 0; i--) {
    if (arr[i][k] != null) return arr[i][k];
  }
  return null;
};
const series = (arr, k) => arr.map((m) => m[k]).filter((v) => v != null);
const maxDefined = (values) => {
  const nums = values.filter((v) => v != null).map(Number);
  return nums.length ? Math.max(...nums) : null;
};
const lastMax = (arr, keys) => maxDefined(keys.map((k) => last(arr, k)));
const seriesMax = (arr, keys) => arr.map((m) => maxDefined(keys.map((k) => m[k]))).filter((v) => v != null);

export default function Dashboard({ status, metrics }) {
  // courbes et textes colorés selon la santé (seuils : docs/GUIDE.md)
  const stat = (label, key, format) => {
    const trend = series(metrics, key);
    return <Stat label={label} value={format(last(metrics, key))}
                 trend={trend} health={metricHealth(key, trend)} />;
  };
  const kvHealth = (key) => metricHealth(key, series(metrics, key));
  const comboMaxFallbackKeys = ["combo_max", "sim_combo_max", "mirror_combo_max", "mirror_sim_combo_max"];
  const combo12FallbackKeys = ["combo12_hits", "sim_combo12_state", "mirror_combo12_hits", "mirror_sim_combo12_state"];
  const freshComboMaxTrend = series(metrics, "fresh_combo_max");
  const freshCombo12Trend = series(metrics, "fresh_combo12_state");
  const freshSparComboMaxTrend = series(metrics, "fresh_spar_combo_max");
  const freshSparCombo12Trend = series(metrics, "fresh_spar_combo12_state");
  const freshRehitComboMaxTrend = series(metrics, "fresh_rehit_combo_max");
  const freshRehitCombo12Trend = series(metrics, "fresh_rehit_combo12_state");
  const freshPressureComboMaxTrend = series(metrics, "fresh_pressure_combo_max");
  const freshPressureCombo12Trend = series(metrics, "fresh_pressure_combo12_state");
  const freshChaseComboMaxTrend = series(metrics, "fresh_chase_combo_max");
  const freshChaseCombo12Trend = series(metrics, "fresh_chase_combo12_state");
  const comboMaxTrend = freshComboMaxTrend.length ? freshComboMaxTrend : seriesMax(metrics, comboMaxFallbackKeys);
  const combo12Trend = freshCombo12Trend.length ? freshCombo12Trend : seriesMax(metrics, combo12FallbackKeys);
  const comboMaxValue = last(metrics, "fresh_combo_max") ?? lastMax(metrics, comboMaxFallbackKeys);
  const combo12Value = last(metrics, "fresh_combo12_state") ?? lastMax(metrics, combo12FallbackKeys);
  const safetyState = last(metrics, "safety_state") ?? "unknown";
  const safetyCheckpoint = last(metrics, "safety_checkpoint") ?? last(metrics, "safety_restored");

  return (
    <>
      <h2 className="title">Dashboard</h2>
      <div style={{ height: 24 }} />

      <div className="grid cols-4">
        {stat("elo", "elo", fmt)}
        {stat("reward", "reward_mean", (v) => fmt(v, 4))}
        <Stat label="safe state" value={safetyState}
              trend={[]} health={safetyHealth(safetyState)} />
        {stat("steps/s", "sps", human)}
      </div>

      <div className="grid cols-2">
        <div className="panel">
          <div className="label">elo</div>
          <BigChart values={series(metrics, "elo")} label="elo"
                    health={metricHealth("elo", series(metrics, "elo"))} />
        </div>
        <div className="panel">
          <div className="label">reward</div>
          <BigChart values={series(metrics, "reward_mean")} label="reward" color="#79d287" />
        </div>
      </div>

      <div className="grid cols-4">
        {stat("entropy", "entropy", (v) => fmt(v, 3))}
        {stat("kl", "approx_kl", (v) => fmt(v, 4))}
        {stat("clip frac", "clip_frac", pct)}
        {stat("value loss", "loss_v", (v) => fmt(v, 3))}
      </div>

      <div className="grid cols-4">
        {stat("eval vs first", "eval_first", pct)}
        {stat("eval vs past", "eval_past", pct)}
        {stat("eval vs bot", "eval_bot", pct)}
        {stat("league WR", "league_winrate", pct)}
      </div>

      <div className="grid cols-3">
        <div className="panel">
          <div className="label">gpu</div>
          {status?.gpu?.available ? (
            <>
              <KV k="device" v={status.gpu.name} hl />
              <KV k="memory" v={`${status.gpu.mem_used_gb} / ${status.gpu.mem_total_gb} GB`} />
            </>
          ) : (
            <div className="empty">no gpu</div>
          )}
        </div>
        <div className="panel">
          <div className="label">training</div>
          <KV k="state" v={status?.training?.running ? "running" : "idle"}
              hl={status?.training?.running} />
          <KV k="run" v={status?.training?.run ?? "—"} />
          <KV k="iteration" v={fmt(last(metrics, "iter"))} />
          <KV k="total steps" v={human(last(metrics, "total_steps"))} hl />
          <KV k="pool" v={fmt(last(metrics, "pool_size"))} />
          <KV k="matches" v={fmt(last(metrics, "matches"))} />
          <KV k="shaping" v={fmt(last(metrics, "shaping"), 4)} />
          <KV k="spawn gap" v={fmt(last(metrics, "spawn_gap"), 1)} />
          <KV k="sprint hits" v={pct(last(metrics, "sprint_hits"))} hl
              health={kvHealth("sprint_hits")} />
          <KV k="combo hits" v={pct(last(metrics, "combo_hits"))} hl
              health={kvHealth("combo_hits")} />
          <KV k="tap total" v={pct(last(metrics, "combo_tap_frac"))} hl
              health={kvHealth("combo_tap_frac")} />
          <KV k="s-tap" v={pct(last(metrics, "combo_s_tap_frac"))} hl
              health={kvHealth("combo_s_tap_frac")} />
          <KV k="z-tap" v={pct(last(metrics, "combo_z_tap_frac"))} hl
              health={kvHealth("combo_z_tap_frac")} />
          <KV k="rechain" v={pct(last(metrics, "rechain_hit_frac"))} hl
              health={kvHealth("rechain_hit_frac")} />
          <KV k="rechain taken" v={pct(last(metrics, "rechain_taken_frac"))} hl
              health={kvHealth("rechain_taken_frac")} />
          <KV k="counter break" v={pct(last(metrics, "counter_break_hit_frac"))} hl
              health={kvHealth("counter_break_hit_frac")} />
          <KV k="counter taken" v={pct(last(metrics, "counter_break_taken_frac"))} hl
              health={kvHealth("counter_break_taken_frac")} />
          <KV k="counter hit/min" v={fmt(last(metrics, "under_combo_counter_hit_rate"), 1)} hl
              health={kvHealth("under_combo_counter_hit_rate")} />
          <KV k="counter hit share" v={pct(last(metrics, "under_combo_counter_hit_frac"))} hl
              health={kvHealth("under_combo_counter_hit_frac")} />
          <KV k="max combo" v={fmt(comboMaxValue)} hl
              health={metricHealth("combo_max", comboMaxTrend)} />
          <KV k="12+ combo state" v={pct(combo12Value)} hl
              health={metricHealth("combo12_hits", combo12Trend)} />
          <KV k="fresh combo12" v={pct(last(metrics, "fresh_combo12_state"))} hl
              health={metricHealth("fresh_combo12_state", freshCombo12Trend)} />
          <KV k="spar combo" v={fmt(last(metrics, "fresh_spar_combo_max"))} hl
              health={metricHealth("fresh_spar_combo_max", freshSparComboMaxTrend)} />
          <KV k="spar combo12" v={pct(last(metrics, "fresh_spar_combo12_state"))} hl
              health={metricHealth("fresh_spar_combo12_state", freshSparCombo12Trend)} />
          <KV k="spar hit/min" v={fmt(last(metrics, "fresh_spar_hit_rate"), 1)} hl
              health={kvHealth("fresh_spar_hit_rate")} />
          <KV k="spar trade" v={pct(last(metrics, "fresh_spar_trade_hit_frac"))} hl
              health={kvHealth("fresh_spar_trade_hit_frac")} />
          <KV k="spar rechain" v={pct(last(metrics, "fresh_spar_rechain_hit_frac"))} hl
              health={kvHealth("fresh_spar_rechain_hit_frac")} />
          <KV k="spar counter" v={pct(last(metrics, "fresh_spar_counter_break_hit_frac"))} hl
              health={kvHealth("fresh_spar_counter_break_hit_frac")} />
          <KV k="rehit combo" v={fmt(last(metrics, "fresh_rehit_combo_max"))} hl
              health={metricHealth("fresh_rehit_combo_max", freshRehitComboMaxTrend)} />
          <KV k="rehit combo12" v={pct(last(metrics, "fresh_rehit_combo12_state"))} hl
              health={metricHealth("fresh_rehit_combo12_state", freshRehitCombo12Trend)} />
          <KV k="rehit rechain" v={pct(last(metrics, "fresh_rehit_rechain_hit_frac"))} hl
              health={kvHealth("fresh_rehit_rechain_hit_frac")} />
          <KV k="rehit taken" v={pct(last(metrics, "fresh_rehit_rechain_taken_frac"))} hl
              health={kvHealth("fresh_rehit_rechain_taken_frac")} />
          <KV k="pressure combo" v={fmt(last(metrics, "fresh_pressure_combo_max"))} hl
              health={metricHealth("fresh_pressure_combo_max", freshPressureComboMaxTrend)} />
          <KV k="pressure combo12" v={pct(last(metrics, "fresh_pressure_combo12_state"))} hl
              health={metricHealth("fresh_pressure_combo12_state", freshPressureCombo12Trend)} />
          <KV k="pressure rechain" v={pct(last(metrics, "fresh_pressure_rechain_hit_frac"))} hl
              health={kvHealth("fresh_pressure_rechain_hit_frac")} />
          <KV k="pressure taken" v={pct(last(metrics, "fresh_pressure_rechain_taken_frac"))} hl
              health={kvHealth("fresh_pressure_rechain_taken_frac")} />
          <KV k="chase combo" v={fmt(last(metrics, "fresh_chase_combo_max"))} hl
              health={metricHealth("fresh_chase_combo_max", freshChaseComboMaxTrend)} />
          <KV k="chase combo12" v={pct(last(metrics, "fresh_chase_combo12_state"))} hl
              health={metricHealth("fresh_chase_combo12_state", freshChaseCombo12Trend)} />
          <KV k="chase hit/min" v={fmt(last(metrics, "fresh_chase_hit_rate"), 1)} hl
              health={kvHealth("fresh_chase_hit_rate")} />
          <KV k="chase trade" v={pct(last(metrics, "fresh_chase_trade_hit_frac"))} hl
              health={kvHealth("fresh_chase_trade_hit_frac")} />
          <KV k="chase rechain" v={pct(last(metrics, "fresh_chase_rechain_hit_frac"))} hl
              health={kvHealth("fresh_chase_rechain_hit_frac")} />
          <KV k="chase counter" v={pct(last(metrics, "fresh_chase_counter_break_hit_frac"))} hl
              health={kvHealth("fresh_chase_counter_break_hit_frac")} />
          <KV k="recover hit/min" v={fmt(last(metrics, "counter_lane_hit_rate"), 1)} hl
              health={kvHealth("counter_lane_hit_rate")} />
          <KV k="recover rechain" v={pct(last(metrics, "counter_lane_rechain_hit_frac"))} hl
              health={kvHealth("counter_lane_rechain_hit_frac")} />
          <KV k="recover counter" v={pct(last(metrics, "counter_lane_break_hit_frac"))} hl
              health={kvHealth("counter_lane_break_hit_frac")} />
          <KV k="recover combo" v={fmt(last(metrics, "counter_lane_combo_max"))} hl
              health={metricHealth("counter_lane_combo_max",
                series(metrics, "counter_lane_combo_max"))} />
          <KV k="fresh sky" v={pct(last(metrics, "fresh_sky_frac"))} hl
              health={kvHealth("fresh_sky_frac")} />
          <KV k="fresh hit/min" v={fmt(last(metrics, "fresh_hit_rate"), 1)} hl />
          <KV k="fresh aim" v={pct(last(metrics, "fresh_aim_body"))} hl />
          <KV k="safe ckpt" v={safetyCheckpoint} hl
              health={safetyHealth(safetyState)} />
          <KV k="safety" v={safetyState} hl
              health={safetyHealth(safetyState)} />
          <KV k="engage" v={pct(last(metrics, "engage_rate"))} hl
              health={kvHealth("engage_rate")} />
          {last(metrics, "pbt_elo") && (
            <KV k="population" v={last(metrics, "pbt_elo")
              .map((e, i) => (i === last(metrics, "pbt_best") ? `★${Math.round(e)}` : Math.round(e)))
              .join(" · ")} hl />
          )}
        </div>
        <div className="panel">
          <div className="label">live</div>
          <KV k="model" v={status?.live?.model?.split(/[\\/]/).pop() ?? "—"} />
          <KV k="latency" v={status?.live?.latency_ms != null
              ? `${status.live.latency_ms} ms` : "—"} />
          <KV k="ticks" v={fmt(status?.live?.tick)} />
        </div>
      </div>
    </>
  );
}

function Stat({ label, value, trend, health }) {
  return (
    <div className="panel">
      <div className="label">{label}</div>
      <div className={"stat-value" + (health ? " " + health : "")}>
        {value ?? "—"}
      </div>
      <div className="stat-trend">
        <Sparkline values={trend?.slice(-80)} health={health} />
      </div>
    </div>
  );
}

function KV({ k, v, hl, health }) {
  return (
    <div className="kv">
      <span className="k">{k}</span>
      <span className={"v" + (health ? " " + health : hl ? " hl" : "")}>
        {v ?? "—"}
      </span>
    </div>
  );
}

const fmt = (v, d = 0) => (v == null ? null : Number(v).toFixed(d));
const pct = (v) => (v == null ? null : `${(v * 100).toFixed(0)}%`);
const safetyHealth = (state) =>
  state === "safe" ? "ok"
    : state === "stop" ? "bad"
    : state === "unknown" ? null
    : "watch";
const human = (v) =>
  v == null ? null
    : v >= 1e9 ? `${(v / 1e9).toFixed(2)}B`
    : v >= 1e6 ? `${(v / 1e6).toFixed(2)}M`
    : `${(v / 1e3).toFixed(0)}k`;
