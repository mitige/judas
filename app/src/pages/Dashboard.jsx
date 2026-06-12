import { BigChart, Sparkline } from "../components/Charts.jsx";

// dernière valeur DÉFINIE (les évals n'existent qu'une itération sur N)
const last = (arr, k) => {
  for (let i = arr.length - 1; i >= 0; i--) {
    if (arr[i][k] != null) return arr[i][k];
  }
  return null;
};
const series = (arr, k) => arr.map((m) => m[k]).filter((v) => v != null);

export default function Dashboard({ status, metrics }) {
  return (
    <>
      <h2 className="title">Dashboard</h2>
      <div style={{ height: 24 }} />

      <div className="grid cols-4">
        <Stat label="elo" value={fmt(last(metrics, "elo"))} trend={series(metrics, "elo")} />
        <Stat label="reward" value={fmt(last(metrics, "reward_mean"), 4)}
              trend={series(metrics, "reward_mean")} />
        <Stat label="win rate" value={pct(last(metrics, "league_winrate"))}
              trend={series(metrics, "league_winrate")} />
        <Stat label="steps/s" value={human(last(metrics, "sps"))}
              trend={series(metrics, "sps")} />
      </div>

      <div className="grid cols-2">
        <div className="panel">
          <div className="label">elo</div>
          <BigChart values={series(metrics, "elo")} label="elo" />
        </div>
        <div className="panel">
          <div className="label">reward</div>
          <BigChart values={series(metrics, "reward_mean")} label="reward" color="#7ee787" />
        </div>
      </div>

      <div className="grid cols-4">
        <Stat label="entropy" value={fmt(last(metrics, "entropy"), 3)}
              trend={series(metrics, "entropy")} />
        <Stat label="kl" value={fmt(last(metrics, "approx_kl"), 4)}
              trend={series(metrics, "approx_kl")} />
        <Stat label="clip frac" value={pct(last(metrics, "clip_frac"))}
              trend={series(metrics, "clip_frac")} />
        <Stat label="value loss" value={fmt(last(metrics, "loss_v"), 3)}
              trend={series(metrics, "loss_v")} />
      </div>

      <div className="grid cols-4">
        <Stat label="eval vs first" value={pct(last(metrics, "eval_first"))}
              trend={series(metrics, "eval_first")} />
        <Stat label="eval vs past" value={pct(last(metrics, "eval_past"))}
              trend={series(metrics, "eval_past")} />
        <Stat label="eval vs bot" value={pct(last(metrics, "eval_bot"))}
              trend={series(metrics, "eval_bot")} />
        <Stat label="hit rate /min" value={fmt(last(metrics, "hit_rate"), 1)}
              trend={series(metrics, "hit_rate")} />
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
          <KV k="sprint hits" v={pct(last(metrics, "sprint_hits"))} hl />
          <KV k="combo hits" v={pct(last(metrics, "combo_hits"))} hl />
          <KV k="engage" v={pct(last(metrics, "engage_rate"))} hl />
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

function Stat({ label, value, trend }) {
  return (
    <div className="panel">
      <div className="label">{label}</div>
      <div className="stat-value">{value ?? "—"}</div>
      <div className="stat-trend">
        <Sparkline values={trend?.slice(-80)} />
      </div>
    </div>
  );
}

function KV({ k, v, hl }) {
  return (
    <div className="kv">
      <span className="k">{k}</span>
      <span className={"v" + (hl ? " hl" : "")}>{v ?? "—"}</span>
    </div>
  );
}

const fmt = (v, d = 0) => (v == null ? null : Number(v).toFixed(d));
const pct = (v) => (v == null ? null : `${(v * 100).toFixed(0)}%`);
const human = (v) =>
  v == null ? null
    : v >= 1e9 ? `${(v / 1e9).toFixed(2)}B`
    : v >= 1e6 ? `${(v / 1e6).toFixed(2)}M`
    : `${(v / 1e3).toFixed(0)}k`;
