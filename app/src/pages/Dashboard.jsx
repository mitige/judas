import { BigChart, Sparkline } from "../components/Charts.jsx";

const last = (arr, k) => (arr.length ? arr[arr.length - 1][k] : null);
const series = (arr, k) => arr.map((m) => m[k]).filter((v) => v != null);

export default function Dashboard({ status, metrics }) {
  const elo = series(metrics, "elo");
  const reward = series(metrics, "reward_mean");
  const wr = series(metrics, "league_winrate");
  const sps = series(metrics, "sps");

  return (
    <>
      <div className="eyebrow">vue d&apos;ensemble</div>
      <h2 className="title">Salle de <em>contrôle</em></h2>
      <p className="subtitle">
        État du système — entraînement, classement et performance du simulateur.
      </p>

      <div className="grid cols-4">
        <Stat label="ELO learner" value={fmt(last(metrics, "elo"))} trend={elo} />
        <Stat label="Reward / step" value={fmt(last(metrics, "reward_mean"), 4)} trend={reward} />
        <Stat label="Winrate league" value={pct(last(metrics, "league_winrate"))} trend={wr} />
        <Stat label="Agent-steps / s" value={human(last(metrics, "sps"))} trend={sps} />
      </div>

      <div className="grid cols-2">
        <div className="panel">
          <div className="label">trajectoire elo</div>
          <BigChart values={elo} label="elo" />
        </div>
        <div className="panel">
          <div className="label">reward moyen</div>
          <BigChart values={reward} label="reward" color="#7ee787" />
        </div>
      </div>

      <div className="grid cols-3">
        <div className="panel">
          <div className="label">gpu</div>
          {status?.gpu?.available ? (
            <>
              <KV k="carte" v={status.gpu.name} hl />
              <KV k="mémoire" v={`${status.gpu.mem_used_gb} / ${status.gpu.mem_total_gb} Go`} />
            </>
          ) : (
            <div className="empty">aucun gpu détecté</div>
          )}
        </div>
        <div className="panel">
          <div className="label">entraînement</div>
          <KV k="état" v={status?.training?.running ? "actif" : "arrêté"}
              hl={status?.training?.running} />
          <KV k="run" v={status?.training?.run ?? "—"} />
          <KV k="itération" v={fmt(last(metrics, "iter"))} />
          <KV k="pool" v={fmt(last(metrics, "pool_size"))} />
        </div>
        <div className="panel">
          <div className="label">bot live</div>
          <KV k="modèle" v={status?.live?.model?.split(/[\\/]/).pop() ?? "—"} />
          <KV k="latence" v={status?.live?.latency_ms != null
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
const pct = (v) => (v == null ? null : `${(v * 100).toFixed(0)} %`);
const human = (v) =>
  v == null ? null : v >= 1e6 ? `${(v / 1e6).toFixed(2)} M` : `${(v / 1e3).toFixed(0)} k`;
