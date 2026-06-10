import { useEffect, useState } from "react";
import { api } from "../api.js";

export default function Models() {
  const [data, setData] = useState({ runs: [], exported: [] });
  const [busy, setBusy] = useState(null);

  const refresh = () => api.models().then(setData).catch(() => {});
  useEffect(() => { refresh(); }, []);

  const doExport = async (run, ckpt) => {
    setBusy(ckpt);
    try {
      await api.exportModel(`runs/${run}/${ckpt}`,
                            `models/${run}-${ckpt.replace(".pt", "")}.pts`);
      refresh();
    } catch {}
    setBusy(null);
  };

  return (
    <>
      <h2 className="title">Models</h2>
      <div style={{ height: 24 }} />

      <div className="panel">
        <div className="label">runs</div>
        {data.runs.length === 0 && <div className="empty">no runs</div>}
        {data.runs.map((r) => (
          <div key={r.name}>
            <div className="kv">
              <span className="k">{r.name}</span>
              <span className="v hl">
                {r.last_metrics
                  ? `elo ${r.last_metrics.elo} · iter ${r.last_metrics.iter}` : ""}
              </span>
            </div>
            <table className="list">
              <tbody>
                {r.checkpoints.slice(-6).reverse().map((c) => (
                  <tr key={c}>
                    <td>{c}</td>
                    <td style={{ textAlign: "right" }}>
                      <button className="btn ghost" disabled={busy === c}
                              onClick={() => doExport(r.name, c)}>
                        {busy === c ? "…" : "export"}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <hr className="sep" />
          </div>
        ))}
      </div>

      <div className="panel">
        <div className="label">exported</div>
        {data.exported.length === 0 && <div className="empty">no exports</div>}
        {data.exported.length > 0 && (
          <table className="list">
            <thead>
              <tr><th>name</th><th>history</th><th>iter</th><th>path</th></tr>
            </thead>
            <tbody>
              {data.exported.map((m) => (
                <tr key={m.path}>
                  <td>{m.name}</td>
                  <td>{m.history ?? "—"}</td>
                  <td>{m.iter ?? "—"}</td>
                  <td>{m.path}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </>
  );
}
