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
      await api.exportModel(`runs/${run}/${ckpt}`, `models/${run}-${ckpt.replace(".pt", "")}.pts`);
      refresh();
    } catch {}
    setBusy(null);
  };

  return (
    <>
      <div className="eyebrow">checkpoints · torchscript</div>
      <h2 className="title">Constellation de <em>modèles</em></h2>
      <p className="subtitle">
        Checkpoints d&apos;entraînement et modèles exportés prêts pour le live.
      </p>

      <div className="panel">
        <div className="label">runs</div>
        {data.runs.length === 0 && <div className="empty">aucun run pour l&apos;instant</div>}
        {data.runs.map((r) => (
          <div key={r.name}>
            <div className="kv">
              <span className="k">{r.name}</span>
              <span className="v hl">
                {r.last_metrics ? `elo ${r.last_metrics.elo} · iter ${r.last_metrics.iter}` : ""}
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
                        {busy === c ? "export…" : "exporter"}
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
        <div className="label">modèles exportés</div>
        {data.exported.length === 0 && <div className="empty">aucun export</div>}
        {data.exported.length > 0 && (
          <table className="list">
            <thead>
              <tr><th>nom</th><th>historique</th><th>itération</th><th>chemin</th></tr>
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
