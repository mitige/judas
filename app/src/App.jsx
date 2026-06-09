import { useEffect, useMemo, useState } from "react";
import { api, connectEvents } from "./api.js";
import Starfield from "./components/Starfield.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import Live from "./pages/Live.jsx";
import Models from "./pages/Models.jsx";
import Training from "./pages/Training.jsx";

const PAGES = [
  { id: "dashboard", label: "Dashboard" },
  { id: "training", label: "Entraînement" },
  { id: "models", label: "Modèles" },
  { id: "live", label: "Live" },
];

export default function App() {
  const [page, setPage] = useState("dashboard");
  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState(null);
  const [metrics, setMetrics] = useState([]);

  // flux temps réel du daemon
  useEffect(() => connectEvents((msg) => {
    if (msg.t === "status") setStatus(msg);
  }, setConnected), []);

  // historique des métriques (poll léger, le flux ne porte que la dernière)
  useEffect(() => {
    let alive = true;
    const load = () =>
      api.metrics().then((m) => alive && setMetrics(m)).catch(() => {});
    load();
    const id = setInterval(load, 5000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  const ctx = useMemo(() => ({ status, metrics, connected }),
                      [status, metrics, connected]);

  return (
    <>
      <Starfield />
      <div className="shell">
        <aside className="rail">
          <div className="brand">
            <span className="orbit" />
            <h1>Judas</h1>
          </div>
          <nav>
            {PAGES.map((p) => (
              <button key={p.id}
                      className={"navlink" + (page === p.id ? " active" : "")}
                      onClick={() => setPage(p.id)}>
                <span className="dot" />
                {p.label}
              </button>
            ))}
          </nav>
          <div className="foot">
            <div className="statusline">
              <span className={"beacon" + (connected ? " on" : " warn")} />
              {connected ? "daemon connecté" : "daemon hors ligne"}
            </div>
            <div className="statusline">
              <span className={"beacon" + (status?.gpu?.available ? " on" : "")} />
              {status?.gpu?.available ? status.gpu.name : "GPU absent"}
            </div>
            <div className="statusline">
              <span className={"beacon" + (status?.training?.running ? " on" : "")} />
              {status?.training?.running ? "entraînement actif" : "entraînement arrêté"}
            </div>
          </div>
        </aside>
        <main className="view">
          {page === "dashboard" && <Dashboard {...ctx} />}
          {page === "training" && <Training {...ctx} />}
          {page === "models" && <Models {...ctx} />}
          {page === "live" && <Live {...ctx} />}
        </main>
      </div>
    </>
  );
}
