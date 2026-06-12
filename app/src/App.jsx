import { useEffect, useMemo, useState } from "react";
import { api, connectEvents } from "./api.js";
import Logo from "./components/Logo.jsx";
import Starfield from "./components/Starfield.jsx";
import { usePersistentState } from "./persistence.mjs";
import Dashboard from "./pages/Dashboard.jsx";
import Live from "./pages/Live.jsx";
import Models from "./pages/Models.jsx";
import Training from "./pages/Training.jsx";

const PAGES = [
  { id: "dashboard", label: "Dashboard" },
  { id: "training", label: "Training" },
  { id: "models", label: "Models" },
  { id: "live", label: "Live" },
];

export default function App() {
  const [page, setPage] = usePersistentState("judas:app:page", "dashboard");
  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState(null);
  const [metrics, setMetrics] = useState([]);
  const activePage = PAGES.some((p) => p.id === page) ? page : "dashboard";

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
            <Logo />
            <h1>Judas</h1>
          </div>
          <nav>
            {PAGES.map((p) => (
              <button key={p.id}
                      className={"navlink" + (activePage === p.id ? " active" : "")}
                      onClick={() => setPage(p.id)}>
                {p.label}
              </button>
            ))}
          </nav>
          <div className="foot">
            <div className={"statusline" + (connected ? " ok" : " warn")}>
              {connected ? "daemon online" : "daemon offline"}
            </div>
            <div className={"statusline" + (status?.gpu?.available ? " ok" : "")}>
              {status?.gpu?.available ? status.gpu.name : "no gpu"}
            </div>
            <div className={"statusline" + (status?.training?.running ? " ok" : "")}>
              {status?.training?.running ? "training" : "idle"}
            </div>
          </div>
        </aside>
        <main className="view">
          {activePage === "dashboard" && <Dashboard {...ctx} />}
          {activePage === "training" && <Training {...ctx} />}
          {activePage === "models" && <Models {...ctx} />}
          {activePage === "live" && <Live {...ctx} />}
        </main>
      </div>
    </>
  );
}
