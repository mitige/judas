import { useEffect, useRef, useState } from "react";
import { api, connectArena } from "./api.js";
import Arena3D from "./components/Arena3D.jsx";
import Starfield from "./components/Starfield.jsx";

export default function App() {
  const stateRef = useRef({ cur: null, prev: null, tCur: 0, tPrev: 0, frame: 0 });
  const [connected, setConnected] = useState(false);
  const [hud, setHud] = useState(null);          // copie basse fréquence pour le HUD
  const [victory, setVictory] = useState(null);  // {winner} affiché brièvement
  const [models, setModels] = useState([]);
  const [sel, setSel] = useState({ a: "", b: "" });
  const [params, setParams] = useState({ cps: 12, rot: 40, arena: 18,
                                         target: 100, sample: true });
  const [speed, setSpeed] = useState(1);
  const [status, setStatus] = useState(null);
  const [err, setErr] = useState(null);

  // ---------------------------------------------------------------- flux WS
  useEffect(() => connectArena((msg) => {
    if (msg.t === "tick" && msg.ready) {
      const S = stateRef.current;
      S.prev = S.cur; S.tPrev = S.tCur;
      S.cur = msg; S.tCur = performance.now();
      setHud(msg);
      if (msg.done) {
        setVictory({ winner: msg.winner });
        setTimeout(() => setVictory(null), 1800);
      }
    } else if (msg.t === "status") {
      setStatus(msg);
    }
  }, setConnected), []);

  // -------------------------------------------------------------- modèles
  const refreshModels = () => api.models().then((d) => {
    const list = [];
    for (const r of d.runs) {
      if (r.latest) list.push({ label: `${r.name} · latest`, path: `runs/${r.name}/latest.pt` });
      for (const c of r.checkpoints.slice(-8))
        list.push({ label: `${r.name} · ${c}`, path: `runs/${r.name}/${c}` });
    }
    for (const m of d.exported) list.push({ label: `export · ${m.name}`, path: m.path });
    setModels(list);
    setSel((s) => ({
      a: s.a || list[0]?.path || "",
      b: s.b || list[0]?.path || "",
    }));
  }).catch(() => {});
  useEffect(() => { refreshModels(); }, []);

  // ------------------------------------------------------------- contrôles
  const load = async () => {
    setErr(null);
    try {
      const st = await api.arenaLoad({
        model_a: sel.a, model_b: sel.b,
        cps: +params.cps, rot_speed: +params.rot,
        arena_size: +params.arena, target_hits: +params.target,
        sample: params.sample,
      });
      setStatus(st);
    } catch (e) { setErr(String(e.message || e)); }
  };
  const control = (body) =>
    api.arenaControl(body).then(setStatus).catch((e) => setErr(String(e.message || e)));

  useEffect(() => {
    const id = setTimeout(() => control({ speed }), 200);
    return () => clearTimeout(id);
  }, [speed]);

  const running = hud?.running ?? status?.running;
  const score = hud?.players?.map((p) => p.hits) ?? [0, 0];
  const names = status?.models ?? [null, null];
  const target = hud?.target_hits ?? status?.target_hits ?? 100;

  return (
    <>
      <Starfield />
      <div className="viz-shell">
        {/* ------------------------------------------------ viewport 3D */}
        <div className="viewport-wrap">
          <Arena3D stateRef={stateRef} />

          <div className="hud-score">
            <div>
              <div className="num a">{String(score[0]).padStart(2, "0")}</div>
              <div className="who">{names[0] ?? "—"}</div>
            </div>
            <div className="vs">contre</div>
            <div>
              <div className="num b">{String(score[1]).padStart(2, "0")}</div>
              <div className="who">{names[1] ?? "—"}</div>
            </div>
          </div>

          <div className="hud-foot">
            <span className={connected ? "on" : ""}>
              {connected ? "● flux connecté" : "○ daemon hors ligne"}
            </span>
            <span>tick {hud?.tick ?? "—"}</span>
            <span>{hud?.step_ms != null ? `${hud.step_ms} ms/step` : ""}</span>
            <span>{running ? `vitesse ×${hud?.speed ?? speed}` : "en pause"}</span>
          </div>

          {victory && (
            <div className="hud-victory">
              <div className={"word " + (victory.winner === 0 ? "a" : victory.winner === 1 ? "b" : "d")}>
                {victory.winner === -1 ? "Égalité" : "Victoire"}
              </div>
              <div className="detail">
                {victory.winner === 0 ? names[0] : victory.winner === 1 ? names[1] : "aucun vainqueur"}
              </div>
            </div>
          )}
        </div>

        {/* ------------------------------------------------- rail droit */}
        <aside className="vrail">
          <div className="vbrand">
            <span className="orbit" />
            <h1>Judas</h1>
            <span className="sub">arène</span>
          </div>

          <div className="panel">
            <div className="label">combattants</div>
            <div className="field" style={{ marginBottom: 10 }}>
              <label style={{ color: "var(--ice)" }}>modèle A</label>
              <select value={sel.a} onChange={(e) => setSel({ ...sel, a: e.target.value })}>
                {models.length === 0 && <option value="">— aucun modèle —</option>}
                {models.map((m) => <option key={"a" + m.path} value={m.path}>{m.label}</option>)}
              </select>
            </div>
            <div className="field">
              <label style={{ color: "var(--ember)" }}>modèle B</label>
              <select value={sel.b} onChange={(e) => setSel({ ...sel, b: e.target.value })}>
                {models.length === 0 && <option value="">— aucun modèle —</option>}
                {models.map((m) => <option key={"b" + m.path} value={m.path}>{m.label}</option>)}
              </select>
            </div>
            <hr className="sep" />
            <div className="grid cols-2">
              <Field label="cps" value={params.cps}
                     onChange={(e) => setParams({ ...params, cps: e.target.value })} />
              <Field label="rotation °/t" value={params.rot}
                     onChange={(e) => setParams({ ...params, rot: e.target.value })} />
              <Field label="arène" value={params.arena}
                     onChange={(e) => setParams({ ...params, arena: e.target.value })} />
              <Field label="hits cible" value={params.target}
                     onChange={(e) => setParams({ ...params, target: e.target.value })} />
            </div>
            <hr className="sep" />
            <label className="toggle">
              <input type="checkbox" checked={params.sample}
                     onChange={(e) => setParams({ ...params, sample: e.target.checked })} />
              échantillonner (sinon déterministe)
            </label>
            <hr className="sep" />
            <div className="controls-row">
              <button className="btn" onClick={load} disabled={!sel.a || !sel.b}>
                charger
              </button>
              <button className="btn ghost" onClick={refreshModels}>↻</button>
              {err && <span className="tag" style={{ color: "var(--ember)" }}>{err}</span>}
            </div>
          </div>

          <div className="panel">
            <div className="label">match</div>
            <div className="controls-row" style={{ marginBottom: 14 }}>
              <button className="btn" onClick={() => control({ running: !running })}
                      disabled={!(status?.ready)}>
                {running ? "❚❚ pause" : "▶ lancer"}
              </button>
              <button className="btn ghost" onClick={() => control({ reset: true })}
                      disabled={!(status?.ready)}>
                réinitialiser
              </button>
            </div>
            <div className="slider">
              <div className="row">
                <span className="field"><label>vitesse</label></span>
                <span className="val">×{speed}</span>
              </div>
              <input type="range" min={0.25} max={16} step={0.25} value={speed}
                     onChange={(e) => setSpeed(+e.target.value)} />
            </div>
          </div>

          <div className="panel">
            <div className="label">score · premier à {target}</div>
            <ScoreBar label={names[0] ?? "A"} cls="a" value={score[0]} max={target} />
            <ScoreBar label={names[1] ?? "B"} cls="b" value={score[1]} max={target} />
          </div>

          <div className="panel">
            <div className="label">session</div>
            <Kv k="matchs joués" v={hud?.matches ?? status?.matches ?? 0} />
            <Kv k="victoires A" v={(hud?.wins ?? status?.wins ?? [0, 0])[0]} hl />
            <Kv k="victoires B" v={(hud?.wins ?? status?.wins ?? [0, 0])[1]} />
            <Kv k="égalités" v={hud?.draws ?? status?.draws ?? 0} />
          </div>
        </aside>
      </div>
    </>
  );
}

function ScoreBar({ label, cls, value, max }) {
  return (
    <div className="scorebar" style={{ marginBottom: 12 }}>
      <div className="meta">
        <span>{label}</span>
        <span>{value} / {max}</span>
      </div>
      <div className="track">
        <div className={"fill " + cls}
             style={{ width: `${Math.min(100, (value / max) * 100)}%` }} />
      </div>
    </div>
  );
}

function Field({ label, value, onChange }) {
  return (
    <div className="field">
      <label>{label}</label>
      <input type="number" value={value} onChange={onChange} />
    </div>
  );
}

function Kv({ k, v, hl }) {
  return (
    <div className="kv">
      <span className="k">{k}</span>
      <span className={"v" + (hl ? " hl" : "")}>{v}</span>
    </div>
  );
}
