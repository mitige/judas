import { useEffect, useRef, useState } from "react";
import { api, connectArena } from "./api.js";
import Arena3D from "./components/Arena3D.jsx";
import Logo from "./components/Logo.jsx";
import { usePersistentState } from "./persistence.mjs";
import Starfield from "./components/Starfield.jsx";

export default function App() {
  const stateRef = useRef({ cur: null, prev: null, tCur: 0, tPrev: 0, frame: 0, resetSeq: 0 });
  const loadingRef = useRef(false);
  const [connected, setConnected] = useState(false);
  const [hud, setHud] = useState(null);          // copie basse fréquence pour le HUD
  const [victory, setVictory] = useState(null);  // {winner} affiché brièvement
  const [models, setModels] = useState([]);
  const [sel, setSel] = usePersistentState("judas:viz:fighters:v4", { a: "", b: "" });
  const [params, setParams] = usePersistentState("judas:viz:params:v3", {
    cps: 10, rot: 190, arena: 40, spawn_gap: 8, target: 50, sample: true,
    kb_h: 0.92, kb_v: 0.90, kb_idle: 0.6, aim_smooth: 0.02,
  });
  const [speed, setSpeed] = usePersistentState("judas:viz:speed", 1);
  const [status, setStatus] = useState(null);
  const [err, setErr] = useState(null);
  const [loading, setLoading] = useState(false);

  const clearArenaFrame = () => {
    const resetSeq = (stateRef.current.resetSeq || 0) + 1;
    stateRef.current = { cur: null, prev: null, tCur: 0, tPrev: 0, frame: 0, resetSeq };
    setHud(null);
    setVictory(null);
  };

  // ---------------------------------------------------------------- flux WS
  useEffect(() => connectArena((msg) => {
    if (loadingRef.current) return;
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
    const list = [
      { label: "script · chase bot", path: "__chase_bot__" },
      { label: "script · combo pad", path: "__combo_pad__" },
      { label: "script · combo spar", path: "__combo_spar__" },
    ];
    const isComboSafeRun = (r) =>
      (r.name === "combo_god_recovery_kb092_combo12"
        || r.name === "combo_god_leaderboard10_combo12"
        || r.name === "combo_god_countertap96_combo12"
        || r.name === "combo_god_directpad_lock_combo12") && r.safe;
    for (const r of d.runs) {
      if (isComboSafeRun(r)) {
        list.push({
          label: `${r.name} · safe latest`,
          path: `runs/${r.name}/safe_latest.pt`,
        });
        continue;
      }
      if (r.best) list.push({ label: `${r.name} · best`, path: `runs/${r.name}/best.pt` });
      if (r.safe) list.push({ label: `${r.name} · safe latest`, path: `runs/${r.name}/safe_latest.pt` });
      if (r.latest) list.push({ label: `${r.name} · latest`, path: `runs/${r.name}/latest.pt` });
      for (const c of r.checkpoints.slice(-8))
        list.push({ label: `${r.name} · ${c}`, path: `runs/${r.name}/${c}` });
    }
    for (const m of d.exported) {
      if (m.export_fresh === false) continue;
      list.push({ label: `export · ${m.name}`, path: m.path });
    }
    setModels(list);
    const isLeaderboardExport = (m) =>
      m?.path?.includes("combo_god_leaderboard10_combo12-safe_latest");
    const isRecoveryExport = (m) =>
      m?.path?.includes("combo_god_recovery_kb092_combo12-safe_latest");
    const isCountertapExport = (m) =>
      m?.path?.includes("combo_god_countertap96_combo12-safe_latest");
    const isRecoverySafeCheckpoint = (m) =>
      m?.path === "runs/combo_god_recovery_kb092_combo12/safe_latest.pt";
    const isLeaderboardSafeCheckpoint = (m) =>
      m?.path === "runs/combo_god_leaderboard10_combo12/safe_latest.pt";
    const isLegacySafeExport = (m) =>
      m?.path?.includes("combo_god_directpad_lock_combo12-safe_latest");
    const isCountertapSafeCheckpoint = (m) =>
      m?.path === "runs/combo_god_countertap96_combo12/safe_latest.pt";
    const isLegacySafeCheckpoint = (m) =>
      m?.path === "runs/combo_god_directpad_lock_combo12/safe_latest.pt";
    const preferredA =
      list.find(isRecoveryExport) ||
      list.find(isRecoverySafeCheckpoint) ||
      list.find(isLeaderboardExport) ||
      list.find(isLeaderboardSafeCheckpoint) ||
      list.find(isCountertapExport) ||
      list.find(isCountertapSafeCheckpoint) ||
      list.find(isLegacySafeExport) ||
      list.find(isLegacySafeCheckpoint) ||
      list.find((m) => (
        m.path !== "__chase_bot__"
        && m.path !== "__combo_pad__"
        && m.path !== "__combo_spar__"
      )) ||
      list[0];
    const preferredB =
      list.find((m) => m.path === "__combo_spar__") ||
      list.find((m) => m.path === "__combo_pad__") ||
      list.find((m) => m.path !== preferredA?.path) ||
      list[0];
    setSel((s) => {
      const exists = (path) => path && list.some((m) => m.path === path);
      const a = exists(s.a) ? s.a : preferredA?.path || "";
      const b = exists(s.b) ? s.b : preferredB?.path || a || "";
      return { a, b };
    });
  }).catch(() => {});
  useEffect(() => { refreshModels(); }, []);

  // ------------------------------------------------------------- contrôles
  const load = async () => {
    if (loading) return;
    setErr(null);
    loadingRef.current = true;
    setLoading(true);
    setStatus((cur) => (cur ? { ...cur, running: false, ready: false } : cur));
    clearArenaFrame();
    try {
      try {
        await api.arenaControl({ running: false });
      } catch {}
      const st = await api.arenaLoad({
        model_a: sel.a, model_b: sel.b,
        cps: +params.cps, rot_speed: +params.rot,
        arena_size: +params.arena, spawn_gap: +params.spawn_gap,
        target_hits: +params.target,
        sample: params.sample,
        kb_h: +params.kb_h, kb_v: +params.kb_v, kb_idle: +params.kb_idle,
        aim_smooth: +params.aim_smooth,
      });
      setStatus(st);
      clearArenaFrame();
    } catch (e) {
      setErr(String(e.message || e));
    } finally {
      loadingRef.current = false;
      setLoading(false);
    }
  };
  const control = (body) => {
    if (loadingRef.current) return Promise.resolve(status);
    return api.arenaControl(body).then(setStatus).catch((e) => setErr(String(e.message || e)));
  };

  useEffect(() => {
    if (loading) return;
    const id = setTimeout(() => control({ speed }), 200);
    return () => clearTimeout(id);
  }, [speed, loading]);

  const running = loading ? false : (status?.running ?? hud?.running);
  const score = hud?.players?.map((p) => p.hits) ?? [0, 0];
  const combos = hud?.players?.map((p) => p.combo ?? 0) ?? status?.combo ?? [0, 0];
  const maxCombos = hud?.players?.map((p) => p.max_combo ?? 0)
    ?? status?.max_combo ?? [0, 0];
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
              <div className="combo-line">combo {combos[0]} / best {maxCombos[0]}</div>
              <div className="who">{names[0] ?? "—"}</div>
            </div>
            <div className="vs">vs</div>
            <div>
              <div className="num b">{String(score[1]).padStart(2, "0")}</div>
              <div className="combo-line">combo {combos[1]} / best {maxCombos[1]}</div>
              <div className="who">{names[1] ?? "—"}</div>
            </div>
          </div>

          <div className="hud-foot">
            <span className={connected ? "on" : "warn"}>
              {connected ? "connected" : "daemon offline"}
            </span>
            <span>tick {hud?.tick ?? "—"}</span>
            <span>{hud?.step_ms != null ? `${hud.step_ms} ms/step` : ""}</span>
            <span>{running ? `×${hud?.speed ?? speed}` : "paused"}</span>
          </div>

          {victory && (
            <div className="hud-victory">
              <div className={"word " + (victory.winner === 0 ? "a" : victory.winner === 1 ? "b" : "d")}>
                {victory.winner === -1 ? "Draw" : "Victory"}
              </div>
              <div className="detail">
                {victory.winner === 0 ? names[0] : victory.winner === 1 ? names[1] : "no winner"}
              </div>
            </div>
          )}
        </div>

        {/* ------------------------------------------------- rail droit */}
        <aside className="vrail">
          <div className="vbrand">
            <Logo size={20} />
            <h1>Judas</h1>
            <span className="sub">arena</span>
          </div>

          <div className="panel">
            <div className="label">fighters</div>
            <div className="field" style={{ marginBottom: 10 }}>
              <label style={{ color: "var(--ice)" }}>model A</label>
              <select value={sel.a} onChange={(e) => setSel({ ...sel, a: e.target.value })}>
                {models.length === 0 && <option value="">—</option>}
                {models.map((m) => <option key={"a" + m.path} value={m.path}>{m.label}</option>)}
              </select>
            </div>
            <div className="field">
              <label style={{ color: "var(--ember)" }}>model B</label>
              <select value={sel.b} onChange={(e) => setSel({ ...sel, b: e.target.value })}>
                {models.length === 0 && <option value="">—</option>}
                {models.map((m) => <option key={"b" + m.path} value={m.path}>{m.label}</option>)}
              </select>
            </div>
            <hr className="sep" />
            <div className="grid cols-2">
              <Field label="cps" value={params.cps}
                     onChange={(e) => setParams({ ...params, cps: e.target.value })} />
              <Field label="rotation °/t" value={params.rot}
                     onChange={(e) => setParams({ ...params, rot: e.target.value })} />
              <Field label="arena" value={params.arena}
                     onChange={(e) => setParams({ ...params, arena: e.target.value })} />
              <Field label="spawn gap" value={params.spawn_gap}
                     onChange={(e) => setParams({ ...params, spawn_gap: e.target.value })} />
              <Field label="target hits" value={params.target}
                     onChange={(e) => setParams({ ...params, target: e.target.value })} />
              <Field label="kb horizontal" value={params.kb_h}
                     onChange={(e) => setParams({ ...params, kb_h: e.target.value })} />
              <Field label="kb vertical" value={params.kb_v}
                     onChange={(e) => setParams({ ...params, kb_v: e.target.value })} />
              <Field label="kb idle" value={params.kb_idle}
                     onChange={(e) => setParams({ ...params, kb_idle: e.target.value })} />
              <Field label="aim smooth" value={params.aim_smooth}
                     onChange={(e) => setParams({ ...params, aim_smooth: e.target.value })} />
            </div>
            <hr className="sep" />
            <label className="toggle">
              <input type="checkbox" checked={params.sample}
                     onChange={(e) => setParams({ ...params, sample: e.target.checked })} />
              sample actions
            </label>
            <hr className="sep" />
            <div className="controls-row">
              <button className="btn" onClick={load} disabled={!sel.a || !sel.b || loading}>
                {loading ? "loading" : "load"}
              </button>
              <button className="btn ghost" onClick={refreshModels}>↻</button>
              {err && <span className="tag" style={{ color: "var(--ember)" }}>{err}</span>}
            </div>
          </div>

          <div className="panel">
            <div className="label">match</div>
            <div className="controls-row" style={{ marginBottom: 14 }}>
              <button className="btn" onClick={() => control({ running: !running })}
                      disabled={loading || !(status?.ready)}>
                {running ? "❚❚ pause" : "▶ play"}
              </button>
              <button className="btn ghost" onClick={() => control({ reset: true })}
                      disabled={loading || !(status?.ready)}>
                reset
              </button>
            </div>
            <div className="slider">
              <div className="row">
                <span className="field"><label>speed</label></span>
                <span className="val">×{speed}</span>
              </div>
              <input type="range" min={0.25} max={16} step={0.25} value={speed}
                     onChange={(e) => setSpeed(+e.target.value)} />
            </div>
          </div>

          <div className="panel">
            <div className="label">score · first to {target}</div>
            <ScoreBar label={names[0] ?? "A"} cls="a" value={score[0]} max={target} />
            <ScoreBar label={names[1] ?? "B"} cls="b" value={score[1]} max={target} />
          </div>

          <div className="panel">
            <div className="label">session</div>
            <Kv k="matches" v={hud?.matches ?? status?.matches ?? 0} />
            <Kv k="wins A" v={(hud?.wins ?? status?.wins ?? [0, 0])[0]} hl />
            <Kv k="wins B" v={(hud?.wins ?? status?.wins ?? [0, 0])[1]} />
            <Kv k="draws" v={hud?.draws ?? status?.draws ?? 0} />
            <Kv k="clicks A" v={(hud?.clicks ?? status?.clicks ?? [0, 0])[0]} hl />
            <Kv k="clicks B" v={(hud?.clicks ?? status?.clicks ?? [0, 0])[1]} />
            <Kv k="combo A" v={combos[0]} hl />
            <Kv k="combo B" v={combos[1]} />
            <Kv k="best combo A" v={maxCombos[0]} hl />
            <Kv k="best combo B" v={maxCombos[1]} />
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
