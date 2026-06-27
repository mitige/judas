import { useEffect, useState } from "react";
import { api } from "../api.js";
import { usePersistentState } from "../persistence.mjs";

const LIVE_DEFAULTS = {
  model: "",
  cps: 10,
  rot: 190,
  counterAssist: false,
  hitSelectAssist: false,
  aimSmoothing: false,
  aimSmoothingStrength: 0.22,
  aimSmoothingSnap: 0.55,
  autoGapple: true,
  autoGappleHealth: 14,
  autoGappleCriticalHealth: 8,
  autoGappleSafeDistance: 11.50,
  autoGappleRetreat: true,
  autoGappleRetreatDistance: 18,
  autoGappleFastRetreat: true,
  autoGappleRetreatHops: true,
  autoGappleSprintHopHold: true,
  autoGappleAvoidObstacles: true,
  autoGappleRetreatStrafe: true,
  autoGappleWallSlide: true,
  autoGappleSpeedLock: true,
  autoGappleVelocityAssist: true,
  autoGappleSpeedFirst: true,
  autoGappleFullSpeed: true,
  autoGappleSpeedFloor: 4.50,
  autoGappleMaxSpeed: 4.80,
  autoGappleAccel: 5.50,
  autoGappleSprintRetap: true,
  autoGappleSprintRetapTicks: 2,
  autoGappleAirControl: true,
  autoGappleStepAssist: true,
  autoGappleStepHeight: 1.20,
  autoGappleFallbackRetreat: true,
  autoGappleRetreatInputLock: true,
  autoGappleForceSprintRetreat: true,
  autoGappleReleaseRetreatOnHit: true,
  autoGappleCriticalRearmOnly: true,
  autoGappleCriticalTrappedEat: true,
  autoGappleRetreatTurnDeg: 360,
  autoGappleEatingRetreatTurnDeg: 360,
  autoGappleRetreatPathHoldTicks: 2,
  autoGappleRetreatStuckAbortTicks: 4,
  autoGappleRetreatMinTicks: 0,
  autoGappleRetreatMaxTicks: 64,
  autoGappleCriticalRetreatMaxTicks: 6,
  autoGappleCriticalEatCommitTicks: 12,
  autoGappleCombatRecoveryTicks: 6,
  autoGappleRetreatStrafeHoldTicks: 5,
  autoGappleRetreatObstacleJumpHoldTicks: 60,
  autoGappleRetreatObstacleEscapeTicks: 120,
  autoGappleRetreatPanicSpeed: true,
  autoGappleRetreatObstacleLookahead: 24.00,
  autoGappleCriticalTrappedStuckTicks: 2,
  autoJump: false,
  knockbackDump: false,
  friendMode: false,
  friends: "",
  arena: { origin_x: 0, origin_z: 0, size_x: 40, size_z: 40, floor_y: 0 },
};

export default function Live({ status }) {
  const [models, setModels] = useState([]);
  const [blockedModels, setBlockedModels] = useState([]);
  const [settings, setSettings] = usePersistentState("judas:app:live:v47", LIVE_DEFAULTS);
  const [err, setErr] = useState(null);
  const {
    model, cps, rot, arena, counterAssist, hitSelectAssist,
    aimSmoothing, aimSmoothingStrength, aimSmoothingSnap,
    autoGapple, autoGappleHealth, autoGappleCriticalHealth,
    autoGappleSafeDistance, autoGappleRetreat,
    autoGappleRetreatDistance, autoGappleFastRetreat,
    autoGappleRetreatHops, autoGappleSprintHopHold,
    autoGappleAvoidObstacles, autoGappleRetreatStrafe,
    autoGappleWallSlide, autoGappleSpeedLock,
    autoGappleVelocityAssist, autoGappleSpeedFirst,
    autoGappleFullSpeed,
    autoGappleSpeedFloor,
    autoGappleMaxSpeed, autoGappleAccel,
    autoGappleSprintRetap, autoGappleSprintRetapTicks,
    autoGappleAirControl,
    autoGappleStepAssist, autoGappleStepHeight,
    autoGappleFallbackRetreat,
    autoGappleRetreatInputLock,
    autoGappleForceSprintRetreat,
    autoGappleReleaseRetreatOnHit,
    autoGappleCriticalRearmOnly,
    autoGappleCriticalTrappedEat,
    autoGappleRetreatTurnDeg, autoGappleEatingRetreatTurnDeg,
    autoGappleRetreatPathHoldTicks, autoGappleRetreatStuckAbortTicks,
    autoGappleRetreatMinTicks, autoGappleRetreatMaxTicks,
    autoGappleCriticalRetreatMaxTicks, autoGappleCriticalEatCommitTicks,
    autoGappleCombatRecoveryTicks,
    autoGappleRetreatStrafeHoldTicks,
    autoGappleRetreatObstacleJumpHoldTicks,
    autoGappleRetreatObstacleEscapeTicks,
    autoGappleRetreatPanicSpeed,
    autoGappleRetreatObstacleLookahead,
    autoGappleCriticalTrappedStuckTicks,
    autoJump, knockbackDump, friendMode, friends,
  } = settings;
  const live = status?.live;
  const onErr = (e) => setErr(String(e?.message || e));

  useEffect(() => {
    api.models().then((d) => {
      const usable = d.exported.filter((m) => m.export_fresh !== false);
      const blocked = d.exported.filter((m) => m.export_fresh === false);
      setModels(usable);
      setBlockedModels(blocked);
      const preferred =
        usable.find((m) => m.path?.includes("combo_god_recovery_kb092_combo12-safe_latest")) ||
        usable.find((m) => m.path?.includes("combo_god_leaderboard10_combo12-safe_latest")) ||
        usable.find((m) => m.path?.includes("combo_god_countertap96_combo12-safe_latest")) ||
        usable.find((m) => m.path?.includes("combo_god_directpad_lock_combo12-safe_latest")) ||
        usable[0];
      const hasModel = (path) => path && usable.some((m) => m.path === path);
      setSettings((cur) => ({
        ...cur,
        model: hasModel(cur.model) ? cur.model : (preferred?.path || ""),
      }));
    }).catch(() => {});
  }, [setSettings]);

  useEffect(() => {
    const id = setTimeout(() => {
      api.liveParams({
        max_cps: +cps,
        max_rot_speed: +rot,
        aim_smoothing: {
          enabled: !!aimSmoothing,
          strength: +aimSmoothingStrength,
          snap_deg: +aimSmoothingSnap,
        },
        hit_select_assist: { enabled: !!hitSelectAssist },
        counter_assist: { enabled: !!counterAssist },
        auto_gapple: {
          enabled: !!autoGapple,
          health_threshold: +autoGappleHealth,
          critical_health_threshold: +autoGappleCriticalHealth,
          safe_distance: +autoGappleSafeDistance,
          retreat_enabled: !!autoGappleRetreat,
          retreat_distance: +autoGappleRetreatDistance,
          fast_retreat: !!autoGappleFastRetreat,
          retreat_hops: !!autoGappleRetreatHops,
          sprint_hop_hold: !!autoGappleSprintHopHold,
          avoid_obstacles: !!autoGappleAvoidObstacles,
          retreat_strafe: !!autoGappleRetreatStrafe,
          wall_slide: !!autoGappleWallSlide,
          retreat_speed_lock: !!autoGappleSpeedLock,
          retreat_velocity_assist: !!autoGappleVelocityAssist,
          retreat_speed_first: !!autoGappleSpeedFirst,
          retreat_full_speed: !!autoGappleFullSpeed,
          retreat_speed_floor: +autoGappleSpeedFloor,
          retreat_max_speed: +autoGappleMaxSpeed,
          retreat_accel: +autoGappleAccel,
          retreat_sprint_retap: !!autoGappleSprintRetap,
          retreat_sprint_retap_ticks: +autoGappleSprintRetapTicks,
          retreat_air_control: !!autoGappleAirControl,
          retreat_step_assist: !!autoGappleStepAssist,
          retreat_step_height: +autoGappleStepHeight,
          fallback_retreat: !!autoGappleFallbackRetreat,
          retreat_input_lock: !!autoGappleRetreatInputLock,
          force_sprint_retreat: !!autoGappleForceSprintRetreat,
          release_retreat_on_hit: !!autoGappleReleaseRetreatOnHit,
          critical_rearm_only: !!autoGappleCriticalRearmOnly,
          critical_trapped_eat: !!autoGappleCriticalTrappedEat,
          retreat_turn_limit_deg: +autoGappleRetreatTurnDeg,
          eating_retreat_turn_limit_deg: +autoGappleEatingRetreatTurnDeg,
          retreat_path_hold_ticks: +autoGappleRetreatPathHoldTicks,
          retreat_stuck_abort_ticks: +autoGappleRetreatStuckAbortTicks,
          retreat_min_ticks: +autoGappleRetreatMinTicks,
          retreat_max_ticks: +autoGappleRetreatMaxTicks,
          critical_retreat_max_ticks: +autoGappleCriticalRetreatMaxTicks,
          critical_eat_commit_ticks: +autoGappleCriticalEatCommitTicks,
          combat_recovery_ticks: +autoGappleCombatRecoveryTicks,
          retreat_strafe_hold_ticks: +autoGappleRetreatStrafeHoldTicks,
          retreat_obstacle_jump_hold_ticks: +autoGappleRetreatObstacleJumpHoldTicks,
          retreat_obstacle_escape_ticks: +autoGappleRetreatObstacleEscapeTicks,
          retreat_panic_speed: !!autoGappleRetreatPanicSpeed,
          retreat_obstacle_lookahead: +autoGappleRetreatObstacleLookahead,
          critical_trapped_stuck_ticks: +autoGappleCriticalTrappedStuckTicks,
        },
        auto_jump: { enabled: !!autoJump },
        knockback_dump: { enabled: !!knockbackDump },
        friends: {
          enabled: !!friendMode,
          names: parseFriends(friends),
        },
      }).catch(() => {});
    }, 250);
    return () => clearTimeout(id);
  }, [
    cps, rot, counterAssist, hitSelectAssist, aimSmoothing,
    aimSmoothingStrength, aimSmoothingSnap, autoGapple, autoGappleHealth,
    autoGappleCriticalHealth, autoGappleSafeDistance,
    autoGappleRetreat, autoGappleRetreatDistance,
    autoGappleFastRetreat, autoGappleRetreatHops,
    autoGappleSprintHopHold,
    autoGappleAvoidObstacles, autoGappleRetreatStrafe,
    autoGappleWallSlide, autoGappleSpeedLock,
    autoGappleVelocityAssist, autoGappleSpeedFirst,
    autoGappleFullSpeed,
    autoGappleSpeedFloor,
    autoGappleMaxSpeed, autoGappleAccel,
    autoGappleSprintRetap, autoGappleSprintRetapTicks,
    autoGappleAirControl,
    autoGappleStepAssist, autoGappleStepHeight,
    autoGappleFallbackRetreat,
    autoGappleRetreatInputLock,
    autoGappleForceSprintRetreat,
    autoGappleReleaseRetreatOnHit,
    autoGappleCriticalRearmOnly,
    autoGappleCriticalTrappedEat,
    autoGappleRetreatTurnDeg, autoGappleEatingRetreatTurnDeg,
    autoGappleRetreatPathHoldTicks, autoGappleRetreatStuckAbortTicks,
    autoGappleRetreatMinTicks, autoGappleRetreatMaxTicks,
    autoGappleCriticalRetreatMaxTicks, autoGappleCriticalEatCommitTicks,
    autoGappleCombatRecoveryTicks,
    autoGappleRetreatStrafeHoldTicks,
    autoGappleRetreatObstacleJumpHoldTicks,
    autoGappleRetreatObstacleEscapeTicks,
    autoGappleRetreatPanicSpeed,
    autoGappleRetreatObstacleLookahead,
    autoGappleCriticalTrappedStuckTicks,
    autoJump, knockbackDump, friendMode, friends,
  ]);

  const load = () => model && api.liveLoad(model).then(() => setErr(null)).catch(onErr);
  const arm = () => api.liveParams({ enabled: true, arena }).then(() => setErr(null)).catch(onErr);
  const kill = () => api.liveKill().then(() => setErr(null)).catch(onErr);
  const setModel = (value) => setSettings((cur) => ({ ...cur, model: value }));
  const setCps = (value) => setSettings((cur) => ({ ...cur, cps: value }));
  const setRot = (value) => setSettings((cur) => ({ ...cur, rot: value }));
  const setCounterAssist = (value) => setSettings((cur) => ({ ...cur, counterAssist: value }));
  const setHitSelectAssist = (value) =>
    setSettings((cur) => ({ ...cur, hitSelectAssist: value }));
  const setAimSmoothing = (value) =>
    setSettings((cur) => ({ ...cur, aimSmoothing: value }));
  const setAimSmoothingStrength = (value) =>
    setSettings((cur) => ({ ...cur, aimSmoothingStrength: value }));
  const setAimSmoothingSnap = (value) =>
    setSettings((cur) => ({ ...cur, aimSmoothingSnap: value }));
  const setAutoGapple = (value) => setSettings((cur) => ({ ...cur, autoGapple: value }));
  const setAutoGappleHealth = (value) =>
    setSettings((cur) => ({ ...cur, autoGappleHealth: value }));
  const setAutoGappleCriticalHealth = (value) =>
    setSettings((cur) => ({ ...cur, autoGappleCriticalHealth: value }));
  const setAutoGappleSafeDistance = (value) =>
    setSettings((cur) => ({ ...cur, autoGappleSafeDistance: value }));
  const setAutoGappleRetreat = (value) =>
    setSettings((cur) => ({ ...cur, autoGappleRetreat: value }));
  const setAutoGappleRetreatDistance = (value) =>
    setSettings((cur) => ({ ...cur, autoGappleRetreatDistance: value }));
  const setAutoGappleFastRetreat = (value) =>
    setSettings((cur) => ({ ...cur, autoGappleFastRetreat: value }));
  const setAutoGappleRetreatHops = (value) =>
    setSettings((cur) => ({ ...cur, autoGappleRetreatHops: value }));
  const setAutoGappleSprintHopHold = (value) =>
    setSettings((cur) => ({ ...cur, autoGappleSprintHopHold: value }));
  const setAutoGappleAvoidObstacles = (value) =>
    setSettings((cur) => ({ ...cur, autoGappleAvoidObstacles: value }));
  const setAutoGappleRetreatStrafe = (value) =>
    setSettings((cur) => ({ ...cur, autoGappleRetreatStrafe: value }));
  const setAutoGappleWallSlide = (value) =>
    setSettings((cur) => ({ ...cur, autoGappleWallSlide: value }));
  const setAutoGappleSpeedLock = (value) =>
    setSettings((cur) => ({ ...cur, autoGappleSpeedLock: value }));
  const setAutoGappleVelocityAssist = (value) =>
    setSettings((cur) => ({ ...cur, autoGappleVelocityAssist: value }));
  const setAutoGappleSpeedFirst = (value) =>
    setSettings((cur) => ({ ...cur, autoGappleSpeedFirst: value }));
  const setAutoGappleFullSpeed = (value) =>
    setSettings((cur) => ({ ...cur, autoGappleFullSpeed: value }));
  const setAutoGappleSpeedFloor = (value) =>
    setSettings((cur) => ({ ...cur, autoGappleSpeedFloor: value }));
  const setAutoGappleMaxSpeed = (value) =>
    setSettings((cur) => ({ ...cur, autoGappleMaxSpeed: value }));
  const setAutoGappleAccel = (value) =>
    setSettings((cur) => ({ ...cur, autoGappleAccel: value }));
  const setAutoGappleSprintRetap = (value) =>
    setSettings((cur) => ({ ...cur, autoGappleSprintRetap: value }));
  const setAutoGappleSprintRetapTicks = (value) =>
    setSettings((cur) => ({ ...cur, autoGappleSprintRetapTicks: value }));
  const setAutoGappleAirControl = (value) =>
    setSettings((cur) => ({ ...cur, autoGappleAirControl: value }));
  const setAutoGappleStepAssist = (value) =>
    setSettings((cur) => ({ ...cur, autoGappleStepAssist: value }));
  const setAutoGappleStepHeight = (value) =>
    setSettings((cur) => ({ ...cur, autoGappleStepHeight: value }));
  const setAutoGappleFallbackRetreat = (value) =>
    setSettings((cur) => ({ ...cur, autoGappleFallbackRetreat: value }));
  const setAutoGappleRetreatInputLock = (value) =>
    setSettings((cur) => ({ ...cur, autoGappleRetreatInputLock: value }));
  const setAutoGappleForceSprintRetreat = (value) =>
    setSettings((cur) => ({ ...cur, autoGappleForceSprintRetreat: value }));
  const setAutoGappleReleaseRetreatOnHit = (value) =>
    setSettings((cur) => ({ ...cur, autoGappleReleaseRetreatOnHit: value }));
  const setAutoGappleCriticalRearmOnly = (value) =>
    setSettings((cur) => ({ ...cur, autoGappleCriticalRearmOnly: value }));
  const setAutoGappleCriticalTrappedEat = (value) =>
    setSettings((cur) => ({ ...cur, autoGappleCriticalTrappedEat: value }));
  const setAutoGappleRetreatTurnDeg = (value) =>
    setSettings((cur) => ({ ...cur, autoGappleRetreatTurnDeg: value }));
  const setAutoGappleEatingRetreatTurnDeg = (value) =>
    setSettings((cur) => ({ ...cur, autoGappleEatingRetreatTurnDeg: value }));
  const setAutoGappleRetreatPathHoldTicks = (value) =>
    setSettings((cur) => ({ ...cur, autoGappleRetreatPathHoldTicks: value }));
  const setAutoGappleRetreatStuckAbortTicks = (value) =>
    setSettings((cur) => ({ ...cur, autoGappleRetreatStuckAbortTicks: value }));
  const setAutoGappleRetreatMinTicks = (value) =>
    setSettings((cur) => ({ ...cur, autoGappleRetreatMinTicks: value }));
  const setAutoGappleRetreatMaxTicks = (value) =>
    setSettings((cur) => ({ ...cur, autoGappleRetreatMaxTicks: value }));
  const setAutoGappleCriticalRetreatMaxTicks = (value) =>
    setSettings((cur) => ({ ...cur, autoGappleCriticalRetreatMaxTicks: value }));
  const setAutoGappleCriticalEatCommitTicks = (value) =>
    setSettings((cur) => ({ ...cur, autoGappleCriticalEatCommitTicks: value }));
  const setAutoGappleCombatRecoveryTicks = (value) =>
    setSettings((cur) => ({ ...cur, autoGappleCombatRecoveryTicks: value }));
  const setAutoGappleRetreatStrafeHoldTicks = (value) =>
    setSettings((cur) => ({ ...cur, autoGappleRetreatStrafeHoldTicks: value }));
  const setAutoGappleRetreatObstacleJumpHoldTicks = (value) =>
    setSettings((cur) => ({ ...cur, autoGappleRetreatObstacleJumpHoldTicks: value }));
  const setAutoGappleRetreatObstacleEscapeTicks = (value) =>
    setSettings((cur) => ({ ...cur, autoGappleRetreatObstacleEscapeTicks: value }));
  const setAutoGappleRetreatPanicSpeed = (value) =>
    setSettings((cur) => ({ ...cur, autoGappleRetreatPanicSpeed: value }));
  const setAutoGappleRetreatObstacleLookahead = (value) =>
    setSettings((cur) => ({ ...cur, autoGappleRetreatObstacleLookahead: value }));
  const setAutoGappleCriticalTrappedStuckTicks = (value) =>
    setSettings((cur) => ({ ...cur, autoGappleCriticalTrappedStuckTicks: value }));
  const setAutoJump = (value) => setSettings((cur) => ({ ...cur, autoJump: value }));
  const setKnockbackDump = (value) =>
    setSettings((cur) => ({ ...cur, knockbackDump: value }));
  const setFriendMode = (value) => setSettings((cur) => ({ ...cur, friendMode: value }));
  const setFriends = (value) => setSettings((cur) => ({ ...cur, friends: value }));
  const setA = (k) => (e) =>
    setSettings((cur) => ({ ...cur, arena: { ...cur.arena, [k]: +e.target.value } }));

  return (
    <>
      <h2 className="title">Live</h2>
      <div style={{ height: 24 }} />

      <div className="grid cols-2">
        <div className="panel">
          <div className="label">model</div>
          <div className="field">
            <label>torchscript</label>
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              {models.length === 0 && <option value="">—</option>}
              {models.map((m) => (
                <option key={m.path} value={m.path}>
                  {m.name}{m.export_status ? ` (${m.export_status})` : ""}
                </option>
              ))}
            </select>
          </div>
          {models.length === 0 && blockedModels.length > 0 && (
            <div className="empty compact">
              no schema 8 combo-safe export available
            </div>
          )}
          {blockedModels.slice(0, 3).map((m) => (
            <div className="kv" key={m.path}>
              <span className="k">{m.name}</span>
              <span className="v bad" title={m.export_error || ""}>
                {m.export_status || "blocked"}
              </span>
            </div>
          ))}
          <hr className="sep" />
          <div className="controls-row">
            <button className="btn" onClick={load} disabled={!model}>load</button>
            <button className="btn" onClick={arm}>arm</button>
            <button className="btn danger" onClick={kill}>kill</button>
            {err && <span className="tag" style={{ color: "var(--ember)" }}>{err}</span>}
          </div>
          <label className="toggle">
            <input
              type="checkbox"
              checked={!!hitSelectAssist}
              onChange={(e) => setHitSelectAssist(e.target.checked)}
            />
            perfect hit-select
          </label>
          <hr className="sep" />
          <Kv k="loaded" v={live?.model?.split(/[\\/]/).pop() ?? "—"} hl={!!live?.model} />
          <Kv k="armed" v={live?.enabled ? "yes" : "no"} hl={live?.enabled} />
          <Kv
            k="hit-select assist"
            v={live?.hit_select_assist?.enabled ? "on" : "off"}
            hl={live?.hit_select_assist?.enabled}
          />
          <Kv k="latency" v={live?.latency_ms != null ? `${live.latency_ms} ms` : "—"} />
          <Kv k="ticks" v={live?.tick ?? "—"} />
          <Kv k="device" v={live?.device ?? "—"} />
        </div>

        <div className="panel">
          <div className="label">humanization · realtime</div>
          <Slider l="cps" v={cps} set={setCps} min={1} max={20} />
          <Slider l="rotation deg/t" v={rot} set={setRot} min={5} max={260} />
          <label className="toggle">
            <input
              type="checkbox"
              checked={!!aimSmoothing}
              onChange={(e) => setAimSmoothing(e.target.checked)}
            />
            aim smoothing
          </label>
          <Kv
            k="aim smoothing"
            v={live?.aim_smoothing?.enabled
              ? `${live.aim_smoothing.strength} / ${live.aim_smoothing.snap_deg}`
              : "off"}
            hl={live?.aim_smoothing?.enabled}
          />
          <Field
            l="smoothing strength"
            v={aimSmoothingStrength}
            on={(e) => setAimSmoothingStrength(+e.target.value)}
          />
          <Field
            l="smoothing snap"
            v={aimSmoothingSnap}
            on={(e) => setAimSmoothingSnap(+e.target.value)}
          />
          <label className="toggle">
            <input
              type="checkbox"
              checked={!!counterAssist}
              onChange={(e) => setCounterAssist(e.target.checked)}
            />
            counter assist
          </label>
          <Kv
            k="counter assist"
            v={live?.counter_assist?.enabled ? "on" : "off"}
            hl={live?.counter_assist?.enabled}
          />
          <hr className="sep" />
          <div className="label">survival</div>
          <label className="toggle">
            <input
              type="checkbox"
              checked={!!autoGapple}
              onChange={(e) => setAutoGapple(e.target.checked)}
            />
            auto gapple
          </label>
          <Field l="gapple hp" v={autoGappleHealth} on={(e) => setAutoGappleHealth(+e.target.value)} />
          <Field
            l="critical hp"
            v={autoGappleCriticalHealth}
            on={(e) => setAutoGappleCriticalHealth(+e.target.value)}
          />
          <Field
            l="safe dist"
            v={autoGappleSafeDistance}
            on={(e) => setAutoGappleSafeDistance(+e.target.value)}
          />
          <label className="toggle">
            <input
              type="checkbox"
              checked={!!autoGappleRetreat}
              onChange={(e) => setAutoGappleRetreat(e.target.checked)}
            />
            retreat before gapple
          </label>
          <Field
            l="retreat dist"
            v={autoGappleRetreatDistance}
            on={(e) => setAutoGappleRetreatDistance(+e.target.value)}
          />
          <label className="toggle">
            <input
              type="checkbox"
              checked={!!autoGappleFastRetreat}
              onChange={(e) => setAutoGappleFastRetreat(e.target.checked)}
            />
            fast retreat
          </label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={!!autoGappleRetreatHops}
              onChange={(e) => setAutoGappleRetreatHops(e.target.checked)}
            />
            retreat hops
          </label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={!!autoGappleSprintHopHold}
              onChange={(e) => setAutoGappleSprintHopHold(e.target.checked)}
            />
            hold sprint-hop
          </label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={!!autoGappleAvoidObstacles}
              onChange={(e) => setAutoGappleAvoidObstacles(e.target.checked)}
            />
            avoid obstacles
          </label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={!!autoGappleRetreatStrafe}
              onChange={(e) => setAutoGappleRetreatStrafe(e.target.checked)}
            />
            retreat strafe
          </label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={!!autoGappleWallSlide}
              onChange={(e) => setAutoGappleWallSlide(e.target.checked)}
            />
            wall slide
          </label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={!!autoGappleSpeedLock}
              onChange={(e) => setAutoGappleSpeedLock(e.target.checked)}
            />
            retreat speed lock
          </label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={!!autoGappleVelocityAssist}
              onChange={(e) => setAutoGappleVelocityAssist(e.target.checked)}
            />
            retreat velocity assist
          </label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={!!autoGappleSpeedFirst}
              onChange={(e) => setAutoGappleSpeedFirst(e.target.checked)}
            />
            speed-first retreat
          </label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={!!autoGappleFullSpeed}
              onChange={(e) => setAutoGappleFullSpeed(e.target.checked)}
            />
            full-speed retreat
          </label>
          <Field
            l="speed floor"
            v={autoGappleSpeedFloor}
            on={(e) => setAutoGappleSpeedFloor(+e.target.value)}
          />
          <Field
            l="max retreat speed"
            v={autoGappleMaxSpeed}
            on={(e) => setAutoGappleMaxSpeed(+e.target.value)}
          />
          <Field
            l="retreat accel"
            v={autoGappleAccel}
            on={(e) => setAutoGappleAccel(+e.target.value)}
          />
          <label className="toggle">
            <input
              type="checkbox"
              checked={!!autoGappleSprintRetap}
              onChange={(e) => setAutoGappleSprintRetap(e.target.checked)}
            />
            sprint retap
          </label>
          <Field
            l="sprint retap ticks"
            v={autoGappleSprintRetapTicks}
            on={(e) => setAutoGappleSprintRetapTicks(+e.target.value)}
          />
          <label className="toggle">
            <input
              type="checkbox"
              checked={!!autoGappleAirControl}
              onChange={(e) => setAutoGappleAirControl(e.target.checked)}
            />
            air control
          </label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={!!autoGappleStepAssist}
              onChange={(e) => setAutoGappleStepAssist(e.target.checked)}
            />
            step assist
          </label>
          <Field
            l="step height"
            v={autoGappleStepHeight}
            on={(e) => setAutoGappleStepHeight(+e.target.value)}
          />
          <label className="toggle">
            <input
              type="checkbox"
              checked={!!autoGappleFallbackRetreat}
              onChange={(e) => setAutoGappleFallbackRetreat(e.target.checked)}
            />
            fallback retreat
          </label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={!!autoGappleRetreatInputLock}
              onChange={(e) => setAutoGappleRetreatInputLock(e.target.checked)}
            />
            retreat input lock
          </label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={!!autoGappleForceSprintRetreat}
              onChange={(e) => setAutoGappleForceSprintRetreat(e.target.checked)}
            />
            force sprint retreat
          </label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={!!autoGappleReleaseRetreatOnHit}
              onChange={(e) => setAutoGappleReleaseRetreatOnHit(e.target.checked)}
            />
            release retreat on hit
          </label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={!!autoGappleCriticalRearmOnly}
              onChange={(e) => setAutoGappleCriticalRearmOnly(e.target.checked)}
            />
            critical rearm only
          </label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={!!autoGappleCriticalTrappedEat}
              onChange={(e) => setAutoGappleCriticalTrappedEat(e.target.checked)}
            />
            critical trapped eat
          </label>
          <Field
            l="retreat turn deg"
            v={autoGappleRetreatTurnDeg}
            on={(e) => setAutoGappleRetreatTurnDeg(+e.target.value)}
          />
          <Field
            l="eat turn deg"
            v={autoGappleEatingRetreatTurnDeg}
            on={(e) => setAutoGappleEatingRetreatTurnDeg(+e.target.value)}
          />
          <Field
            l="path hold ticks"
            v={autoGappleRetreatPathHoldTicks}
            on={(e) => setAutoGappleRetreatPathHoldTicks(+e.target.value)}
          />
          <Field
            l="stuck abort ticks"
            v={autoGappleRetreatStuckAbortTicks}
            on={(e) => setAutoGappleRetreatStuckAbortTicks(+e.target.value)}
          />
          <Field
            l="min retreat ticks"
            v={autoGappleRetreatMinTicks}
            on={(e) => setAutoGappleRetreatMinTicks(+e.target.value)}
          />
          <Field
            l="max retreat ticks"
            v={autoGappleRetreatMaxTicks}
            on={(e) => setAutoGappleRetreatMaxTicks(+e.target.value)}
          />
          <Field
            l="critical retreat ticks"
            v={autoGappleCriticalRetreatMaxTicks}
            on={(e) => setAutoGappleCriticalRetreatMaxTicks(+e.target.value)}
          />
          <Field
            l="critical eat commit ticks"
            v={autoGappleCriticalEatCommitTicks}
            on={(e) => setAutoGappleCriticalEatCommitTicks(+e.target.value)}
          />
          <Field
            l="combat recovery ticks"
            v={autoGappleCombatRecoveryTicks}
            on={(e) => setAutoGappleCombatRecoveryTicks(+e.target.value)}
          />
          <Field
            l="strafe hold ticks"
            v={autoGappleRetreatStrafeHoldTicks}
            on={(e) => setAutoGappleRetreatStrafeHoldTicks(+e.target.value)}
          />
          <Field
            l="obstacle jump hold ticks"
            v={autoGappleRetreatObstacleJumpHoldTicks}
            on={(e) => setAutoGappleRetreatObstacleJumpHoldTicks(+e.target.value)}
          />
          <Field
            l="obstacle escape ticks"
            v={autoGappleRetreatObstacleEscapeTicks}
            on={(e) => setAutoGappleRetreatObstacleEscapeTicks(+e.target.value)}
          />
          <label className="toggle">
            <input
              type="checkbox"
              checked={!!autoGappleRetreatPanicSpeed}
              onChange={(e) => setAutoGappleRetreatPanicSpeed(e.target.checked)}
            />
            panic retreat speed
          </label>
          <Field
            l="obstacle lookahead"
            v={autoGappleRetreatObstacleLookahead}
            on={(e) => setAutoGappleRetreatObstacleLookahead(+e.target.value)}
          />
          <Field
            l="trapped stuck ticks"
            v={autoGappleCriticalTrappedStuckTicks}
            on={(e) => setAutoGappleCriticalTrappedStuckTicks(+e.target.value)}
          />
          <Kv
            k="auto gapple"
            v={live?.auto_gapple?.enabled ? `on @ ${live.auto_gapple.health_threshold}` : "off"}
            hl={live?.auto_gapple?.enabled}
          />
          <label className="toggle">
            <input
              type="checkbox"
              checked={!!autoJump}
              onChange={(e) => setAutoJump(e.target.checked)}
            />
            auto jump
          </label>
          <Kv
            k="auto jump"
            v={live?.auto_jump?.enabled ? "on" : "off"}
            hl={live?.auto_jump?.enabled}
          />
          <label className="toggle">
            <input
              type="checkbox"
              checked={!!knockbackDump}
              onChange={(e) => setKnockbackDump(e.target.checked)}
            />
            kb dump
          </label>
          <Kv
            k="kb dump"
            v={live?.knockback_dump?.enabled ? "on" : "off"}
            hl={live?.knockback_dump?.enabled}
          />
          <hr className="sep" />
          <div className="label">friends</div>
          <label className="toggle">
            <input
              type="checkbox"
              checked={!!friendMode}
              onChange={(e) => setFriendMode(e.target.checked)}
            />
            friend mode
          </label>
          <TextArea l="names" v={friends} on={(e) => setFriends(e.target.value)} />
          <Kv
            k="friend mode"
            v={live?.friends?.enabled ? `${live.friends.names?.length || 0}` : "off"}
            hl={live?.friends?.enabled}
          />
          <hr className="sep" />
          <div className="label">arena</div>
          <div className="grid cols-2">
            <Field l="origin x" v={arena.origin_x} on={setA("origin_x")} />
            <Field l="origin z" v={arena.origin_z} on={setA("origin_z")} />
            <Field l="size x" v={arena.size_x} on={setA("size_x")} />
            <Field l="size z" v={arena.size_z} on={setA("size_z")} />
            <Field l="floor y" v={arena.floor_y} on={setA("floor_y")} />
          </div>
        </div>
      </div>
    </>
  );
}

function Slider({ l, v, set, min, max }) {
  return (
    <div className="slider" style={{ marginBottom: 16 }}>
      <div className="row">
        <span className="field"><label>{l}</label></span>
        <span className="val">{v}</span>
      </div>
      <input type="range" min={min} max={max} value={v}
             onChange={(e) => set(+e.target.value)} />
    </div>
  );
}

function Field({ l, v, on }) {
  return (
    <div className="field">
      <label>{l}</label>
      <input type="number" value={v} onChange={on} />
    </div>
  );
}

function TextArea({ l, v, on }) {
  return (
    <div className="field">
      <label>{l}</label>
      <textarea value={v} onChange={on} rows={3} />
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

function parseFriends(value) {
  return String(value || "")
    .split(/[\s,;]+/)
    .map((name) => name.trim())
    .filter(Boolean);
}
