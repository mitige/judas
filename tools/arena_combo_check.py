from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_MODEL_A = "models/combo_god_leaderboard10_combo12-safe_latest.pts"
DEFAULT_MODEL_B = "models/combo_god_leaderboard10_combo12-safe_latest.pts"
SCRIPTED_MODELS = {
    "__chase_bot__",
    "__combo_pad__",
    "__combo_spar__",
    "chase_bot",
    "combo_pad",
    "combo_spar",
    "bot:chase",
    "bot:combo_pad",
    "bot:combo_spar",
}


@dataclass(frozen=True)
class MatchEvent:
    winner: int
    combo: tuple[int, int]
    max_combo: tuple[int, int]
    wins: tuple[int, int]
    draws: int


@dataclass(frozen=True)
class ArenaComboReport:
    matches: int
    ticks: int
    wins: tuple[int, int]
    draws: int
    max_combo: tuple[int, int]
    hits: tuple[int, int]
    swings: tuple[int, int]
    hits_per_min: tuple[float, float]
    swings_per_min: tuple[float, float]
    avg_dist: float
    close_frac: float
    avg_abs_pitch: tuple[float, float]
    sky_frac: tuple[float, float]
    back_frac: tuple[float, float]
    tap_back_frac: tuple[float, float]
    strafe_frac: tuple[float, float]
    opener_strafe_frac: tuple[float, float]
    opener_strafe_hold_frac: tuple[float, float]
    opener_pressure_frac: tuple[float, float]
    jump_frac: tuple[float, float]
    events: tuple[MatchEvent, ...]
    sample: bool
    target_hits: int
    spawn_gap: float
    arena_size: float
    cps: float
    rot_speed: float

    @property
    def best_combo(self) -> int:
        return max(self.max_combo) if self.max_combo else 0


@dataclass(frozen=True)
class ArenaComboThresholds:
    required_matches: int = 8
    min_combo: int = 12
    max_draws: int = 0
    min_wins_a: int = 1
    min_wins_b: int = 0
    min_hits_a: float = 18.0
    min_hits_b: float = 0.0
    min_combo_b: int = 0
    min_close_frac: float = 0.08
    max_sky_frac_a: float = 0.02
    max_sky_frac_b: float = 0.02
    max_back_frac_a: float = 0.01
    max_back_frac_b: float = 0.01
    min_strafe_frac_a: float = 0.50
    min_strafe_frac_b: float = 0.50
    min_opener_strafe_frac_a: float = 0.75
    min_opener_strafe_frac_b: float = 0.75
    min_opener_strafe_hold_frac_a: float = 0.70
    min_opener_strafe_hold_frac_b: float = 0.70
    min_opener_pressure_frac_a: float = 0.60
    min_opener_pressure_frac_b: float = 0.60
    max_strafe_frac_a: float = 1.0
    max_strafe_frac_b: float = 1.0
    max_jump_frac_a: float = 0.01
    max_jump_frac_b: float = 0.01


def run_arena_combo_check(
    model_a: str = DEFAULT_MODEL_A,
    model_b: str = DEFAULT_MODEL_B,
    *,
    matches: int = 8,
    max_steps: int = 200_000,
    cps: float = 10.0,
    close_dist: float = 3.25,
    sky_pitch_deg: float = 60.0,
    opener_ticks: int = 20,
) -> ArenaComboReport:
    from serve.arena import ArenaSession

    session = ArenaSession()
    status = session.load(model_a, model_b, cps=cps)
    events: list[MatchEvent] = []
    ticks = 0
    hits = [0, 0]
    swings = [0, 0]
    dist_sum = 0.0
    close_ticks = 0
    pitch_abs_sum = [0.0, 0.0]
    sky_ticks = [0, 0]
    back_ticks = [0, 0]
    tap_back_ticks = [0, 0]
    strafe_ticks = [0, 0]
    opener_strafe_ticks = [0, 0]
    opener_strafe_pos_ticks = [0, 0]
    opener_strafe_neg_ticks = [0, 0]
    opener_pressure_ticks = [0, 0]
    opener_samples = 0
    jump_ticks = [0, 0]
    measured_ticks = 0
    match_tick = 0
    for _ in range(max_steps):
        state = session.step()
        ticks += 1
        players = state.get("players") or []
        if len(players) >= 2:
            a, b = players[0], players[1]
            dx = float(a["x"]) - float(b["x"])
            dz = float(a["z"]) - float(b["z"])
            dist = (dx * dx + dz * dz) ** 0.5
            dist_sum += dist
            close_ticks += int(dist <= close_dist)
            measured_ticks += 1
            for i, p in enumerate(players[:2]):
                pitch = abs(float(p["pitch"]))
                pitch_abs_sum[i] += pitch
                sky_ticks[i] += int(pitch >= sky_pitch_deg)
                swings[i] += int(bool(p.get("swing")))
                hits[i] += int(bool(p.get("landed")))
                fwd = float(p.get("forward", 0.0))
                sprinting = bool(p.get("sprint", False))
                back_ticks[i] += int(fwd < -0.5 and sprinting)
                tap_back_ticks[i] += int(fwd < -0.5 and not sprinting)
                strafe_active = abs(float(p.get("strafe", 0.0))) > 0.5
                strafe_ticks[i] += int(strafe_active)
                if match_tick < opener_ticks:
                    opener_strafe_ticks[i] += int(strafe_active)
                    opener_strafe_pos_ticks[i] += int(float(p.get("strafe", 0.0)) > 0.5)
                    opener_strafe_neg_ticks[i] += int(float(p.get("strafe", 0.0)) < -0.5)
                    opener_pressure_ticks[i] += int(fwd > 0.5 and strafe_active)
                jump_ticks[i] += int(bool(p.get("jump", False)))
            if match_tick < opener_ticks:
                opener_samples += 1
        match_tick += 1
        if state.get("done"):
            events.append(MatchEvent(
                winner=int(state["winner"]),
                combo=tuple(int(v) for v in state["combo"]),
                max_combo=tuple(int(v) for v in state["max_combo"]),
                wins=tuple(int(v) for v in state["wins"]),
                draws=int(state["draws"]),
            ))
            match_tick = 0
            if len(events) >= matches:
                break
    minutes = max(ticks / (20.0 * 60.0), 1e-6)
    measured = max(measured_ticks, 1)
    return ArenaComboReport(
        matches=len(events),
        ticks=ticks,
        wins=tuple(int(v) for v in session.wins),
        draws=int(session.draws),
        max_combo=tuple(int(v) for v in session.max_combo),
        hits=tuple(int(v) for v in hits),
        swings=tuple(int(v) for v in swings),
        hits_per_min=tuple(float(v) / minutes for v in hits),
        swings_per_min=tuple(float(v) / minutes for v in swings),
        avg_dist=dist_sum / measured,
        close_frac=close_ticks / measured,
        avg_abs_pitch=tuple(v / measured for v in pitch_abs_sum),
        sky_frac=tuple(v / measured for v in sky_ticks),
        back_frac=tuple(v / measured for v in back_ticks),
        tap_back_frac=tuple(v / measured for v in tap_back_ticks),
        strafe_frac=tuple(v / measured for v in strafe_ticks),
        opener_strafe_frac=tuple(v / max(opener_samples, 1) for v in opener_strafe_ticks),
        opener_strafe_hold_frac=tuple(
            max(opener_strafe_pos_ticks[i], opener_strafe_neg_ticks[i])
            / max(opener_samples, 1)
            for i in range(2)
        ),
        opener_pressure_frac=tuple(v / max(opener_samples, 1) for v in opener_pressure_ticks),
        jump_frac=tuple(v / measured for v in jump_ticks),
        events=tuple(events),
        sample=bool(status["sample"]),
        target_hits=int(status["target_hits"]),
        spawn_gap=float(status["spawn_gap"]),
        arena_size=float(status["arena"]["sx"]),
        cps=float(session.cfg.cps_min),
        rot_speed=float(session.cfg.rot_speed_min),
    )


def report_passes(report: ArenaComboReport, *, required_matches: int = 8,
                  min_combo: int = 12, max_draws: int = 0,
                  min_wins_a: int = 1, min_wins_b: int = 0,
                  min_hits_a: float = 18.0, min_hits_b: float = 0.0,
                  min_combo_b: int = 0,
                  min_close_frac: float = 0.08,
                  max_sky_frac_a: float = 0.02,
                  max_sky_frac_b: float = 0.02,
                  max_back_frac_a: float = 0.01,
                  max_back_frac_b: float = 0.01,
                  max_tap_back_frac_a: float = 0.0,
                  max_tap_back_frac_b: float = 0.0,
                  min_strafe_frac_a: float = 0.50,
                  min_strafe_frac_b: float = 0.50,
                  min_opener_strafe_frac_a: float = 0.75,
                  min_opener_strafe_frac_b: float = 0.75,
                  min_opener_strafe_hold_frac_a: float = 0.70,
                  min_opener_strafe_hold_frac_b: float = 0.70,
                  min_opener_pressure_frac_a: float = 0.60,
                  min_opener_pressure_frac_b: float = 0.60,
                  max_strafe_frac_a: float = 1.0,
                  max_strafe_frac_b: float = 1.0,
                  max_jump_frac_a: float = 0.01,
                  max_jump_frac_b: float = 0.01) -> bool:
    return (
        report.matches >= required_matches
        and report.draws <= max_draws
        and report.wins[0] >= min_wins_a
        and report.wins[1] >= min_wins_b
        and report.max_combo[0] >= min_combo
        and report.max_combo[1] >= min_combo_b
        and report.hits_per_min[0] >= min_hits_a
        and report.hits_per_min[1] >= min_hits_b
        and report.close_frac >= min_close_frac
        and report.sky_frac[0] <= max_sky_frac_a
        and report.sky_frac[1] <= max_sky_frac_b
        and report.back_frac[0] <= max_back_frac_a
        and report.back_frac[1] <= max_back_frac_b
        and report.tap_back_frac[0] <= max_tap_back_frac_a
        and report.tap_back_frac[1] <= max_tap_back_frac_b
        and report.strafe_frac[0] >= min_strafe_frac_a
        and report.strafe_frac[1] >= min_strafe_frac_b
        and report.opener_strafe_frac[0] >= min_opener_strafe_frac_a
        and report.opener_strafe_frac[1] >= min_opener_strafe_frac_b
        and report.opener_strafe_hold_frac[0] >= min_opener_strafe_hold_frac_a
        and report.opener_strafe_hold_frac[1] >= min_opener_strafe_hold_frac_b
        and report.opener_pressure_frac[0] >= min_opener_pressure_frac_a
        and report.opener_pressure_frac[1] >= min_opener_pressure_frac_b
        and report.strafe_frac[0] <= max_strafe_frac_a
        and report.strafe_frac[1] <= max_strafe_frac_b
        and report.jump_frac[0] <= max_jump_frac_a
        and report.jump_frac[1] <= max_jump_frac_b
    )


def verdict_text(report: ArenaComboReport, *, required_matches: int = 8,
                 min_combo: int = 12, max_draws: int = 0,
                 min_wins_a: int = 1, min_wins_b: int = 0,
                 min_hits_a: float = 18.0, min_hits_b: float = 0.0,
                 min_combo_b: int = 0,
                 min_close_frac: float = 0.08,
                 max_sky_frac_a: float = 0.02,
                 max_sky_frac_b: float = 0.02,
                 max_back_frac_a: float = 0.01,
                 max_back_frac_b: float = 0.01,
                 max_tap_back_frac_a: float = 0.0,
                 max_tap_back_frac_b: float = 0.0,
                 min_strafe_frac_a: float = 0.50,
                 min_strafe_frac_b: float = 0.50,
                 min_opener_strafe_frac_a: float = 0.75,
                 min_opener_strafe_frac_b: float = 0.75,
                 min_opener_strafe_hold_frac_a: float = 0.70,
                 min_opener_strafe_hold_frac_b: float = 0.70,
                 min_opener_pressure_frac_a: float = 0.60,
                 min_opener_pressure_frac_b: float = 0.60,
                 max_strafe_frac_a: float = 1.0,
                 max_strafe_frac_b: float = 1.0,
                 max_jump_frac_a: float = 0.01,
                 max_jump_frac_b: float = 0.01) -> str:
    verdict = "PASS" if report_passes(
        report, required_matches=required_matches,
        min_combo=min_combo, max_draws=max_draws,
        min_wins_a=min_wins_a, min_wins_b=min_wins_b,
        min_hits_a=min_hits_a, min_hits_b=min_hits_b,
        min_combo_b=min_combo_b,
        min_close_frac=min_close_frac, max_sky_frac_a=max_sky_frac_a,
        max_sky_frac_b=max_sky_frac_b,
        max_back_frac_a=max_back_frac_a, max_back_frac_b=max_back_frac_b,
        max_tap_back_frac_a=max_tap_back_frac_a,
        max_tap_back_frac_b=max_tap_back_frac_b,
        min_strafe_frac_a=min_strafe_frac_a,
        min_strafe_frac_b=min_strafe_frac_b,
        min_opener_strafe_frac_a=min_opener_strafe_frac_a,
        min_opener_strafe_frac_b=min_opener_strafe_frac_b,
        min_opener_strafe_hold_frac_a=min_opener_strafe_hold_frac_a,
        min_opener_strafe_hold_frac_b=min_opener_strafe_hold_frac_b,
        min_opener_pressure_frac_a=min_opener_pressure_frac_a,
        min_opener_pressure_frac_b=min_opener_pressure_frac_b,
        max_strafe_frac_a=max_strafe_frac_a,
        max_strafe_frac_b=max_strafe_frac_b,
        max_jump_frac_a=max_jump_frac_a, max_jump_frac_b=max_jump_frac_b,
    ) else "FAIL"
    return (
        f"{verdict} matches={report.matches}/{required_matches} "
        f"wins={report.wins[0]}/{report.wins[1]} "
        f"min_wins={min_wins_a}/{min_wins_b} "
        f"draws={report.draws} max_draws={max_draws} "
        f"max_combo={report.max_combo[0]}/{report.max_combo[1]} "
        f"model_combo={report.max_combo[0]}/{report.max_combo[1]} "
        f"threshold={min_combo}/{min_combo_b} "
        f"hits_min={report.hits_per_min[0]:.1f}/{report.hits_per_min[1]:.1f} "
        f"min_hits={min_hits_a:.1f}/{min_hits_b:.1f} "
        f"close={report.close_frac:.2f} min_close={min_close_frac:.2f} "
        f"dist={report.avg_dist:.2f} "
        f"pitch_abs={report.avg_abs_pitch[0]:.1f}/{report.avg_abs_pitch[1]:.1f} "
        f"sky={report.sky_frac[0]:.2f}/{report.sky_frac[1]:.2f} "
        f"max_sky={max_sky_frac_a:.2f}/{max_sky_frac_b:.2f} "
        f"escape={report.back_frac[0]:.2f}/{report.strafe_frac[0]:.2f}/{report.jump_frac[0]:.2f} "
        f"limit={max_back_frac_a:.2f}/{min_strafe_frac_a:.2f}-{max_strafe_frac_a:.2f}/{max_jump_frac_a:.2f} "
        f"opener_strafe={report.opener_strafe_frac[0]:.2f} "
        f"min_opener_strafe={min_opener_strafe_frac_a:.2f} "
        f"opener_strafe_hold={report.opener_strafe_hold_frac[0]:.2f} "
        f"min_opener_strafe_hold={min_opener_strafe_hold_frac_a:.2f} "
        f"opener_pressure={report.opener_pressure_frac[0]:.2f} "
        f"min_opener_pressure={min_opener_pressure_frac_a:.2f} "
        f"tap_back={report.tap_back_frac[0]:.2f} max_tap_back={max_tap_back_frac_a:.2f} "
        f"escape_b={report.back_frac[1]:.2f}/{report.strafe_frac[1]:.2f}/{report.jump_frac[1]:.2f} "
        f"limit_b={max_back_frac_b:.2f}/{min_strafe_frac_b:.2f}-{max_strafe_frac_b:.2f}/{max_jump_frac_b:.2f} "
        f"opener_strafe_b={report.opener_strafe_frac[1]:.2f} "
        f"min_opener_strafe_b={min_opener_strafe_frac_b:.2f} "
        f"opener_strafe_hold_b={report.opener_strafe_hold_frac[1]:.2f} "
        f"min_opener_strafe_hold_b={min_opener_strafe_hold_frac_b:.2f} "
        f"opener_pressure_b={report.opener_pressure_frac[1]:.2f} "
        f"min_opener_pressure_b={min_opener_pressure_frac_b:.2f} "
        f"tap_back_b={report.tap_back_frac[1]:.2f} max_tap_back_b={max_tap_back_frac_b:.2f} "
        f"sample={str(report.sample).lower()} target={report.target_hits} "
        f"arena={report.arena_size:.1f} gap={report.spawn_gap:.1f} "
        f"cps={report.cps:.1f} rot={report.rot_speed:.1f}"
    )


def events_text(events: Iterable[MatchEvent]) -> str:
    parts = []
    for idx, event in enumerate(events, start=1):
        parts.append(
            f"#{idx}:w={event.winner}:combo={event.combo[0]}/{event.combo[1]}:"
            f"best={event.max_combo[0]}/{event.max_combo[1]}:draws={event.draws}"
        )
    return " ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", default=DEFAULT_MODEL_A)
    parser.add_argument("--model-b", default=DEFAULT_MODEL_B)
    parser.add_argument("--matches", type=int, default=8)
    parser.add_argument("--min-combo", type=int, default=12)
    parser.add_argument("--min-combo-b", type=int, default=0)
    parser.add_argument("--max-draws", type=int, default=0)
    parser.add_argument("--min-wins-a", type=int, default=1)
    parser.add_argument("--min-wins-b", type=int, default=0)
    parser.add_argument("--min-hits-a", type=float, default=18.0)
    parser.add_argument("--min-hits-b", type=float, default=0.0)
    parser.add_argument("--min-close-frac", type=float, default=0.08)
    parser.add_argument("--max-sky-frac-a", type=float, default=0.02)
    parser.add_argument("--max-sky-frac-b", type=float, default=0.02)
    parser.add_argument("--max-back-frac-a", type=float, default=0.01)
    parser.add_argument("--max-back-frac-b", type=float, default=0.01)
    parser.add_argument("--max-tap-back-frac-a", type=float, default=0.0)
    parser.add_argument("--max-tap-back-frac-b", type=float, default=0.0)
    parser.add_argument("--min-strafe-frac-a", type=float, default=0.50)
    parser.add_argument("--min-strafe-frac-b", type=float, default=0.50)
    parser.add_argument("--min-opener-strafe-frac-a", type=float, default=0.75)
    parser.add_argument("--min-opener-strafe-frac-b", type=float, default=0.75)
    parser.add_argument("--min-opener-strafe-hold-frac-a", type=float, default=0.70)
    parser.add_argument("--min-opener-strafe-hold-frac-b", type=float, default=0.70)
    parser.add_argument("--min-opener-pressure-frac-a", type=float, default=0.60)
    parser.add_argument("--min-opener-pressure-frac-b", type=float, default=0.60)
    parser.add_argument("--max-strafe-frac-a", type=float, default=1.0)
    parser.add_argument("--max-strafe-frac-b", type=float, default=1.0)
    parser.add_argument("--max-jump-frac-a", type=float, default=0.01)
    parser.add_argument("--max-jump-frac-b", type=float, default=0.01)
    parser.add_argument("--close-dist", type=float, default=3.25)
    parser.add_argument("--sky-pitch-deg", type=float, default=60.0)
    parser.add_argument("--opener-ticks", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=200_000)
    parser.add_argument("--cps", type=float, default=10.0)
    parser.add_argument("--events", action="store_true")
    args = parser.parse_args(argv)

    for model in (args.model_a, args.model_b):
        if model not in SCRIPTED_MODELS and not Path(model).exists():
            print(f"MISSING model={model}")
            return 2

    report = run_arena_combo_check(
        args.model_a, args.model_b,
        matches=args.matches,
        max_steps=args.max_steps,
        cps=args.cps,
        close_dist=args.close_dist,
        sky_pitch_deg=args.sky_pitch_deg,
        opener_ticks=args.opener_ticks,
    )
    print(verdict_text(
        report, required_matches=args.matches,
        min_combo=args.min_combo, max_draws=args.max_draws,
        min_wins_a=args.min_wins_a, min_wins_b=args.min_wins_b,
        min_hits_a=args.min_hits_a, min_hits_b=args.min_hits_b,
        min_combo_b=args.min_combo_b,
        min_close_frac=args.min_close_frac,
        max_sky_frac_a=args.max_sky_frac_a,
        max_sky_frac_b=args.max_sky_frac_b,
        max_back_frac_a=args.max_back_frac_a,
        max_back_frac_b=args.max_back_frac_b,
        max_tap_back_frac_a=args.max_tap_back_frac_a,
        max_tap_back_frac_b=args.max_tap_back_frac_b,
        min_strafe_frac_a=args.min_strafe_frac_a,
        min_strafe_frac_b=args.min_strafe_frac_b,
        min_opener_strafe_frac_a=args.min_opener_strafe_frac_a,
        min_opener_strafe_frac_b=args.min_opener_strafe_frac_b,
        min_opener_strafe_hold_frac_a=args.min_opener_strafe_hold_frac_a,
        min_opener_strafe_hold_frac_b=args.min_opener_strafe_hold_frac_b,
        min_opener_pressure_frac_a=args.min_opener_pressure_frac_a,
        min_opener_pressure_frac_b=args.min_opener_pressure_frac_b,
        max_strafe_frac_a=args.max_strafe_frac_a,
        max_strafe_frac_b=args.max_strafe_frac_b,
        max_jump_frac_a=args.max_jump_frac_a,
        max_jump_frac_b=args.max_jump_frac_b,
    ))
    if args.events:
        print(events_text(report.events))
    return 0 if report_passes(
        report, required_matches=args.matches,
        min_combo=args.min_combo, max_draws=args.max_draws,
        min_wins_a=args.min_wins_a, min_wins_b=args.min_wins_b,
        min_hits_a=args.min_hits_a, min_hits_b=args.min_hits_b,
        min_combo_b=args.min_combo_b,
        min_close_frac=args.min_close_frac,
        max_sky_frac_a=args.max_sky_frac_a,
        max_sky_frac_b=args.max_sky_frac_b,
        max_back_frac_a=args.max_back_frac_a,
        max_back_frac_b=args.max_back_frac_b,
        max_tap_back_frac_a=args.max_tap_back_frac_a,
        max_tap_back_frac_b=args.max_tap_back_frac_b,
        min_strafe_frac_a=args.min_strafe_frac_a,
        min_strafe_frac_b=args.min_strafe_frac_b,
        min_opener_strafe_frac_a=args.min_opener_strafe_frac_a,
        min_opener_strafe_frac_b=args.min_opener_strafe_frac_b,
        min_opener_strafe_hold_frac_a=args.min_opener_strafe_hold_frac_a,
        min_opener_strafe_hold_frac_b=args.min_opener_strafe_hold_frac_b,
        min_opener_pressure_frac_a=args.min_opener_pressure_frac_a,
        min_opener_pressure_frac_b=args.min_opener_pressure_frac_b,
        max_strafe_frac_a=args.max_strafe_frac_a,
        max_strafe_frac_b=args.max_strafe_frac_b,
        max_jump_frac_a=args.max_jump_frac_a,
        max_jump_frac_b=args.max_jump_frac_b,
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
