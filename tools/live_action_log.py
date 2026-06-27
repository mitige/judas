from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


LIVE_GUARDED_MODEL_MARKERS = (
    "combo_god_leaderboard10_combo12",
    "combo_god_attn96_combo12",
    "combo_god_consistent",
    "combo_god_candidate_freshopt",
    "combo_god_bodyaim96_combo12",
)
DEFAULT_REQUIRE_MODEL = "|".join(LIVE_GUARDED_MODEL_MARKERS)


@dataclass(frozen=True)
class LiveActionReport:
    samples: int
    model: str
    back: int
    tap_back: int
    strafe: int
    strafe_flips: int
    strafe_runs: int
    opener_samples: int
    opener_strafe: int
    opener_strafe_hold: int
    opener_pressure: int
    attack: int
    jump: int
    sky: int
    pitch_abs_p95: float
    pitch_err_abs_p95: float
    max_own_combo: int
    max_opp_combo: int
    own_hit_events: int
    own_hit_wtap: int
    under_combo_samples: int
    under_combo_attack: int
    under_combo_counter_hits: int
    far_under_combo_samples: int
    far_under_combo_attack: int

    @property
    def escape(self) -> int:
        return self.back + self.jump


def latest_session_lines(lines: Iterable[str]) -> list[str]:
    all_lines = [raw.rstrip("\n") for raw in lines]
    session: list[str] = []
    seen_start = False
    for line in all_lines:
        if line.startswith("event=start"):
            session = [line]
            seen_start = True
        elif seen_start:
            session.append(line)
    return session if seen_start else all_lines


def analyze_lines(lines: Iterable[str], sky_pitch_deg: float = 60.0,
                  opener_ticks: int = 20,
                  far_under_dist: float = 3.30) -> LiveActionReport:
    samples = 0
    model = ""
    back = tap_back = strafe = strafe_flips = strafe_runs = attack = jump = sky = 0
    opener_samples = opener_strafe = opener_strafe_pos = opener_strafe_neg = opener_pressure = 0
    max_own_combo = max_opp_combo = 0
    own_combo = opp_combo = 0
    own_hit_events = own_hit_wtap = 0
    under_combo_samples = under_combo_attack = under_combo_counter_hits = 0
    far_under_combo_samples = far_under_combo_attack = 0
    prev_own_hits: int | None = None
    prev_opp_hits: int | None = None
    pending_own_hit_wtap = 0
    prev_strafe = 0
    pitch_abs: list[float] = []
    pitch_err_abs: list[float] = []

    for raw in lines:
        fields = _fields(raw.strip())
        if not fields:
            continue
        if fields.get("event") == "start":
            model = fields.get("model", model)
            continue
        if "tick" not in fields:
            continue
        samples += 1
        model = fields.get("model", model)
        forward = int(float(fields.get("forward", "0")))
        strafe_v = int(float(fields.get("strafe", "0")))
        jump_v = fields.get("jump", "false").lower() == "true"
        sprint_v = fields.get("sprint", "false").lower() == "true"
        attack_v = fields.get("attack", "false").lower() == "true"
        own_pitch = abs(float(fields.get("ownPitch", "0")))
        pitch_err = abs(float(fields.get("pitchErr", "0")))
        own_hurt = int(float(fields.get("ownHurt", "0")))
        opp_hurt = int(float(fields.get("oppHurt", "0")))
        own_hits = int(float(fields.get("ownHits", "0")))
        opp_hits = int(float(fields.get("oppHits", "0")))
        dist = float(fields.get("dist", "0"))
        wtap_release = (
            forward == 0 and not sprint_v and strafe_v != 0 and not jump_v
        )

        back += int(forward < 0 and sprint_v)
        tap_back += int(forward < 0 and not sprint_v)
        strafe += int(strafe_v != 0)
        if samples <= opener_ticks:
            opener_samples += 1
            opener_strafe += int(strafe_v != 0)
            opener_strafe_pos += int(strafe_v > 0)
            opener_strafe_neg += int(strafe_v < 0)
            opener_pressure += int(forward > 0 and strafe_v != 0)
        if strafe_v != 0 and strafe_v != prev_strafe:
            strafe_runs += 1
        if prev_strafe != 0 and strafe_v != 0 and strafe_v != prev_strafe:
            strafe_flips += 1
        prev_strafe = strafe_v
        attack += int(attack_v)
        jump += int(jump_v)
        sky += int(own_pitch >= sky_pitch_deg)
        pitch_abs.append(own_pitch)
        pitch_err_abs.append(pitch_err)
        under_combo_now = own_hurt > opp_hurt + 1
        if under_combo_now:
            under_combo_samples += 1
            under_combo_attack += int(attack_v)
            if dist >= far_under_dist:
                far_under_combo_samples += 1
                far_under_combo_attack += int(attack_v)
        if prev_own_hits is not None and prev_opp_hits is not None:
            if own_hits < prev_own_hits or opp_hits < prev_opp_hits:
                own_combo = 0
                opp_combo = 0
                pending_own_hit_wtap = 0
            elif pending_own_hit_wtap:
                own_hit_wtap += pending_own_hit_wtap * int(wtap_release)
                pending_own_hit_wtap = 0
            own_delta = max(0, own_hits - prev_own_hits)
            opp_delta = max(0, opp_hits - prev_opp_hits)
            if own_delta > 0:
                own_hit_events += 1
                under_combo_counter_hits += int(under_combo_now)
                if wtap_release:
                    own_hit_wtap += 1
                else:
                    pending_own_hit_wtap = 1
            if own_delta and not opp_delta:
                own_combo += own_delta
                opp_combo = 0
            elif opp_delta and not own_delta:
                opp_combo += opp_delta
                own_combo = 0
            elif own_delta and opp_delta:
                own_combo = own_delta
                opp_combo = opp_delta
            max_own_combo = max(max_own_combo, own_combo)
            max_opp_combo = max(max_opp_combo, opp_combo)
        prev_own_hits = own_hits
        prev_opp_hits = opp_hits

    return LiveActionReport(
        samples=samples,
        model=model,
        back=back,
        tap_back=tap_back,
        strafe=strafe,
        strafe_flips=strafe_flips,
        strafe_runs=strafe_runs,
        opener_samples=opener_samples,
        opener_strafe=opener_strafe,
        opener_strafe_hold=max(opener_strafe_pos, opener_strafe_neg),
        opener_pressure=opener_pressure,
        attack=attack,
        jump=jump,
        sky=sky,
        pitch_abs_p95=_percentile(pitch_abs, 0.95),
        pitch_err_abs_p95=_percentile(pitch_err_abs, 0.95),
        max_own_combo=max_own_combo,
        max_opp_combo=max_opp_combo,
        own_hit_events=own_hit_events,
        own_hit_wtap=own_hit_wtap,
        under_combo_samples=under_combo_samples,
        under_combo_attack=under_combo_attack,
        under_combo_counter_hits=under_combo_counter_hits,
        far_under_combo_samples=far_under_combo_samples,
        far_under_combo_attack=far_under_combo_attack,
    )


def verdict_text(report: LiveActionReport, *,
                 min_samples: int = 20,
                 max_sky_frac: float = 0.02,
                 max_pitch_err_p95: float = 45.0,
                 max_tap_back_frac: float = 0.0,
                 max_attack_cps: float = 10.0,
                 min_strafe_frac: float = 0.50,
                 min_opener_strafe_frac: float = 0.75,
                 min_opener_strafe_hold_frac: float = 0.70,
                 min_opener_pressure_frac: float = 0.60,
                 max_strafe_flip_frac: float = 0.10,
                 min_strafe_hold_avg: float = 3.0,
                 min_max_own_combo: int = 0,
                 max_max_opp_combo: int = 999999,
                 min_hit_wtap_frac: float = 0.75,
                 min_under_combo_attack_frac: float = 0.0,
                 min_under_combo_counter_hit_frac: float = 0.0,
                 max_far_under_combo_frac: float = 1.0,
                 require_model: str = "") -> str:
    denom = max(1, report.samples)
    sky_frac = report.sky / denom
    tap_back_frac = report.tap_back / denom
    attack_cps = 20.0 * report.attack / denom
    strafe_frac = report.strafe / denom
    opener_strafe_frac = report.opener_strafe / max(1, report.opener_samples)
    opener_strafe_hold_frac = report.opener_strafe_hold / max(1, report.opener_samples)
    opener_pressure_frac = report.opener_pressure / max(1, report.opener_samples)
    strafe_flip_frac = report.strafe_flips / denom
    strafe_hold_avg = report.strafe / max(1, report.strafe_runs)
    under_combo_attack_frac = (
        report.under_combo_attack / max(1, report.under_combo_samples)
    )
    under_combo_counter_hit_frac = (
        report.under_combo_counter_hits / max(1, report.under_combo_samples)
    )
    hit_wtap_frac = report.own_hit_wtap / max(1, report.own_hit_events)
    hit_wtap_ok = (
        min_hit_wtap_frac < 0.0
        or report.own_hit_events == 0
        or hit_wtap_frac >= min_hit_wtap_frac
    )
    far_under_combo_frac = (
        report.far_under_combo_samples / max(1, report.under_combo_samples)
    )
    ok = (
        report.samples >= min_samples
        and report.escape == 0
        and tap_back_frac <= max_tap_back_frac
        and attack_cps <= max_attack_cps
        and strafe_frac >= min_strafe_frac
        and opener_strafe_frac >= min_opener_strafe_frac
        and opener_strafe_hold_frac >= min_opener_strafe_hold_frac
        and opener_pressure_frac >= min_opener_pressure_frac
        and strafe_flip_frac <= max_strafe_flip_frac
        and strafe_hold_avg >= min_strafe_hold_avg
        and sky_frac <= max_sky_frac
        and report.pitch_err_abs_p95 <= max_pitch_err_p95
        and report.max_own_combo >= min_max_own_combo
        and report.max_opp_combo <= max_max_opp_combo
        and hit_wtap_ok
        and under_combo_attack_frac >= min_under_combo_attack_frac
        and under_combo_counter_hit_frac >= min_under_combo_counter_hit_frac
        and far_under_combo_frac <= max_far_under_combo_frac
        and _model_matches(require_model, report.model)
    )
    status = "PASS" if ok else "FAIL"
    return (
        f"{status} samples={report.samples} min_samples={min_samples} "
        f"model={report.model or '-'} require_model={require_model or '-'} "
        f"escape={report.back}/{report.jump} "
        f"strafe={report.strafe} strafe_frac={strafe_frac:.3f} "
        f"min_strafe_frac={min_strafe_frac:.3f} "
        f"opener_strafe={report.opener_strafe}/{report.opener_samples} "
        f"opener_strafe_frac={opener_strafe_frac:.3f} "
        f"min_opener_strafe_frac={min_opener_strafe_frac:.3f} "
        f"opener_strafe_hold={report.opener_strafe_hold}/{report.opener_samples} "
        f"opener_strafe_hold_frac={opener_strafe_hold_frac:.3f} "
        f"min_opener_strafe_hold_frac={min_opener_strafe_hold_frac:.3f} "
        f"opener_pressure={report.opener_pressure}/{report.opener_samples} "
        f"opener_pressure_frac={opener_pressure_frac:.3f} "
        f"min_opener_pressure_frac={min_opener_pressure_frac:.3f} "
        f"strafe_flips={report.strafe_flips} "
        f"strafe_flip_frac={strafe_flip_frac:.3f} "
        f"max_strafe_flip_frac={max_strafe_flip_frac:.3f} "
        f"strafe_runs={report.strafe_runs} "
        f"strafe_hold_avg={strafe_hold_avg:.2f} "
        f"min_strafe_hold_avg={min_strafe_hold_avg:.2f} "
        f"tap_back={report.tap_back} tap_back_frac={tap_back_frac:.3f} "
        f"max_tap_back_frac={max_tap_back_frac:.3f} "
        f"attack={report.attack} attack_cps={attack_cps:.2f} "
        f"max_attack_cps={max_attack_cps:.2f} "
        f"sky={report.sky} sky_frac={sky_frac:.3f} max_sky_frac={max_sky_frac:.3f} "
        f"pitch_p95={report.pitch_abs_p95:.2f} "
        f"pitch_err_p95={report.pitch_err_abs_p95:.2f} "
        f"max_pitch_err_p95={max_pitch_err_p95:.2f} "
        f"max_own_combo={report.max_own_combo} min_max_own_combo={min_max_own_combo} "
        f"max_opp_combo={report.max_opp_combo} max_max_opp_combo={max_max_opp_combo} "
        f"hit_wtap={report.own_hit_wtap}/{report.own_hit_events} "
        f"hit_wtap_frac={hit_wtap_frac:.3f} "
        f"min_hit_wtap_frac={min_hit_wtap_frac:.3f} "
        f"under_combo_attack={report.under_combo_attack}/{report.under_combo_samples} "
        f"under_combo_attack_frac={under_combo_attack_frac:.3f} "
        f"min_under_combo_attack_frac={min_under_combo_attack_frac:.3f} "
        f"under_combo_counter_hit={report.under_combo_counter_hits}/{report.under_combo_samples} "
        f"under_combo_counter_hit_frac={under_combo_counter_hit_frac:.3f} "
        f"min_under_combo_counter_hit_frac={min_under_combo_counter_hit_frac:.3f} "
        f"far_under_combo={report.far_under_combo_samples}/{report.under_combo_samples} "
        f"far_under_combo_frac={far_under_combo_frac:.3f} "
        f"max_far_under_combo_frac={max_far_under_combo_frac:.3f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--sky-pitch-deg", type=float, default=60.0)
    parser.add_argument("--max-sky-frac", type=float, default=0.02)
    parser.add_argument("--max-pitch-err-p95", type=float, default=45.0)
    parser.add_argument("--max-tap-back-frac", type=float, default=0.0)
    parser.add_argument("--max-attack-cps", type=float, default=10.0)
    parser.add_argument("--min-strafe-frac", type=float, default=0.50)
    parser.add_argument("--opener-ticks", type=int, default=20)
    parser.add_argument("--min-opener-strafe-frac", type=float, default=0.75)
    parser.add_argument("--min-opener-strafe-hold-frac", type=float, default=0.70)
    parser.add_argument("--min-opener-pressure-frac", type=float, default=0.60)
    parser.add_argument("--max-strafe-flip-frac", type=float, default=0.10)
    parser.add_argument("--min-strafe-hold-avg", type=float, default=3.0)
    parser.add_argument("--min-max-own-combo", type=int, default=0)
    parser.add_argument("--max-max-opp-combo", type=int, default=999999)
    parser.add_argument("--min-hit-wtap-frac", type=float, default=0.75)
    parser.add_argument("--min-under-combo-attack-frac", type=float, default=0.0)
    parser.add_argument("--min-under-combo-counter-hit-frac", type=float, default=0.0)
    parser.add_argument("--max-far-under-combo-frac", type=float, default=1.0)
    parser.add_argument("--far-under-dist", type=float, default=3.30)
    parser.add_argument("--require-model", default=DEFAULT_REQUIRE_MODEL)
    parser.add_argument("log", type=Path)
    args = parser.parse_args(argv)

    lines = args.log.read_text(errors="replace").splitlines() if args.log.exists() else []
    if not args.all:
        lines = latest_session_lines(lines)
    report = analyze_lines(
        lines,
        sky_pitch_deg=args.sky_pitch_deg,
        opener_ticks=args.opener_ticks,
        far_under_dist=args.far_under_dist,
    )
    text = verdict_text(
        report,
        min_samples=args.min_samples,
        max_sky_frac=args.max_sky_frac,
        max_pitch_err_p95=args.max_pitch_err_p95,
        max_tap_back_frac=args.max_tap_back_frac,
        max_attack_cps=args.max_attack_cps,
        min_strafe_frac=args.min_strafe_frac,
        min_opener_strafe_frac=args.min_opener_strafe_frac,
        min_opener_strafe_hold_frac=args.min_opener_strafe_hold_frac,
        min_opener_pressure_frac=args.min_opener_pressure_frac,
        max_strafe_flip_frac=args.max_strafe_flip_frac,
        min_strafe_hold_avg=args.min_strafe_hold_avg,
        min_max_own_combo=args.min_max_own_combo,
        max_max_opp_combo=args.max_max_opp_combo,
        min_hit_wtap_frac=args.min_hit_wtap_frac,
        min_under_combo_attack_frac=args.min_under_combo_attack_frac,
        min_under_combo_counter_hit_frac=args.min_under_combo_counter_hit_frac,
        max_far_under_combo_frac=args.max_far_under_combo_frac,
        require_model=args.require_model,
    )
    print(text)
    if not args.strict:
        return 1 if report.samples == 0 else 0
    return 0 if text.startswith("PASS ") else 1


def _fields(line: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in line.split():
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k] = v
    return out


def _model_matches(require_model: str, model: str) -> bool:
    if not require_model:
        return True
    required = [part.strip() for part in require_model.split("|") if part.strip()]
    return any(part in model for part in required)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * q))
    return ordered[max(0, min(len(ordered) - 1, idx))]


if __name__ == "__main__":
    raise SystemExit(main())
