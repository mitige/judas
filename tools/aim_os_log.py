from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class AimSample:
    yaw_err: float
    pitch_err: float
    cmd_yaw: float
    cmd_pitch: float
    dx: int
    dy: int
    stall: int
    sent_yaw: float = 0.0
    sent_pitch: float = 0.0
    applied_yaw: float = 0.0
    applied_pitch: float = 0.0
    pending_yaw: float = 0.0
    pending_pitch: float = 0.0
    step_yaw: float = 0.0
    step_pitch: float = 0.0
    yaw_sign: int = 1
    pitch_sign: int = 1


@dataclass(frozen=True)
class AimReport:
    samples: int
    yaw_abs_mean: float
    pitch_abs_mean: float
    yaw_abs_p95: float
    pitch_abs_p95: float
    yaw_abs_max: float
    pitch_abs_max: float
    max_stall: int
    moving_ticks: int
    yaw_bad_apply_ticks: int = 0
    pitch_bad_apply_ticks: int = 0
    yaw_no_apply_ticks: int = 0
    pitch_no_apply_ticks: int = 0
    yaw_cmd_drift_p95: float = 0.0
    pitch_cmd_drift_p95: float = 0.0
    yaw_cmd_drift_max: float = 0.0
    pitch_cmd_drift_max: float = 0.0

    @property
    def divergent(self) -> bool:
        if self.samples < 8:
            return False
        limit = max(3, int(self.samples * 0.15))
        return self.yaw_bad_apply_ticks >= limit or self.pitch_bad_apply_ticks >= limit

    @property
    def one_to_one(self) -> bool:
        return (
            self.samples >= 20
            and self.yaw_cmd_drift_p95 <= 1.25
            and self.pitch_cmd_drift_p95 <= 1.25
            and self.yaw_cmd_drift_max <= 2.50
            and self.pitch_cmd_drift_max <= 2.50
        )

    @property
    def precise(self) -> bool:
        return (
            self.samples >= 20
            and not self.divergent
            and self.one_to_one
            and self.yaw_abs_p95 <= 5.0
            and self.pitch_abs_p95 <= 5.0
            and self.yaw_abs_max <= 15.0
            and self.pitch_abs_max <= 15.0
            and self.max_stall < 10
        )


def parse_lines(lines: Iterable[str]) -> list[AimSample]:
    samples: list[AimSample] = []
    for raw in lines:
        fields = _fields(raw.strip())
        if not fields or "yawErr" not in fields or "pitchErr" not in fields:
            continue
        try:
            samples.append(AimSample(
                yaw_err=float(fields["yawErr"]),
                pitch_err=float(fields["pitchErr"]),
                cmd_yaw=float(fields.get("cmdYaw", 0.0)),
                cmd_pitch=float(fields.get("cmdPitch", 0.0)),
                dx=int(fields.get("dx", 0)),
                dy=int(fields.get("dy", 0)),
                stall=int(fields.get("stall", 0)),
                sent_yaw=float(fields.get("sentYaw", 0.0)),
                sent_pitch=float(fields.get("sentPitch", 0.0)),
                applied_yaw=float(fields.get("appliedYaw", 0.0)),
                applied_pitch=float(fields.get("appliedPitch", 0.0)),
                pending_yaw=float(fields.get("pendingYaw", 0.0)),
                pending_pitch=float(fields.get("pendingPitch", 0.0)),
                step_yaw=float(fields.get("stepYaw", 0.0)),
                step_pitch=float(fields.get("stepPitch", 0.0)),
                yaw_sign=int(fields.get("yawSign", 1)),
                pitch_sign=int(fields.get("pitchSign", 1)),
            ))
        except ValueError:
            continue
    return samples


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


def analyze_lines(lines: Iterable[str]) -> AimReport:
    samples = parse_lines(lines)
    if not samples:
        return AimReport(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)
    yaw = [abs(s.yaw_err) for s in samples]
    pitch = [abs(s.pitch_err) for s in samples]
    yaw_bad, pitch_bad, yaw_no_apply, pitch_no_apply = _movement_diagnostics(samples)
    yaw_cmd_drift, pitch_cmd_drift = _command_drifts(samples)
    return AimReport(
        samples=len(samples),
        yaw_abs_mean=sum(yaw) / len(yaw),
        pitch_abs_mean=sum(pitch) / len(pitch),
        yaw_abs_p95=_percentile(yaw, 0.95),
        pitch_abs_p95=_percentile(pitch, 0.95),
        yaw_abs_max=max(yaw),
        pitch_abs_max=max(pitch),
        max_stall=max(s.stall for s in samples),
        moving_ticks=sum(1 for s in samples if s.dx != 0 or s.dy != 0),
        yaw_bad_apply_ticks=yaw_bad,
        pitch_bad_apply_ticks=pitch_bad,
        yaw_no_apply_ticks=yaw_no_apply,
        pitch_no_apply_ticks=pitch_no_apply,
        yaw_cmd_drift_p95=_percentile(yaw_cmd_drift, 0.95),
        pitch_cmd_drift_p95=_percentile(pitch_cmd_drift, 0.95),
        yaw_cmd_drift_max=max(yaw_cmd_drift) if yaw_cmd_drift else 0.0,
        pitch_cmd_drift_max=max(pitch_cmd_drift) if pitch_cmd_drift else 0.0,
    )


def verdict_text(report: AimReport, lines: Iterable[str] | None = None) -> str:
    if report.samples == 0:
        verdict = "NO_TARGET" if lines is not None and _has_event(lines, "event=no_target") else "NO_SAMPLES"
    elif report.samples < 20:
        verdict = "WARMUP"
    elif report.max_stall >= 10:
        verdict = "STALL"
    elif report.divergent:
        verdict = "DIVERGE"
    elif not report.one_to_one:
        verdict = "NOT_1TO1"
    elif report.precise:
        verdict = "PRECISE"
    else:
        verdict = "LOOSE"
    return (
        f"{verdict} samples={report.samples} moving={report.moving_ticks} "
        f"yaw_mean={report.yaw_abs_mean:.2f} pitch_mean={report.pitch_abs_mean:.2f} "
        f"yaw_p95={report.yaw_abs_p95:.2f} pitch_p95={report.pitch_abs_p95:.2f} "
        f"yaw_max={report.yaw_abs_max:.2f} pitch_max={report.pitch_abs_max:.2f} "
        f"max_stall={report.max_stall} "
        f"bad_apply={report.yaw_bad_apply_ticks}/{report.pitch_bad_apply_ticks} "
        f"no_apply={report.yaw_no_apply_ticks}/{report.pitch_no_apply_ticks} "
        f"cmd_drift_p95={report.yaw_cmd_drift_p95:.2f}/{report.pitch_cmd_drift_p95:.2f} "
        f"cmd_drift_max={report.yaw_cmd_drift_max:.2f}/{report.pitch_cmd_drift_max:.2f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--all", action="store_true",
                        help="analyze the whole append-only log instead of the latest session")
    parser.add_argument("log", type=Path)
    args = parser.parse_args(argv)

    lines = args.log.read_text(errors="replace").splitlines() if args.log.exists() else []
    if not args.all:
        lines = latest_session_lines(lines)
    report = analyze_lines(lines)
    print(verdict_text(report, lines))
    if args.strict:
        return 0 if report.precise else 1
    return 1 if report.samples == 0 or report.max_stall >= 10 or report.divergent else 0


def _command_drifts(samples: list[AimSample]) -> tuple[list[float], list[float]]:
    yaw_acc = 0.0
    pitch_acc = 0.0
    yaw: list[float] = []
    pitch: list[float] = []
    for s in samples:
        yaw_acc += s.sent_yaw - s.cmd_yaw
        pitch_acc += s.sent_pitch - s.cmd_pitch
        yaw.append(abs(yaw_acc))
        pitch.append(abs(pitch_acc))
    return yaw, pitch


def _movement_diagnostics(samples: list[AimSample]) -> tuple[int, int, int, int]:
    yaw_bad = 0
    pitch_bad = 0
    yaw_no_apply = 0
    pitch_no_apply = 0
    for prev, cur in zip(samples, samples[1:]):
        if _bad_apply(prev.yaw_err, cur.applied_yaw):
            yaw_bad += 1
        if _bad_apply(prev.pitch_err, cur.applied_pitch):
            pitch_bad += 1
        if abs(prev.sent_yaw) > 0.5 and abs(cur.applied_yaw) <= _axis_step(cur.step_yaw):
            yaw_no_apply += 1
        if abs(prev.sent_pitch) > 0.5 and abs(cur.applied_pitch) <= _axis_step(cur.step_pitch):
            pitch_no_apply += 1
    return yaw_bad, pitch_bad, yaw_no_apply, pitch_no_apply


def _bad_apply(prev_error: float, applied: float) -> bool:
    return abs(prev_error) > 2.0 and abs(applied) > 0.5 and prev_error * applied < 0.0


def _axis_step(step: float) -> float:
    return max(0.05, abs(step) * 0.5)


def _has_event(lines: Iterable[str], event: str) -> bool:
    return any(line.startswith(event) for line in lines)


def _fields(line: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in line.split():
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k] = v
    return out


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * q))
    return ordered[max(0, min(len(ordered) - 1, idx))]


if __name__ == "__main__":
    raise SystemExit(main())