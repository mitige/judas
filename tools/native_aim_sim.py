from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from random import Random


DEFAULT_JAVA = Path("mod/src/main/java/dev/judas/bridge/ActionApplier.java")
DEFAULT_BASE_STEP = 0.15
BASE_STEPS = (0.0096, 0.05145, 0.15, 0.6144)


@dataclass(frozen=True)
class NativeAimConstants:
    native_max: float
    max_counts: int
    aim_lock_blend: float
    aim_fine_lock_deg: float
    aim_fine_lock_blend: float
    demand_gain: float
    fine_one_to_one_deg: float
    sign_flip_ticks: int
    cmd_flip_guard_ticks: int
    pending_stale_ticks: int
    reversal_settle_ticks: int


@dataclass(frozen=True)
class NativeAimReport:
    cases: int
    failures: int
    worst_final_error: float
    worst_abs_error: float
    worst_growth_over_limit: float
    sign_corrections: int
    reversal_flips: int
    nonzero_reversal_sends: int
    max_abs_counts: int
    worst_command_error: float = 0.0

    @property
    def stable(self) -> bool:
        return self.failures == 0 and self.nonzero_reversal_sends == 0

    @property
    def field_stable(self) -> bool:
        return (
            self.nonzero_reversal_sends == 0
            and self.worst_command_error <= 0.50
            and self.worst_final_error <= 110.0
            and self.worst_growth_over_limit <= 100.0
        )


def load_constants(path: Path = DEFAULT_JAVA) -> NativeAimConstants:
    source = path.read_text(encoding="utf-8")
    return NativeAimConstants(
        native_max=_java_float(source, "NATIVE_MAX_DEG_PER_TICK"),
        max_counts=_java_int(source, "NATIVE_MAX_COUNTS_PER_TICK"),
        aim_lock_blend=_java_float(source, "AIM_LOCK_BLEND"),
        aim_fine_lock_deg=_java_float(source, "AIM_FINE_LOCK_DEG"),
        aim_fine_lock_blend=_java_float(source, "AIM_FINE_LOCK_BLEND"),
        demand_gain=_java_float(source, "NATIVE_DEMAND_GAIN"),
        fine_one_to_one_deg=_java_float(source, "NATIVE_FINE_ONE_TO_ONE_DEG"),
        sign_flip_ticks=_java_int(source, "NATIVE_SIGN_FLIP_TICKS"),
        cmd_flip_guard_ticks=_java_int(source, "NATIVE_CMD_FLIP_GUARD_TICKS"),
        pending_stale_ticks=_java_int(source, "NATIVE_PENDING_STALE_TICKS"),
        reversal_settle_ticks=_java_int(source, "NATIVE_REVERSAL_SETTLE_TICKS"),
    )


def simulate_suite(constants: NativeAimConstants) -> NativeAimReport:
    cases = 0
    failures = 0
    worst_final = 0.0
    worst_abs = 0.0
    worst_growth_over_limit = 0.0
    corrections = 0
    reversal_flips = 0
    nonzero_reversal_sends = 0
    max_abs_counts = 0
    worst_command_error = 0.0

    def record(result: dict, *, expected_corrections: int, tolerance: float,
               max_abs_limit: float) -> None:
        nonlocal cases, failures, worst_final, worst_abs, corrections
        nonlocal reversal_flips, nonzero_reversal_sends, max_abs_counts
        nonlocal worst_command_error, worst_growth_over_limit
        cases += 1
        corrections += result["corrections"]
        reversal_flips += result["reversal_flips"]
        nonzero_reversal_sends += result["nonzero_reversal_sends"]
        worst_final = max(worst_final, abs(result["final_error"]))
        worst_abs = max(worst_abs, result["max_abs_error"])
        worst_growth_over_limit = max(
            worst_growth_over_limit,
            result["max_abs_error"] - max_abs_limit,
        )
        max_abs_counts = max(max_abs_counts, result["max_abs_counts"])
        worst_command_error = max(worst_command_error, result.get("max_command_error", 0.0))
        if result.get("max_command_error", 0.0) > tolerance:
            failures += 1
        if result["max_abs_error"] > max_abs_limit + 1.0e-6:
            failures += 1
        if abs(result["final_error"]) > max(2.0, tolerance * 2.0):
            failures += 1
        if result["corrections"] != expected_corrections:
            failures += 1
        if result["max_abs_counts"] > constants.max_counts:
            failures += 1

    for base_step in BASE_STEPS:
        tolerance = max(0.35, base_step * 3.0)
        for initial_error in (35.0, 90.0):
            for latency_ticks in (0, 1, 2):
                for os_sign in (1.0, -1.0):
                    for os_gain in (0.45, 0.70, 1.0, 1.35, 1.80):
                        result = _run_case(
                            constants, initial_error, os_gain, os_sign,
                            latency_ticks=latency_ticks, seed=None,
                            base_step=base_step,
                        )
                        allowed_growth = (
                            1.0e-6 if os_sign > 0.0
                            else constants.native_max * os_gain + 0.5
                        )
                        record(
                            result,
                            expected_corrections=1 if os_sign < 0.0 else 0,
                            tolerance=tolerance,
                            max_abs_limit=initial_error + allowed_growth,
                        )

    for base_step in BASE_STEPS:
        tolerance = max(0.35, base_step * 3.0)
        for seed in range(100):
            for os_gain in (0.45, 0.70, 1.0, 1.35, 1.80):
                result = _run_case(
                    constants, 35.0, os_gain, 1.0,
                    latency_ticks=None, seed=seed, base_step=base_step,
                )
                record(result, expected_corrections=0, tolerance=tolerance,
                       max_abs_limit=60.0)

            result = _run_case(
                constants, 35.0, 1.0, -1.0,
                latency_ticks=None, seed=seed, base_step=base_step,
            )
            record(result, expected_corrections=1, tolerance=tolerance,
                   max_abs_limit=60.0)

            for os_gain in (0.45, 0.70, 1.0, 1.35, 1.80):
                result = _run_case(
                    constants, 35.0, os_gain, 1.0,
                    latency_ticks=None, seed=seed, base_step=base_step,
                    policy_mode="weak_jitter",
                )
                record(result, expected_corrections=0, tolerance=tolerance,
                       max_abs_limit=60.0)

    return NativeAimReport(
        cases=cases,
        failures=failures,
        worst_final_error=worst_final,
        worst_abs_error=worst_abs,
        worst_growth_over_limit=worst_growth_over_limit,
        sign_corrections=corrections,
        reversal_flips=reversal_flips,
        nonzero_reversal_sends=nonzero_reversal_sends,
        max_abs_counts=max_abs_counts,
        worst_command_error=worst_command_error,
    )


def verdict_text(report: NativeAimReport) -> str:
    verdict = "PASS" if report.field_stable else "FAIL"
    return (
        f"{verdict} cases={report.cases} failures={report.failures} "
        f"worst_final={report.worst_final_error:.3f} "
        f"worst_abs={report.worst_abs_error:.3f} "
        f"worst_growth_over_limit={report.worst_growth_over_limit:.3f} "
        f"sign_corrections={report.sign_corrections} "
        f"reversal_flips={report.reversal_flips} "
        f"nonzero_reversal_sends={report.nonzero_reversal_sends} "
        f"max_counts={report.max_abs_counts} "
        f"worst_cmd_err={report.worst_command_error:.4f}"
    )


def _run_case(constants: NativeAimConstants, initial_error: float, os_gain: float,
              os_sign: float, *, latency_ticks: int | None, seed: int | None,
              base_step: float = DEFAULT_BASE_STEP,
              policy_mode: str = "strong") -> dict:
    err = initial_error
    residual = 0.0
    applied = 0.0
    yaw_sign = 1
    diverge_ticks = 0
    intended_dir = 0
    last_dx = 0
    native_step = base_step
    native_pending = 0.0
    pending_stale = 0
    last_cmd_dir = 0
    cmd_guard_ticks = 0
    reversal_settle_ticks = 0
    corrections = 0
    max_abs_error = abs(err)
    reversal_flips = 0
    nonzero_reversal_sends = 0
    max_abs_counts = 0
    max_command_error = 0.0

    delivery = [0] * latency_ticks if latency_ticks is not None else None
    inflight: list[tuple[int, int]] = []
    rng = Random(seed or 0)
    ticks = 320 if seed is not None else 260

    for tick in range(ticks):
        if delivery is None:
            delivered = sum(dx for due, dx in inflight if due <= tick)
            inflight = [(due, dx) for due, dx in inflight if due > tick]
            applied = os_sign * delivered * base_step * os_gain
            err -= applied
        # Mirror ActionApplier command-fidelity mode: use the vanilla mouse step
        # for the command conversion, and keep OS gain/latency only in applied.
        native_step = base_step
        native_pending = _update_native_pending(
            constants, native_pending, applied, base_step,
        )
        policy_cmd = _policy_command(err, tick, rng, policy_mode)
        cmd = _stabilize_axis(constants, policy_cmd, err, 40.0, 0.05)
        pending_stale = _update_pending_stale(
            constants, pending_stale, native_pending, applied, cmd, base_step,
        )
        if pending_stale >= constants.pending_stale_ticks:
            native_pending = 0.0
            pending_stale = 0
        cmd_dir = _command_dir(cmd)
        flipped = False
        if cmd_dir != 0 and last_cmd_dir != 0 and cmd_dir != last_cmd_dir:
            cmd_guard_ticks = constants.cmd_flip_guard_ticks
            diverge_ticks = 0
            native_pending = 0.0
            pending_stale = 0
            reversal_settle_ticks = constants.reversal_settle_ticks
            reversal_flips += 1
            flipped = True
        elif cmd_guard_ticks > 0:
            cmd_guard_ticks -= 1
        if cmd_dir != 0:
            last_cmd_dir = cmd_dir
        if abs(cmd) < 1.0e-6:
            residual = 0.0
            native_pending = 0.0
            pending_stale = 0
        if residual * cmd < 0.0:
            residual = 0.0
            native_pending = 0.0
            pending_stale = 0
        if cmd_guard_ticks <= 0 and _diverged(intended_dir, cmd, applied, base_step):
            diverge_ticks += 1
            if diverge_ticks >= constants.sign_flip_ticks:
                yaw_sign = -yaw_sign
                diverge_ticks = 0
                native_pending = 0.0
                pending_stale = 0
                cmd_guard_ticks = constants.cmd_flip_guard_ticks
                reversal_settle_ticks = constants.reversal_settle_ticks
                intended_dir = 0
                corrections += 1
        elif abs(applied) > 1.0e-6:
            diverge_ticks = 0

        hold = reversal_settle_ticks > 0
        if hold:
            reversal_settle_ticks -= 1
            residual = 0.0
            demand = 0.0
            send = 0.0
        else:
            demand = _native_demand(constants, residual, cmd, applied, base_step)
            send = _native_issue(constants, demand, native_pending, cmd, base_step)
        dx = _clamp_counts(round(yaw_sign * send / native_step), constants.max_counts)
        sent = yaw_sign * dx * native_step
        if flipped and dx != 0 and sent * cmd < 0.0:
            nonzero_reversal_sends += 1
        native_pending += sent
        last_dx = dx
        intended_dir = (1 if sent > 0.0 else -1) if dx else _pending_dir(native_pending, base_step)
        max_abs_counts = max(max_abs_counts, abs(dx))
        residual = 0.0 if hold else demand - sent
        max_command_error = max(max_command_error, abs(residual))

        if delivery is None:
            if dx:
                delay = rng.randrange(3)
                if rng.random() < 0.10:
                    delay += 1
                inflight.append((tick + delay, dx))
        else:
            delivery.append(dx)
            delivered = delivery.pop(0)
            applied = os_sign * delivered * base_step * os_gain
            err -= applied
        max_abs_error = max(max_abs_error, abs(err))

    return {
        "final_error": err,
        "max_abs_error": max_abs_error,
        "corrections": corrections,
        "reversal_flips": reversal_flips,
        "nonzero_reversal_sends": nonzero_reversal_sends,
        "max_abs_counts": max_abs_counts,
        "max_command_error": max_command_error,
    }


def _native_demand(c: NativeAimConstants, residual: float, cmd: float,
                   applied: float, base_step: float) -> float:
    if abs(cmd) < 1.0e-6:
        return 0.0
    demand = residual + cmd
    if demand * cmd < 0.0:
        return 0.0
    fine_cap = min(abs(cmd), c.fine_one_to_one_deg)
    damped_cap = abs(cmd) * c.demand_gain + base_step
    cap = max(base_step, min(c.native_max, max(fine_cap, damped_cap)))
    return _clamp_mag(demand, cap)


def _native_issue(c: NativeAimConstants, demand: float, pending: float,
                  cmd: float, base_step: float) -> float:
    if abs(cmd) < 1.0e-6:
        return 0.0
    if demand * cmd < 0.0:
        return 0.0
    return _clamp_mag(demand, max(base_step, c.native_max))


def _update_native_pending(c: NativeAimConstants, pending: float,
                           applied: float, base_step: float) -> float:
    if abs(pending) <= base_step * 0.5:
        return 0.0
    if pending * applied > 0.0 and abs(applied) > base_step * 0.5:
        return 0.0
    return _clamp_mag(pending, c.native_max)


def _update_pending_stale(c: NativeAimConstants, stale: int, pending: float,
                          applied: float, cmd: float, base_step: float) -> int:
    if abs(pending) <= base_step * 0.5 or abs(cmd) <= base_step or abs(applied) > base_step * 0.5:
        return 0
    return stale + 1


def _policy_command(err: float, tick: int, rng: Random, mode: str) -> float:
    sign = 1.0 if err > 0.0 else -1.0
    if mode == "weak_jitter":
        cmd = sign * rng.uniform(2.0, 8.0)
        if tick % 11 in (4, 5) or rng.random() < 0.12:
            cmd = -cmd
        return cmd
    return sign * 40.0


def _stabilize_axis(c: NativeAimConstants, cmd: float, err: float,
                    limit: float, deadband: float) -> float:
    abs_err = abs(err)
    if abs_err <= deadband:
        return 0.0
    return _clamp_mag(err, limit)


def _diverged(intended_dir: int, cmd: float, applied: float, base_step: float) -> bool:
    return (
        intended_dir != 0
        and cmd * intended_dir > 0.0
        and abs(applied) > max(1.0e-6, base_step * 0.5)
        and applied * intended_dir < 0.0
    )


def _pending_dir(pending: float, base_step: float) -> int:
    if abs(pending) <= base_step * 0.5:
        return 0
    return 1 if pending > 0.0 else -1

def _command_dir(cmd: float) -> int:
    if abs(cmd) <= 1.0e-6:
        return 0
    return 1 if cmd > 0.0 else -1


def _clamp_counts(counts: int, max_counts: int) -> int:
    return max(-max_counts, min(max_counts, counts))


def _clamp_mag(v: float, m: float) -> float:
    return -m if v < -m else (m if v > m else v)




def _java_float(source: str, name: str) -> float:
    match = re.search(rf"private static final double {name} = ([0-9.]+);", source)
    if not match:
        raise ValueError(f"missing Java double constant: {name}")
    return float(match.group(1))


def _java_int(source: str, name: str) -> int:
    match = re.search(rf"private static final int {name} = ([0-9]+);", source)
    if not match:
        raise ValueError(f"missing Java int constant: {name}")
    return int(match.group(1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--java", type=Path, default=DEFAULT_JAVA)
    args = parser.parse_args(argv)
    report = simulate_suite(load_constants(args.java))
    print(verdict_text(report))
    return 0 if report.field_stable else 1


if __name__ == "__main__":
    raise SystemExit(main())
