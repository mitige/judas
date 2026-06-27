from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class PacketOrderReport:
    attacks: int
    ok_attacks: int
    bad_pre_attack: int
    guard_injections: int
    probe_installs: int
    reset_markers: int

    @property
    def clean(self) -> bool:
        return (
            self.attacks > 0
            and self.bad_pre_attack == 0
            and self.guard_injections == 0
        )


@dataclass(frozen=True)
class ServerChatReport:
    packet_order_failures: int


def analyze_lines(lines: Iterable[str]) -> PacketOrderReport:
    attacks = 0
    ok_attacks = 0
    bad_pre_attack = 0
    guard_injections = 0
    probe_installs = 0
    reset_markers = 0

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if " probe=installed" in line:
            probe_installs += 1
        if " guard=injected" in line:
            guard_injections += 1
        if " seq=" not in line:
            continue
        attacks += 1
        if "ok=true" in line:
            ok_attacks += 1
        if "BAD pre-attack" in line or "ok=false" in line:
            bad_pre_attack += 1
        seq = _field(line, "seq")
        if seq is not None:
            reset_markers += seq.split(">").count("R")

    return PacketOrderReport(
        attacks=attacks,
        ok_attacks=ok_attacks,
        bad_pre_attack=bad_pre_attack,
        guard_injections=guard_injections,
        probe_installs=probe_installs,
        reset_markers=reset_markers,
    )


def analyze_server_chat_lines(lines: Iterable[str]) -> ServerChatReport:
    failures = 0
    for raw in lines:
        line = raw.strip()
        lower = line.lower()
        if "failed" not in lower:
            continue
        if "packetorderb" in lower and "pre-attack" in lower:
            failures += 1
        elif "packet order" in lower:
            failures += 1
    return ServerChatReport(packet_order_failures=failures)


def verdict_text(report: PacketOrderReport) -> str:
    if report.bad_pre_attack:
        verdict = "BAD"
    elif report.guard_injections:
        verdict = "GUARDED"
    elif report.clean:
        verdict = "CLEAN"
    elif report.probe_installs == 0:
        verdict = "NO_PROBE"
    else:
        verdict = "NO_ATTACKS"
    return (
        f"{verdict} attacks={report.attacks} ok={report.ok_attacks} "
        f"bad_pre_attack={report.bad_pre_attack} "
        f"guard_injections={report.guard_injections} "
        f"probe_installs={report.probe_installs} resets={report.reset_markers}"
    )


def server_verdict_text(report: ServerChatReport) -> str:
    if report.packet_order_failures:
        verdict = "SERVER_BAD"
    else:
        verdict = "SERVER_CLEAN"
    return f"{verdict} packet_order_chat_failures={report.packet_order_failures}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail unless the log is CLEAN: attacks observed, no BAD, no guard injection",
    )
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--server-offset", type=int, default=0)
    parser.add_argument("log", type=Path)
    args = parser.parse_args(argv)

    local_lines = args.log.read_text(errors="replace").splitlines() if args.log.exists() else []
    report = analyze_lines(local_lines)
    print(verdict_text(report))
    server_report = ServerChatReport(packet_order_failures=0)
    if args.server_log is not None and args.server_log.exists():
        server_report = analyze_server_chat_lines(_read_from_offset(args.server_log, args.server_offset))
        print(server_verdict_text(server_report))
    if args.strict:
        return 0 if report.clean and server_report.packet_order_failures == 0 else 1
    return 1 if report.bad_pre_attack or server_report.packet_order_failures else 0


def _read_from_offset(path: Path, offset: int) -> list[str]:
    with path.open("rb") as handle:
        size = path.stat().st_size
        if offset > size:
            offset = 0
        handle.seek(max(0, offset))
        data = handle.read()
    return data.decode(errors="replace").splitlines()


def _field(line: str, name: str) -> str | None:
    prefix = name + "="
    for part in line.split():
        if part.startswith(prefix):
            return part[len(prefix):]
    return None


if __name__ == "__main__":
    raise SystemExit(main())
