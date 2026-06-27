from __future__ import annotations

import argparse
import os
import socket
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.aim_os_log import (
    analyze_lines as analyze_aim_lines,
    latest_session_lines as latest_aim_session_lines,
    verdict_text as aim_verdict_text,
)
from tools.live_action_log import (
    DEFAULT_REQUIRE_MODEL,
    analyze_lines as analyze_live_lines,
    latest_session_lines as latest_live_session_lines,
    verdict_text as live_verdict_text,
)
from tools.packet_order_log import (
    ServerChatReport,
    analyze_lines as analyze_packet_lines,
    analyze_server_chat_lines,
    server_verdict_text,
    verdict_text as packet_verdict_text,
)


SAFE_MODEL_MARKER = DEFAULT_REQUIRE_MODEL


@dataclass(frozen=True)
class ComponentStatus:
    name: str
    state: str
    detail: str

    @property
    def ok(self) -> bool:
        return self.state == "PASS"

    @property
    def failed(self) -> bool:
        return self.state in {"FAIL", "STALE"}

    def line(self) -> str:
        return f"{self.name} {self.state} {self.detail}"


def default_minecraft_dir() -> Path:
    appdata = os.environ.get("APPDATA", "")
    return Path(appdata) / ".minecraft" if appdata else Path(".minecraft")


def latest_session(path: Path, latest_fn) -> list[str]:
    if not path.exists():
        return []
    return latest_fn(path.read_text(errors="replace").splitlines())


def path_detail(path: Path) -> str:
    try:
        p = path.resolve()
    except OSError:
        p = path
    if not path.exists():
        return f"path={p} exists=false"
    try:
        stat = path.stat()
        size = stat.st_size
        mtime = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
    except OSError:
        size = -1
        mtime = "unknown"
    return f"path={p} exists=true size={size} mtime={mtime}"


def daemon_detail(host: str = "127.0.0.1", port: int = 8765) -> str:
    try:
        with socket.create_connection((host, port), timeout=0.20):
            return f"daemon={host}:{port}:up"
    except OSError:
        return f"daemon={host}:{port}:down"


def parse_fresh_after(value: str) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError as exc:
        raise SystemExit(f"invalid --fresh-after: {value}") from exc


def stale_against_fresh_after(path: Path, fresh_after: float | None) -> bool:
    if fresh_after is None or not path.exists():
        return False
    return path.stat().st_mtime < fresh_after


def aim_status(log: Path, mods_dir: Path, *, allow_stale: bool = False,
               fresh_after: float | None = None) -> ComponentStatus:
    lines = latest_session(log, latest_aim_session_lines)
    report = analyze_aim_lines(lines)
    detail = f"{aim_verdict_text(report, lines)} {path_detail(log)}"
    if stale_against_fresh_after(log, fresh_after):
        return ComponentStatus("AIM_OS", "STALE", detail)
    if not allow_stale and is_stale_against_active_jar(log, mods_dir):
        return ComponentStatus("AIM_OS", "STALE", detail)
    if report.precise:
        return ComponentStatus("AIM_OS", "PASS", detail)
    if report.samples < 20:
        return ComponentStatus("AIM_OS", "INCOMPLETE", detail)
    return ComponentStatus("AIM_OS", "FAIL", detail)


def live_status(log: Path, *, min_samples: int = 20,
                max_attack_cps: float = 10.0,
                min_strafe_frac: float = 0.50,
                opener_ticks: int = 20,
                min_opener_strafe_frac: float = 0.75,
                min_opener_strafe_hold_frac: float = 0.70,
                min_opener_pressure_frac: float = 0.60,
                max_strafe_flip_frac: float = 0.10,
                min_strafe_hold_avg: float = 3.0,
                min_hit_wtap_frac: float = 0.75,
                fresh_after: float | None = None) -> ComponentStatus:
    lines = latest_session(log, latest_live_session_lines)
    report = analyze_live_lines(lines, opener_ticks=opener_ticks)
    detail = (
        live_verdict_text(
            report,
            min_samples=min_samples,
            max_attack_cps=max_attack_cps,
            min_strafe_frac=min_strafe_frac,
            min_opener_strafe_frac=min_opener_strafe_frac,
            min_opener_strafe_hold_frac=min_opener_strafe_hold_frac,
            min_opener_pressure_frac=min_opener_pressure_frac,
            max_strafe_flip_frac=max_strafe_flip_frac,
            min_strafe_hold_avg=min_strafe_hold_avg,
            min_hit_wtap_frac=min_hit_wtap_frac,
            require_model=SAFE_MODEL_MARKER,
        )
        + f" {path_detail(log)} {daemon_detail()}"
    )
    if stale_against_fresh_after(log, fresh_after):
        return ComponentStatus("LIVE_ACTIONS", "STALE", detail)
    if detail.startswith("PASS "):
        return ComponentStatus("LIVE_ACTIONS", "PASS", detail)
    if report.samples < min_samples:
        return ComponentStatus("LIVE_ACTIONS", "INCOMPLETE", detail)
    return ComponentStatus("LIVE_ACTIONS", "FAIL", detail)


def packet_status(log: Path, minecraft_log: Path, session: Path,
                  fresh_after: float | None = None) -> ComponentStatus:
    lines = log.read_text(errors="replace").splitlines() if log.exists() else []
    report = analyze_packet_lines(lines)
    server_report = ServerChatReport(packet_order_failures=0)
    if minecraft_log.exists():
        server_report = analyze_server_chat_lines(
            read_server_log_from_session(minecraft_log, session)
        )
    detail = packet_verdict_text(report)
    server_detail = server_verdict_text(server_report)
    detail = (
        f"{detail} {server_detail} "
        f"{path_detail(log)} minecraft_log={path_detail(minecraft_log)}"
    )
    if stale_against_fresh_after(log, fresh_after):
        return ComponentStatus("PACKET_ORDER", "STALE", detail)
    if report.clean and server_report.packet_order_failures == 0:
        return ComponentStatus("PACKET_ORDER", "PASS", detail)
    if report.bad_pre_attack or report.guard_injections or server_report.packet_order_failures:
        return ComponentStatus("PACKET_ORDER", "FAIL", detail)
    return ComponentStatus("PACKET_ORDER", "INCOMPLETE", detail)


def summary_state(parts: list[ComponentStatus]) -> str:
    if all(p.ok for p in parts):
        return "PASS"
    if any(p.failed for p in parts):
        return "FAIL"
    return "INCOMPLETE"


def is_stale_against_active_jar(log: Path, mods_dir: Path) -> bool:
    if not log.exists() or not mods_dir.exists():
        return False
    jars = sorted(
        mods_dir.glob("judas-bridge-*.jar"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return bool(jars and log.stat().st_mtime < jars[0].stat().st_mtime)


def read_server_log_from_session(minecraft_log: Path, session: Path) -> list[str]:
    offset = 0
    log_path = minecraft_log
    if session.exists():
        values: dict[str, str] = {}
        for raw in session.read_text(errors="replace").splitlines():
            if "=" in raw:
                k, v = raw.split("=", 1)
                values[k] = v
        if values.get("minecraft_log"):
            log_path = Path(values["minecraft_log"])
        if values.get("minecraft_log_size"):
            try:
                offset = int(values["minecraft_log_size"])
            except ValueError:
                offset = 0
    if not log_path.exists():
        return []
    size = log_path.stat().st_size
    if offset > size:
        offset = 0
    with log_path.open("rb") as handle:
        handle.seek(max(0, offset))
        return handle.read().decode(errors="replace").splitlines()


def main(argv: list[str] | None = None) -> int:
    mc = default_minecraft_dir()
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--allow-stale", action="store_true")
    parser.add_argument("--aim-log", type=Path, default=mc / "judas-aim-os.log")
    parser.add_argument("--live-log", type=Path, default=Path("runs/judas-live-actions.log"))
    parser.add_argument("--packet-log", type=Path, default=mc / "judas-packet-order.log")
    parser.add_argument("--mods-dir", type=Path, default=mc / "mods")
    parser.add_argument("--minecraft-log", type=Path, default=mc / "logs/latest.log")
    parser.add_argument("--packet-session", type=Path, default=mc / "judas-packet-order-session.txt")
    parser.add_argument("--min-live-samples", type=int, default=20)
    parser.add_argument("--max-live-attack-cps", type=float, default=10.0)
    parser.add_argument("--min-live-strafe-frac", type=float, default=0.50)
    parser.add_argument("--live-opener-ticks", type=int, default=20)
    parser.add_argument("--min-live-opener-strafe-frac", type=float, default=0.75)
    parser.add_argument("--min-live-opener-strafe-hold-frac", type=float, default=0.70)
    parser.add_argument("--min-live-opener-pressure-frac", type=float, default=0.60)
    parser.add_argument("--max-live-strafe-flip-frac", type=float, default=0.10)
    parser.add_argument("--min-live-strafe-hold-avg", type=float, default=3.0)
    parser.add_argument("--min-live-hit-wtap-frac", type=float, default=0.75)
    parser.add_argument("--fresh-after", default="")
    args = parser.parse_args(argv)
    fresh_after = parse_fresh_after(args.fresh_after)

    parts = [
        aim_status(args.aim_log, args.mods_dir, allow_stale=args.allow_stale,
                   fresh_after=fresh_after),
        live_status(args.live_log, min_samples=args.min_live_samples,
                    max_attack_cps=args.max_live_attack_cps,
                    min_strafe_frac=args.min_live_strafe_frac,
                    opener_ticks=args.live_opener_ticks,
                    min_opener_strafe_frac=args.min_live_opener_strafe_frac,
                    min_opener_strafe_hold_frac=args.min_live_opener_strafe_hold_frac,
                    min_opener_pressure_frac=args.min_live_opener_pressure_frac,
                    max_strafe_flip_frac=args.max_live_strafe_flip_frac,
                    min_strafe_hold_avg=args.min_live_strafe_hold_avg,
                    min_hit_wtap_frac=args.min_live_hit_wtap_frac,
                    fresh_after=fresh_after),
        packet_status(args.packet_log, args.minecraft_log, args.packet_session,
                      fresh_after=fresh_after),
    ]
    for part in parts:
        print(part.line())
    summary = summary_state(parts)
    print(f"SUMMARY {summary}")
    if args.strict:
        return 0 if summary == "PASS" else 1
    return 1 if summary == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
