from pathlib import Path

from tools.aim_os_log import analyze_lines, latest_session_lines, main, verdict_text


ROOT = Path(__file__).resolve().parents[1]


def _log_path(name: str) -> Path:
    root = ROOT / ".pytest-local"
    root.mkdir(exist_ok=True)
    path = root / name
    if path.exists():
        path.unlink()
    return path


def _line(tick: int, yaw: float, pitch: float, dx: int = 1, dy: int = 1,
          stall: int = 0, cmd_yaw: float = 1.0,
          cmd_pitch: float = 1.0, sent_yaw: float = 1.0,
          sent_pitch: float = 1.0, applied_yaw: float = 1.0,
          applied_pitch: float = 1.0) -> str:
    return (
        f"tick={tick} player=me yawErr={yaw:.3f} pitchErr={pitch:.3f} "
        f"cmdYaw={cmd_yaw:.3f} cmdPitch={cmd_pitch:.3f} dx={dx} dy={dy} "
        f"sentYaw={sent_yaw:.3f} sentPitch={sent_pitch:.3f} "
        f"appliedYaw={applied_yaw:.3f} appliedPitch={applied_pitch:.3f} "
        f"pendingYaw=0.000 pendingPitch=0.000 stepYaw=0.15000 "
        f"stepPitch=0.15000 stall={stall} yawSign=1 pitchSign=1"
    )


def test_aim_os_log_reports_precise_when_errors_stay_low():
    report = analyze_lines(_line(i, 1.5, -1.0) for i in range(40))

    assert report.samples == 40
    assert report.one_to_one is True
    assert report.precise is True
    assert "PRECISE" in verdict_text(report)
    assert "cmd_drift_p95=" in verdict_text(report)


def test_aim_os_log_reports_loose_when_p95_error_is_high():
    lines = [_line(i, 2.0, 1.0) for i in range(35)]
    lines += [_line(i, 12.0, 1.0) for i in range(35, 40)]
    report = analyze_lines(lines)

    assert report.precise is False
    assert report.one_to_one is True
    assert "LOOSE" in verdict_text(report)


def test_aim_os_log_reports_not_1to1_when_sent_command_drifts():
    lines = [
        _line(i, 1.0, 1.0, cmd_yaw=4.0, cmd_pitch=4.0,
              sent_yaw=1.0, sent_pitch=1.0)
        for i in range(30)
    ]
    report = analyze_lines(lines)

    assert report.one_to_one is False
    assert report.yaw_cmd_drift_max > 20.0
    assert "NOT_1TO1" in verdict_text(report)


def test_aim_os_log_reports_stall():
    report = analyze_lines([_line(i, 1.0, 1.0, dx=0, dy=0, stall=12) for i in range(25)])

    assert report.max_stall == 12
    assert "STALL" in verdict_text(report)


def test_aim_os_log_reports_diverge_when_applied_motion_increases_error():
    lines = [_line(i, 8.0, 1.0, applied_yaw=-1.0) for i in range(20)]
    report = analyze_lines(lines)

    assert report.divergent is True
    assert report.yaw_bad_apply_ticks >= 3
    assert "DIVERGE" in verdict_text(report)


def test_aim_os_log_counts_sent_motion_that_was_not_applied():
    lines = [
        _line(i, 8.0, 1.0, cmd_yaw=2.0, sent_yaw=2.0, applied_yaw=0.0)
        for i in range(6)
    ]
    report = analyze_lines(lines)

    assert report.yaw_no_apply_ticks >= 5
    assert "no_apply=" in verdict_text(report)


def test_aim_os_log_reports_warmup_before_enough_samples():
    report = analyze_lines(_line(i, 1.0, 1.0) for i in range(8))

    assert report.precise is False
    assert "WARMUP" in verdict_text(report)


def test_aim_os_log_cli_strict(capsys):
    log = _log_path("strict.log")
    log.write_text("\n".join(_line(i, 1.0, 1.0) for i in range(30)))

    assert main(["--strict", str(log)]) == 0
    assert "PRECISE" in capsys.readouterr().out


def test_aim_os_log_cli_strict_rejects_not_1to1(capsys):
    log = _log_path("not_1to1.log")
    log.write_text("\n".join(
        _line(i, 1.0, 1.0, cmd_yaw=4.0, cmd_pitch=4.0,
              sent_yaw=1.0, sent_pitch=1.0)
        for i in range(30)
    ))

    assert main(["--strict", str(log)]) == 1
    assert "NOT_1TO1" in capsys.readouterr().out


def test_aim_os_log_cli_strict_rejects_missing_log(capsys):
    log = _log_path("missing.log")
    assert main(["--strict", str(log)]) == 1
    assert "NO_SAMPLES" in capsys.readouterr().out


def test_aim_os_log_cli_reports_no_target(capsys):
    log = _log_path("no_target.log")
    lines = [
        "event=start player=me sensitivity=0.5 invert=false",
        "event=no_target tick=0 player=me cmdYaw=1.000 cmdPitch=0.000 dx=0 dy=0",
    ]
    log.write_text("\n".join(lines))

    report = analyze_lines(lines)
    assert report.samples == 0
    assert "NO_TARGET" in verdict_text(report, lines)
    assert main(["--strict", str(log)]) == 1
    assert "NO_TARGET" in capsys.readouterr().out


def test_latest_session_lines_ignores_previous_runs():
    lines = ["event=start player=me"]
    lines += [_line(i, 18.0, 12.0) for i in range(30)]
    lines += ["event=start player=me"]
    lines += [_line(i, 1.0, 1.0) for i in range(30)]

    latest = latest_session_lines(lines)
    report = analyze_lines(latest)

    assert latest[0].startswith("event=start")
    assert report.samples == 30
    assert report.precise is True


def test_aim_os_log_cli_strict_uses_latest_session_by_default():
    log = _log_path("latest.log")
    log.write_text(
        "event=start player=me\n"
        + "\n".join(_line(i, 20.0, 12.0) for i in range(30))
        + "\nevent=start player=me\n"
        + "\n".join(_line(i, 1.0, 1.0) for i in range(30))
    )

    assert main(["--strict", str(log)]) == 0
    assert main(["--strict", "--all", str(log)]) == 1


def test_aim_os_check_script_defaults_to_minecraft_log():
    source = (ROOT / "scripts/check_aim_os.ps1").read_text()

    assert ".minecraft\\judas-aim-os.log" in source
    assert "tools\\aim_os_log.py" in source
    assert "[string]$ModsDir" in source
    assert "[switch]$AllowStale" in source
    assert "[string]$FreshAfter" in source
    assert "Resolve-FreshAfterUtc" in source
    assert "freshAfter=" in source
    assert "judas-bridge-*.jar" in source
    assert "STALE_LOG" in source
    assert "LastWriteTime" in source
    assert "[switch]$Reset" in source
    assert "Remove-Item" in source
    assert "--strict" in source
    assert "--all" in source


def test_aim_os_check_bat_wraps_powershell_with_bypass():
    source = (ROOT / "scripts/check_aim_os.bat").read_text()

    assert "ExecutionPolicy Bypass" in source
    assert "check_aim_os.ps1" in source


def test_guide_documents_aim_os_runtime_check():
    guide = (ROOT / "docs/GUIDE.md").read_text(encoding="utf-8")

    assert "Verifier l'aim souris OS" in guide
    assert "scripts\\prepare_aim_os_test.bat" in guide
    assert "scripts\\prepare_aim_os_test.bat -StopMinecraft" in guide
    assert "scripts\\field_test_aim_os.bat" in guide
    assert "scripts\\check_live_actions.bat -Strict" in guide
    assert "scripts\\check_packet_order.bat -Strict" in guide
    assert "judas-live-actions.log" in guide
    assert "scripts\\check_mod_deploy.bat" in guide
    assert "scripts\\check_mod_deploy.bat -RequireWritable" in guide
    assert "OK_DEPLOY" in guide
    assert "STALE_DEPLOY" in guide
    assert "LOCKED_DEPLOY" in guide
    assert "scripts\\watch_deploy_aim_os.bat" in guide
    assert "scripts\\check_aim_os.bat -Reset" in guide
    assert "scripts\\check_aim_os.bat -Strict" in guide
    assert "scripts\\watch_aim_os.bat" in guide
    assert "PRECISE" in guide
    assert "NOT_1TO1" in guide
    assert "LOOSE" in guide
    assert "STALL" in guide
    assert "DIVERGE" in guide
    assert "NO_SAMPLES" in guide
    assert "NO_TARGET" in guide
    assert "STALE_LOG" in guide
    assert "-FreshAfter" in guide
    assert "-AllowStale" in guide
