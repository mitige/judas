from pathlib import Path

from tools.live_ws_check import LiveWsResult, _cases, format_result


ROOT = Path(__file__).resolve().parents[1]


def test_live_ws_check_formats_movement_metrics():
    ok = LiveWsResult("front", 16, 0, 0, 0, 0.0, 1.4, {"forward": 1, "jump": False})
    bad = LiveWsResult("front", 16, 0, 0, 2, 0.0, 1.4, {"forward": 1, "jump": True})
    flippy = LiveWsResult(
        "front",
        16,
        0,
        16,
        0,
        0.0,
        1.4,
        {"forward": 1, "jump": False},
        strafe_sign_flips=1,
        max_strafe_sign_flips=0,
    )
    circling = LiveWsResult(
        "under_combo_far_reentry",
        16,
        0,
        16,
        0,
        0.0,
        1.4,
        {"forward": 1, "jump": False},
        sprint=16,
        attack=1,
        max_attack=0,
        max_strafe=0,
    )

    assert format_result(ok).startswith("PASS front")
    assert "back_strafe_jump=0/0/0" in format_result(ok)
    assert "neutral=0 sprint=0 attack=0" in format_result(ok)
    assert "max_strafe=- max_attack=-" in format_result(ok)
    assert "strafe_flips=0" in format_result(ok)
    assert format_result(bad).startswith("FAIL front")
    assert "back_strafe_jump=0/0/2" in format_result(bad)
    assert format_result(flippy).startswith("FAIL front")
    assert "strafe_flips=1" in format_result(flippy)
    assert format_result(circling).startswith("FAIL under_combo_far_reentry")
    assert "max_strafe=0 max_attack=0" in format_result(circling)


def test_live_ws_check_includes_close_combo_and_counter_cases():
    names = {case.name for case in _cases()}

    assert {
        "opener_front_strafe",
        "opener_right_strafe",
        "opener_left_strafe",
        "post_opener_strafe",
        "post_opener_reset_cap",
        "combo_too_close_s_tap",
        "combo_wait_rehit",
        "combo_press_rehit",
        "combo_landed_reset",
        "under_combo_counter",
        "under_combo_far_reentry",
    } <= names
    for name in ("opener_front_strafe", "opener_right_strafe", "opener_left_strafe"):
        opener = next(case for case in _cases() if case.name == name)
        assert opener.max_back == 0
        assert opener.min_strafe >= 12
        assert opener.min_sprint >= 16
        assert opener.max_neutral == 0
        assert opener.max_strafe_sign_flips == 0
    post = next(case for case in _cases() if case.name == "post_opener_strafe")
    reset = next(case for case in _cases() if case.name == "post_opener_reset_cap")
    assert post.ticks == 64
    assert post.measure_from == 48
    assert post.min_strafe >= 16
    assert post.min_sprint >= 16
    assert post.max_neutral == 0
    assert post.max_strafe_sign_flips == 0
    assert reset.ticks == 64
    assert reset.measure_from == 48
    assert reset.min_strafe >= 16
    assert reset.max_neutral <= 2
    assert reset.min_sprint >= 14
    assert reset.max_strafe_sign_flips == 0
    close = next(case for case in _cases() if case.name == "combo_too_close_s_tap")
    wait = next(case for case in _cases() if case.name == "combo_wait_rehit")
    press = next(case for case in _cases() if case.name == "combo_press_rehit")
    landed = next(case for case in _cases() if case.name == "combo_landed_reset")
    under = next(case for case in _cases() if case.name == "under_combo_counter")
    far_under = next(case for case in _cases() if case.name == "under_combo_far_reentry")
    assert close.max_back == 0
    assert close.min_strafe >= 1
    assert close.min_sprint >= 1
    assert close.min_attack >= 1
    assert wait.max_back == 0
    assert wait.min_strafe >= 1
    assert wait.min_sprint >= 1
    assert wait.min_attack >= 1
    assert press.max_back == 0
    assert press.min_strafe >= 1
    assert press.min_sprint >= 16
    assert landed.max_back == 0
    assert landed.min_strafe >= 1
    assert landed.min_sprint >= 1
    assert landed.min_attack >= 1
    assert under.min_attack >= 8
    assert under.max_back == 0
    assert under.min_strafe >= 1
    assert far_under.max_back == 0
    assert far_under.min_sprint >= 16
    assert far_under.max_strafe == 0
    assert far_under.max_attack == 0


def test_live_ws_check_scripts_wire_safe_model():
    ps1 = (ROOT / "scripts/check_live_ws.ps1").read_text(encoding="utf-8")
    bat = (ROOT / "scripts/check_live_ws.bat").read_text(encoding="utf-8")

    assert "combo_god_leaderboard10_combo12-safe_latest.pts" in ps1
    assert "combo_god_countertap96_combo12-safe_latest.pts" in ps1
    assert "combo_god_directpad_lock_combo12-safe_latest.pts" in ps1
    assert "tools\\live_ws_check.py" in ps1
    assert "ExecutionPolicy Bypass" in bat
    assert "check_live_ws.ps1" in bat


def test_judas_live_defaults_use_combo_safe_runtime():
    source = (ROOT / "scripts/judas_live.ps1").read_text(encoding="utf-8")

    assert 'runs/combo_god_leaderboard10_combo12/safe_latest.pt' in source
    assert 'models/combo_god_leaderboard10_combo12-safe_latest.pts' in source
    assert 'runs/combo_god_countertap96_combo12/safe_latest.pt' in source
    assert 'models/combo_god_countertap96_combo12-safe_latest.pts' in source
    assert 'runs/combo_god_directpad_lock_combo12/safe_latest.pt' in source
    assert 'models/combo_god_directpad_lock_combo12-safe_latest.pts' in source
    assert 'runs/god/best.pt' not in source
    assert '$MaxCps      = 10.0' in source
    assert '$MaxRotSpeed = 195.0' in source
    assert '$SizeX   = 40.0' in source
    assert '$SizeZ   = 40.0' in source
    assert "judas_live_daemon.pid" in source
    assert "judas-live-actions.log" in source
    assert "$ActionLogExplicit" in source
    assert "JUDAS_LIVE_ACTION_LOG" in source
    assert "$ActionLog" in source
    assert "-PassThru" in source
    assert "Set-Content -LiteralPath $PidFile" in source
    assert "Assert-FreshExport $Checkpoint $Out" in source
    assert "AllowStaleExport" in source
    assert "Stale export" in source
    assert "Export source mismatch" in source
    assert "source_sha256" in source
    assert "source_size" in source
    assert "Assert-SafeCheckpointContract $Checkpoint" in source
    assert "safe_latest.meta.json" in source
    assert "score_schema" in source
    assert "opener_pressure_frac" in source
    assert "opener_strafe_hold_frac" in source
    assert "combo_z_tap_frac" in source
    assert "combo_s_tap_frac" in source
    assert "hit_wtap_frac" in source
    assert "under_combo_counter_hit_frac" in source
    assert "Assert-CounterMetric" in source
    assert "under_combo_avoidance_score_bonus" in source
    assert "requires_counter_recovery" in source
    assert "under_combo_hit_select_clean_frac" in source
    assert "under_combo_hit_select_trade_frac" in source
    assert "score_schema=$schema < 8" in source
    assert "Combo safe metadata too old" in source
    assert "Export checkpoint hash mismatch" in source
    assert "Export checkpoint size mismatch" in source
    assert "ForceDaemon" in source
    assert "Get-OwnedDaemonPids" in source
    assert "ParentProcessId" in source
    assert "ProcessId=$($proc.ParentProcessId)" in source
    assert "Stopped old repo daemon" in source
    assert "PYTHONPATH" in source
    assert "VIRTUAL_ENV" in source
    assert "_base_executable" in source


def test_check_safe_export_is_passive_and_matches_judas_live_defaults():
    ps1 = (ROOT / "scripts/check_safe_export.ps1").read_text(encoding="utf-8")
    bat = (ROOT / "scripts/check_safe_export.bat").read_text(encoding="utf-8")

    assert "combo_god_leaderboard10_combo12/safe_latest.pt" in ps1
    assert "combo_god_leaderboard10_combo12-safe_latest.pts" in ps1
    assert "combo_god_countertap96_combo12/safe_latest.pt" in ps1
    assert "combo_god_countertap96_combo12-safe_latest.pts" in ps1
    assert "combo_god_directpad_lock_combo12/safe_latest.pt" in ps1
    assert "combo_god_directpad_lock_combo12-safe_latest.pts" in ps1
    assert "ChangeExtension($exportAbs, \".json\")" in ps1
    assert "source_sha256" in ps1
    assert "source_size" in ps1
    assert "safe_latest.meta.json" in ps1
    assert "score_schema" in ps1
    assert "OLD_SAFE_SCHEMA" in ps1
    assert "opener_pressure_frac" in ps1
    assert "opener_strafe_hold_frac" in ps1
    assert "combo_z_tap_frac" in ps1
    assert "combo_s_tap_frac" in ps1
    assert "hit_wtap_frac" in ps1
    assert "under_combo_counter_hit_frac" in ps1
    assert "Assert-CounterMetric" in ps1
    assert "under_combo_avoidance_score_bonus" in ps1
    assert "requires_counter_recovery" in ps1
    assert "under_combo_hit_select_clean_frac" in ps1
    assert "under_combo_hit_select_trade_frac" in ps1
    assert "score_schema=$schema min=8" in ps1
    assert "SAFE_META_LOW_METRIC" in ps1
    assert "SAFE_META_HIGH_METRIC" in ps1
    assert "OK_EXPORT" in ps1
    assert "Start-Process" not in ps1
    assert "serve.daemon" not in ps1
    assert "train.export" not in ps1
    assert "ExecutionPolicy Bypass" in bat
    assert "check_safe_export.ps1" in bat


def test_stop_judas_live_stops_pid_file_and_orphan_daemons():
    ps1 = (ROOT / "scripts/stop_judas_live.ps1").read_text(encoding="utf-8")
    bat = (ROOT / "scripts/stop_judas_live.bat").read_text(encoding="utf-8")

    assert "judas_live_daemon.pid" in ps1
    assert "taskkill.exe /PID $ProcId /T /F" in ps1
    assert "Get-OwnedDaemonPids" in ps1
    assert "ParentProcessId" in ps1
    assert "ProcessId=$($proc.ParentProcessId)" in ps1
    assert "[int]$DaemonPort = 8765" in ps1
    assert "Get-NetTCPConnection -LocalPort $DaemonPort -State Listen" in ps1
    assert "orphan_daemon=stopped" in ps1
    assert "orphan_daemon=none" in ps1
    assert "Remove-Item -LiteralPath $PidFile" in ps1
    assert "ExecutionPolicy Bypass" in bat
    assert "stop_judas_live.ps1" in bat


def test_app_daemon_launcher_is_single_run_guarded():
    daemon_bat = (ROOT / "scripts/daemon.bat").read_text(encoding="utf-8")
    start_daemon = (ROOT / "scripts/start_judas_daemon.ps1").read_text(encoding="utf-8")
    app_bat = (ROOT / "scripts/app.bat").read_text(encoding="utf-8")
    run_bat = (ROOT / "run.bat").read_text(encoding="utf-8")

    assert "start_judas_daemon.ps1" in daemon_bat
    assert "ExecutionPolicy Bypass" in daemon_bat
    assert "python -m serve.daemon" not in daemon_bat

    assert "judas_live_daemon.pid" in start_daemon
    assert "ALREADY_RUNNING" in start_daemon
    assert "ALREADY_LISTENING" in start_daemon
    assert "STOPPED_OLD_DAEMON" in start_daemon
    assert "Get-OwnedDaemonPids" in start_daemon
    assert "ParentProcessId" in start_daemon
    assert "ProcessId=$($proc.ParentProcessId)" in start_daemon
    assert "taskkill.exe /PID $ProcId /T /F" in start_daemon
    assert "-WindowStyle Hidden -PassThru" in start_daemon
    assert "PYTHONPATH" in start_daemon
    assert "VIRTUAL_ENV" in start_daemon
    assert "_base_executable" in start_daemon
    assert "ActionLog" in start_daemon
    assert "judas-live-actions.log" in start_daemon
    assert "JUDAS_LIVE_ACTION_LOG" in start_daemon

    assert "call scripts\\daemon.bat" in app_bat
    assert "cmd /k scripts\\daemon.bat" not in app_bat
    assert "start \"judas-daemon\"" not in app_bat
    assert "netstat -ano" not in app_bat

    assert 'if "%c%"=="1" call scripts\\daemon.bat' in run_bat
    assert 'if "%c%"=="1" start "judas-daemon" cmd /k scripts\\daemon.bat' not in run_bat
