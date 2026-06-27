from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_field_preflight_is_passive_and_chains_required_checks():
    source = (ROOT / "scripts/check_field_preflight.ps1").read_text()

    assert "check_safe_export.ps1" in source
    assert "check_mod_deploy.ps1" in source
    assert "check_field_status.ps1" in source
    assert "-RequireWritable" in source
    assert "PROCESS_CLEAN" in source
    assert "PROCESS_DIRTY" in source
    assert "LIVE_PORT down" in source
    assert "LIVE_PORT up" in source
    assert "MINECRAFT_STATUS missing" in source
    assert "MINECRAFT_STATUS running" in source
    assert "RequireMinecraft" in source
    assert "Get-MinecraftProcessIds" in source
    assert "Test-MinecraftRunning" in source
    assert "PREFLIGHT PASS" in source
    assert "PREFLIGHT FAIL" in source
    assert "RequireField" in source
    assert "Start-Process" not in source
    assert "Remove-Item" not in source
    assert "taskkill" not in source
    assert "stop_judas" not in source
    assert "SAFE_EXPORT" in source
    assert "MOD_DEPLOY" in source
    assert "FIELD_STATUS" in source
    assert "$code = Invoke-PreflightStep" not in source
    assert "$code = Test-RepoProcessClean" not in source
    assert "script:failures += 1" in source


def test_field_preflight_bat_wraps_powershell_with_bypass():
    source = (ROOT / "scripts/check_field_preflight.bat").read_text()

    assert "ExecutionPolicy Bypass" in source
    assert "check_field_preflight.ps1" in source


def test_arm_combo_god_field_loads_safe_and_refuses_no_world_before_arm():
    ps1 = (ROOT / "scripts/arm_combo_god_field.ps1").read_text()
    bat = (ROOT / "scripts/arm_combo_god_field.bat").read_text()

    assert "check_safe_export.ps1" in ps1
    assert "judas_live.ps1" in ps1
    assert "control_judas_mod.ps1" in ps1
    assert "combo_god_leaderboard10_combo12-safe_latest.pts" in ps1
    assert "-NoExport -NoLaunch" in ps1
    assert "ReadyTimeoutSeconds" in ps1
    assert "Test-MinecraftRunning" in ps1
    assert "FIELD_NOT_READY result=no_minecraft_process" in ps1
    assert "Get-CimInstance Win32_Process" in ps1
    assert "LiveTarget" in ps1
    assert '"status_live"' in ps1
    assert '"status"' in ps1
    assert "-Command $statusCommand" in ps1
    assert "[Console]::Out.WriteLine($_)" in ps1
    assert "FIELD_WAITING result=$statusResult" in ps1
    assert "$statusResult -ne \"ok\"" in ps1
    assert "FIELD_NOT_READY result=$statusResult" in ps1
    assert '"arm_native_live"' in ps1
    assert '"arm_native"' in ps1
    assert "-Command $armCommand" in ps1
    assert "$armResult -ne \"armed\"" in ps1
    assert '"live_target"' in ps1
    assert "FIELD_ARMED" in ps1
    assert "Start-Process" not in ps1
    assert "taskkill" not in ps1
    assert "Remove-Item" not in ps1
    assert "ExecutionPolicy Bypass" in bat
    assert "arm_combo_god_field.ps1" in bat
