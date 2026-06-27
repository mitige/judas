from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_watch_deploy_aim_os_waits_for_unlock_then_prepares():
    source = (ROOT / "scripts/watch_deploy_aim_os.ps1").read_text()

    assert "check_mod_deploy.ps1" in source
    assert "-RequireWritable" in source
    assert "LOCKED_DEPLOY" in source
    assert "prepare_aim_os_test.bat" in source
    assert "Start-Sleep" in source
    assert "TimeoutSeconds" in source
    assert "$TimeoutSeconds = 120" in source
    assert "BuildTimeoutSeconds" in source
    assert "-BuildTimeoutSeconds $BuildTimeoutSeconds" in source
    assert "$LASTEXITCODE" in source


def test_watch_deploy_aim_os_bat_wraps_powershell_with_bypass():
    source = (ROOT / "scripts/watch_deploy_aim_os.bat").read_text()

    assert "ExecutionPolicy Bypass" in source
    assert "watch_deploy_aim_os.ps1" in source
