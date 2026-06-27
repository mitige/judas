from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_watch_aim_os_script_polls_strict_check_and_waits_for_warmup():
    source = (ROOT / "scripts/watch_aim_os.ps1").read_text()

    assert "check_aim_os.ps1" in source
    assert "-Strict" in source
    assert "Start-Sleep" in source
    assert "PRECISE" in source
    assert "STALL" in source
    assert "DIVERGE" in source
    assert "LOOSE" in source
    assert "MinLooseSamples" in source
    assert "FreshAfter" in source
    assert '$checkArgs += @("-FreshAfter", $FreshAfter)' in source
    assert "TimeoutSeconds" in source
    assert "$TimeoutSeconds = 120" in source
    assert "TIMEOUT_AIM_OS" in source
    assert "Get-LogStatus" in source
    assert "Get-MinecraftStatus" in source
    assert "press O for OS mouse, then K" in source


def test_watch_aim_os_bat_wraps_powershell_with_bypass():
    source = (ROOT / "scripts/watch_aim_os.bat").read_text()

    assert "ExecutionPolicy Bypass" in source
    assert "watch_aim_os.ps1" in source
