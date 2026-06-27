from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_watch_packet_order_script_polls_strict_check_and_stops_on_clean_or_bad():
    source = (ROOT / "scripts/watch_packet_order.ps1").read_text()

    assert "check_packet_order.ps1" in source
    assert "-Strict" in source
    assert "Start-Sleep" in source
    assert "CLEAN" in source
    assert "SERVER_BAD" in source
    assert "BAD " in source
    assert "$TimeoutSeconds = 120" in source


def test_watch_packet_order_bat_wraps_powershell_with_bypass():
    source = (ROOT / "scripts/watch_packet_order.bat").read_text()

    assert "ExecutionPolicy Bypass" in source
    assert "watch_packet_order.ps1" in source
