from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_prepare_packet_order_test_builds_deploys_and_resets_log():
    source = (ROOT / "scripts/prepare_packet_order_test.bat").read_text()

    assert "build_mod.bat" in source
    assert "-Clean -ModsDir" in source
    assert "-BuildTimeoutSeconds" in source
    assert "BUILD_TIMEOUT=300" in source
    assert "check_packet_order.bat" in source
    assert "-Reset" in source
    assert "%APPDATA%\\.minecraft\\mods" in source


def test_prepare_packet_order_test_accepts_stop_minecraft_option():
    source = (ROOT / "scripts/prepare_packet_order_test.bat").read_text()

    assert "-StopMinecraft" in source
    assert "STOP_MINECRAFT" in source
    assert "BUILD_TIMEOUT" in source
    assert "SCRIPT_DIR" in source
    assert "shift" in source
