from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_guide_documents_packet_order_runtime_check():
    guide = (ROOT / "docs/GUIDE.md").read_text(encoding="utf-8")

    assert "PacketOrder serveur" in guide
    assert "Fermez Minecraft" in guide
    assert "scripts\\prepare_packet_order_test.bat -StopMinecraft" in guide
    assert "scripts\\prepare_packet_order_test.bat" in guide
    assert "scripts\\watch_packet_order.bat" in guide
    assert "scripts\\check_packet_order.bat" in guide
    assert "scripts\\check_packet_order.bat -Strict" in guide
    assert "CLEAN" in guide
    assert "GUARDED" in guide
    assert "BAD" in guide
