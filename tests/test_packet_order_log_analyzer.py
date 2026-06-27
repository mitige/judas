from pathlib import Path

from tools.packet_order_log import analyze_lines, analyze_server_chat_lines, verdict_text


ROOT = Path(__file__).resolve().parents[1]


def test_packet_order_log_analyzer_reports_clean_final_order():
    result = analyze_lines([
        "tick=10 player=me seq=A>I order=OK A->I ok=true",
        "tick=11 player=me seq=A>S>I order=OK A->I ok=true",
    ])

    assert result.attacks == 2
    assert result.bad_pre_attack == 0
    assert result.guard_injections == 0
    assert result.clean is True
    assert "CLEAN" in verdict_text(result)


def test_packet_order_log_analyzer_reports_guard_injections():
    result = analyze_lines([
        "tick=20 player=me guard=injected total=1",
        "tick=20 player=me seq=A>I order=OK A->I ok=true",
    ])

    assert result.attacks == 1
    assert result.guard_injections == 1
    assert result.clean is False
    assert "GUARDED" in verdict_text(result)


def test_packet_order_log_analyzer_distinguishes_missing_probe_from_no_attacks():
    missing = analyze_lines([])
    installed = analyze_lines(["tick=1 player=me probe=installed"])

    assert missing.probe_installs == 0
    assert installed.probe_installs == 1
    assert "NO_PROBE" in verdict_text(missing)
    assert "NO_ATTACKS" in verdict_text(installed)


def test_packet_order_log_analyzer_reports_bad_pre_attack():
    result = analyze_lines([
        "tick=30 player=me seq=I order=BAD pre-attack ok=false",
        "tick=31 player=me guard=injected total=1",
    ])

    assert result.attacks == 1
    assert result.bad_pre_attack == 1
    assert result.guard_injections == 1
    assert result.clean is False
    assert "BAD" in verdict_text(result)


def test_server_chat_analyzer_detects_grim_and_vulcan_packet_order_failures():
    result = analyze_server_chat_lines([
        "[00:30:31] [Client thread/INFO]: [CHAT] Grim > boitedenuit failed PacketOrderB (x1) pre-attack",
        "[00:30:17] [Client thread/INFO]: [CHAT] Vulcan > boitedenuit failed Badpackets (Packet Order) x0.0",
        "[00:30:32] [Client thread/INFO]: [CHAT] Grim > other failed Simulation (x1) .022588 /gl 21",
    ])

    assert result.packet_order_failures == 2


def test_packet_order_log_cli_reads_file(tmp_path: Path, capsys):
    log = tmp_path / "judas-packet-order.log"
    log.write_text("tick=1 player=me seq=A>I order=OK A->I ok=true\n")

    from tools.packet_order_log import main

    assert main([str(log)]) == 0
    assert "attacks=1" in capsys.readouterr().out


def test_packet_order_log_cli_strict_requires_clean_order(tmp_path: Path):
    log = tmp_path / "judas-packet-order.log"
    log.write_text(
        "tick=1 player=me guard=injected total=1\n"
        "tick=1 player=me seq=A>I order=OK A->I ok=true\n"
    )

    from tools.packet_order_log import main

    assert main(["--strict", str(log)]) == 1


def test_packet_order_log_cli_strict_rejects_missing_attacks(tmp_path: Path):
    log = tmp_path / "judas-packet-order.log"
    log.write_text("tick=1 player=me guard=injected total=1\n")

    from tools.packet_order_log import main

    assert main(["--strict", str(log)]) == 1


def test_packet_order_log_cli_strict_reports_missing_local_log_as_no_attacks(tmp_path: Path, capsys):
    log = tmp_path / "missing.log"

    from tools.packet_order_log import main

    assert main(["--strict", str(log)]) == 1
    assert "NO_PROBE" in capsys.readouterr().out


def test_packet_order_log_cli_strict_rejects_server_chat_failures_after_offset(tmp_path: Path):
    log = tmp_path / "judas-packet-order.log"
    log.write_text("tick=1 player=me seq=A>I order=OK A->I ok=true\n")
    server_log = tmp_path / "latest.log"
    prefix = "[00:00:00] [Client thread/INFO]: old PacketOrderB pre-attack\n"
    server_log.write_text(
        prefix
        + "[00:00:01] [Client thread/INFO]: [CHAT] Grim > boitedenuit failed PacketOrderB (x1) pre-attack\n"
    )

    from tools.packet_order_log import main

    assert main([
        "--strict",
        "--server-log",
        str(server_log),
        "--server-offset",
        str(len(prefix)),
        str(log),
    ]) == 1


def test_packet_order_log_cli_scans_from_start_when_server_log_was_truncated(tmp_path: Path, capsys):
    log = tmp_path / "judas-packet-order.log"
    log.write_text("tick=1 player=me seq=A>I order=OK A->I ok=true\n")
    server_log = tmp_path / "latest.log"
    server_log.write_text(
        "[00:00:01] [Client thread/INFO]: [CHAT] Grim > boitedenuit failed PacketOrderB (x1) pre-attack\n"
    )

    from tools.packet_order_log import main

    assert main([
        "--strict",
        "--server-log",
        str(server_log),
        "--server-offset",
        "999999",
        str(log),
    ]) == 1
    assert "SERVER_BAD" in capsys.readouterr().out


def test_packet_order_log_cli_scans_server_chat_when_local_log_is_missing(tmp_path: Path, capsys):
    log = tmp_path / "missing.log"
    server_log = tmp_path / "latest.log"
    server_log.write_text(
        "[00:00:01] [Client thread/INFO]: [CHAT] Grim > boitedenuit failed PacketOrderB (x1) pre-attack\n"
    )

    from tools.packet_order_log import main

    assert main(["--server-log", str(server_log), str(log)]) == 1
    out = capsys.readouterr().out
    assert "NO_PROBE" in out
    assert "SERVER_BAD" in out


def test_packet_order_check_script_uses_default_minecraft_log():
    source = (ROOT / "scripts/check_packet_order.ps1").read_text()

    assert ".minecraft\\judas-packet-order.log" in source
    assert "tools\\packet_order_log.py" in source


def test_packet_order_check_script_can_reset_log_before_runtime_test():
    source = (ROOT / "scripts/check_packet_order.ps1").read_text()

    assert "[switch]$Reset" in source
    assert "Remove-Item" in source
    assert "packet-order log reset" in source


def test_packet_order_check_script_forwards_strict_mode():
    source = (ROOT / "scripts/check_packet_order.ps1").read_text()

    assert "[switch]$Strict" in source
    assert "--strict" in source


def test_packet_order_check_script_records_latest_log_offset_on_reset():
    source = (ROOT / "scripts/check_packet_order.ps1").read_text()

    assert "judas-packet-order-session.txt" in source
    assert "minecraft_log_size" in source
    assert "--server-log" in source
    assert "--server-offset" in source
    assert "Log introuvable" not in source


def test_packet_order_check_bat_wraps_powershell_with_bypass():
    source = (ROOT / "scripts/check_packet_order.bat").read_text()

    assert "ExecutionPolicy Bypass" in source
    assert "check_packet_order.ps1" in source
