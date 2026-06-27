import subprocess
import sys
from pathlib import Path

from tools.field_test_status import main


ROOT = Path(__file__).resolve().parents[1]


def _aim_line(tick: int) -> str:
    return (
        f"tick={tick} player=me yawErr=1.000 pitchErr=1.000 "
        "cmdYaw=1.000 cmdPitch=1.000 dx=1 dy=1 "
        "sentYaw=1.000 sentPitch=1.000 appliedYaw=1.000 appliedPitch=1.000 "
        "pendingYaw=0.000 pendingPitch=0.000 stepYaw=0.15000 stepPitch=0.15000 "
        "stall=0 yawSign=1 pitchSign=1"
    )


def _live_line(tick: int) -> str:
    attack = "true" if tick % 2 == 0 else "false"
    return (
        f"tick={tick} model=models/combo_god_leaderboard10_combo12-safe_latest.pts "
        f"forward=1 strafe=1 jump=false sprint=true attack={attack} "
        "dyaw=0.0 dpitch=1.0 ownPitch=1.000 yawErr=0.0 pitchErr=1.000"
    )


def test_field_status_reports_pass_when_all_runtime_proofs_are_present(tmp_path, capsys):
    aim = tmp_path / "judas-aim-os.log"
    live = tmp_path / "judas-live-actions.log"
    packet = tmp_path / "judas-packet-order.log"
    mc_log = tmp_path / "latest.log"
    session = tmp_path / "session.txt"
    mods = tmp_path / "mods"
    mods.mkdir()

    aim.write_text("event=start\n" + "\n".join(_aim_line(i) for i in range(24)))
    live.write_text("event=start model=models/combo_god_leaderboard10_combo12-safe_latest.pts\n"
                    + "\n".join(_live_line(i) for i in range(24)))
    packet.write_text("event=start probe=installed\ntick=1 seq=A>R>ATK ok=true\n")
    mc_log.write_text("server clean\n")
    session.write_text(f"minecraft_log={mc_log}\nminecraft_log_size=0\n")

    code = main([
        "--strict",
        "--aim-log", str(aim),
        "--live-log", str(live),
        "--packet-log", str(packet),
        "--mods-dir", str(mods),
        "--minecraft-log", str(mc_log),
        "--packet-session", str(session),
    ])

    out = capsys.readouterr().out
    assert code == 0
    assert "AIM_OS PASS" in out
    assert "mtime=" in out
    assert "LIVE_ACTIONS PASS" in out
    assert "min_strafe_frac=0.500" in out
    assert "opener_strafe=20/20" in out
    assert "min_opener_strafe_frac=0.750" in out
    assert "opener_strafe_hold=20/20" in out
    assert "min_opener_strafe_hold_frac=0.700" in out
    assert "opener_pressure=20/20" in out
    assert "min_opener_pressure_frac=0.600" in out
    assert "attack_cps=10.00" in out
    assert "max_attack_cps=10.00" in out
    assert "hit_wtap=0/0" in out
    assert "min_hit_wtap_frac=0.750" in out
    assert "strafe_flips=0" in out
    assert "strafe_hold_avg=24.00" in out
    assert "PACKET_ORDER PASS" in out
    assert "SUMMARY PASS" in out


def test_field_status_reports_incomplete_without_runtime_logs(tmp_path, capsys):
    code = main([
        "--aim-log", str(tmp_path / "missing-aim.log"),
        "--live-log", str(tmp_path / "missing-live.log"),
        "--packet-log", str(tmp_path / "missing-packet.log"),
        "--mods-dir", str(tmp_path / "mods"),
        "--minecraft-log", str(tmp_path / "latest.log"),
        "--packet-session", str(tmp_path / "session.txt"),
    ])

    out = capsys.readouterr().out
    assert code == 0
    assert "AIM_OS INCOMPLETE" in out
    assert "AIM_OS INCOMPLETE NO_SAMPLES" in out
    assert "LIVE_ACTIONS INCOMPLETE" in out
    assert "path=" in out
    assert "exists=false" in out
    assert "daemon=127.0.0.1:8765:" in out
    assert "PACKET_ORDER INCOMPLETE" in out
    assert "SUMMARY INCOMPLETE" in out


def test_field_status_marks_existing_old_logs_stale_with_fresh_after(tmp_path, capsys):
    aim = tmp_path / "judas-aim-os.log"
    live = tmp_path / "judas-live-actions.log"
    packet = tmp_path / "judas-packet-order.log"
    mc_log = tmp_path / "latest.log"
    session = tmp_path / "session.txt"
    mods = tmp_path / "mods"
    mods.mkdir()

    aim.write_text("event=start\n" + "\n".join(_aim_line(i) for i in range(24)))
    live.write_text("event=start model=models/combo_god_leaderboard10_combo12-safe_latest.pts\n"
                    + "\n".join(_live_line(i) for i in range(24)))
    packet.write_text("event=start probe=installed\ntick=1 seq=A>R>ATK ok=true\n")
    mc_log.write_text("server clean\n")
    session.write_text(f"minecraft_log={mc_log}\nminecraft_log_size=0\n")

    code = main([
        "--strict",
        "--fresh-after", "9999999999",
        "--aim-log", str(aim),
        "--live-log", str(live),
        "--packet-log", str(packet),
        "--mods-dir", str(mods),
        "--minecraft-log", str(mc_log),
        "--packet-session", str(session),
    ])

    out = capsys.readouterr().out
    assert code == 1
    assert "AIM_OS STALE" in out
    assert "LIVE_ACTIONS STALE" in out
    assert "PACKET_ORDER STALE" in out
    assert "SUMMARY FAIL" in out


def test_field_status_script_is_passive_and_wraps_python():
    ps1 = (ROOT / "scripts/check_field_status.ps1").read_text()
    bat = (ROOT / "scripts/check_field_status.bat").read_text()
    guide = (ROOT / "docs/GUIDE.md").read_text()

    assert "tools\\field_test_status.py" in ps1
    assert "FreshAfter" in ps1
    assert "--fresh-after" in ps1
    assert "judas-live-actions.log" in ps1
    assert "--strict" in ps1
    assert "MaxLiveAttackCps" in ps1
    assert "MaxLiveStrafeFlipFrac" in ps1
    assert "LiveOpenerTicks" in ps1
    assert "MinLiveOpenerStrafeFrac" in ps1
    assert "MinLiveOpenerStrafeHoldFrac" in ps1
    assert "MinLiveOpenerPressureFrac" in ps1
    assert "MinLiveStrafeHoldAvg" in ps1
    assert "MinLiveHitWtapFrac" in ps1
    assert "$MinLiveStrafeFrac = 0.50" in ps1
    assert "--max-live-attack-cps" in ps1
    assert "--live-opener-ticks" in ps1
    assert "--min-live-opener-strafe-frac" in ps1
    assert "--min-live-opener-strafe-hold-frac" in ps1
    assert "--min-live-opener-pressure-frac" in ps1
    assert "--max-live-strafe-flip-frac" in ps1
    assert "--min-live-strafe-hold-avg" in ps1
    assert "--min-live-hit-wtap-frac" in ps1
    assert "Remove-Item" not in ps1
    assert "Start-Process" not in ps1
    assert "field_test_status.py" not in bat
    assert "check_field_status.ps1" in bat
    assert "path=... exists=... size=... mtime=..." in guide
    assert "daemon=127.0.0.1:8765:up/down" in guide
    assert "runs\\judas-live-actions.log" in guide


def test_field_status_tool_runs_when_called_by_file_path(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/field_test_status.py"),
            "--aim-log", str(tmp_path / "missing-aim.log"),
            "--live-log", str(tmp_path / "missing-live.log"),
            "--packet-log", str(tmp_path / "missing-packet.log"),
            "--mods-dir", str(tmp_path / "mods"),
            "--minecraft-log", str(tmp_path / "latest.log"),
            "--packet-session", str(tmp_path / "session.txt"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "SUMMARY INCOMPLETE" in result.stdout
    assert result.stderr == ""
