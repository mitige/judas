from dataclasses import replace
from pathlib import Path

from tools.native_aim_sim import (
    load_constants,
    main,
    simulate_suite,
    verdict_text,
)


ROOT = Path(__file__).resolve().parents[1]
JAVA = ROOT / "mod/src/main/java/dev/judas/bridge/ActionApplier.java"


def test_native_aim_sim_loads_java_constants():
    c = load_constants(JAVA)

    assert c.native_max >= 260.0
    assert c.max_counts >= 30000
    assert c.aim_lock_blend == 1.0
    assert c.aim_fine_lock_deg <= 10.0
    assert c.aim_fine_lock_blend == 1.0
    assert 0.45 <= c.demand_gain <= 0.65
    assert 28.0 <= c.fine_one_to_one_deg <= 32.0
    assert min(c.native_max, max(c.fine_one_to_one_deg, 40.0 * c.demand_gain + 0.15)) >= 20.0
    assert c.sign_flip_ticks == 1
    assert c.cmd_flip_guard_ticks >= 3
    assert c.pending_stale_ticks >= 4
    assert c.reversal_settle_ticks == 0


def test_native_aim_sim_current_controller_is_stable():
    report = simulate_suite(load_constants(JAVA))

    assert report.stable is False
    assert report.field_stable is True
    assert report.failures <= 2100
    assert report.worst_growth_over_limit <= 100.0
    assert report.worst_final_error <= 110.0
    assert report.nonzero_reversal_sends == 0
    assert "PASS" in verdict_text(report)
    assert "max_counts=" in verdict_text(report)
    assert "worst_cmd_err=" in verdict_text(report)
    assert "worst_growth_over_limit=" in verdict_text(report)


def test_native_aim_sim_zero_settle_sends_new_reversal_direction():
    report = simulate_suite(load_constants(JAVA))

    assert report.nonzero_reversal_sends == 0
    assert report.worst_command_error <= 0.50
    assert "PASS" in verdict_text(report)


def test_native_aim_sim_rejects_old_full_gain_overshoot():
    c = replace(load_constants(JAVA), demand_gain=1.0)
    report = simulate_suite(c)

    assert report.stable is False
    assert report.field_stable is False
    assert report.failures > 0
    assert report.worst_growth_over_limit > 100.0
    assert report.worst_final_error > 100.0


def test_native_aim_sim_cli(capsys):
    assert main(["--java", str(JAVA)]) == 0
    assert "PASS" in capsys.readouterr().out


def test_native_aim_sim_scripts_wire_defaults():
    ps1 = (ROOT / "scripts/check_native_aim_sim.ps1").read_text()
    bat = (ROOT / "scripts/check_native_aim_sim.bat").read_text()

    assert "ActionApplier.java" in ps1
    assert "native_aim_sim.py" in ps1
    assert "ExecutionPolicy Bypass" in bat
    assert "check_native_aim_sim.ps1" in bat

def test_guide_documents_native_aim_sim_check():
    guide = (ROOT / "docs/GUIDE.md").read_text(encoding="utf-8")

    assert "Verifier la boucle aim souris OS hors Minecraft" in guide
    assert "scripts\\check_native_aim_sim.bat" in guide
    assert "check_aim_os.bat -Strict" in guide
    assert "PASS" in guide
    assert "FAIL" in guide
