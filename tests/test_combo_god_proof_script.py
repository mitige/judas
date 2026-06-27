from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_combo_god_proof_script_chains_arena_live_and_field_status():
    source = (ROOT / "scripts/prove_combo_god.ps1").read_text()

    assert "check_arena_combo.ps1" in source
    assert "judas_live.ps1" in source
    assert "check_live_ws.ps1" in source
    assert "check_live_actions.ps1" in source
    assert "check_field_status.ps1" in source
    assert "combo-proof-live-actions.log" in source
    assert "combo-proof-live-daemon.pid" in source
    assert "runs\\judas-live-actions.log" not in source
    assert "train_combo_god" not in source
    assert "python -m train.run" not in source


def test_combo_god_proof_script_stops_live_and_keeps_field_optional():
    source = (ROOT / "scripts/prove_combo_god.ps1").read_text()

    assert "finally" in source
    assert "Stop-ProofLive" in source
    assert "-ActionLog" in source
    assert "-NoExport" in source
    assert "-ForceDaemon" in source
    assert "-AllowStaleExport" not in source
    assert "-NoLaunch" in source
    assert "-RequireField" in source
    assert "$proofStartedAt = (Get-Date).ToUniversalTime().ToString(\"o\")" in source
    assert '"-LiveLog", $SyntheticActionLog' in source
    assert '"-FreshAfter", $proofStartedAt' in source
    assert "if ($RequireField) { $fieldArgs += \"-Strict\" }" in source
    assert "FIELD_STATUS failed" in source
    assert source.index("Stop-ProofLive") < source.index('Write-Output "---FIELD_STATUS---"')


def test_combo_god_proof_bat_wraps_powershell_with_bypass():
    source = (ROOT / "scripts/prove_combo_god.bat").read_text()

    assert "ExecutionPolicy Bypass" in source
    assert "prove_combo_god.ps1" in source
