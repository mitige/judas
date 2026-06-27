from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_check_mod_deploy_compares_active_jar_hashes_read_only():
    source = (ROOT / "scripts/check_mod_deploy.ps1").read_text()

    assert ".minecraft\\mods" in source
    assert "mod\\build\\libs\\judas-bridge-0.1.0.jar" in source
    assert "Get-ChildItem -Path $ModsDir -Filter \"judas-bridge-*.jar\" -File" in source
    assert "Where-Object { $_.Name -notmatch '\\.disabled-' }" in source
    assert "Get-FileHash" in source
    assert "OK_DEPLOY" in source
    assert "STALE_DEPLOY" in source
    assert "MULTIPLE_DEPLOY" in source
    assert "[switch]$RequireWritable" in source
    assert "LOCKED_DEPLOY" in source
    assert "freshness=$freshness" in source
    assert "RestartManager.NativeMethods" in source
    assert "RmGetList" in source
    assert "lockers=" in source
    assert source.index("LOCKED_DEPLOY") < source.index("STALE_DEPLOY")
    assert "[System.IO.FileShare]::None" in source
    assert "MISSING_DEPLOY" in source
    assert "Copy-Item" not in source
    assert "Move-Item" not in source
    assert "Remove-Item" not in source


def test_check_mod_deploy_bat_wraps_powershell_with_bypass():
    source = (ROOT / "scripts/check_mod_deploy.bat").read_text()

    assert "ExecutionPolicy Bypass" in source
    assert "check_mod_deploy.ps1" in source
