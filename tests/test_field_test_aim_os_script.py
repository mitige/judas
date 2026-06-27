from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_field_test_aim_os_chains_deploy_check_and_aim_watch():
    source = (ROOT / "scripts/field_test_aim_os.ps1").read_text()

    assert "watch_deploy_aim_os.ps1" in source
    assert "check_mod_deploy.ps1" in source
    assert "check_aim_os.ps1" in source
    assert "judas_live.ps1" in source
    assert "-NoExport" in source
    assert "-ForceDaemon" in source
    assert "-AllowStaleExport" not in source
    assert "check_live_ws.ps1" in source
    assert "check_live_actions.ps1" in source
    assert "check_packet_order.ps1" in source
    assert "check_field_status.ps1" in source
    assert "control_judas_mod.ps1" in source
    assert "watch_aim_os.ps1" in source
    assert "stop_judas_live.ps1" in source
    assert "DeployTimeoutSeconds" in source
    assert "BuildTimeoutSeconds" in source
    assert "AimTimeoutSeconds" in source
    assert "MinLooseSamples" in source
    assert "NoLive" in source
    assert "KeepLive" in source
    assert "KeepOtherJudasProcesses" in source
    assert "KeepAimLog" in source
    assert "ProofLog" in source
    assert "NoProofLog" in source
    assert "SkipLivePrecheck" in source
    assert "SkipPacketOrder" in source
    assert "SkipModControl" in source
    assert "UseDeployedMod" in source
    assert "RequireMinecraft" in source
    assert "LiveActionLog" in source
    assert "judas-live-actions.log" in source
    assert "PacketLog" in source
    assert "MinecraftLog" in source
    assert "PacketSession" in source
    assert "judas-packet-order.log" in source
    assert "judas-packet-order-session.txt" in source
    assert "-ActionLog" in source
    assert "-FreshAfter" in source
    assert "-Strict" in source
    assert "$stepOutput = & powershell" in source
    assert "$stepExitCode = $LASTEXITCODE" in source
    assert "return [int]$stepExitCode" in source
    assert "stop_judas_ui.ps1" in source
    assert "stop_judas_train.ps1" in source
    assert "stop_combo_god.ps1" in source
    assert "Stop-OtherJudasProcesses" in source
    assert "pre-stop app/viz failed" in source
    assert "pre-stop training failed" in source
    assert "pre-stop combo training failed" in source
    assert "Prepare-DeployedMod" in source
    assert "Get-MinecraftProcessIds" in source
    assert "Assert-MinecraftRunning" in source
    assert "minecraft_not_running" in source
    assert "judas-control.txt" in source
    assert "Using deployed Judas jar" in source
    assert "-RequireWritable" in source
    assert "check_mod_deploy deployed mod failed" in source
    assert "check_aim_os reset failed" in source
    assert "post-precheck reset" in source
    assert "arm_native" in source
    assert "control_judas_mod arm_native failed" in source
    assert "check_packet_order reset failed" in source
    assert "check_packet_order failed" in source
    assert "watch_aim_os failed" in source
    assert "runs\\field_proof" in source
    assert "Start-Transcript" in source
    assert "Stop-Transcript" in source
    assert "proof_log=$ProofLog" in source
    assert "cleanup live failed" in source
    assert "field status failed" in source
    assert "transcript stop failed" in source
    assert "$DeployTimeoutSeconds = 120" in source
    assert "$BuildTimeoutSeconds = 300" in source
    assert "$AimTimeoutSeconds = 120" in source
    assert "live safe model OK" in source
    assert source.index("    Assert-MinecraftRunning") < source.index("    Stop-OtherJudasProcesses")
    assert source.index("Stop-OtherJudasProcesses") < source.index("pre-stop live daemon failed")
    assert source.index("Start-Transcript") < source.index("    Stop-OtherJudasProcesses")
    assert source.index("Invoke-PowerShellStep $watchDeploy") < source.index('Invoke-PowerShellStep $checkDeploy @("-ModsDir", $ModsDir)')
    assert '"-BuildTimeoutSeconds", [string]$BuildTimeoutSeconds' in source
    prepare_call = source.index("    Prepare-DeployedMod")
    assert prepare_call < source.index("Invoke-PowerShellStep $checkAim")
    assert source.index("Invoke-PowerShellStep $checkAim") < source.index('"-Log", $Log')
    assert source.index('"-Log", $Log') < source.index('"-Reset"')
    assert source.index('"-Log", $PacketLog') < source.index("check_packet_order reset failed")
    assert source.index('"-MinecraftLog", $MinecraftLog') < source.index("check_packet_order reset failed")
    assert source.index('"-Session", $PacketSession') < source.index("check_packet_order reset failed")
    assert source.index("Invoke-PowerShellStep $checkAim") < source.index("Invoke-PowerShellStep $checkPacketOrder @(")
    assert prepare_call < source.index("Invoke-PowerShellStep $judasLive")
    assert prepare_call < source.index("Invoke-PowerShellStep $checkPacketOrder @(")
    assert source.index("Invoke-PowerShellStep $checkPacketOrder @(") < source.index("Invoke-PowerShellStep $judasLive")
    assert source.index("Invoke-PowerShellStep $judasLive") < source.index("Invoke-PowerShellStep $checkLive @(")
    assert source.index("Invoke-PowerShellStep $checkLive @(") < source.index("Invoke-PowerShellStep $watchAim")
    assert source.index("Invoke-PowerShellStep $watchAim") < source.index('"-FreshAfter", $fieldStartedAt')
    assert source.index('"-FreshAfter", $fieldStartedAt') < source.index("watch_aim_os failed")
    assert source.index("watch_aim_os failed") < source.rindex("Invoke-PowerShellStep $checkLiveActions @(")
    assert source.index("check_live_ws failed") < source.index("check_live_actions post-precheck reset failed")
    assert source.index("check_live_actions post-precheck reset failed") < source.index("Invoke-PowerShellStep $controlMod")
    assert source.index("Invoke-PowerShellStep $controlMod") < source.index("Invoke-PowerShellStep $watchAim")
    assert source.index("Invoke-PowerShellStep $watchAim") < source.rindex("Invoke-PowerShellStep $checkLiveActions @(")
    assert source.index("Invoke-PowerShellStep $watchAim") < source.rindex("Invoke-PowerShellStep $checkPacketOrder @(")
    assert source.rindex('"-Log", $PacketLog') < source.index("check_packet_order failed")
    assert source.rindex('"-MinecraftLog", $MinecraftLog') < source.index("check_packet_order failed")
    assert source.rindex('"-Session", $PacketSession') < source.index("check_packet_order failed")
    assert "$LASTEXITCODE" in source
    assert "finally" in source
    assert "Show-FieldStatus" in source
    assert "---FIELD_STATUS---" in source
    assert "-AimLog $Log" in source
    assert "-LiveLog $LiveActionLog" in source
    assert "-PacketLog $PacketLog" in source
    assert "-MinecraftLog $MinecraftLog" in source
    assert "-PacketSession $PacketSession" in source
    assert "-FreshAfter $fieldStartedAt" in source
    assert "ToUniversalTime().ToString(\"o\")" in source
    assert "Stop-OwnedLive" in source
    assert source.rindex("Stop-OwnedLive") < source.rindex("Show-FieldStatus")
    assert source.rindex("Show-FieldStatus") < source.rindex("Stop-Transcript")
    assert "pre-stop live daemon failed" in source
    assert source.index("pre-stop live daemon failed") < prepare_call
    assert_minecraft_call = source.index("    Assert-MinecraftRunning")
    assert assert_minecraft_call < source.index("    Stop-OtherJudasProcesses")
    assert source.index("pre-stop live daemon failed") < prepare_call


def test_field_test_aim_os_bat_wraps_powershell_with_bypass():
    source = (ROOT / "scripts/field_test_aim_os.bat").read_text()
    quick = (ROOT / "scripts/field_test_aim_os_quick.bat").read_text()

    assert "ExecutionPolicy Bypass" in source
    assert "field_test_aim_os.ps1" in source
    assert "ExecutionPolicy Bypass" in quick
    assert "field_test_aim_os.ps1" in quick
    assert "-UseDeployedMod" in quick
    assert "-RequireMinecraft" in quick


def test_control_judas_mod_script_and_mod_file_commands_are_wired():
    ps1 = (ROOT / "scripts/control_judas_mod.ps1").read_text()
    bat = (ROOT / "scripts/control_judas_mod.bat").read_text()
    mod = (ROOT / "mod/src/main/java/dev/judas/bridge/JudasMod.java").read_text()

    assert "judas-control.txt" in ps1
    assert "judas-bridge-status.log" in ps1
    assert "dump_screen" in ps1
    assert "arm_native_live" in ps1
    assert "status_live" in ps1
    assert "CONTROL_ACK_TIMEOUT" in ps1
    assert "-RequireAck" in ps1
    assert "ToUnixTimeMilliseconds" in ps1
    assert "$ackMillis -ge ($startMillis - 500)" in ps1
    assert "Write-ControlCommand" in ps1
    assert "[System.IO.FileShare]::ReadWrite" in ps1
    assert "Set-Content -LiteralPath $control" not in ps1
    assert "control_judas_mod.ps1" in bat
    assert "ExecutionPolicy Bypass" in bat

    assert "judas-control.txt" in mod
    assert "judas-bridge-status.log" in mod
    assert "\"arm_native\".equals(command)" in mod
    assert "\"arm_native_live\".equals(command)" in mod
    assert "\"status_live\".equals(command)" in mod
    assert "applier.setNative(true)" in mod
    assert "armBot(mc, command, true)" in mod
    assert "armBot(mc, command, false)" in mod
    assert "not_live_match" in mod
    assert "blocked:" in mod
    assert "ws.isOpen()" in mod
    assert "handleControlFile(mc);\n        if (mc.thePlayer == null || mc.theWorld == null)" in mod
    assert 'logStatus(command, "none".equals(reason) ? "ok" : "ok:" + reason)' in mod
