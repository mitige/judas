<#
.SYNOPSIS
  Bounded field test for OS mouse aim with the combo-safe live model.

.DESCRIPTION
  Deploys/checks the mod, starts judas_live with the safe combo model, runs a
  synthetic WebSocket precheck, then waits for a fresh aim OS verdict.
  The live daemon started by this script is stopped on exit unless -KeepLive is set.
#>
[CmdletBinding()]
param(
    [string]$ModsDir = "$env:APPDATA\.minecraft\mods",
    [string]$Log = "$env:APPDATA\.minecraft\judas-aim-os.log",
    [int]$IntervalSeconds = 2,
    [int]$DeployTimeoutSeconds = 120,
    [int]$BuildTimeoutSeconds = 300,
    [int]$AimTimeoutSeconds = 120,
    [int]$MinLooseSamples = 40,
    [string]$LivePidFile = "",
    [string]$LiveActionLog = "",
    [string]$PacketLog = "$env:APPDATA\.minecraft\judas-packet-order.log",
    [string]$MinecraftLog = "$env:APPDATA\.minecraft\logs\latest.log",
    [string]$PacketSession = "$env:APPDATA\.minecraft\judas-packet-order-session.txt",
    [string]$ProofLog = "",
    [switch]$NoLive,
    [switch]$KeepLive,
    [switch]$KeepOtherJudasProcesses,
    [switch]$KeepAimLog,
    [switch]$NoProofLog,
    [switch]$SkipLivePrecheck,
    [switch]$SkipPacketOrder,
    [switch]$SkipModControl,
    [switch]$UseDeployedMod,
    [switch]$RequireMinecraft
)

$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
if (-not $LivePidFile) {
    $LivePidFile = Join-Path $repo "runs\judas_live_daemon.pid"
}
if (-not $LiveActionLog) {
    $LiveActionLog = Join-Path $repo "runs\judas-live-actions.log"
}
if (-not $ProofLog) {
    $proofDir = Join-Path $repo "runs\field_proof"
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $ProofLog = Join-Path $proofDir "field_$stamp.log"
}

$watchDeploy = Join-Path $PSScriptRoot "watch_deploy_aim_os.ps1"
$checkDeploy = Join-Path $PSScriptRoot "check_mod_deploy.ps1"
$checkAim = Join-Path $PSScriptRoot "check_aim_os.ps1"
$watchAim = Join-Path $PSScriptRoot "watch_aim_os.ps1"
$judasLive = Join-Path $PSScriptRoot "judas_live.ps1"
$stopLive = Join-Path $PSScriptRoot "stop_judas_live.ps1"
$stopUi = Join-Path $PSScriptRoot "stop_judas_ui.ps1"
$stopTrain = Join-Path $PSScriptRoot "stop_judas_train.ps1"
$stopCombo = Join-Path $PSScriptRoot "stop_combo_god.ps1"
$checkLive = Join-Path $PSScriptRoot "check_live_ws.ps1"
$checkLiveActions = Join-Path $PSScriptRoot "check_live_actions.ps1"
$checkPacketOrder = Join-Path $PSScriptRoot "check_packet_order.ps1"
$checkFieldStatus = Join-Path $PSScriptRoot "check_field_status.ps1"
$controlMod = Join-Path $PSScriptRoot "control_judas_mod.ps1"

function Invoke-PowerShellStep([string]$Script, [string[]]$ArgsList) {
    $stepOutput = & powershell -NoProfile -ExecutionPolicy Bypass -File $Script @ArgsList 2>&1
    $stepExitCode = $LASTEXITCODE
    foreach ($line in $stepOutput) {
        Write-Host $line
    }
    return [int]$stepExitCode
}

function Stop-OwnedLive {
    if (-not $NoLive -and -not $KeepLive) {
        & powershell -NoProfile -ExecutionPolicy Bypass -File $stopLive -PidFile $LivePidFile
    }
}

function Stop-OtherJudasProcesses {
    if ($KeepOtherJudasProcesses) { return }
    $exit = Invoke-PowerShellStep $stopUi @("-Surface", "all")
    if ($exit -ne 0) { throw "pre-stop app/viz failed code=$exit" }
    $exit = Invoke-PowerShellStep $stopTrain @()
    if ($exit -ne 0) { throw "pre-stop training failed code=$exit" }
    $exit = Invoke-PowerShellStep $stopCombo @()
    if ($exit -ne 0) { throw "pre-stop combo training failed code=$exit" }
}

function Get-MinecraftProcessIds {
    try {
        return @(Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object {
            $_.Name -match "java|javaw|Minecraft|prismlauncher|MultiMC|lunar|badlion" -and
            $_.CommandLine -and (
                $_.CommandLine -match "net\.minecraft\.client" -or
                $_.CommandLine -match "\.minecraft" -or
                $_.CommandLine -match "minecraft" -or
                $_.CommandLine -match "fabric" -or
                $_.CommandLine -match "forge"
            )
        } | Select-Object -ExpandProperty ProcessId)
    } catch {
        return @()
    }
}

function Assert-MinecraftRunning {
    if (-not $RequireMinecraft) { return }
    $minecraftPids = @(Get-MinecraftProcessIds)
    if ($minecraftPids.Count -eq 0) {
        throw "minecraft_not_running: launch Minecraft before field proof, enter the arena, then run this script to arm Judas through judas-control.txt"
    }
    Write-Host ("[aim_os] Minecraft detected pids={0}" -f ($minecraftPids -join ",")) -ForegroundColor Cyan
}

function Prepare-DeployedMod {
    if ($UseDeployedMod) {
        Write-Host "[aim_os] Using deployed Judas jar; build/deploy skipped." -ForegroundColor Cyan
        $exit = Invoke-PowerShellStep $checkDeploy @(
            "-ModsDir", $ModsDir,
            "-RequireWritable"
        )
        if ($exit -ne 0) { throw "check_mod_deploy deployed mod failed code=$exit" }
        return
    }

    $exit = Invoke-PowerShellStep $watchDeploy @(
        "-ModsDir", $ModsDir,
        "-IntervalSeconds", [string]$IntervalSeconds,
        "-TimeoutSeconds", [string]$DeployTimeoutSeconds,
        "-BuildTimeoutSeconds", [string]$BuildTimeoutSeconds
    )
    if ($exit -ne 0) { throw "watch_deploy_aim_os failed code=$exit" }

    $exit = Invoke-PowerShellStep $checkDeploy @("-ModsDir", $ModsDir)
    if ($exit -ne 0) { throw "check_mod_deploy failed code=$exit" }
}

function Show-FieldStatus {
    Write-Output "---FIELD_STATUS---"
    & powershell -NoProfile -ExecutionPolicy Bypass -File $checkFieldStatus `
        -AimLog $Log `
        -LiveLog $LiveActionLog `
        -PacketLog $PacketLog `
        -ModsDir $ModsDir `
        -MinecraftLog $MinecraftLog `
        -PacketSession $PacketSession `
        -FreshAfter $fieldStartedAt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[aim_os] FIELD_STATUS diagnostic failed code=$LASTEXITCODE" -ForegroundColor Yellow
    }
}

$exitCode = 0
$fieldStartedAt = (Get-Date).ToUniversalTime().ToString("o")
$transcriptStarted = $false
try {
    if (-not $NoProofLog) {
        $proofParent = Split-Path $ProofLog -Parent
        if ($proofParent -and -not (Test-Path -LiteralPath $proofParent)) {
            New-Item -ItemType Directory -Path $proofParent -Force | Out-Null
        }
        Start-Transcript -LiteralPath $ProofLog -Force | Out-Null
        $transcriptStarted = $true
        Write-Host "[aim_os] Proof log: $ProofLog" -ForegroundColor DarkGray
    }

    Assert-MinecraftRunning
    Stop-OtherJudasProcesses

    if (-not $NoLive) {
        & powershell -NoProfile -ExecutionPolicy Bypass -File $stopLive -PidFile $LivePidFile
        if ($LASTEXITCODE -ne 0) { throw "pre-stop live daemon failed code=$LASTEXITCODE" }
    }

    Prepare-DeployedMod

    if (-not $KeepAimLog) {
        $exitCode = Invoke-PowerShellStep $checkAim @(
            "-Log", $Log,
            "-Reset"
        )
        if ($exitCode -ne 0) { throw "check_aim_os reset failed code=$exitCode" }
    }

    if (-not $SkipPacketOrder) {
        $exitCode = Invoke-PowerShellStep $checkPacketOrder @(
            "-Log", $PacketLog,
            "-MinecraftLog", $MinecraftLog,
            "-Session", $PacketSession,
            "-Reset"
        )
        if ($exitCode -ne 0) { throw "check_packet_order reset failed code=$exitCode" }
    }

    if (-not $NoLive) {
        $exitCode = Invoke-PowerShellStep $checkLiveActions @(
            "-Log", $LiveActionLog,
            "-Reset"
        )
        if ($exitCode -ne 0) { throw "check_live_actions reset failed code=$exitCode" }

        $exitCode = Invoke-PowerShellStep $judasLive @(
            "-NoExport",
            "-NoLaunch",
            "-ForceDaemon",
            "-PidFile", $LivePidFile,
            "-ActionLog", $LiveActionLog
        )
        if ($exitCode -ne 0) { throw "judas_live failed code=$exitCode" }

        if (-not $SkipLivePrecheck) {
            $exitCode = Invoke-PowerShellStep $checkLive @(
                "-NoLoad",
                "-Ticks", "16"
            )
            if ($exitCode -ne 0) { throw "check_live_ws failed code=$exitCode" }

            $exitCode = Invoke-PowerShellStep $checkLiveActions @(
                "-Log", $LiveActionLog,
                "-Reset"
            )
            if ($exitCode -ne 0) { throw "check_live_actions post-precheck reset failed code=$exitCode" }
        }

        if (-not $SkipModControl) {
            $exitCode = Invoke-PowerShellStep $controlMod @(
                "-Command", "arm_native",
                "-WaitSeconds", "8",
                "-RequireAck"
            )
            if ($exitCode -ne 0) { throw "control_judas_mod arm_native failed code=$exitCode" }
        }
    }

    Write-Host "[aim_os] Jar OK, live safe model OK. Start Minecraft, press O for OS mouse, K to arm; waiting for aim verdict..." -ForegroundColor Cyan
    $exitCode = Invoke-PowerShellStep $watchAim @(
        "-Log", $Log,
        "-IntervalSeconds", [string]$IntervalSeconds,
        "-TimeoutSeconds", [string]$AimTimeoutSeconds,
        "-MinLooseSamples", [string]$MinLooseSamples,
        "-FreshAfter", $fieldStartedAt
    )
    if ($exitCode -ne 0) { throw "watch_aim_os failed code=$exitCode" }
    if ($exitCode -eq 0 -and -not $NoLive) {
        $exitCode = Invoke-PowerShellStep $checkLiveActions @(
            "-Log", $LiveActionLog,
            "-Strict",
            "-MinSamples", "20"
        )
        if ($exitCode -ne 0) { throw "check_live_actions failed code=$exitCode" }
    }
    if ($exitCode -eq 0 -and -not $SkipPacketOrder) {
        $exitCode = Invoke-PowerShellStep $checkPacketOrder @(
            "-Log", $PacketLog,
            "-MinecraftLog", $MinecraftLog,
            "-Session", $PacketSession,
            "-Strict"
        )
        if ($exitCode -ne 0) { throw "check_packet_order failed code=$exitCode" }
    }
} catch {
    if ($exitCode -eq 0) { $exitCode = 1 }
    Write-Host "[aim_os] FAILED $($_.Exception.Message)" -ForegroundColor Red
} finally {
    try {
        Stop-OwnedLive
    } catch {
        if ($exitCode -eq 0) { $exitCode = 1 }
        Write-Host "[aim_os] cleanup live failed $($_.Exception.Message)" -ForegroundColor Yellow
    }
    try {
        Show-FieldStatus
    } catch {
        if ($exitCode -eq 0) { $exitCode = 1 }
        Write-Host "[aim_os] field status failed $($_.Exception.Message)" -ForegroundColor Yellow
    }
    if ($transcriptStarted) {
        try {
            Stop-Transcript | Out-Null
        } catch {
            if ($exitCode -eq 0) { $exitCode = 1 }
            Write-Host "[aim_os] transcript stop failed $($_.Exception.Message)" -ForegroundColor Yellow
        }
        Write-Output "proof_log=$ProofLog"
    }
}

exit $exitCode
