<#
.SYNOPSIS
  Short non-training proof for the current combo god model.

.DESCRIPTION
  Runs the arena two-role proof, starts a temporary live daemon with a dedicated
  synthetic action log, runs the live WebSocket check, verifies no escape/sky in
  that synthetic live path, then stops the daemon. Field status is printed at
  the end but only gates the exit code when -RequireField is passed.
#>
[CmdletBinding()]
param(
    [string]$PidFile = "",
    [string]$SyntheticActionLog = "",
    [switch]$SkipArena,
    [switch]$SkipLive,
    [switch]$RequireField
)

$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
if (-not $PidFile) {
    $PidFile = Join-Path $repo "runs\combo-proof-live-daemon.pid"
}
if (-not $SyntheticActionLog) {
    $SyntheticActionLog = Join-Path $repo "runs\combo-proof-live-actions.log"
}

$checkArena = Join-Path $PSScriptRoot "check_arena_combo.ps1"
$judasLive = Join-Path $PSScriptRoot "judas_live.ps1"
$checkLiveWs = Join-Path $PSScriptRoot "check_live_ws.ps1"
$checkLiveActions = Join-Path $PSScriptRoot "check_live_actions.ps1"
$checkFieldStatus = Join-Path $PSScriptRoot "check_field_status.ps1"
$stopLive = Join-Path $PSScriptRoot "stop_judas_live.ps1"

function Invoke-Step([string]$Name, [string]$Script, [string[]]$ArgsList) {
    Write-Output "---$Name---"
    & powershell -NoProfile -ExecutionPolicy Bypass -File $Script @ArgsList
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed code=$LASTEXITCODE"
    }
}

function Stop-ProofLive {
    & powershell -NoProfile -ExecutionPolicy Bypass -File $stopLive -PidFile $PidFile
}

$exitCode = 0
$proofStartedAt = (Get-Date).ToUniversalTime().ToString("o")
try {
    Stop-ProofLive

    if (-not $SkipArena) {
        Invoke-Step "ARENA_PROOF" $checkArena @("-Events")
    }

    if (-not $SkipLive) {
        Invoke-Step "LIVE_ACTION_LOG_RESET" $checkLiveActions @(
            "-Log", $SyntheticActionLog,
            "-Reset"
        )
        Invoke-Step "LIVE_START" $judasLive @(
            "-NoExport",
            "-NoLaunch",
            "-ForceDaemon",
            "-PidFile", $PidFile,
            "-ActionLog", $SyntheticActionLog
        )
        Invoke-Step "LIVE_WS_PROOF" $checkLiveWs @(
            "-NoLoad",
            "-Ticks", "24"
        )
        Invoke-Step "LIVE_ACTION_PROOF" $checkLiveActions @(
            "-Log", $SyntheticActionLog,
            "-Strict",
            "-MinSamples", "20"
        )
    }

    Stop-ProofLive
    Write-Output "---FIELD_STATUS---"
    $fieldArgs = @(
        "-LiveLog", $SyntheticActionLog,
        "-FreshAfter", $proofStartedAt
    )
    if ($RequireField) { $fieldArgs += "-Strict" }
    & powershell -NoProfile -ExecutionPolicy Bypass -File $checkFieldStatus @fieldArgs
    if ($RequireField -and $LASTEXITCODE -ne 0) {
        throw "FIELD_STATUS failed code=$LASTEXITCODE"
    }
} catch {
    $exitCode = 1
    Write-Host "[combo-proof] FAILED $($_.Exception.Message)" -ForegroundColor Red
} finally {
    Stop-ProofLive
}

exit $exitCode
