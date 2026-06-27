<#
.SYNOPSIS
  Loads the validated combo-god live model and arms the Judas mod only when
  Minecraft is already in a valid boxing world.

.DESCRIPTION
  This is intentionally bounded: no training, no process killing, and no blind
  arm command while the mod reports no_world/screen/no_focus/not_boxing_match.
#>
[CmdletBinding()]
param(
    [string]$Checkpoint = "runs/combo_god_leaderboard10_combo12/safe_latest.pt",
    [string]$Out = "models/combo_god_leaderboard10_combo12-safe_latest.pts",
    [double]$MaxCps = 10.0,
    [double]$MaxRotSpeed = 195.0,
    [string]$MinecraftDir = "$env:APPDATA\.minecraft",
    [int]$WaitSeconds = 8,
    [int]$ReadyTimeoutSeconds = 0,
    [switch]$LiveTarget
)

$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
$checkSafe = Join-Path $PSScriptRoot "check_safe_export.ps1"
$judasLive = Join-Path $PSScriptRoot "judas_live.ps1"
$controlMod = Join-Path $PSScriptRoot "control_judas_mod.ps1"

function Invoke-Step([string]$Label, [scriptblock]$Block) {
    Write-Output "FIELD_STEP $Label"
    & $Block
    $code = $LASTEXITCODE
    if ($null -ne $code -and $code -ne 0) {
        Write-Output "FIELD_FAIL step=$Label code=$code"
        exit $code
    }
}

function Get-ControlAckResult([string[]]$Lines, [string]$Command) {
    $ack = $Lines | Where-Object { $_ -like "CONTROL_ACK * command=$Command *" } | Select-Object -Last 1
    if (-not $ack) { return $null }
    if ($ack -match " result=([^ ]+) ") { return $Matches[1] }
    return $null
}

function Test-MinecraftRunning([string]$GameDir) {
    $escaped = [Regex]::Escape($GameDir)
    try {
        $procs = @(Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object {
            ($_.Name -match '^(java|javaw|Minecraft)(\.exe)?$') -and
            (
                $_.CommandLine -match 'net\.minecraft\.client' -or
                $_.CommandLine -match '--gameDir' -or
                $_.CommandLine -match $escaped
            )
        })
        return $procs.Count -gt 0
    } catch {
        Write-Output "FIELD_WARN minecraft_process_check=$($_.Exception.GetType().Name)"
        return $true
    }
}

function Get-ModStatusResult {
    if (-not (Test-MinecraftRunning $MinecraftDir)) {
        Write-Output "FIELD_NOT_READY result=no_minecraft_process"
        exit 3
    }
    $statusCommand = if ($LiveTarget) { "status_live" } else { "status" }
    $statusLines = & powershell -NoProfile -ExecutionPolicy Bypass -File $controlMod `
        -Command $statusCommand -MinecraftDir $MinecraftDir -WaitSeconds $WaitSeconds -RequireAck 2>&1
    $statusCode = $LASTEXITCODE
    $statusLines | ForEach-Object { [Console]::Out.WriteLine($_) }
    if ($statusCode -ne 0) {
        Write-Output "FIELD_FAIL step=mod_status code=$statusCode"
        exit $statusCode
    }
    $statusResult = Get-ControlAckResult $statusLines $statusCommand
    if (-not $statusResult) {
        Write-Output "FIELD_FAIL step=mod_status reason=no_ack_result"
        exit 2
    }
    return $statusResult
}

Invoke-Step "safe_export" {
    powershell -NoProfile -ExecutionPolicy Bypass -File $checkSafe `
        -Checkpoint $Checkpoint -Export $Out
}

Invoke-Step "live_load" {
    powershell -NoProfile -ExecutionPolicy Bypass -File $judasLive `
        -Checkpoint $Checkpoint -Out $Out -MaxCps $MaxCps -MaxRotSpeed $MaxRotSpeed `
        -NoExport -NoLaunch
}

$deadline = (Get-Date).AddSeconds([Math]::Max(0, $ReadyTimeoutSeconds))
$statusResult = $null
do {
    $statusResult = Get-ModStatusResult
    if ($statusResult -eq "ok") { break }
    if ((Get-Date) -ge $deadline) { break }
    Write-Output "FIELD_WAITING result=$statusResult"
    Start-Sleep -Seconds 1
} while ($true)

if ($statusResult -ne "ok") {
    Write-Output "FIELD_NOT_READY result=$statusResult"
    exit 3
}

$armCommand = if ($LiveTarget) { "arm_native_live" } else { "arm_native" }
$armLines = & powershell -NoProfile -ExecutionPolicy Bypass -File $controlMod `
    -Command $armCommand -MinecraftDir $MinecraftDir -WaitSeconds $WaitSeconds -RequireAck 2>&1
$armCode = $LASTEXITCODE
$armLines | ForEach-Object { Write-Output $_ }
if ($armCode -ne 0) {
    Write-Output "FIELD_FAIL step=arm_native code=$armCode"
    exit $armCode
}

$armResult = Get-ControlAckResult $armLines $armCommand
if ($armResult -ne "armed") {
    Write-Output "FIELD_FAIL step=arm_native result=$armResult"
    exit 2
}

$mode = if ($LiveTarget) { "live_target" } else { "boxing" }
Write-Output "FIELD_ARMED mode=$mode model=$Out cps=$MaxCps rot=$MaxRotSpeed"
exit 0
