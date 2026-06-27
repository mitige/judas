<#
.SYNOPSIS
  Passive field-proof status: aim OS, live actions, and packet order.
#>
[CmdletBinding()]
param(
    [string]$AimLog = "$env:APPDATA\.minecraft\judas-aim-os.log",
    [string]$LiveLog = "",
    [string]$PacketLog = "$env:APPDATA\.minecraft\judas-packet-order.log",
    [string]$ModsDir = "$env:APPDATA\.minecraft\mods",
    [string]$MinecraftLog = "$env:APPDATA\.minecraft\logs\latest.log",
    [string]$PacketSession = "$env:APPDATA\.minecraft\judas-packet-order-session.txt",
    [string]$FreshAfter = "",
    [int]$MinLiveSamples = 20,
    [double]$MaxLiveAttackCps = 10.0,
    [double]$MinLiveStrafeFrac = 0.50,
    [int]$LiveOpenerTicks = 20,
    [double]$MinLiveOpenerStrafeFrac = 0.75,
    [double]$MinLiveOpenerStrafeHoldFrac = 0.70,
    [double]$MinLiveOpenerPressureFrac = 0.60,
    [double]$MaxLiveStrafeFlipFrac = 0.10,
    [double]$MinLiveStrafeHoldAvg = 3.0,
    [double]$MinLiveHitWtapFrac = 0.75,
    [switch]$AllowStale,
    [switch]$Strict
)

$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
if (-not $LiveLog) {
    $LiveLog = Join-Path $repo "runs\judas-live-actions.log"
}
$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $cmd) { throw "Python introuvable." }
    $python = $cmd.Source
}

$argsList = @(
    "--aim-log", $AimLog,
    "--live-log", $LiveLog,
    "--packet-log", $PacketLog,
    "--mods-dir", $ModsDir,
    "--minecraft-log", $MinecraftLog,
    "--packet-session", $PacketSession,
    "--min-live-samples", [string]$MinLiveSamples,
    "--max-live-attack-cps", [string]$MaxLiveAttackCps,
    "--min-live-strafe-frac", [string]$MinLiveStrafeFrac,
    "--live-opener-ticks", [string]$LiveOpenerTicks,
    "--min-live-opener-strafe-frac", [string]$MinLiveOpenerStrafeFrac,
    "--min-live-opener-strafe-hold-frac", [string]$MinLiveOpenerStrafeHoldFrac,
    "--min-live-opener-pressure-frac", [string]$MinLiveOpenerPressureFrac,
    "--max-live-strafe-flip-frac", [string]$MaxLiveStrafeFlipFrac,
    "--min-live-strafe-hold-avg", [string]$MinLiveStrafeHoldAvg,
    "--min-live-hit-wtap-frac", [string]$MinLiveHitWtapFrac
)
if ($FreshAfter) { $argsList += @("--fresh-after", $FreshAfter) }
if ($AllowStale) { $argsList += "--allow-stale" }
if ($Strict) { $argsList += "--strict" }

& $python (Join-Path $repo "tools\field_test_status.py") @argsList
exit $LASTEXITCODE
