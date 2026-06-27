<#
.SYNOPSIS
  Checks the structured live action log for escape and sky aiming.
#>
[CmdletBinding()]
param(
    [string]$Log = "",
    [switch]$Reset,
    [switch]$Strict,
    [switch]$All,
    [int]$MinSamples = 20,
    [double]$MaxAttackCps = 10.0,
    [double]$MinStrafeFrac = 0.50,
    [int]$OpenerTicks = 20,
    [double]$MinOpenerStrafeFrac = 0.75,
    [double]$MinOpenerStrafeHoldFrac = 0.70,
    [double]$MinOpenerPressureFrac = 0.60,
    [double]$MaxStrafeFlipFrac = 0.10,
    [double]$MinStrafeHoldAvg = 3.0,
    [double]$MinHitWtapFrac = 0.75,
    [double]$MinUnderComboCounterHitFrac = 0.0,
    [string]$RequireModel = "combo_god_recovery_kb092_combo12|combo_god_leaderboard10_combo12|combo_god_attn96_combo12|combo_god_consistent|combo_god_candidate_freshopt|combo_god_bodyaim96_combo12"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
if (-not $Log) {
    $Log = Join-Path $repo "runs\judas-live-actions.log"
}
$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $cmd) { throw "Python introuvable." }
    $python = $cmd.Source
}

if ($Reset) {
    $dir = Split-Path $Log -Parent
    if ($dir -and -not (Test-Path -LiteralPath $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    Remove-Item -LiteralPath $Log -Force -ErrorAction SilentlyContinue
    Write-Host "live action log reset: $Log"
    exit 0
}

$argsList = @(
    "--min-samples", [string]$MinSamples,
    "--max-attack-cps", [string]$MaxAttackCps,
    "--min-strafe-frac", [string]$MinStrafeFrac,
    "--opener-ticks", [string]$OpenerTicks,
    "--min-opener-strafe-frac", [string]$MinOpenerStrafeFrac,
    "--min-opener-strafe-hold-frac", [string]$MinOpenerStrafeHoldFrac,
    "--min-opener-pressure-frac", [string]$MinOpenerPressureFrac,
    "--max-strafe-flip-frac", [string]$MaxStrafeFlipFrac,
    "--min-strafe-hold-avg", [string]$MinStrafeHoldAvg,
    "--min-hit-wtap-frac", [string]$MinHitWtapFrac,
    "--min-under-combo-counter-hit-frac", [string]$MinUnderComboCounterHitFrac,
    "--require-model", $RequireModel
)
if ($Strict) { $argsList += "--strict" }
if ($All) { $argsList += "--all" }
$argsList += $Log

& $python (Join-Path $repo "tools\live_action_log.py") @argsList
exit $LASTEXITCODE
