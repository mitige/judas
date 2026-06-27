<#
.SYNOPSIS
  Verifie le chemin daemon /live WebSocket avec le modele safe combo god.
#>
[CmdletBinding()]
param(
    [string]$HostName = "127.0.0.1",
    [int]$Port = 8765,
    [string]$Model = "models\combo_god_recovery_kb092_combo12-safe_latest.pts",
    [int]$Ticks = 16,
    [switch]$NoLoad
)
$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
if (-not $PSBoundParameters.ContainsKey("Model")) {
    $preferred = Join-Path $repo "models\combo_god_recovery_kb092_combo12-safe_latest.pts"
    $leaderboard = Join-Path $repo "models\combo_god_leaderboard10_combo12-safe_latest.pts"
    $counter = Join-Path $repo "models\combo_god_countertap96_combo12-safe_latest.pts"
    $legacy = Join-Path $repo "models\combo_god_directpad_lock_combo12-safe_latest.pts"
    if (-not (Test-Path -LiteralPath $preferred) -and (Test-Path -LiteralPath $leaderboard)) {
        $Model = "models\combo_god_leaderboard10_combo12-safe_latest.pts"
    } elseif (-not (Test-Path -LiteralPath $preferred) -and (Test-Path -LiteralPath $counter)) {
        $Model = "models\combo_god_countertap96_combo12-safe_latest.pts"
    } elseif (-not (Test-Path -LiteralPath $preferred) -and (Test-Path -LiteralPath $legacy)) {
        $Model = "models\combo_god_directpad_lock_combo12-safe_latest.pts"
    }
}
$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $cmd) { throw "Python introuvable." }
    $python = $cmd.Source
}
$argsList = @(
    "--host", $HostName,
    "--port", $Port,
    "--model", $Model,
    "--ticks", $Ticks
)
if ($NoLoad) { $argsList += "--no-load" }
& $python (Join-Path $repo "tools\live_ws_check.py") @argsList
exit $LASTEXITCODE
