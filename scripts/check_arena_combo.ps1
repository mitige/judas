<#
.SYNOPSIS
  Rejoue des matchs dans le visualiseur et echoue si le modele combo ne boxe
  pas vraiment en role A puis en role B.
#>
[CmdletBinding()]
param(
    [string]$ModelA = "models/combo_god_recovery_kb092_combo12-safe_latest.pts",
    [string]$ModelB = "__combo_spar__",
    [int]$Matches = 4,
    [int]$MinCombo = 12,
    [int]$MaxDraws = 0,
    [int]$MinWinsA = 1,
    [int]$MinWinsB = 0,
    [double]$MinHitsA = 18.0,
    [double]$MinHitsB = 0.0,
    [int]$MinComboB = 0,
    [int]$MirrorMaxDraws = -1,
    [double]$MinCloseFrac = 0.04,
    [double]$MaxSkyFracA = 0.02,
    [double]$MaxSkyFracB = 0.02,
    [double]$MaxBackFracA = 0.01,
    [double]$MaxBackFracB = 0.01,
    [double]$MaxTapBackFracA = 0.0,
    [double]$MaxTapBackFracB = 0.0,
    [double]$MinStrafeFracA = 0.50,
    [double]$MinStrafeFracB = 0.50,
    [double]$MinOpenerStrafeFracA = 0.75,
    [double]$MinOpenerStrafeFracB = 0.75,
    [double]$MinOpenerStrafeHoldFracA = 0.70,
    [double]$MinOpenerStrafeHoldFracB = 0.70,
    [double]$MinOpenerPressureFracA = 0.60,
    [double]$MinOpenerPressureFracB = 0.60,
    [double]$MaxStrafeFracA = 1.0,
    [double]$MaxStrafeFracB = 1.0,
    [double]$MaxJumpFracA = 0.01,
    [double]$MaxJumpFracB = 0.01,
    [double]$CloseDist = 3.25,
    [double]$SkyPitchDeg = 60.0,
    [double]$Cps = 10.0,
    [int]$MaxSteps = 60000,
    [switch]$NoRoleProof,
    [switch]$NoMirrorProof,
    [switch]$Events
)
$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
if ($MirrorMaxDraws -lt 0) {
    $MirrorMaxDraws = $Matches
}
if (-not $PSBoundParameters.ContainsKey("ModelA")) {
    $preferred = Join-Path $repo "models\combo_god_recovery_kb092_combo12-safe_latest.pts"
    $leaderboard = Join-Path $repo "models\combo_god_leaderboard10_combo12-safe_latest.pts"
    $counter = Join-Path $repo "models\combo_god_countertap96_combo12-safe_latest.pts"
    $legacy = Join-Path $repo "models\combo_god_directpad_lock_combo12-safe_latest.pts"
    if (-not (Test-Path -LiteralPath $preferred) -and (Test-Path -LiteralPath $leaderboard)) {
        $ModelA = "models/combo_god_leaderboard10_combo12-safe_latest.pts"
    } elseif (-not (Test-Path -LiteralPath $preferred) -and (Test-Path -LiteralPath $counter)) {
        $ModelA = "models/combo_god_countertap96_combo12-safe_latest.pts"
    } elseif (-not (Test-Path -LiteralPath $preferred) -and (Test-Path -LiteralPath $legacy)) {
        $ModelA = "models/combo_god_directpad_lock_combo12-safe_latest.pts"
    }
}
$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $cmd) { throw "Python introuvable." }
    $python = $cmd.Source
}
$argsList = @(
    "--model-a", $ModelA,
    "--model-b", $ModelB,
    "--matches", $Matches,
    "--min-combo", $MinCombo,
    "--min-combo-b", $MinComboB,
    "--max-draws", $MaxDraws,
    "--min-wins-a", $MinWinsA,
    "--min-wins-b", $MinWinsB,
    "--min-hits-a", $MinHitsA,
    "--min-hits-b", $MinHitsB,
    "--min-close-frac", $MinCloseFrac,
    "--max-sky-frac-a", $MaxSkyFracA,
    "--max-sky-frac-b", 1.0,
    "--max-back-frac-a", $MaxBackFracA,
    "--max-back-frac-b", 1.0,
    "--max-tap-back-frac-a", $MaxTapBackFracA,
    "--max-tap-back-frac-b", 1.0,
    "--min-strafe-frac-a", $MinStrafeFracA,
    "--min-strafe-frac-b", 0.0,
    "--min-opener-strafe-frac-a", $MinOpenerStrafeFracA,
    "--min-opener-strafe-frac-b", 0.0,
    "--min-opener-strafe-hold-frac-a", $MinOpenerStrafeHoldFracA,
    "--min-opener-strafe-hold-frac-b", 0.0,
    "--min-opener-pressure-frac-a", $MinOpenerPressureFracA,
    "--min-opener-pressure-frac-b", 0.0,
    "--max-strafe-frac-a", $MaxStrafeFracA,
    "--max-strafe-frac-b", 1.0,
    "--max-jump-frac-a", $MaxJumpFracA,
    "--max-jump-frac-b", 1.0,
    "--close-dist", $CloseDist,
    "--sky-pitch-deg", $SkyPitchDeg,
    "--cps", $Cps,
    "--max-steps", $MaxSteps
)
if ($Events) { $argsList += "--events" }
Write-Output "---ROLE A PROOF---"
& $python (Join-Path $repo "tools\arena_combo_check.py") @argsList
$code = $LASTEXITCODE
if ($code -ne 0 -or $NoRoleProof) { exit $code }

$roleBArgs = @(
    "--model-a", "__combo_spar__",
    "--model-b", $ModelA,
    "--matches", $Matches,
    "--min-combo", 0,
    "--min-combo-b", $MinCombo,
    "--max-draws", $MaxDraws,
    "--min-wins-a", 0,
    "--min-wins-b", 1,
    "--min-hits-a", 0,
    "--min-hits-b", $MinHitsA,
    "--min-close-frac", $MinCloseFrac,
    "--max-sky-frac-a", 1.0,
    "--max-sky-frac-b", $MaxSkyFracB,
    "--max-back-frac-a", 1.0,
    "--max-back-frac-b", $MaxBackFracB,
    "--max-tap-back-frac-a", 1.0,
    "--max-tap-back-frac-b", $MaxTapBackFracB,
    "--min-strafe-frac-a", 0.0,
    "--min-strafe-frac-b", $MinStrafeFracB,
    "--min-opener-strafe-frac-a", 0.0,
    "--min-opener-strafe-frac-b", $MinOpenerStrafeFracB,
    "--min-opener-strafe-hold-frac-a", 0.0,
    "--min-opener-strafe-hold-frac-b", $MinOpenerStrafeHoldFracB,
    "--min-opener-pressure-frac-a", 0.0,
    "--min-opener-pressure-frac-b", $MinOpenerPressureFracB,
    "--max-strafe-frac-a", 1.0,
    "--max-strafe-frac-b", $MaxStrafeFracB,
    "--max-jump-frac-a", 1.0,
    "--max-jump-frac-b", $MaxJumpFracB,
    "--close-dist", $CloseDist,
    "--sky-pitch-deg", $SkyPitchDeg,
    "--cps", $Cps,
    "--max-steps", $MaxSteps
)
if ($Events) { $roleBArgs += "--events" }
Write-Output "---ROLE B PROOF---"
& $python (Join-Path $repo "tools\arena_combo_check.py") @roleBArgs
$code = $LASTEXITCODE
if ($code -ne 0 -or $NoMirrorProof) { exit $code }

$mirrorMinCloseFrac = [Math]::Min($MinCloseFrac, 0.03)
$mirrorArgs = @(
    "--model-a", $ModelA,
    "--model-b", $ModelA,
    "--matches", $Matches,
    "--min-combo", 1,
    "--min-combo-b", 1,
    "--max-draws", $MirrorMaxDraws,
    "--min-wins-a", 0,
    "--min-wins-b", 0,
    "--min-hits-a", $MinHitsA,
    "--min-hits-b", $MinHitsA,
    "--min-close-frac", $mirrorMinCloseFrac,
    "--max-sky-frac-a", $MaxSkyFracA,
    "--max-sky-frac-b", $MaxSkyFracB,
    "--max-back-frac-a", $MaxBackFracA,
    "--max-back-frac-b", $MaxBackFracB,
    "--max-tap-back-frac-a", $MaxTapBackFracA,
    "--max-tap-back-frac-b", $MaxTapBackFracB,
    "--min-strafe-frac-a", $MinStrafeFracA,
    "--min-strafe-frac-b", $MinStrafeFracB,
    "--min-opener-strafe-frac-a", $MinOpenerStrafeFracA,
    "--min-opener-strafe-frac-b", $MinOpenerStrafeFracB,
    "--min-opener-strafe-hold-frac-a", $MinOpenerStrafeHoldFracA,
    "--min-opener-strafe-hold-frac-b", $MinOpenerStrafeHoldFracB,
    "--min-opener-pressure-frac-a", $MinOpenerPressureFracA,
    "--min-opener-pressure-frac-b", $MinOpenerPressureFracB,
    "--max-strafe-frac-a", $MaxStrafeFracA,
    "--max-strafe-frac-b", $MaxStrafeFracB,
    "--max-jump-frac-a", $MaxJumpFracA,
    "--max-jump-frac-b", $MaxJumpFracB,
    "--close-dist", $CloseDist,
    "--sky-pitch-deg", $SkyPitchDeg,
    "--cps", $Cps,
    "--max-steps", $MaxSteps
)
if ($Events) { $mirrorArgs += "--events" }
Write-Output "---MIRROR PROOF---"
& $python (Join-Path $repo "tools\arena_combo_check.py") @mirrorArgs
exit $LASTEXITCODE
