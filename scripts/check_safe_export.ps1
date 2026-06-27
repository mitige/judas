<#
.SYNOPSIS
  Passive freshness check for the combo-safe TorchScript export.
#>
[CmdletBinding()]
param(
    [string]$Checkpoint = "runs/combo_god_recovery_kb092_combo12/safe_latest.pt",
    [string]$Export = "models/combo_god_recovery_kb092_combo12-safe_latest.pts"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
if (-not $PSBoundParameters.ContainsKey("Checkpoint")) {
    $preferred = Join-Path $repo "runs\combo_god_recovery_kb092_combo12\safe_latest.pt"
    $leaderboard = Join-Path $repo "runs\combo_god_leaderboard10_combo12\safe_latest.pt"
    $counter = Join-Path $repo "runs\combo_god_countertap96_combo12\safe_latest.pt"
    $legacy = Join-Path $repo "runs\combo_god_directpad_lock_combo12\safe_latest.pt"
    if (-not (Test-Path -LiteralPath $preferred) -and (Test-Path -LiteralPath $leaderboard)) {
        $Checkpoint = "runs/combo_god_leaderboard10_combo12/safe_latest.pt"
        if (-not $PSBoundParameters.ContainsKey("Export")) {
            $Export = "models/combo_god_leaderboard10_combo12-safe_latest.pts"
        }
    } elseif (-not (Test-Path -LiteralPath $preferred) -and (Test-Path -LiteralPath $counter)) {
        $Checkpoint = "runs/combo_god_countertap96_combo12/safe_latest.pt"
        if (-not $PSBoundParameters.ContainsKey("Export")) {
            $Export = "models/combo_god_countertap96_combo12-safe_latest.pts"
        }
    } elseif (-not (Test-Path -LiteralPath $preferred) -and (Test-Path -LiteralPath $legacy)) {
        $Checkpoint = "runs/combo_god_directpad_lock_combo12/safe_latest.pt"
        if (-not $PSBoundParameters.ContainsKey("Export")) {
            $Export = "models/combo_god_directpad_lock_combo12-safe_latest.pts"
        }
    }
}

function Resolve-RepoPath([string]$PathValue) {
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return [System.IO.Path]::GetFullPath($PathValue)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $repo $PathValue))
}

function Fail([string]$Code, [string]$Message) {
    Write-Output "$Code $Message"
    exit 1
}

function Has-Field($Obj, [string]$Name) {
    return $Obj.PSObject.Properties.Name -contains $Name
}

function Get-RequiredNumber($Obj, [string]$Name) {
    if (-not (Has-Field $Obj $Name)) {
        Fail "SAFE_META_MISSING_FIELD" "field=$Name"
    }
    try {
        return [double]$Obj.$Name
    } catch {
        Fail "SAFE_META_BAD_FIELD" "field=$Name value=$($Obj.$Name)"
    }
}

function Assert-MinMetric($Obj, [string]$Metric, [string]$Threshold) {
    $value = Get-RequiredNumber $Obj $Metric
    $min = Get-RequiredNumber $Obj $Threshold
    if ($min -ge 0.0 -and $value -lt $min) {
        Fail "SAFE_META_LOW_METRIC" "$Metric=$value min=$min"
    }
}

function Assert-MaxMetric($Obj, [string]$Metric, [string]$Threshold) {
    $value = Get-RequiredNumber $Obj $Metric
    $max = Get-RequiredNumber $Obj $Threshold
    if ($max -ge 0.0 -and $value -gt $max) {
        Fail "SAFE_META_HIGH_METRIC" "$Metric=$value max=$max"
    }
}

function Assert-CounterMetric($Obj) {
    $value = Get-RequiredNumber $Obj "under_combo_counter_hit_frac"
    $min = Get-RequiredNumber $Obj "safety_min_under_combo_counter_hit_frac"
    $avoidanceBonus = 0.0
    if (Has-Field $Obj "under_combo_avoidance_score_bonus") {
        try { $avoidanceBonus = [double]$Obj.under_combo_avoidance_score_bonus } catch { $avoidanceBonus = 0.0 }
    }
    if ($min -ge 0.0 -and $value -lt $min -and $avoidanceBonus -le 0.0) {
        Fail "SAFE_META_LOW_METRIC" "under_combo_counter_hit_frac=$value min=$min"
    }
}

function Should-CheckCounterRecovery($Obj) {
    if (-not (Has-Field $Obj "requires_counter_recovery")) { return $false }
    try { return [System.Convert]::ToBoolean($Obj.requires_counter_recovery) } catch { return $false }
}

function Should-CheckOpenerMetrics($Obj) {
    if (Has-Field $Obj "opener_samples") {
        try { return ([double]$Obj.opener_samples) -gt 0.0 } catch { return $true }
    }
    foreach ($metric in @("opener_strafe_frac", "opener_strafe_hold_frac", "opener_pressure_frac")) {
        if (Has-Field $Obj $metric) {
            try {
                if ([double]$Obj.$metric -gt 0.0) { return $true }
            } catch {
                return $true
            }
        }
    }
    return $false
}

$checkpointAbs = Resolve-RepoPath $Checkpoint
$exportAbs = Resolve-RepoPath $Export
if (-not (Test-Path -LiteralPath $checkpointAbs)) {
    Fail "MISSING_CHECKPOINT" "checkpoint=$Checkpoint"
}
if (-not (Test-Path -LiteralPath $exportAbs)) {
    Fail "MISSING_EXPORT" "export=$Export"
}

$checkpointInfo = Get-Item -LiteralPath $checkpointAbs
$exportInfo = Get-Item -LiteralPath $exportAbs
if ($exportInfo.LastWriteTime -lt $checkpointInfo.LastWriteTime) {
    Fail "STALE_EXPORT" "export=$Export exportTime=$($exportInfo.LastWriteTime.ToString('o')) checkpoint=$Checkpoint checkpointTime=$($checkpointInfo.LastWriteTime.ToString('o'))"
}

$metaPath = [System.IO.Path]::ChangeExtension($exportAbs, ".json")
if (-not (Test-Path -LiteralPath $metaPath)) {
    Fail "MISSING_EXPORT_META" "meta=$metaPath"
}

$meta = Get-Content -LiteralPath $metaPath -Raw | ConvertFrom-Json
if ($meta.source) {
    $sourceAbs = Resolve-RepoPath ([string]$meta.source)
    if ($sourceAbs -ne $checkpointAbs) {
        Fail "EXPORT_SOURCE_MISMATCH" "source=$($meta.source) expected=$Checkpoint"
    }
}

$fields = $meta.PSObject.Properties.Name
if ($fields -contains "source_size") {
    $expectedSize = [int64]$meta.source_size
    if ($checkpointInfo.Length -ne $expectedSize) {
        Fail "EXPORT_SIZE_MISMATCH" "metadata=$expectedSize actual=$($checkpointInfo.Length)"
    }
}
if ($fields -contains "source_sha256") {
    $actualHash = (Get-FileHash -LiteralPath $checkpointAbs -Algorithm SHA256).Hash.ToLowerInvariant()
    $expectedHash = ([string]$meta.source_sha256).ToLowerInvariant()
    if ($actualHash -ne $expectedHash) {
        Fail "EXPORT_HASH_MISMATCH" "metadata=$expectedHash actual=$actualHash"
    }
}

$safeMetaPath = Join-Path (Split-Path $checkpointAbs -Parent) "safe_latest.meta.json"
if (-not (Test-Path -LiteralPath $safeMetaPath)) {
    Fail "MISSING_SAFE_META" "meta=$safeMetaPath"
}

$safeMeta = Get-Content -LiteralPath $safeMetaPath -Raw | ConvertFrom-Json
$safeFields = $safeMeta.PSObject.Properties.Name
if (-not ($safeFields -contains "score_schema")) {
    Fail "SAFE_META_MISSING_FIELD" "field=score_schema"
}
$schema = [int]$safeMeta.score_schema
if ($schema -lt 8) {
    Fail "OLD_SAFE_SCHEMA" "score_schema=$schema min=8"
}

Assert-MaxMetric $safeMeta "back_frac" "safety_back_frac"
Assert-MinMetric $safeMeta "strafe_frac" "safety_min_strafe_frac"
if (Should-CheckOpenerMetrics $safeMeta) {
    Assert-MinMetric $safeMeta "opener_strafe_frac" "safety_min_opener_strafe_frac"
    Assert-MinMetric $safeMeta "opener_strafe_hold_frac" "safety_min_opener_strafe_hold_frac"
    Assert-MinMetric $safeMeta "opener_pressure_frac" "safety_min_opener_pressure_frac"
}
Assert-MinMetric $safeMeta "combo_tap_frac" "safety_min_combo_tap_frac"
Assert-MinMetric $safeMeta "combo_z_tap_frac" "safety_min_combo_z_tap_frac"
Assert-MaxMetric $safeMeta "combo_s_tap_frac" "safety_max_combo_s_tap_frac"
Assert-MinMetric $safeMeta "hit_wtap_frac" "safety_min_hit_wtap_frac"
Assert-CounterMetric $safeMeta
if (Should-CheckCounterRecovery $safeMeta) {
    Assert-MinMetric $safeMeta "under_combo_hit_select_clean_frac" "safety_min_under_combo_hit_select_clean_frac"
    Assert-MaxMetric $safeMeta "under_combo_hit_select_trade_frac" "safety_max_under_combo_hit_select_trade_frac"
}

Write-Output "OK_EXPORT checkpoint=$Checkpoint export=$Export meta=$metaPath safe_meta=$safeMetaPath"
exit 0
