<#
.SYNOPSIS
  Judas Live launcher for the Forge 1.8.9 POV harness.

.DESCRIPTION
  1. Exports the safe combo checkpoint to TorchScript unless -NoExport is set.
  2. Starts the serve daemon when it is not already listening.
  3. Loads the combo-safe model and applies live parameters.
  4. Optionally launches the Prism/MultiMC instance.

.EXAMPLE
  ./scripts/judas_live.ps1 -Server 192.168.1.50 -Port 25565
  ./scripts/judas_live.ps1 -Checkpoint runs/combo_god_recovery_kb092_combo12/safe_latest.pt -NoLaunch
#>
[CmdletBinding()]
param(
    [string]$Checkpoint = "runs/combo_god_recovery_kb092_combo12/safe_latest.pt",
    [string]$Out        = "models/combo_god_recovery_kb092_combo12-safe_latest.pts",
    [string]$Server     = "",
    [int]   $Port       = 25565,
    [string]$Instance   = "JudasLive",
    [double]$MaxCps      = 10.0,
    [double]$MaxRotSpeed = 195.0,
    [double]$OriginX = 0.0,
    [double]$OriginZ = 0.0,
    [double]$SizeX   = 40.0,
    [double]$SizeZ   = 40.0,
    [double]$FloorY  = 0.0,
    [string]$DaemonHost = "127.0.0.1",
    [int]   $DaemonPort = 8765,
    [string]$PidFile = "",
    [string]$ActionLog = "",
    [switch]$NoExport,
    [switch]$AllowStaleExport,
    [switch]$ForceDaemon,
    [switch]$NoLaunch
)

$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
$base = "http://${DaemonHost}:${DaemonPort}"
$ActionLogExplicit = $PSBoundParameters.ContainsKey("ActionLog")
if (-not $PSBoundParameters.ContainsKey("Checkpoint")) {
    $preferred = Join-Path $repo "runs\combo_god_recovery_kb092_combo12\safe_latest.pt"
    $leaderboard = Join-Path $repo "runs\combo_god_leaderboard10_combo12\safe_latest.pt"
    $counter = Join-Path $repo "runs\combo_god_countertap96_combo12\safe_latest.pt"
    $legacy = Join-Path $repo "runs\combo_god_directpad_lock_combo12\safe_latest.pt"
    if (-not (Test-Path -LiteralPath $preferred) -and (Test-Path -LiteralPath $leaderboard)) {
        $Checkpoint = "runs/combo_god_leaderboard10_combo12/safe_latest.pt"
        if (-not $PSBoundParameters.ContainsKey("Out")) {
            $Out = "models/combo_god_leaderboard10_combo12-safe_latest.pts"
        }
    } elseif (-not (Test-Path -LiteralPath $preferred) -and (Test-Path -LiteralPath $counter)) {
        $Checkpoint = "runs/combo_god_countertap96_combo12/safe_latest.pt"
        if (-not $PSBoundParameters.ContainsKey("Out")) {
            $Out = "models/combo_god_countertap96_combo12-safe_latest.pts"
        }
    } elseif (-not (Test-Path -LiteralPath $preferred) -and (Test-Path -LiteralPath $legacy)) {
        $Checkpoint = "runs/combo_god_directpad_lock_combo12/safe_latest.pt"
        if (-not $PSBoundParameters.ContainsKey("Out")) {
            $Out = "models/combo_god_directpad_lock_combo12-safe_latest.pts"
        }
    }
}
if (-not $PidFile) {
    $PidFile = Join-Path $repo "runs\judas_live_daemon.pid"
}
if (-not $ActionLog) {
    $ActionLog = Join-Path $repo "runs\judas-live-actions.log"
}

function Prepend-EnvPath([string]$Name, [string]$Value) {
    if (-not $Value -or -not (Test-Path -LiteralPath $Value)) { return }
    $current = [string](Get-Item -Path "Env:$Name" -ErrorAction SilentlyContinue).Value
    $parts = @($current -split [System.IO.Path]::PathSeparator | Where-Object { $_ })
    if ($parts -notcontains $Value) {
        Set-Item -Path "Env:$Name" -Value ($Value + [System.IO.Path]::PathSeparator + $current)
    }
}

function Resolve-Python {
    $venvRoot = Join-Path $repo ".venv"
    $venv = Join-Path $venvRoot "Scripts\python.exe"
    if (Test-Path -LiteralPath $venv) {
        $sitePackages = Join-Path $venvRoot "Lib\site-packages"
        Prepend-EnvPath "PYTHONPATH" $sitePackages
        Prepend-EnvPath "PATH" (Join-Path $venvRoot "Scripts")
        $env:VIRTUAL_ENV = $venvRoot
        try {
            $base = (& $venv -c "import sys; print(getattr(sys, '_base_executable', sys.executable))" 2>$null | Select-Object -First 1).Trim()
            if ($base -and (Test-Path -LiteralPath $base)) {
                return $base
            }
        } catch {
            return $venv
        }
        return $venv
    }
    Write-Warning ".venv not found; using python from PATH."
    return "python"
}
$py = Resolve-Python

function Test-Daemon {
    try {
        Invoke-RestMethod -Uri "$base/status" -TimeoutSec 2 -Method Get | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Stop-ProcessTree([int]$ProcId) {
    $proc = Get-Process -Id $ProcId -ErrorAction SilentlyContinue
    if (-not $proc) { return $false }
    & taskkill.exe /PID $ProcId /T /F | Out-Null
    Start-Sleep -Milliseconds 500
    return $true
}

function Read-PidFile([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    $text = Get-Content -LiteralPath $Path -Raw -ErrorAction SilentlyContinue
    if ($text -match "^\s*(\d+)\b") { return [int]$Matches[1] }
    return $null
}

function Get-OwnedDaemonPids {
    $repoNeedle = [System.IO.Path]::GetFullPath($repo)
    $connections = Get-NetTCPConnection -LocalPort $DaemonPort -State Listen -ErrorAction SilentlyContinue
    foreach ($conn in $connections) {
        $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$($conn.OwningProcess)" -ErrorAction SilentlyContinue
        $seen = @{}
        while ($proc -and -not $seen.ContainsKey([int]$proc.ProcessId)) {
            $seen[[int]$proc.ProcessId] = $true
            $cmd = [string]$proc.CommandLine
            $exe = [string]$proc.ExecutablePath
            if (($cmd -like "*serve.daemon*" -or $exe -like "*python*") -and ($cmd.Contains($repoNeedle) -or $exe.Contains($repoNeedle))) {
                [int]$proc.ProcessId
                break
            }
            if (-not $proc.ParentProcessId) { break }
            $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$($proc.ParentProcessId)" -ErrorAction SilentlyContinue
        }
    }
}

function Resolve-RepoPath([string]$PathValue) {
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return [System.IO.Path]::GetFullPath($PathValue)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $repo $PathValue))
}

function Has-Field($Obj, [string]$Name) {
    return $Obj.PSObject.Properties.Name -contains $Name
}

function Get-RequiredNumber($Obj, [string]$Name) {
    if (-not (Has-Field $Obj $Name)) {
        throw "Combo safe metadata missing field: $Name"
    }
    try {
        return [double]$Obj.$Name
    } catch {
        throw "Combo safe metadata field is not numeric: $Name=$($Obj.$Name)"
    }
}

function Assert-MinMetric($Obj, [string]$Metric, [string]$Threshold) {
    $value = Get-RequiredNumber $Obj $Metric
    $min = Get-RequiredNumber $Obj $Threshold
    if ($min -ge 0.0 -and $value -lt $min) {
        throw "Combo safe metadata failed: $Metric=$value < $Threshold=$min"
    }
}

function Assert-MaxMetric($Obj, [string]$Metric, [string]$Threshold) {
    $value = Get-RequiredNumber $Obj $Metric
    $max = Get-RequiredNumber $Obj $Threshold
    if ($max -ge 0.0 -and $value -gt $max) {
        throw "Combo safe metadata failed: $Metric=$value > $Threshold=$max"
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
        throw "Combo safe metadata failed: under_combo_counter_hit_frac=$value < safety_min_under_combo_counter_hit_frac=$min"
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

function Assert-SafeCheckpointContract([string]$CheckpointPath) {
    $checkpointAbs = Resolve-RepoPath $CheckpointPath
    $normalized = $checkpointAbs.Replace("/", "\")
    $comboSafe = $false
    foreach ($runName in @("combo_god_recovery_kb092_combo12", "combo_god_leaderboard10_combo12", "combo_god_countertap96_combo12", "combo_god_directpad_lock_combo12")) {
        if ($normalized.EndsWith("\runs\$runName\safe_latest.pt")) {
            $comboSafe = $true
            break
        }
    }
    if (-not $comboSafe) { return }
    if (-not (Test-Path -LiteralPath $checkpointAbs)) {
        throw "Missing checkpoint: $CheckpointPath"
    }
    $safeMetaPath = Join-Path (Split-Path $checkpointAbs -Parent) "safe_latest.meta.json"
    if (-not (Test-Path -LiteralPath $safeMetaPath)) {
        throw "Missing combo safe metadata: $safeMetaPath"
    }
    $safeMeta = Get-Content -LiteralPath $safeMetaPath -Raw | ConvertFrom-Json
    if (-not (Has-Field $safeMeta "score_schema")) {
        throw "Combo safe metadata missing field: score_schema"
    }
    $schema = [int]$safeMeta.score_schema
    if ($schema -lt 8) {
        throw "Combo safe metadata too old: score_schema=$schema < 8"
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
    Write-Host "[judas-live] Combo safe metadata OK: $safeMetaPath" -ForegroundColor DarkGray
}

function Assert-FreshExport([string]$CheckpointPath, [string]$ExportPath) {
    $exportAbs = Resolve-RepoPath $ExportPath
    if (-not (Test-Path -LiteralPath $exportAbs)) {
        throw "Missing exported model: $ExportPath. Run without -NoExport first."
    }
    $checkpointAbs = Resolve-RepoPath $CheckpointPath
    if (-not (Test-Path -LiteralPath $checkpointAbs)) {
        throw "Missing checkpoint: $CheckpointPath"
    }
    $exportInfo = Get-Item -LiteralPath $exportAbs
    $checkpointInfo = Get-Item -LiteralPath $checkpointAbs
    if ($exportInfo.LastWriteTime -lt $checkpointInfo.LastWriteTime) {
        throw ("Stale export: {0} is older than {1}. Run without -NoExport or pass -AllowStaleExport." -f `
            $ExportPath, $CheckpointPath)
    }
    $metaPath = [System.IO.Path]::ChangeExtension($exportAbs, ".json")
    if (-not (Test-Path -LiteralPath $metaPath)) {
        throw "Missing export metadata: $metaPath. Run without -NoExport first."
    }
    $meta = Get-Content -LiteralPath $metaPath -Raw | ConvertFrom-Json
    if ($meta.source) {
        $sourceAbs = Resolve-RepoPath ([string]$meta.source)
        if ($sourceAbs -ne $checkpointAbs) {
            throw ("Export source mismatch: metadata source={0}, expected={1}. Run without -NoExport." -f `
                $meta.source, $CheckpointPath)
        }
    }
    $metaFields = $meta.PSObject.Properties.Name
    if ($metaFields -contains "source_size") {
        $expectedSize = [int64]$meta.source_size
        if ($checkpointInfo.Length -ne $expectedSize) {
            throw ("Export checkpoint size mismatch: metadata={0}, actual={1}. Run without -NoExport." -f `
                $expectedSize, $checkpointInfo.Length)
        }
    }
    if ($metaFields -contains "source_sha256") {
        $actualHash = (Get-FileHash -LiteralPath $checkpointAbs -Algorithm SHA256).Hash.ToLowerInvariant()
        $expectedHash = ([string]$meta.source_sha256).ToLowerInvariant()
        if ($actualHash -ne $expectedHash) {
            throw ("Export checkpoint hash mismatch: metadata={0}, actual={1}. Run without -NoExport." -f `
            $expectedHash, $actualHash)
        }
    }
    Assert-SafeCheckpointContract $CheckpointPath
    Write-Host "[judas-live] Export freshness OK: $ExportPath" -ForegroundColor DarkGray
}

if (-not $NoExport) {
    Assert-SafeCheckpointContract $Checkpoint
    Write-Host "[judas-live] Export $Checkpoint -> $Out" -ForegroundColor Cyan
    & $py -m train.export $Checkpoint --out $Out
    if ($LASTEXITCODE -ne 0) { throw "train.export failed." }
} else {
    Write-Host "[judas-live] Export skipped (-NoExport); using $Out." -ForegroundColor DarkGray
    if ($AllowStaleExport) {
        Write-Warning "AllowStaleExport enabled; skipping checkpoint/export freshness check."
        Assert-SafeCheckpointContract $Checkpoint
    } else {
        Assert-FreshExport $Checkpoint $Out
    }
}

if ($ForceDaemon) {
    $oldPid = Read-PidFile $PidFile
    if ($null -ne $oldPid) {
        if (Stop-ProcessTree $oldPid) {
            Write-Host "[judas-live] Stopped old daemon pid=$oldPid" -ForegroundColor DarkGray
        }
    }
    Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
    $owned = @(Get-OwnedDaemonPids | Sort-Object -Unique)
    foreach ($ownedPid in $owned) {
        if (Stop-ProcessTree $ownedPid) {
            Write-Host "[judas-live] Stopped old repo daemon pid=$ownedPid" -ForegroundColor DarkGray
        }
    }
}

if (Test-Daemon) {
    if ($ActionLogExplicit) {
        throw "Daemon already active on $base; cannot attach JUDAS_LIVE_ACTION_LOG after start."
    }
    Write-Warning "Daemon already active on $base; action log remains whatever the daemon was started with."
    Write-Host "[judas-live] Daemon already active on $base" -ForegroundColor DarkGray
} else {
    Write-Host "[judas-live] Starting daemon ($base)..." -ForegroundColor Cyan
    if ($ActionLog) {
        $actionLogDir = Split-Path $ActionLog -Parent
        if ($actionLogDir -and -not (Test-Path -LiteralPath $actionLogDir)) {
            New-Item -ItemType Directory -Path $actionLogDir -Force | Out-Null
        }
        $env:JUDAS_LIVE_ACTION_LOG = $ActionLog
    }
    $daemonProc = Start-Process -FilePath $py `
        -ArgumentList @("-m", "serve.daemon", "--host", $DaemonHost, "--port", "$DaemonPort") `
        -WorkingDirectory $repo -WindowStyle Hidden -PassThru
    $pidDir = Split-Path $PidFile -Parent
    if ($pidDir -and -not (Test-Path -LiteralPath $pidDir)) {
        New-Item -ItemType Directory -Path $pidDir -Force | Out-Null
    }
    Set-Content -LiteralPath $PidFile -Value $daemonProc.Id -Encoding ascii
    Write-Host "[judas-live] Daemon pid=$($daemonProc.Id) pid_file=$PidFile" -ForegroundColor DarkGray
    $deadline = (Get-Date).AddSeconds(30)
    while (-not (Test-Daemon)) {
        if ((Get-Date) -gt $deadline) { throw "Daemon did not answer within 30 seconds." }
        Start-Sleep -Milliseconds 500
    }
    Write-Host "[judas-live] Daemon ready." -ForegroundColor Green
}

Write-Host "[judas-live] Loading model: $Out" -ForegroundColor Cyan
$loadBody = @{ model = $Out } | ConvertTo-Json -Compress
$load = Invoke-RestMethod -Uri "$base/live/load" -Method Post -Body $loadBody -ContentType "application/json"
Write-Host ("[judas-live] Model loaded (history={0})." -f $load.history) -ForegroundColor Green

$paramsBody = @{
    max_cps       = $MaxCps
    max_rot_speed = $MaxRotSpeed
    enabled       = $true
    arena = @{
        origin_x = $OriginX
        origin_z = $OriginZ
        size_x   = $SizeX
        size_z   = $SizeZ
        floor_y  = $FloorY
    }
} | ConvertTo-Json -Compress
Invoke-RestMethod -Uri "$base/live/params" -Method Post -Body $paramsBody -ContentType "application/json" | Out-Null
Write-Host ("[judas-live] Params: cps={0} rot={1} arena=({2},{3}) {4}x{5} floor={6}" -f `
    $MaxCps, $MaxRotSpeed, $OriginX, $OriginZ, $SizeX, $SizeZ, $FloorY) -ForegroundColor Green

if ($NoLaunch) {
    Write-Host "[judas-live] -NoLaunch: start the $Instance instance manually." -ForegroundColor DarkGray
} else {
    $prism = Get-Command prismlauncher -ErrorAction SilentlyContinue
    if (-not $prism) {
        $prism = Get-Command multimc -ErrorAction SilentlyContinue
    }
    if ($prism) {
        Write-Host "[judas-live] Launching $Instance via $($prism.Source)" -ForegroundColor Cyan
        & $prism.Source --launch $Instance
    } else {
        Write-Warning "Prism/MultiMC not found on PATH."
        Write-Host "  -> Launch your Forge 1.8.9 + OptiFine + judas-bridge.jar instance manually." -ForegroundColor Yellow
    }
}

if ($Server) {
    Write-Host ""
    Write-Host "[judas-live] Join server: $Server`:$Port" -ForegroundColor Cyan
}
Write-Host ""
Write-Host "[judas-live] In game: enter a 1v1 box, K toggles the bot, L is kill-switch." -ForegroundColor Green
Write-Host "             Use O for OS mouse mode before field-testing aim." -ForegroundColor DarkGray
