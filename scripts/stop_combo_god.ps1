param(
  [string]$Run = "combo_god_recovery_kb092_combo12"
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runDir = Join-Path $root "runs\$Run"
$pidFile = Join-Path $runDir "train.pid"

Write-Output "run=$Run"

function Read-PidFile([string]$Path) {
  if (-not (Test-Path -LiteralPath $Path)) { return $null }
  $text = Get-Content -LiteralPath $Path -Raw -ErrorAction SilentlyContinue
  if ($text -match "^\s*(\d+)\b") { return [int]$Matches[1] }
  return $null
}

function Stop-ProcessTree([int]$ProcId) {
  $proc = Get-Process -Id $ProcId -ErrorAction SilentlyContinue
  if (-not $proc) { return $false }
  & taskkill.exe /PID $ProcId /T /F | Out-Null
  Start-Sleep -Milliseconds 750
  return $true
}

function Get-AncestorPids {
  $seen = @{}
  $cur = Get-CimInstance Win32_Process -Filter "ProcessId=$PID" -ErrorAction SilentlyContinue
  while ($cur -and $cur.ParentProcessId -and -not $seen.ContainsKey([int]$cur.ParentProcessId)) {
    $parentId = [int]$cur.ParentProcessId
    $seen[$parentId] = $true
    $cur = Get-CimInstance Win32_Process -Filter "ProcessId=$parentId" -ErrorAction SilentlyContinue
  }
  return @($seen.Keys)
}

function Test-ComboTrainingProcess($ProcInfo) {
  $cmd = [string]$ProcInfo.CommandLine
  $exe = [string]$ProcInfo.ExecutablePath
  if (-not $cmd) { return $false }
  $rootNeedle = [System.IO.Path]::GetFullPath($root)
  $inRepo = $cmd.Contains($rootNeedle) -or ($exe -and $exe.Contains($rootNeedle))
  $comboNeedles = @(
    "train/configs/combo_god_recovery_kb092_combo12.json",
    "train\configs\combo_god_recovery_kb092_combo12.json",
    "train/configs/combo_god_leaderboard10_combo12.json",
    "train\configs\combo_god_leaderboard10_combo12.json",
    "train/configs/combo_god_attn96_combo12.json",
    "train\configs\combo_god_attn96_combo12.json",
    "combo_god_recovery_kb092_combo12",
    "combo_god_leaderboard10_combo12",
    "combo_god_countertap96_combo12",
    "combo_god_directpad_lock_combo12",
    "scripts\train_combo_god.bat",
    "scripts/train_combo_god.bat"
  )
  foreach ($needle in $comboNeedles) {
    if ($cmd.Contains($needle)) { return $true }
  }
  return ($inRepo -and $cmd.Contains("-m train.run"))
}

function Get-OwnedTrainingPids {
  $excluded = @{}
  $excluded[[int]$PID] = $true
  foreach ($ancestor in Get-AncestorPids) { $excluded[[int]$ancestor] = $true }
  Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object {
      $_.Name -match '^(python|pythonw|cmd|powershell|pwsh)\.exe$' -and
      -not $excluded.ContainsKey([int]$_.ProcessId) -and
      (Test-ComboTrainingProcess $_)
    } |
    Select-Object -ExpandProperty ProcessId
}

function Stop-OrphanTraining {
  $orphans = @(Get-OwnedTrainingPids | Sort-Object -Unique)
  foreach ($orphanPid in $orphans) {
    if (Stop-ProcessTree $orphanPid) {
      Write-Output "orphan_training=stopped pid=$orphanPid"
    }
  }
  if ($orphans.Count -eq 0) {
    Write-Output "orphan_training=none"
  }
}

if (-not (Test-Path -LiteralPath $pidFile)) {
  Write-Output "process=no pid file"
  Stop-OrphanTraining
  exit 0
}

$procId = Read-PidFile $pidFile
if ($null -eq $procId) {
  Write-Output "process=invalid pid_file=$pidFile"
  Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
  Stop-OrphanTraining
  exit 0
}

$proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
if (-not $proc) {
  Write-Output "process=exited pid=$procId"
  Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
  Stop-OrphanTraining
  exit 0
}

$stopStart = $proc.StartTime
try {
  Stop-ProcessTree $procId | Out-Null
} catch {
  Write-Output "taskkill=failed $($_.Exception.Message)"
}
$still = Get-Process -Id $procId -ErrorAction SilentlyContinue
if ($still) {
  try {
    Stop-Process -Id $procId -Force -ErrorAction Stop
    Write-Output "fallback=stopped_wrapper pid=$procId"
  } catch {
    Write-Output "fallback=wrapper_failed pid=$procId error=$($_.Exception.Message)"
  }
  $nearStart = Get-Process -Name python,pythonw -ErrorAction SilentlyContinue | Where-Object {
    $_.StartTime -ge $stopStart.AddSeconds(-5) -and $_.StartTime -le $stopStart.AddSeconds(15)
  }
  foreach ($child in $nearStart) {
    try {
      Stop-Process -Id $child.Id -Force -ErrorAction Stop
      Write-Output "fallback=stopped_python pid=$($child.Id)"
    } catch {
      Write-Output "fallback=python_failed pid=$($child.Id) error=$($_.Exception.Message)"
    }
  }
  Start-Sleep -Milliseconds 750
}

$still = Get-Process -Id $procId -ErrorAction SilentlyContinue
$pyStill = Get-Process -Name python,pythonw -ErrorAction SilentlyContinue | Where-Object {
  $_.StartTime -ge $stopStart.AddSeconds(-5) -and $_.StartTime -le $stopStart.AddSeconds(15)
}
if ($still -or $pyStill) {
  Write-Output "process=still_running pid=$procId"
  exit 1
}

Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
Write-Output "process=stopped pid=$procId"
Stop-OrphanTraining
