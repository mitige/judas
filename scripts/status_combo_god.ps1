param(
  [string]$Run = "combo_god_recovery_kb092_combo12",
  [int]$Tail = 3
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runDir = Join-Path $root "runs\$Run"
$metrics = Join-Path $runDir "metrics.jsonl"
$pidFile = Join-Path $runDir "train.pid"

function Read-PidFile([string]$Path) {
  if (-not (Test-Path -LiteralPath $Path)) { return $null }
  $text = Get-Content -LiteralPath $Path -Raw -ErrorAction SilentlyContinue
  if ($text -match "^\s*(\d+)\b") { return [int]$Matches[1] }
  return $null
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

Write-Output "run=$Run"
if (Test-Path -LiteralPath $pidFile) {
  $pidValue = Read-PidFile $pidFile
  if ($null -ne $pidValue) {
    $proc = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
    if ($proc) {
      Write-Output "process=running pid=$pidValue cpu=$([math]::Round($proc.CPU, 2)) start=$($proc.StartTime)"
    } else {
      Write-Output "process=exited pid=$pidValue"
      Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
    }
  } else {
    Write-Output "process=invalid pid_file=$pidFile"
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
  }
} else {
  Write-Output "process=no pid file"
}

$orphans = @(Get-OwnedTrainingPids | Sort-Object -Unique)
if ($orphans.Count -gt 0) {
  Write-Output ("orphan_training=running pids=" + ($orphans -join ","))
} else {
  Write-Output "orphan_training=none"
}

if (Test-Path -LiteralPath $metrics) {
  $item = Get-Item -LiteralPath $metrics
  Write-Output "metrics=$metrics updated=$($item.LastWriteTime) bytes=$($item.Length)"
  Write-Output "---METRICS TAIL---"
  Get-Content -LiteralPath $metrics -Tail $Tail
} else {
  Write-Output "metrics=missing $metrics"
}
