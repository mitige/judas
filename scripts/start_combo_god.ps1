param(
  [string]$Run = "combo_god_recovery_kb092_combo12",
  [switch]$Force,
  [int]$Iters = 0,
  [int]$TimeoutMinutes = 20,
  [string]$Resume = "",
  [int]$Seed = -1,
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runDir = Join-Path $root "runs\$Run"
New-Item -ItemType Directory -Path $runDir -Force | Out-Null

$pidFile = Join-Path $runDir "train.pid"

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

if (Test-Path -LiteralPath $pidFile) {
  $oldPid = Read-PidFile $pidFile
  if ($null -eq $oldPid) {
    Write-Output "REMOVED_INVALID_PID_FILE path=$pidFile"
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
  } else {
    $oldProc = Get-Process -Id $oldPid -ErrorAction SilentlyContinue
    if ($oldProc) {
      if (-not $Force) {
        Write-Output "ALREADY_RUNNING pid=$oldPid run=$Run"
        Write-Output "pid_file=$pidFile"
        Write-Output "Use scripts\stop_combo_god.bat before restarting, or pass -Force."
        exit 0
      }
      $stopped = Stop-ProcessTree $oldPid
      if ($stopped) { Write-Output "STOPPED_OLD pid=$oldPid run=$Run" }
    }
  }
}
Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue

if ($Force) {
  $orphans = @(Get-OwnedTrainingPids | Sort-Object -Unique)
  foreach ($orphanPid in $orphans) {
    if (Stop-ProcessTree $orphanPid) {
      Write-Output "STOPPED_ORPHAN_TRAINING pid=$orphanPid run=$Run"
    }
  }
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$out = Join-Path $runDir "train_$stamp.out.log"
$err = Join-Path $runDir "train_$stamp.err.log"
$launch = Join-Path $runDir "launch_$stamp.ps1"
$launchLog = Join-Path $runDir "launch_$stamp.log"
$pythonExe = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
  throw "Missing venv python: $pythonExe. Run setup.bat first."
}

$cfg = "train/configs/$Run.json"
if (-not (Test-Path -LiteralPath (Join-Path $root $cfg))) {
  throw "Missing combo config: $cfg"
}
if ($Seed -lt -1) {
  throw "Seed must be >= 0, or -1 to use the config default."
}
if ($Seed -ge 0) {
  $seedCfg = Join-Path $runDir "config_seed_$stamp.json"
  $cfgObj = Get-Content -LiteralPath (Join-Path $root $cfg) -Raw | ConvertFrom-Json
  $cfgObj.seed = $Seed
  $cfgObj | ConvertTo-Json -Depth 64 | Set-Content -LiteralPath $seedCfg -Encoding UTF8
  $cfg = "runs/$Run/config_seed_$stamp.json"
}
$args = @("-m", "train.run", "--config", $cfg)
if ($Resume) {
  if (-not (Test-Path -LiteralPath (Join-Path $root $Resume))) {
    throw "Missing resume checkpoint: $Resume"
  }
  $args += @("--resume", $Resume)
} else {
  $resumeCandidates = @(
    "runs/$Run/safe_latest.pt",
    "runs/$Run/latest.pt",
    "runs/combo_god_leaderboard10_combo12/safe_latest.pt",
    "runs/combo_god_leaderboard10_combo12/latest.pt",
    "runs/combo_god_countertap96_combo12/safe_latest.pt",
    "runs/combo_god_countertap96_combo12/latest.pt",
    "runs/combo_god_directpad_lock_combo12/safe_latest.pt",
    "runs/combo_god_directpad_lock_combo12/latest.pt",
    "runs/$Run/seed_from_directpad_iter60.pt",
    "runs/combo_god_directpad_fast_combo12/ckpt_000060.pt",
    "runs/combo_god_directpad_fast_combo12/seed_from_gap8_iter15.pt",
    "runs/combo_god_mirror_gap8_combo12/latest.pt",
    "runs/combo_god_mirror96_combo12/ckpt_000015.pt",
    "runs/god/best.pt"
  )
  foreach ($candidate in $resumeCandidates) {
    if (Test-Path -LiteralPath (Join-Path $root $candidate)) {
      $args += @("--resume", $candidate)
      break
    }
  }
}
if ($Iters -gt 0) { $args += @("--iters", [string]$Iters) }
if ($ExtraArgs) { $args += $ExtraArgs }

if ($TimeoutMinutes -lt 0) {
  throw "TimeoutMinutes must be >= 0. Use 0 to disable the watchdog."
}
$timeoutSeconds = [int64]$TimeoutMinutes * 60

function Quote-PowerShellLiteral([string]$Value) {
  return "'" + $Value.Replace("'", "''") + "'"
}

$argsLiteral = "@(" + (($args | ForEach-Object { Quote-PowerShellLiteral $_ }) -join ", ") + ")"
$launchText = @(
  '$ErrorActionPreference = "Stop"',
  ('Set-Content -LiteralPath ' + (Quote-PowerShellLiteral $pidFile) + ' -Value $PID -Encoding ascii'),
  ('Set-Content -LiteralPath ' + (Quote-PowerShellLiteral $launchLog) + ' -Value ("wrapper_pid=" + $PID) -Encoding ascii'),
  ('Add-Content -LiteralPath ' + (Quote-PowerShellLiteral $launchLog) + ' -Value ' + (Quote-PowerShellLiteral ("python=" + $pythonExe))),
  ('$pythonExe = ' + (Quote-PowerShellLiteral $pythonExe)),
  ('$out = ' + (Quote-PowerShellLiteral $out)),
  ('$err = ' + (Quote-PowerShellLiteral $err)),
  ('$pidFile = ' + (Quote-PowerShellLiteral $pidFile)),
  ('$timeoutSeconds = ' + [string]$timeoutSeconds),
  ('Add-Content -LiteralPath ' + (Quote-PowerShellLiteral $launchLog) + ' -Value ("timeout_seconds=" + $timeoutSeconds)'),
  ('$argsList = ' + $argsLiteral),
  ('Add-Content -LiteralPath ' + (Quote-PowerShellLiteral $launchLog) + ' -Value ("args=" + ($argsList -join " "))'),
  ('Set-Location -LiteralPath ' + (Quote-PowerShellLiteral $root)),
  ('$envBat = ' + (Quote-PowerShellLiteral (Join-Path $root "scripts\env.bat"))),
  'if (Test-Path -LiteralPath $envBat) {',
  '  $envDump = & cmd.exe /d /c ("call ""{0}"" >nul && set" -f $envBat)',
  '  foreach ($line in $envDump) {',
  '    $eq = $line.IndexOf("=")',
  '    if ($eq -gt 0) {',
  '      [Environment]::SetEnvironmentVariable($line.Substring(0, $eq), $line.Substring($eq + 1), "Process")',
  '    }',
  '  }',
  ('  Add-Content -LiteralPath ' + (Quote-PowerShellLiteral $launchLog) + ' -Value "env_bat=loaded"'),
  '}',
  ('$env:TORCH_EXTENSIONS_DIR = ' + (Quote-PowerShellLiteral (Join-Path $root "torch_extensions_judas"))),
  '$ErrorActionPreference = "Continue"',
  'try {',
  '  if ($timeoutSeconds -gt 0) {',
  ('    $child = Start-Process -FilePath $pythonExe -ArgumentList $argsList -WorkingDirectory ' + (Quote-PowerShellLiteral $root) + ' -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err -PassThru'),
  ('    Add-Content -LiteralPath ' + (Quote-PowerShellLiteral $launchLog) + ' -Value ("child_pid=" + $child.Id)'),
  '    $timeoutMs = [int]([Math]::Min([int64]$timeoutSeconds * 1000, [int64][int]::MaxValue))',
  '    if (-not $child.WaitForExit($timeoutMs)) {',
  '      & taskkill.exe /PID $child.Id /T /F | Out-Null',
  ('      Add-Content -LiteralPath ' + (Quote-PowerShellLiteral $launchLog) + ' -Value ("timeout_killed_pid=" + $child.Id)'),
  '      Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue',
  '      exit 124',
  '    }',
  '    [void]$child.WaitForExit()',
  '    $child.Refresh()',
  '    $code = [int]$child.ExitCode',
  '  } else {',
  '    & $pythonExe @argsList > $out 2> $err',
  '    $code = if ($LASTEXITCODE -ne $null) { $LASTEXITCODE } else { 0 }',
  '  }',
  '  $ErrorActionPreference = "Stop"',
  ('  Add-Content -LiteralPath ' + (Quote-PowerShellLiteral $launchLog) + ' -Value ("exit_code=" + $code)'),
  '  Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue',
  '  exit $code',
  '} catch {',
  ('  Add-Content -LiteralPath ' + (Quote-PowerShellLiteral $launchLog) + ' -Value ("wrapper_error=" + $_.Exception.Message)'),
  '  Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue',
  '  throw',
  '}'
) -join [Environment]::NewLine
[System.IO.File]::WriteAllText($launch, $launchText, [System.Text.UTF8Encoding]::new($false))

$psExe = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$launchCmd = '"' + $psExe + '" -NoProfile -ExecutionPolicy Bypass -File "' + $launch + '"'
$shell = New-Object -ComObject WScript.Shell
[void]$shell.Run($launchCmd, 0, $false)

$procId = $null
for ($i = 0; $i -lt 50; $i++) {
  if (Test-Path -LiteralPath $pidFile) {
    $pidText = (Get-Content -LiteralPath $pidFile -TotalCount 1 -ErrorAction SilentlyContinue).Trim()
    $parsed = 0
    if ([int]::TryParse($pidText, [ref]$parsed)) {
      $proc = Get-Process -Id $parsed -ErrorAction SilentlyContinue
      if ($proc) { $procId = $parsed; break }
    }
  }
  Start-Sleep -Milliseconds 100
}
if (-not $procId) {
  throw "Detached launch did not report a live PID; script=$launch"
}

Write-Output "STARTED pid=$procId run=$Run iters=$Iters timeout_minutes=$TimeoutMinutes"
if ($Seed -ge 0) { Write-Output "seed=$Seed config=$cfg" }
Write-Output "out=$out"
Write-Output "err=$err"
Write-Output "metrics=$(Join-Path $runDir 'metrics.jsonl')"
