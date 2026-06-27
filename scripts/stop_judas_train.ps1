<#
.SYNOPSIS
  Stops old Judas train.run processes owned by this repo.
#>
[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
$repoNeedle = [System.IO.Path]::GetFullPath($repo)

function Stop-ProcessTree([int]$ProcId) {
    $proc = Get-Process -Id $ProcId -ErrorAction SilentlyContinue
    if (-not $proc) { return $false }
    & taskkill.exe /PID $ProcId /T /F | Out-Null
    Start-Sleep -Milliseconds 500
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

function Test-TrainProcess($ProcInfo) {
    $cmd = [string]$ProcInfo.CommandLine
    $exe = [string]$ProcInfo.ExecutablePath
    if (-not $cmd) { return $false }
    $inRepo = $cmd.Contains($repoNeedle) -or ($exe -and $exe.Contains($repoNeedle))
    if (-not $inRepo) { return $false }
    if ($cmd.Contains("-m train.run")) { return $true }
    if ($cmd.Contains("train\run.py") -or $cmd.Contains("train/run.py")) { return $true }
    if ($cmd.Contains("scripts\train.bat") -or $cmd.Contains("scripts/train.bat")) { return $true }
    return $false
}

$excluded = @{}
$excluded[[int]$PID] = $true
foreach ($ancestor in Get-AncestorPids) { $excluded[[int]$ancestor] = $true }

$targets = @(
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -match '^(python|pythonw|cmd|powershell|pwsh)\.exe$' -and
            -not $excluded.ContainsKey([int]$_.ProcessId) -and
            (Test-TrainProcess $_)
        } |
        Select-Object -ExpandProperty ProcessId
) | Sort-Object -Unique

if ($targets.Count -eq 0) {
    Write-Output "train_process=none"
    exit 0
}

$failures = 0
foreach ($procId in $targets) {
    if (Stop-ProcessTree $procId) {
        Write-Output "train_process=stopped pid=$procId"
    } else {
        Write-Output "train_process=missing pid=$procId"
    }
    if (Get-Process -Id $procId -ErrorAction SilentlyContinue) {
        Write-Output "train_process=still_running pid=$procId"
        $failures += 1
    }
}

if ($failures -ne 0) { exit 1 }
exit 0
