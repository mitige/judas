<#
.SYNOPSIS
  Starts one guarded Judas daemon for app/viz/manual use.
#>
[CmdletBinding()]
param(
    [string]$DaemonHost = "127.0.0.1",
    [int]$DaemonPort = 8765,
    [string]$PidFile = "",
    [string]$ActionLog = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
if (-not $PidFile) {
    $PidFile = Join-Path $repo "runs\judas_live_daemon.pid"
}
if (-not $ActionLog) {
    $ActionLog = Join-Path $repo "runs\judas-live-actions.log"
}
$base = "http://${DaemonHost}:${DaemonPort}"

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
    return "python"
}

function Read-PidFile([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    $text = Get-Content -LiteralPath $Path -Raw -ErrorAction SilentlyContinue
    if ($text -match "^\s*(\d+)\b") { return [int]$Matches[1] }
    return $null
}

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

$pidDir = Split-Path $PidFile -Parent
if ($pidDir -and -not (Test-Path -LiteralPath $pidDir)) {
    New-Item -ItemType Directory -Path $pidDir -Force | Out-Null
}

$oldPid = Read-PidFile $PidFile
if ($null -ne $oldPid) {
    $oldProc = Get-Process -Id $oldPid -ErrorAction SilentlyContinue
    if ($oldProc) {
        if (-not $Force) {
            Write-Output "ALREADY_RUNNING pid=$oldPid pid_file=$PidFile"
            exit 0
        }
        if (Stop-ProcessTree $oldPid) { Write-Output "STOPPED_OLD pid=$oldPid" }
    }
} elseif (Test-Path -LiteralPath $PidFile) {
    Write-Output "REMOVED_INVALID_PID_FILE path=$PidFile"
}
Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue

if ($Force) {
    $owned = @(Get-OwnedDaemonPids | Sort-Object -Unique)
    foreach ($ownedPid in $owned) {
        if (Stop-ProcessTree $ownedPid) { Write-Output "STOPPED_OLD_DAEMON pid=$ownedPid" }
    }
}

if (Test-Daemon) {
    Write-Output "ALREADY_LISTENING base=$base"
    exit 0
}

$py = Resolve-Python
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
Set-Content -LiteralPath $PidFile -Value $daemonProc.Id -Encoding ascii
Write-Output "STARTED pid=$($daemonProc.Id) pid_file=$PidFile base=$base"

$deadline = (Get-Date).AddSeconds(30)
while (-not (Test-Daemon)) {
    if ((Get-Date) -gt $deadline) { throw "Daemon did not answer within 30 seconds." }
    Start-Sleep -Milliseconds 500
}
Write-Output "READY base=$base"
