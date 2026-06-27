<#
.SYNOPSIS
  Stops the Judas live daemon started by judas_live.ps1.
#>
[CmdletBinding()]
param(
    [string]$PidFile = "",
    [int]$DaemonPort = 8765
)

$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
if (-not $PidFile) {
    $PidFile = Join-Path $repo "runs\judas_live_daemon.pid"
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

function Stop-OwnedOrphanDaemons([int[]]$SkipPids) {
    $owned = @(Get-OwnedDaemonPids | Sort-Object -Unique | Where-Object { $SkipPids -notcontains $_ })
    if ($owned.Count -eq 0) {
        Write-Output "orphan_daemon=none"
        return
    }
    foreach ($ownedPid in $owned) {
        if (Stop-ProcessTree $ownedPid) {
            Write-Output "orphan_daemon=stopped pid=$ownedPid"
        } else {
            Write-Output "orphan_daemon=missing pid=$ownedPid"
        }
        if (Get-Process -Id $ownedPid -ErrorAction SilentlyContinue) {
            Write-Output "orphan_daemon=still_running pid=$ownedPid"
            $script:orphanFailures += 1
        }
    }
}

Write-Output "pid_file=$PidFile"
$stoppedPid = 0
$exitCode = 0
$script:orphanFailures = 0
if (-not (Test-Path -LiteralPath $PidFile)) {
    Write-Output "live_daemon=no pid file"
} else {
    $text = Get-Content -LiteralPath $PidFile -Raw -ErrorAction SilentlyContinue
    if ($text -notmatch "^\s*(\d+)\b") {
        Write-Output "live_daemon=invalid pid file"
        Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
    } else {
        $procId = [int]$Matches[1]
        $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
        if (-not $proc) {
            Write-Output "live_daemon=exited pid=$procId"
            Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
        } else {
            try {
                Stop-ProcessTree $procId | Out-Null
            } catch {
                Write-Output "taskkill=failed $($_.Exception.Message)"
            }
            $still = Get-Process -Id $procId -ErrorAction SilentlyContinue
            if ($still) {
                Write-Output "live_daemon=still_running pid=$procId"
                $exitCode = 1
            } else {
                Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
                Write-Output "live_daemon=stopped pid=$procId"
                $stoppedPid = $procId
            }
        }
    }
}

Stop-OwnedOrphanDaemons @($stoppedPid)
if ($script:orphanFailures -ne 0) {
    $exitCode = 1
}
exit $exitCode
