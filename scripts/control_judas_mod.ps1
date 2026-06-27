<#
.SYNOPSIS
  Sends a bounded control command to the loaded Judas Bridge mod.

.PARAMETER RequireAck
  Use -RequireAck to fail unless the mod writes a fresh non-blocked status ack.
#>
[CmdletBinding()]
param(
    [ValidateSet("arm", "arm_native", "arm_live", "arm_native_live", "native_on", "native_off", "disarm", "kill", "status", "status_live", "dump_screen")]
    [string]$Command = "arm_native",
    [string]$MinecraftDir = "$env:APPDATA\.minecraft",
    [int]$WaitSeconds = 8,
    [switch]$RequireAck
)

$ErrorActionPreference = "Stop"
$control = Join-Path $MinecraftDir "judas-control.txt"
$status = Join-Path $MinecraftDir "judas-bridge-status.log"
$start = Get-Date
$startMillis = [int64]([DateTimeOffset]$start).ToUnixTimeMilliseconds()

if (-not (Test-Path -LiteralPath $MinecraftDir)) {
    throw "MinecraftDir introuvable: $MinecraftDir"
}

function Write-ControlCommand([string]$Path, [string]$Value) {
    $bytes = [System.Text.Encoding]::ASCII.GetBytes($Value + [Environment]::NewLine)
    $lastError = $null
    for ($i = 0; $i -lt 10; $i++) {
        try {
            $stream = [System.IO.FileStream]::new(
                $Path,
                [System.IO.FileMode]::Create,
                [System.IO.FileAccess]::Write,
                [System.IO.FileShare]::ReadWrite
            )
            try {
                $stream.Write($bytes, 0, $bytes.Length)
                $stream.Flush()
                return
            } finally {
                $stream.Dispose()
            }
        } catch {
            $lastError = $_
            Start-Sleep -Milliseconds 50
        }
    }
    throw $lastError
}

Write-ControlCommand $control $Command
Write-Output "CONTROL_SENT command=$Command path=$control"

$deadline = (Get-Date).AddSeconds([Math]::Max(0, $WaitSeconds))
do {
    if (Test-Path -LiteralPath $status) {
        $lines = @(Get-Content -LiteralPath $status -Tail 20 -ErrorAction SilentlyContinue)
        $ack = $null
        foreach ($line in (($lines | Where-Object { $_ -like "*command=$Command *" }) | Select-Object -Last 20)) {
            if ($line -match "^time=(\d+) ") {
                $ackMillis = [int64]$Matches[1]
                if ($ackMillis -ge ($startMillis - 500)) {
                    $ack = $line
                }
            }
        }
        if ($ack) {
            $info = Get-Item -LiteralPath $status
            if ($info.LastWriteTime -ge $start.AddSeconds(-1)) {
                if ($ack -match "result=(blocked:[^ ]+|unknown)") {
                    Write-Output "CONTROL_NACK $ack"
                    exit 2
                }
                Write-Output "CONTROL_ACK $ack"
                exit 0
            }
        }
    }
    if (-not $RequireAck) { break }
    Start-Sleep -Milliseconds 250
} while ((Get-Date) -lt $deadline)

if ($RequireAck) {
    if (Test-Path -LiteralPath $status) {
        Write-Output "---STATUS TAIL---"
        Get-Content -LiteralPath $status -Tail 20
    } else {
        Write-Output "status_log=missing path=$status"
    }
    Write-Output "CONTROL_ACK_TIMEOUT command=$Command wait=${WaitSeconds}s"
    exit 1
}

if (Test-Path -LiteralPath $status) {
    Write-Output "---STATUS TAIL---"
    Get-Content -LiteralPath $status -Tail 5
}
