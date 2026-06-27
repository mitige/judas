<#
.SYNOPSIS
  Surveille .minecraft/judas-aim-os.log jusqu'a PRECISE ou echec aim OS clair.
#>
[CmdletBinding()]
param(
    [string]$Log = "$env:APPDATA\.minecraft\judas-aim-os.log",
    [int]$IntervalSeconds = 2,
    [int]$TimeoutSeconds = 120,
    [int]$MinLooseSamples = 40,
    [string]$FreshAfter = ""
)
$ErrorActionPreference = "Stop"
$check = Join-Path $PSScriptRoot "check_aim_os.ps1"
$started = Get-Date

function Get-LogStatus([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        return "log=$Path exists=false"
    }
    $item = Get-Item -LiteralPath $Path
    return "log=$($item.FullName) exists=true size=$($item.Length) mtime=$($item.LastWriteTime.ToString("s"))"
}

function Get-MinecraftStatus {
    try {
        $matches = @(Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object {
            $_.Name -match "java|javaw" -and $_.CommandLine -and (
                $_.CommandLine -match "net\.minecraft\.client" -or
                $_.CommandLine -match "\.minecraft" -or
                $_.CommandLine -match "minecraft"
            )
        })
        if ($matches.Count -gt 0) {
            $pids = ($matches | Select-Object -ExpandProperty ProcessId) -join ","
            return "minecraft=running pids=$pids"
        }
    } catch {
        return "minecraft=unknown"
    }
    return "minecraft=not_found"
}

while ($true) {
    $checkArgs = @("-Strict", "-Log", $Log)
    if ($FreshAfter) { $checkArgs += @("-FreshAfter", $FreshAfter) }
    $output = & powershell -NoProfile -ExecutionPolicy Bypass -File $check @checkArgs 2>&1
    $exitCode = $LASTEXITCODE
    $text = ($output | Out-String).TrimEnd()
    if ($text) {
        Write-Host $text
    }

    if ($text -match "(?m)^PRECISE\b") {
        exit 0
    }
    if ($text -match "(?m)^(STALL|DIVERGE|NOT_1TO1)\b") {
        exit 1
    }
    if ($text -match "(?m)^LOOSE\b") {
        $samples = 0
        if ($text -match "samples=(\d+)") {
            $samples = [int]$Matches[1]
        }
        if ($samples -ge $MinLooseSamples) {
            exit 1
        }
    }
    if ($TimeoutSeconds -gt 0 -and ((Get-Date) - $started).TotalSeconds -ge $TimeoutSeconds) {
        Write-Host ("TIMEOUT_AIM_OS {0} {1} hint=""launch Minecraft, press O for OS mouse, then K to arm Judas""" -f `
            (Get-LogStatus $Log), (Get-MinecraftStatus))
        exit 2
    }

    Start-Sleep -Seconds $IntervalSeconds
}
