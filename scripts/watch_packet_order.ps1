<#
.SYNOPSIS
  Surveille le test packet-order jusqu'a CLEAN ou echec serveur/local.
#>
[CmdletBinding()]
param(
    [string]$Log = "$env:APPDATA\.minecraft\judas-packet-order.log",
    [string]$MinecraftLog = "$env:APPDATA\.minecraft\logs\latest.log",
    [string]$Session = "$env:APPDATA\.minecraft\judas-packet-order-session.txt",
    [int]$IntervalSeconds = 2,
    [int]$TimeoutSeconds = 120
)
$ErrorActionPreference = "Stop"
$check = Join-Path $PSScriptRoot "check_packet_order.ps1"
$started = Get-Date

while ($true) {
    $checkArgs = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $check,
        "-Strict",
        "-Log", $Log,
        "-MinecraftLog", $MinecraftLog,
        "-Session", $Session
    )
    $output = & powershell @checkArgs 2>&1
    $exitCode = $LASTEXITCODE
    $text = ($output | Out-String).TrimEnd()
    if ($text) {
        Write-Host $text
    }

    $localClean = $text -match "(?m)^CLEAN\b"
    $serverClean = $text -match "(?m)^SERVER_CLEAN\b"
    $localBad = $text -match "(?m)^BAD "
    $serverBad = $text -match "(?m)^SERVER_BAD\b"

    if ($localBad -or $serverBad) {
        exit 1
    }
    if ($exitCode -eq 0 -and $localClean -and $serverClean) {
        exit 0
    }
    if ($TimeoutSeconds -gt 0 -and ((Get-Date) - $started).TotalSeconds -ge $TimeoutSeconds) {
        exit 2
    }

    Start-Sleep -Seconds $IntervalSeconds
}
