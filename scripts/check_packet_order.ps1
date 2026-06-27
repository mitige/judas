<#
.SYNOPSIS
  Resume .minecraft/judas-packet-order.log apres un test serveur.
#>
[CmdletBinding()]
param(
    [string]$Log = "$env:APPDATA\.minecraft\judas-packet-order.log",
    [string]$MinecraftLog = "$env:APPDATA\.minecraft\logs\latest.log",
    [string]$Session = "$env:APPDATA\.minecraft\judas-packet-order-session.txt",
    [switch]$Reset,
    [switch]$Strict
)
$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $cmd) { throw "Python introuvable." }
    $python = $cmd.Source
}
if ($Reset) {
    $dir = Split-Path $Log -Parent
    if ($dir -and -not (Test-Path $dir)) {
        New-Item -ItemType Directory -Force $dir | Out-Null
    }
    if (Test-Path $Log) {
        Remove-Item -LiteralPath $Log -Force
    }
    $minecraftLogSize = 0
    if (Test-Path $MinecraftLog) {
        $minecraftLogSize = (Get-Item $MinecraftLog).Length
    }
    @(
        "minecraft_log=$MinecraftLog",
        "minecraft_log_size=$minecraftLogSize"
    ) | Set-Content -LiteralPath $Session -Encoding ascii
    Write-Host "packet-order log reset: $Log"
    Write-Host "packet-order session: $Session"
    exit 0
}
$analyzerArgs = @()
if ($Strict) { $analyzerArgs += "--strict" }
if (Test-Path $Session) {
    $sessionData = @{}
    Get-Content -LiteralPath $Session | ForEach-Object {
        $parts = $_.Split("=", 2)
        if ($parts.Count -eq 2) { $sessionData[$parts[0]] = $parts[1] }
    }
    if ($sessionData.ContainsKey("minecraft_log") -and $sessionData.ContainsKey("minecraft_log_size")) {
        $analyzerArgs += @("--server-log", $sessionData["minecraft_log"])
        $analyzerArgs += @("--server-offset", $sessionData["minecraft_log_size"])
    }
}
$analyzerArgs += $Log
& $python (Join-Path $repo "tools\packet_order_log.py") @analyzerArgs
exit $LASTEXITCODE
