<#
.SYNOPSIS
  Resume .minecraft/judas-aim-os.log apres un test en entree souris OS.
#>
[CmdletBinding()]
param(
    [string]$Log = "$env:APPDATA\.minecraft\judas-aim-os.log",
    [string]$ModsDir = "$env:APPDATA\.minecraft\mods",
    [switch]$Reset,
    [switch]$AllowStale,
    [string]$FreshAfter = "",
    [switch]$Strict,
    [switch]$All
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
    Write-Host "aim-os log reset: $Log"
    exit 0
}
function Resolve-FreshAfterUtc([string]$Value) {
    if (-not $Value) { return $null }
    $doubleValue = 0.0
    if ([double]::TryParse($Value, [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$doubleValue)) {
        return [DateTimeOffset]::FromUnixTimeSeconds([int64]$doubleValue).UtcDateTime
    }
    try {
        return ([DateTimeOffset]::Parse($Value, [Globalization.CultureInfo]::InvariantCulture)).UtcDateTime
    } catch {
        throw "Invalid FreshAfter value: $Value"
    }
}

if (-not $AllowStale -and $FreshAfter -and (Test-Path $Log)) {
    $freshAfterUtc = Resolve-FreshAfterUtc $FreshAfter
    $logInfo = Get-Item -LiteralPath $Log
    if ($freshAfterUtc -and $logInfo.LastWriteTimeUtc -lt $freshAfterUtc) {
        Write-Host ("STALE_LOG log={0} logTime={1:o} freshAfter={2:o}" -f `
            $logInfo.FullName, $logInfo.LastWriteTimeUtc, $freshAfterUtc)
        exit 1
    }
}
if (-not $AllowStale -and (Test-Path $Log) -and (Test-Path $ModsDir)) {
    $activeJars = @(Get-ChildItem -Path $ModsDir -Filter "judas-bridge-*.jar" -File | Sort-Object LastWriteTime -Descending)
    if ($activeJars.Count -gt 0) {
        $logInfo = Get-Item -LiteralPath $Log
        $jarInfo = $activeJars[0]
        if ($logInfo.LastWriteTime -lt $jarInfo.LastWriteTime) {
            Write-Host ("STALE_LOG log={0} logTime={1:o} jar={2} jarTime={3:o}" -f `
                $logInfo.FullName, $logInfo.LastWriteTime, $jarInfo.FullName, $jarInfo.LastWriteTime)
            exit 1
        }
    }
}

$analyzerArgs = @()
if ($Strict) { $analyzerArgs += "--strict" }
if ($All) { $analyzerArgs += "--all" }
$analyzerArgs += $Log
& $python (Join-Path $repo "tools\aim_os_log.py") @analyzerArgs
exit $LASTEXITCODE
