<#
.SYNOPSIS
  Attend que la jar Judas dans mods/ soit deverrouillee, puis prepare l'aim OS.
#>
[CmdletBinding()]
param(
    [string]$ModsDir = "$env:APPDATA\.minecraft\mods",
    [int]$IntervalSeconds = 2,
    [int]$TimeoutSeconds = 120,
    [int]$BuildTimeoutSeconds = 300
)
$ErrorActionPreference = "Stop"
$check = Join-Path $PSScriptRoot "check_mod_deploy.ps1"
$prepare = Join-Path $PSScriptRoot "prepare_aim_os_test.bat"
$started = Get-Date

while ($true) {
    $output = & powershell -NoProfile -ExecutionPolicy Bypass -File $check -ModsDir $ModsDir -RequireWritable 2>&1
    $text = ($output | Out-String).TrimEnd()
    if ($text) { Write-Host $text }

    if ($text -notmatch "(?m)^LOCKED_DEPLOY\b") {
        & cmd /c "`"$prepare`" `"$ModsDir`" -BuildTimeoutSeconds $BuildTimeoutSeconds"
        exit $LASTEXITCODE
    }
    if ($TimeoutSeconds -gt 0 -and ((Get-Date) - $started).TotalSeconds -ge $TimeoutSeconds) {
        exit 2
    }
    Start-Sleep -Seconds $IntervalSeconds
}
