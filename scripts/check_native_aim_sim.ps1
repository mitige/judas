<#
.SYNOPSIS
  Simule la boucle souris OS Judas contre gain, latence, jitter et inversion.
#>
[CmdletBinding()]
param(
    [string]$Java = "mod\src\main\java\dev\judas\bridge\ActionApplier.java"
)
$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $cmd) { throw "Python introuvable." }
    $python = $cmd.Source
}
& $python (Join-Path $repo "tools\native_aim_sim.py") --java $Java
exit $LASTEXITCODE