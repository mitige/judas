<#
.SYNOPSIS
  Stops old Judas app/viz dev UI processes owned by this repo.
#>
[CmdletBinding()]
param(
    [ValidateSet("app", "viz", "all")]
    [string]$Surface = "all"
)

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

function Test-Needle([string]$Text, [string[]]$Needles) {
    foreach ($needle in $Needles) {
        if ($Text.Contains($needle)) { return $true }
    }
    return $false
}

function Test-UiProcess($ProcInfo) {
    $cmd = [string]$ProcInfo.CommandLine
    $exe = [string]$ProcInfo.ExecutablePath
    if (-not $cmd) { return $false }
    if (-not ($cmd.Contains($repoNeedle) -or ($exe -and $exe.Contains($repoNeedle)))) { return $false }

    $appNeedles = @(
        "\app\",
        "/app/",
        "judas-app",
        "scripts\app.bat",
        "scripts/app.bat",
        "localhost:5173"
    )
    $vizNeedles = @(
        "\viz\",
        "/viz/",
        "judas-viz",
        "scripts\viz.bat",
        "scripts/viz.bat",
        "localhost:5174"
    )

    $isApp = Test-Needle $cmd $appNeedles
    $isViz = Test-Needle $cmd $vizNeedles
    if ($Surface -eq "app") { return $isApp }
    if ($Surface -eq "viz") { return $isViz }
    return ($isApp -or $isViz)
}

$excluded = @{}
$excluded[[int]$PID] = $true
foreach ($ancestor in Get-AncestorPids) { $excluded[[int]$ancestor] = $true }

$targets = @(
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -match '^(cmd|powershell|pwsh|node|npm|electron)\.exe$' -and
            -not $excluded.ContainsKey([int]$_.ProcessId) -and
            (Test-UiProcess $_)
        } |
        Select-Object -ExpandProperty ProcessId
) | Sort-Object -Unique

if ($targets.Count -eq 0) {
    Write-Output "ui_process=none surface=$Surface"
    exit 0
}

$failures = 0
foreach ($procId in $targets) {
    if (Stop-ProcessTree $procId) {
        Write-Output "ui_process=stopped surface=$Surface pid=$procId"
    } else {
        Write-Output "ui_process=missing surface=$Surface pid=$procId"
    }
    if (Get-Process -Id $procId -ErrorAction SilentlyContinue) {
        Write-Output "ui_process=still_running surface=$Surface pid=$procId"
        $failures += 1
    }
}

if ($failures -ne 0) { exit 1 }
exit 0
