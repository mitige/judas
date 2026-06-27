<#
.SYNOPSIS
  Passive preflight before running the Minecraft field proof.
#>
[CmdletBinding()]
param(
    [string]$ModsDir = "$env:APPDATA\.minecraft\mods",
    [string]$AimLog = "$env:APPDATA\.minecraft\judas-aim-os.log",
    [string]$LiveLog = "",
    [string]$PacketLog = "$env:APPDATA\.minecraft\judas-packet-order.log",
    [string]$MinecraftLog = "$env:APPDATA\.minecraft\logs\latest.log",
    [string]$PacketSession = "$env:APPDATA\.minecraft\judas-packet-order-session.txt",
    [string]$FreshAfter = "",
    [switch]$RequireMinecraft,
    [switch]$RequireField
)

$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
if (-not $LiveLog) {
    $LiveLog = Join-Path $repo "runs\judas-live-actions.log"
}

$checkSafeExport = Join-Path $PSScriptRoot "check_safe_export.ps1"
$checkModDeploy = Join-Path $PSScriptRoot "check_mod_deploy.ps1"
$checkFieldStatus = Join-Path $PSScriptRoot "check_field_status.ps1"

function Invoke-PreflightStep([string]$Name, [string]$Script, [string[]]$ArgsList) {
    Write-Output "---$Name---"
    & powershell -NoProfile -ExecutionPolicy Bypass -File $Script @ArgsList
    if ($LASTEXITCODE -ne 0) { $script:failures += 1 }
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

function Get-RepoProcesses {
    $repoNeedle = [System.IO.Path]::GetFullPath($repo)
    $excluded = @{}
    $excluded[[int]$PID] = $true
    foreach ($ancestor in Get-AncestorPids) { $excluded[[int]$ancestor] = $true }
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object {
            $_.CommandLine -and
            $_.CommandLine.Contains($repoNeedle) -and
            $_.Name -match '^(python|pythonw|cmd|powershell|pwsh|node|npm|java|gradle|electron)\.exe$' -and
            -not $excluded.ContainsKey([int]$_.ProcessId)
        } |
        Select-Object ProcessId, Name, CommandLine
}

function Test-LivePort {
    $ports = @(Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue)
    if ($ports.Count -eq 0) {
        Write-Output "LIVE_PORT down"
        return
    }
    $owners = ($ports | Select-Object -ExpandProperty OwningProcess -Unique) -join ","
    Write-Output "LIVE_PORT up owners=$owners"
    $script:failures += 1
}

function Test-RepoProcessClean {
    $rows = @(Get-RepoProcesses)
    if ($rows.Count -eq 0) {
        Write-Output "PROCESS_CLEAN repo=$repo"
        return
    }
    Write-Output "PROCESS_DIRTY count=$($rows.Count) repo=$repo"
    foreach ($row in $rows | Select-Object -First 12) {
        $cmd = ([string]$row.CommandLine).Replace("`r", " ").Replace("`n", " ")
        if ($cmd.Length -gt 220) { $cmd = $cmd.Substring(0, 220) + "..." }
        Write-Output "PROCESS pid=$($row.ProcessId) name=$($row.Name) cmd=$cmd"
    }
    $script:failures += 1
}

function Get-MinecraftProcessIds {
    try {
        return @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
            $_.Name -match "java|javaw|Minecraft|prismlauncher|MultiMC|lunar|badlion" -and
            $_.CommandLine -and (
                $_.CommandLine -match "net\.minecraft\.client" -or
                $_.CommandLine -match "\.minecraft" -or
                $_.CommandLine -match "minecraft" -or
                $_.CommandLine -match "fabric" -or
                $_.CommandLine -match "forge"
            )
        } | Select-Object -ExpandProperty ProcessId)
    } catch {
        return @()
    }
}

function Test-MinecraftRunning {
    $minecraftPids = @(Get-MinecraftProcessIds)
    if ($minecraftPids.Count -eq 0) {
        Write-Output "MINECRAFT_STATUS missing"
        if ($RequireMinecraft) { $script:failures += 1 }
        return
    }
    Write-Output ("MINECRAFT_STATUS running pids={0}" -f ($minecraftPids -join ","))
}

$failures = 0

Invoke-PreflightStep "SAFE_EXPORT" $checkSafeExport @()

Invoke-PreflightStep "MOD_DEPLOY" $checkModDeploy @(
    "-ModsDir", $ModsDir,
    "-RequireWritable"
)

Write-Output "---PROCESS_PREFLIGHT---"
Test-RepoProcessClean
Test-LivePort
Test-MinecraftRunning

Write-Output "---FIELD_STATUS---"
$fieldArgs = @(
    "-AimLog", $AimLog,
    "-LiveLog", $LiveLog,
    "-PacketLog", $PacketLog,
    "-ModsDir", $ModsDir,
    "-MinecraftLog", $MinecraftLog,
    "-PacketSession", $PacketSession
)
if ($FreshAfter) { $fieldArgs += @("-FreshAfter", $FreshAfter) }
if ($RequireField) { $fieldArgs += "-Strict" }
& powershell -NoProfile -ExecutionPolicy Bypass -File $checkFieldStatus @fieldArgs
$fieldCode = $LASTEXITCODE
if ($RequireField -and $fieldCode -ne 0) { $failures += 1 }

if ($failures -eq 0) {
    Write-Output "PREFLIGHT PASS"
    exit 0
}
Write-Output "PREFLIGHT FAIL failures=$failures"
exit 1
