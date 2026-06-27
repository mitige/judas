<#
.SYNOPSIS
  Verifie que la jar Judas active dans mods/ est celle construite localement.
#>
[CmdletBinding()]
param(
    [string]$ModsDir = "$env:APPDATA\.minecraft\mods",
    [string]$Jar = "",
    [switch]$RequireWritable
)
$ErrorActionPreference = "Stop"
function Get-LockingProcessSummary([string]$Path) {
    try {
        if (-not ("RestartManager.NativeMethods" -as [type])) {
            Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
namespace RestartManager {
  public enum RM_APP_TYPE { RmUnknownApp = 0, RmMainWindow = 1, RmOtherWindow = 2, RmService = 3, RmExplorer = 4, RmConsole = 5, RmCritical = 1000 }
  [StructLayout(LayoutKind.Sequential)]
  public struct RM_UNIQUE_PROCESS { public int dwProcessId; public System.Runtime.InteropServices.ComTypes.FILETIME ProcessStartTime; }
  [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
  public struct RM_PROCESS_INFO {
    public RM_UNIQUE_PROCESS Process;
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 256)] public string strAppName;
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 64)] public string strServiceShortName;
    public RM_APP_TYPE ApplicationType;
    public uint AppStatus;
    public uint TSSessionId;
    [MarshalAs(UnmanagedType.Bool)] public bool bRestartable;
  }
  public static class NativeMethods {
    [DllImport("rstrtmgr.dll", CharSet = CharSet.Unicode)]
    public static extern int RmStartSession(out uint pSessionHandle, int dwSessionFlags, string strSessionKey);
    [DllImport("rstrtmgr.dll", CharSet = CharSet.Unicode)]
    public static extern int RmRegisterResources(uint pSessionHandle, uint nFiles, string[] rgsFilenames, uint nApplications, RM_UNIQUE_PROCESS[] rgApplications, uint nServices, string[] rgsServiceNames);
    [DllImport("rstrtmgr.dll")]
    public static extern int RmGetList(uint dwSessionHandle, out uint pnProcInfoNeeded, ref uint pnProcInfo, [In, Out] RM_PROCESS_INFO[] rgAffectedApps, ref uint lpdwRebootReasons);
    [DllImport("rstrtmgr.dll")]
    public static extern int RmEndSession(uint pSessionHandle);
  }
}
"@
        }

        $session = [uint32]0
        $key = [Guid]::NewGuid().ToString()
        $rc = [RestartManager.NativeMethods]::RmStartSession([ref]$session, 0, $key)
        if ($rc -ne 0) { return "rm_start=$rc" }
        try {
            $resolved = (Resolve-Path -LiteralPath $Path -ErrorAction Stop).Path
            $files = [string[]]@($resolved)
            $rc = [RestartManager.NativeMethods]::RmRegisterResources($session, [uint32]1, $files, 0, $null, 0, $null)
            if ($rc -ne 0) { return "rm_register=$rc" }

            $needed = [uint32]0
            $count = [uint32]0
            $reasons = [uint32]0
            $rc = [RestartManager.NativeMethods]::RmGetList($session, [ref]$needed, [ref]$count, $null, [ref]$reasons)
            if ($rc -eq 234 -and $needed -gt 0) {
                $count = $needed
                $apps = New-Object RestartManager.RM_PROCESS_INFO[] $count
                $rc = [RestartManager.NativeMethods]::RmGetList($session, [ref]$needed, [ref]$count, $apps, [ref]$reasons)
                if ($rc -eq 0 -and $count -gt 0) {
                    $items = for ($i = 0; $i -lt $count; $i++) {
                        $app = $apps[$i]
                        $name = if ($app.strAppName) { $app.strAppName } else { "unknown" }
                        "pid=$($app.Process.dwProcessId),app=$name"
                    }
                    return ($items -join ";")
                }
            }
            if ($rc -eq 0) { return "none" }
            return "rm_getlist=$rc"
        } finally {
            if ($session -ne 0) { [void][RestartManager.NativeMethods]::RmEndSession($session) }
        }
    } catch {
        $msg = ($_.Exception.Message -replace "\s+", " ").Trim()
        if ($msg.Length -gt 120) { $msg = $msg.Substring(0, 120) }
        return "rm_error=$($_.Exception.GetType().Name):$msg"
    }
}
$repo = Split-Path $PSScriptRoot -Parent
if (-not $Jar) {
    $Jar = Join-Path $repo "mod\build\libs\judas-bridge-0.1.0.jar"
}
if (-not (Test-Path $Jar)) {
    Write-Host "MISSING_BUILD jar=$Jar"
    exit 1
}
if (-not (Test-Path $ModsDir)) {
    Write-Host "MISSING_MODS mods=$ModsDir"
    exit 1
}
$active = @(Get-ChildItem -Path $ModsDir -Filter "judas-bridge-*.jar" -File |
    Where-Object { $_.Name -notmatch '\.disabled-' })
if ($active.Count -eq 0) {
    Write-Host "MISSING_DEPLOY mods=$ModsDir"
    exit 1
}
if ($active.Count -gt 1) {
    $names = ($active | ForEach-Object { $_.Name }) -join ","
    Write-Host "MULTIPLE_DEPLOY count=$($active.Count) jars=$names"
    exit 1
}
$sourceHash = (Get-FileHash -LiteralPath $Jar -Algorithm SHA256).Hash
$deployed = $active[0].FullName
$deployHash = (Get-FileHash -LiteralPath $deployed -Algorithm SHA256).Hash
if ($RequireWritable) {
    $stream = $null
    try {
        $stream = [System.IO.File]::Open(
            $deployed,
            [System.IO.FileMode]::Open,
            [System.IO.FileAccess]::ReadWrite,
            [System.IO.FileShare]::None)
    } catch {
        $lockers = Get-LockingProcessSummary $deployed
        $freshness = if ($sourceHash -eq $deployHash) { "fresh" } else { "stale" }
        Write-Host "LOCKED_DEPLOY jar=$deployed freshness=$freshness source=$sourceHash deployed=$deployHash lockers=$lockers"
        exit 1
    } finally {
        if ($stream) { $stream.Dispose() }
    }
}
if ($sourceHash -ne $deployHash) {
    Write-Host "STALE_DEPLOY jar=$deployed source=$sourceHash deployed=$deployHash"
    exit 1
}

Write-Host "OK_DEPLOY jar=$deployed sha256=$deployHash"
exit 0
