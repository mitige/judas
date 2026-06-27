<#
.SYNOPSIS
  Compile le mod Judas Bridge (Forge 1.8.9) -> mod/build/libs/judas-bridge-0.1.0.jar

.DESCRIPTION
  Gradle tourne sur JDK 17 (requis par le plugin architectury-pack200 qui cible
  Java 16+), avec Zulu 8 comme toolchain de compilation. Par defaut le cache
  outil est %LOCALAPPDATA%\judas-build-tools; definissez JUDAS_BUILD_TOOLS pour
  rediriger le gradle-home vers un dossier writable du workspace.

.EXAMPLE
  ./scripts/build_mod.ps1
  ./scripts/build_mod.ps1 -Clean
#>
[CmdletBinding()]
param(
    [string]$Jdk17 = "C:\Program Files\Zulu\zulu-17",
    [string]$Jdk8  = "C:\Program Files\Zulu\zulu-8",
    [string]$ModsDir = "",   # si fourni : copie le jar dans ce dossier mods/ apres build
    [int]$BuildTimeoutSeconds = 300,
    [switch]$StopMinecraft,
    [switch]$Clean
)
$ErrorActionPreference = "Stop"
function Quote-ProcessArg([string]$Arg) {
    if ($Arg -notmatch '[\s"]') { return $Arg }
    return '"' + ($Arg -replace '"', '\"') + '"'
}

function Write-LogTail([string]$Path, [int]$Lines = 80) {
    if (Test-Path -LiteralPath $Path) {
        Write-Host "--- $Path tail ---" -ForegroundColor DarkGray
        Get-Content -LiteralPath $Path -Tail $Lines
    }
}

function Wait-JudasJarWritable([string]$Path) {
    $deadline = (Get-Date).AddSeconds(20)
    while ($true) {
        $stream = $null
        try {
            $stream = [System.IO.File]::Open(
                $Path,
                [System.IO.FileMode]::Open,
                [System.IO.FileAccess]::ReadWrite,
                [System.IO.FileShare]::None)
            return
        } catch {
            if ((Get-Date) -ge $deadline) {
                throw "Impossible de desactiver l'ancien jar Judas: $Path. Ferme Minecraft puis relance le script de preparation avec -StopMinecraft."
            }
            Start-Sleep -Milliseconds 500
        } finally {
            if ($stream) { $stream.Dispose() }
        }
    }
}
$repo = Split-Path $PSScriptRoot -Parent
$portableTools = "$env:LOCALAPPDATA\judas-build-tools"
$tools = if ($env:JUDAS_BUILD_TOOLS) { $env:JUDAS_BUILD_TOOLS } else { $portableTools }
$ghome = Join-Path $tools "ghome"

# Gradle portable, sinon PATH. Le Gradle portable peut rester dans LOCALAPPDATA
# meme si le gradle-home est redirige via JUDAS_BUILD_TOOLS.
$gradleCandidates = @(
    (Join-Path $tools "gradle-7.5.1\bin\gradle.bat"),
    (Join-Path $portableTools "gradle-7.5.1\bin\gradle.bat")
) | Select-Object -Unique
$gradle = $gradleCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $gradle) {
    $g = Get-Command gradle -ErrorAction SilentlyContinue
    if (-not $g) {
        throw "Gradle introuvable ($($gradleCandidates -join ', ') absent et pas sur le PATH)."
    }
    $gradle = $g.Source
}
foreach ($j in @($Jdk17, $Jdk8)) {
    if (-not (Test-Path "$j\bin\java.exe")) { throw "JDK introuvable: $j" }
}

if ($StopMinecraft) {
    $mcRoot = Join-Path $env:APPDATA ".minecraft"
    $mcJre = Join-Path $mcRoot "jre"
    $ids = New-Object 'System.Collections.Generic.HashSet[int]'

    Get-Process java, javaw, Minecraft, PrismLauncher, MultiMC, Lunar, Badlion -ErrorAction SilentlyContinue |
        Where-Object {
            ($_.Path -and $_.Path.StartsWith($mcJre, [System.StringComparison]::OrdinalIgnoreCase)) -or
            ($_.MainWindowTitle -match 'Minecraft')
        } | ForEach-Object { [void]$ids.Add([int]$_.Id) }

    try {
        Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object {
            ($_.Name -match '^(java|javaw|Minecraft|PrismLauncher|MultiMC|Lunar|Badlion)') -and
            ($_.CommandLine -match '\.minecraft|net\.minecraft\.client|--gameDir|--assetsDir')
        } | ForEach-Object { [void]$ids.Add([int]$_.ProcessId) }
    } catch {
        Write-Host "[build_mod] diagnostic process Minecraft limite: $($_.Exception.Message)" -ForegroundColor DarkYellow
    }

    foreach ($procId in $ids) {
        try {
            $proc = Get-Process -Id $procId -ErrorAction Stop
            Write-Host "[build_mod] stop Minecraft -> $($proc.ProcessName) pid=$procId" -ForegroundColor Yellow
            Stop-Process -Id $procId -Force
            Wait-Process -Id $procId -Timeout 15 -ErrorAction SilentlyContinue
        } catch {
            Write-Host "[build_mod] process deja ferme ou inaccessible pid=$procId" -ForegroundColor DarkYellow
        }
    }
}

if ($ModsDir) {
    if (-not (Test-Path $ModsDir)) { throw "ModsDir introuvable: $ModsDir" }
    $staleJars = @(Get-ChildItem -Path $ModsDir -Filter "judas-bridge-*.jar" -File)
    foreach ($stale in $staleJars) {
        Wait-JudasJarWritable $stale.FullName
    }
}

New-Item -ItemType Directory -Force $ghome | Out-Null
Set-Content -LiteralPath (Join-Path $ghome "gradle.properties") `
    -Value "org.gradle.java.installations.paths=$($Jdk8 -replace '\\', '/')" -Encoding ascii

$env:JAVA_HOME = $Jdk17
$env:Path = "$Jdk17\bin;$env:Path"

$gargs = @(
    '-g', $ghome,
    '-p', (Join-Path $repo 'mod'),
    '-Dorg.gradle.jvmargs=-Xmx512m -XX:CICompilerCount=2 -XX:ActiveProcessorCount=2'
)
if ($Clean) { $gargs += 'clean' }
$gargs += @('build', '--no-daemon', '--max-workers=1')

Write-Host "[build_mod] gradle (JDK17 runtime, Zulu8 toolchain) ..." -ForegroundColor Cyan
$logDir = Join-Path $repo "runs\build_mod"
New-Item -ItemType Directory -Force $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outLog = Join-Path $logDir "gradle_$stamp.out.log"
$errLog = Join-Path $logDir "gradle_$stamp.err.log"
$argLine = ($gargs | ForEach-Object { Quote-ProcessArg $_ }) -join " "
Write-Host "[build_mod] timeout=${BuildTimeoutSeconds}s out=$outLog err=$errLog" -ForegroundColor DarkGray
$gradleProc = Start-Process -FilePath $gradle `
    -ArgumentList $argLine `
    -WorkingDirectory $repo `
    -WindowStyle Hidden `
    -RedirectStandardOutput $outLog `
    -RedirectStandardError $errLog `
    -PassThru
if ($BuildTimeoutSeconds -gt 0) {
    $finished = $gradleProc.WaitForExit($BuildTimeoutSeconds * 1000)
} else {
    $gradleProc.WaitForExit()
    $finished = $true
}
if (-not $finished) {
    & taskkill.exe /PID $gradleProc.Id /T /F | Out-Null
    Write-LogTail $outLog
    Write-LogTail $errLog
    throw "Build timeout apres $BuildTimeoutSeconds secondes (pid=$($gradleProc.Id))."
}
$gradleProc.WaitForExit()
$gradleProc.Refresh()
$exitCode = $gradleProc.ExitCode
if ($null -eq $exitCode) {
    $outText = ""
    if (Test-Path -LiteralPath $outLog) {
        $outText = Get-Content -LiteralPath $outLog -Raw -ErrorAction SilentlyContinue
    }
    if ($outText -match "BUILD SUCCESSFUL") {
        $exitCode = 0
    }
}
if ($null -eq $exitCode -or $exitCode -ne 0) {
    Write-LogTail $outLog
    Write-LogTail $errLog
    throw "Build echoue (exit $exitCode)."
}

$jar = Join-Path $repo 'mod\build\libs\judas-bridge-0.1.0.jar'
$jarInfo = Get-Item $jar -ErrorAction Stop
if ($jarInfo.Length -lt 10000) {
    throw "Jar invalide ou corrompu: $jar ($($jarInfo.Length) octets)."
}
Write-Host "[build_mod] OK -> $jar" -ForegroundColor Green
if ($ModsDir) {
    $stamp = Get-Date -Format "yyyyMMddHHmmss"
    $modsParent = Split-Path $ModsDir -Parent
    $modsLeaf = Split-Path $ModsDir -Leaf
    $featherOrig = Join-Path $modsParent "mods.orig"
    $featherUserMods = Join-Path $env:APPDATA ".feather\user-mods\1.8.9"
    $featherMode = $modsLeaf -ieq "mods" -and (
        (Test-Path -LiteralPath $featherOrig) -or
        (Test-Path -LiteralPath $featherUserMods)
    )
    $cleanDirs = @($ModsDir)
    if ($featherMode) {
        foreach ($candidateDir in @($featherOrig, $featherUserMods)) {
            if (Test-Path -LiteralPath $candidateDir) { $cleanDirs += $candidateDir }
        }
    }
    foreach ($dir in ($cleanDirs | Select-Object -Unique)) {
        $staleJars = @(Get-ChildItem -Path $dir -Filter "judas-bridge-*.jar" -File)
        foreach ($stale in $staleJars) {
            $staleJar = $stale.FullName
            $disabled = Join-Path $dir ($stale.Name + ".disabled-$stamp")
            Wait-JudasJarWritable $staleJar
            try {
                Move-Item -LiteralPath $staleJar -Destination $disabled -Force
            } catch {
                throw "Impossible de desactiver l'ancien jar Judas: $staleJar. Ferme Minecraft puis relance le script de preparation avec -StopMinecraft."
            }
            Write-Host "[build_mod] ancien jar Judas desactive -> $disabled" -ForegroundColor Yellow
        }
    }
    $sourceHash = (Get-FileHash -LiteralPath $jar -Algorithm SHA256).Hash
    $deployTargets = @(
        @{ Dir = $ModsDir; Name = (Split-Path $jar -Leaf) }
    )
    if ($featherMode) {
        $deployTargets = @(
            @{ Dir = $ModsDir; Name = "judas-bridge-0.1.0.temp.jar" }
        )
        foreach ($candidateDir in @($featherOrig, $featherUserMods)) {
            if (Test-Path -LiteralPath $candidateDir) {
                $deployTargets += @{ Dir = $candidateDir; Name = (Split-Path $jar -Leaf) }
            }
        }
    }
    foreach ($target in $deployTargets) {
        $destJar = Join-Path ([string]$target.Dir) ([string]$target.Name)
        Copy-Item $jar -Destination $destJar -Force
        $destHash = (Get-FileHash -LiteralPath $destJar -Algorithm SHA256).Hash
        if ($sourceHash -ne $destHash) {
            throw "Jar deploye non identique: $destJar. source=$sourceHash dest=$destHash"
        }
        Write-Host "[build_mod] deploye -> $destJar sha256=$destHash" -ForegroundColor Green
    }
}
