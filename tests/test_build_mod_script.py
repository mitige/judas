from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_build_mod_uses_low_memory_gradle_profile_and_rejects_corrupt_jar():
    source = (ROOT / "scripts/build_mod.ps1").read_text()
    guide = (ROOT / "docs/GUIDE.md").read_text()

    assert "-Dorg.gradle.jvmargs=-Xmx512m -XX:CICompilerCount=2 -XX:ActiveProcessorCount=2" in source
    assert "--max-workers=1" in source
    assert "BuildTimeoutSeconds = 300" in source
    assert "Start-Process -FilePath $gradle" in source
    assert "$gradleProc.WaitForExit($BuildTimeoutSeconds * 1000)" in source
    assert "$gradleProc.WaitForExit()" in source
    assert "$exitCode = $gradleProc.ExitCode" in source
    assert "BUILD SUCCESSFUL" in source
    assert "taskkill.exe /PID $gradleProc.Id /T /F" in source
    assert "runs\\build_mod" in source
    assert "gradle_$stamp.out.log" in source
    assert "gradle_$stamp.err.log" in source
    assert "Build timeout apres" in source
    assert "$jarInfo.Length -lt 10000" in source
    assert "Jar invalide" in source
    assert "BuildTimeoutSeconds 600" in guide
    assert "runs\\build_mod\\gradle_*.out.log" in guide


def test_build_mod_disables_stale_judas_jars_before_deploying():
    source = (ROOT / "scripts/build_mod.ps1").read_text()

    assert "judas-bridge-*.jar" in source
    assert "mods.orig" in source
    assert ".feather\\user-mods\\1.8.9" in source
    assert "$featherUserMods" in source
    assert "$featherMode" in source
    assert "judas-bridge-0.1.0.temp.jar" in source
    assert "$deployTargets" in source
    assert "Move-Item" in source
    assert ".disabled-" in source
    assert "$staleJar = $stale.FullName" in source
    assert "[System.IO.FileShare]::None" in source
    assert "Wait-JudasJarWritable" in source
    assert "Start-Sleep -Milliseconds 500" in source
    assert "Ferme Minecraft" in source
    assert "script de preparation avec -StopMinecraft" in source
    assert "Copy-Item $jar -Destination $destJar -Force" in source
    assert "Get-FileHash" in source
    assert "Jar deploye non identique" in source
    assert "sha256=" in source


def test_build_mod_can_stop_minecraft_before_lock_check_when_requested():
    source = (ROOT / "scripts/build_mod.ps1").read_text()

    assert "[switch]$StopMinecraft" in source
    assert "Stop-Process" in source
    assert ".minecraft" in source
    assert '"jre"' in source
    assert "Wait-Process" in source
    assert "Get-CimInstance Win32_Process" in source
    assert "CommandLine" in source
    assert "net\\.minecraft\\.client" in source
    assert "diagnostic process Minecraft limite" in source
    assert "foreach ($procId in $ids)" in source
    assert "foreach ($pid in $ids)" not in source


def test_build_mod_allows_workspace_gradle_home_override():
    source = (ROOT / "scripts/build_mod.ps1").read_text()

    assert "JUDAS_BUILD_TOOLS" in source
    assert "portableTools" in source
    assert "gradleCandidates" in source
    assert "Join-Path $tools \"ghome\"" in source
