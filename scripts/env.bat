@echo off
rem Environnement commun : venv + MSVC (pour le JIT CUDA).
rem L'arch CUDA est auto-detectee par sim/judas_sim.py (TORCH_CUDA_ARCH_LIST
rem peut etre exportee manuellement pour forcer une arch).
cd /d "%~dp0.."
if not exist .venv\Scripts\activate.bat (
  echo [!] .venv introuvable : lancer setup.bat depuis la racine du projet.
  exit /b 1
)
call .venv\Scripts\activate.bat || exit /b 1

python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if errorlevel 1 (
  echo [!] .venv utilise Python ^< 3.10, incompatible avec ce projet.
  echo     Supprimer puis recreer le venv :
  echo       rmdir /s /q .venv
  echo       setup.bat
  exit /b 1
)

where cl >nul 2>nul
if not errorlevel 1 goto :eof

set "VS_INSTALLER=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer"
if not exist "%VS_INSTALLER%\vswhere.exe" goto :msvc_check
rem VsDevCmd appelle vswhere.exe sans chemin : l'Installer doit etre sur le PATH
set "PATH=%VS_INSTALLER%;%PATH%"
for /f "usebackq tokens=*" %%i in (`vswhere.exe -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find Common7\Tools\VsDevCmd.bat`) do call "%%i" -arch=x64 >nul

:msvc_check
where cl >nul 2>nul
if errorlevel 1 echo [!] MSVC introuvable : installer "Build Tools C++" ou lancer depuis "x64 Native Tools Command Prompt"
