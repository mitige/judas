@echo off
rem Environnement commun : venv + MSVC (pour le JIT CUDA) + arch 3060
cd /d "%~dp0.."
call .venv\Scripts\activate.bat
set TORCH_CUDA_ARCH_LIST=8.6

where cl >nul 2>nul
if not errorlevel 1 goto :eof

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" (
  for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find Common7\Tools\VsDevCmd.bat`) do (
    call "%%i" -arch=x64 >nul
  )
)

where cl >nul 2>nul
if errorlevel 1 echo [!] MSVC introuvable : installer "Build Tools C++" ou lancer depuis "x64 Native Tools Command Prompt"
