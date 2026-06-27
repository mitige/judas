@echo off
setlocal
rem Build le mod, le copie dans mods/, puis reset le log packet-order.
set "SCRIPT_DIR=%~dp0"
set "MODS_DIR=%APPDATA%\.minecraft\mods"
set "STOP_MINECRAFT="
set "BUILD_TIMEOUT=300"

:parse_args
if "%~1"=="" goto run_prepare
if /I "%~1"=="-StopMinecraft" (
  set "STOP_MINECRAFT=-StopMinecraft"
  shift
  goto parse_args
)
if /I "%~1"=="-BuildTimeoutSeconds" (
  set "BUILD_TIMEOUT=%~2"
  shift
  shift
  goto parse_args
)
set "MODS_DIR=%~1"
shift
goto parse_args

:run_prepare
call "%SCRIPT_DIR%build_mod.bat" %STOP_MINECRAFT% -Clean -ModsDir "%MODS_DIR%" -BuildTimeoutSeconds %BUILD_TIMEOUT%
if errorlevel 1 exit /b %errorlevel%

call "%SCRIPT_DIR%check_packet_order.bat" -Reset
if errorlevel 1 exit /b %errorlevel%

echo [packet_order] Pret: jar deploye dans "%MODS_DIR%" et log reset.
exit /b 0
