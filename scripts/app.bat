@echo off
setlocal
cd /d "%~dp0.."

if /i not "%JUDAS_SKIP_UI_STOP%"=="1" (
  call scripts\stop_judas_ui.bat -Surface app
  if errorlevel 1 exit /b %ERRORLEVEL%
)

if /i not "%JUDAS_SKIP_DAEMON%"=="1" (
  call scripts\daemon.bat
  if errorlevel 1 exit /b %ERRORLEVEL%
)

cd /d "%~dp0..\app"
if not exist node_modules call npm install --no-audit --no-fund
call npm run dev
