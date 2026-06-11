@echo off
setlocal
cd /d "%~dp0.."

if /i not "%JUDAS_SKIP_DAEMON%"=="1" (
  netstat -ano -p tcp | findstr /R /C:":8765 .*LISTENING" >nul
  if errorlevel 1 (
    echo Starting judas daemon on 127.0.0.1:8765...
    start "judas-daemon" /min cmd /k scripts\daemon.bat
    timeout /t 2 /nobreak >nul
  )
)

cd /d "%~dp0..\app"
if not exist node_modules call npm install --no-audit --no-fund
call npm run dev
