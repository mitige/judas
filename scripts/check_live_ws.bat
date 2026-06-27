@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0check_live_ws.ps1" %*
exit /b %ERRORLEVEL%