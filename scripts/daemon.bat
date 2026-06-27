@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_judas_daemon.ps1" %*
exit /b %ERRORLEVEL%
