@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop_judas_ui.ps1" %*
exit /b %ERRORLEVEL%
