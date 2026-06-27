@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop_judas_train.ps1" %*
exit /b %ERRORLEVEL%
