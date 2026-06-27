@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0check_field_preflight.ps1" %*
exit /b %ERRORLEVEL%
