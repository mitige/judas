@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0check_safe_export.ps1" %*
exit /b %ERRORLEVEL%
