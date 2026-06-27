@echo off
rem Wrapper : surveille le log aim OS sans modifier l'ExecutionPolicy.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0watch_aim_os.ps1" %*