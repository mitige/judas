@echo off
rem Wrapper : analyse le log aim OS sans modifier l'ExecutionPolicy.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0check_aim_os.ps1" %*