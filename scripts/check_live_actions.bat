@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0check_live_actions.ps1" %*
