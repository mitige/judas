@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0check_field_status.ps1" %*
