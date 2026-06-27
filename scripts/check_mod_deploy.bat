@echo off
rem Wrapper : verifie que la jar active dans mods/ correspond au build local.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0check_mod_deploy.ps1" %*