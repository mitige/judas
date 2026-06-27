@echo off
rem Wrapper : lance judas_live.ps1 sans modifier l'ExecutionPolicy du systeme.
rem Le flag -ExecutionPolicy Bypass ne vaut que pour ce processus PowerShell.
rem Usage : judas_live.bat -Server 192.168.1.50 -Port 25565
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0judas_live.ps1" %*
