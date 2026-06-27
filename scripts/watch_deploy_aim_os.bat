@echo off
rem Wrapper : attend que Minecraft libere la jar puis prepare l'aim OS.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0watch_deploy_aim_os.ps1" %*