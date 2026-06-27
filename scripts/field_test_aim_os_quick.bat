@echo off
rem Wrapper : flux terrain aim OS rapide, reutilise le jar deja deploye.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0field_test_aim_os.ps1" -UseDeployedMod -RequireMinecraft %*
