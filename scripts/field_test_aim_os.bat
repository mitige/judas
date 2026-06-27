@echo off
rem Wrapper : flux terrain complet aim OS (deploy jar puis watch PRECISE).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0field_test_aim_os.ps1" %*