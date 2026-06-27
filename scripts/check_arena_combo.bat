@echo off
rem Wrapper : verifie que l'arene combo ne retombe pas en draw miroir.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0check_arena_combo.ps1" %*