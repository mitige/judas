@echo off
rem Wrapper : compile le mod sans modifier l'ExecutionPolicy du systeme.
rem Usage : build_mod.bat   (ou build_mod.bat -Clean)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_mod.ps1" %*
