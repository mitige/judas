@echo off
rem Wrapper : surveille le log packet-order sans modifier l'ExecutionPolicy.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0watch_packet_order.ps1" %*
