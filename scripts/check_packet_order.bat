@echo off
rem Wrapper : analyse le log packet-order sans modifier l'ExecutionPolicy.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0check_packet_order.ps1" %*
