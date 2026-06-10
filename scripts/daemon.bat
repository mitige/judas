@echo off
call "%~dp0env.bat"
python -m serve.daemon %*
