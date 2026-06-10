@echo off
call "%~dp0env.bat"
python -m pytest tests
