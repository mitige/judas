@echo off
call "%~dp0env.bat" || exit /b 1
python -m train.run --config train/configs/boxing.json %*
