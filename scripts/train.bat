@echo off
call "%~dp0env.bat"
python -m train.run --config train/configs/boxing.json %*
