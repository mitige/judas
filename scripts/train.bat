@echo off
if /i not "%JUDAS_SKIP_TRAIN_STOP%"=="1" (
  call "%~dp0stop_judas_train.bat"
  if errorlevel 1 exit /b %ERRORLEVEL%
)
call "%~dp0env.bat" || exit /b 1
python -m train.run --config train/configs/boxing.json %*
