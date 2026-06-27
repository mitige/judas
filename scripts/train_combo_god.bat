@echo off
rem Legacy safe wrapper. The real launcher writes runs\combo_god_recovery_kb092_combo12\train.pid
rem and stops the previous owned run before starting a new bounded one.
call "%~dp0start_combo_god.bat" -Force -Iters 8 -TimeoutMinutes 20 %*
exit /b %ERRORLEVEL%
