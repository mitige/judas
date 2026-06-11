@echo off
call "%~dp0env.bat" || exit /b 1
python -m pytest tests || exit /b 1
where node >nul 2>nul || goto :eof
echo === tests Node (persistence) ===
node --test tools/persistence.test.mjs || exit /b 1
