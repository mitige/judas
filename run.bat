@echo off
setlocal
cd /d "%~dp0"
:menu
echo.
echo  ============ JUDAS ============
echo   [1] daemon        (API + live + arena)
echo   [2] training      (PPO self-play)
echo   [3] app           (Electron - controle)
echo   [4] viz           (Electron - arene 3D)
echo   [5] tests
echo   [6] verify + bench
echo   [q] quitter
echo  ===============================
set /p c=^> 
if "%c%"=="1" start "judas-daemon" cmd /k scripts\daemon.bat
if "%c%"=="2" start "judas-train" cmd /k scripts\train.bat
if "%c%"=="3" start "judas-app" cmd /k scripts\app.bat
if "%c%"=="4" start "judas-viz" cmd /k scripts\viz.bat
if "%c%"=="5" call scripts\tests.bat
if "%c%"=="6" call scripts\verify.bat
if /i "%c%"=="q" exit /b 0
goto menu
