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
echo   [7] combo god consistent (8 iters, force stop old)
echo   [8] stop combo god
echo   [9] field preflight (no start)
echo   [p] combo proof local (no training)
echo   [f] field proof quick (deployed mod, no rebuild)
echo   [0] stop app/viz/live/train
echo   [q] quitter
echo  ===============================
set /p c=^> 
if "%c%"=="1" call scripts\daemon.bat
if "%c%"=="2" (
  call scripts\stop_judas_train.bat
  call scripts\stop_combo_god.bat
  start "judas-train" cmd /k scripts\train.bat
)
if "%c%"=="3" start "judas-app" cmd /k scripts\app.bat
if "%c%"=="4" start "judas-viz" cmd /k scripts\viz.bat
if "%c%"=="5" call scripts\tests.bat
if "%c%"=="6" call scripts\verify.bat
if "%c%"=="7" (
  call scripts\start_combo_god.bat -Force -Iters 8 -TimeoutMinutes 20
  call scripts\status_combo_god.bat -Tail 3
)
if "%c%"=="8" (
  call scripts\stop_combo_god.bat
  call scripts\status_combo_god.bat -Tail 3
)
if "%c%"=="9" call scripts\check_field_preflight.bat
if /i "%c%"=="p" call scripts\prove_combo_god.bat
if /i "%c%"=="f" call scripts\field_test_aim_os_quick.bat
if "%c%"=="0" (
  call scripts\stop_judas_ui.bat -Surface all
  call scripts\stop_judas_live.bat
  call scripts\stop_judas_train.bat
  call scripts\stop_combo_god.bat
)
if /i "%c%"=="q" exit /b 0
goto menu
