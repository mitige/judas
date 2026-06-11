@echo off
call "%~dp0env.bat" || exit /b 1
echo === equivalence sim_ref ^<-^> CUDA (build double) ===
python -m sim.verify || exit /b 1
echo.
echo === benchmark (build float32) ===
python -m sim.bench || exit /b 1
