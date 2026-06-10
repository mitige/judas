@echo off
call "%~dp0env.bat"
echo === equivalence sim_ref ^<-^> CUDA (build double) ===
python -m sim.verify
echo.
echo === benchmark (build float32) ===
python -m sim.bench
