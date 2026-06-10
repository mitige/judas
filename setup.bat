@echo off
setlocal
cd /d "%~dp0"
echo === Judas setup (CUDA 12.8/12.9, RTX 3060) ===

where python >nul 2>nul || (echo [!] Python 3.10+ requis dans le PATH & exit /b 1)

if not exist .venv (
  echo - creation du venv...
  python -m venv .venv
)
call .venv\Scripts\activate.bat

echo - installation de PyTorch cu128...
python -m pip install --upgrade pip -q
pip install torch --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (echo [!] echec installation torch cu128 & exit /b 1)

echo - installation de Judas...
pip install -e .[dev] -q

echo - verification...
python -c "import torch; print('torch', torch.__version__, '^| cuda', torch.version.cuda, '^| gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'ABSENT')"

echo.
echo Setup termine. Lancer run.bat
endlocal
