@echo off
setlocal
cd /d "%~dp0"
echo === Judas setup (GPU NVIDIA, CUDA 12.8/12.9) ===

set "PYTHON_CMD="
where py >nul 2>nul
if not errorlevel 1 (
  py -3.11 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
  if not errorlevel 1 set "PYTHON_CMD=py -3.11"
)

if not defined PYTHON_CMD (
  where python >nul 2>nul || (echo [!] Python 3.10+ requis dans le PATH & exit /b 1)
  python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
  if errorlevel 1 (
    echo [!] Python 3.10+ requis. Le python du PATH est trop ancien.
    echo     Installer Python 3.11 ou lancer setup.bat depuis un PATH Python 3.10+.
    exit /b 1
  )
  set "PYTHON_CMD=python"
)

if not exist .venv (
  echo - creation du venv...
  %PYTHON_CMD% -m venv .venv
)
call .venv\Scripts\activate.bat || (echo [!] impossible d'activer .venv & exit /b 1)

python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if errorlevel 1 (
  echo [!] .venv utilise Python ^< 3.10.
  echo     Supprimer puis recreer le venv :
  echo       rmdir /s /q .venv
  echo       setup.bat
  exit /b 1
)

echo - installation de PyTorch cu128...
python -m pip install --upgrade pip -q
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (echo [!] echec installation torch cu128 & exit /b 1)

echo - installation de Judas...
python -m pip install -e ".[dev]" -q
if errorlevel 1 (echo [!] echec installation Judas & exit /b 1)

echo - verification...
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'ABSENT')"

echo.
echo Setup termine. Lancer run.bat
endlocal
