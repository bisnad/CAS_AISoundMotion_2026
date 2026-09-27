@echo off
setlocal

set "VENV_DIR=%~1"
if "%VENV_DIR%"=="" set "VENV_DIR=venv_unloop"

where py >nul 2>nul
if %errorlevel%==0 (
    py -3 -m venv "%VENV_DIR%"
) else (
    python -m venv "%VENV_DIR%"
)

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Could not find an activation script in %VENV_DIR% 1>&2
    exit /b 1
)

call "%VENV_DIR%\Scripts\activate.bat"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .

echo Virtual environment ready in %VENV_DIR%
