@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "VENV_DIR=%~1"
if "%VENV_DIR%"=="" set "VENV_DIR=%SCRIPT_DIR%venv_unloop"

if not exist "%VENV_DIR%" (
    if exist "%SCRIPT_DIR%venv" (
        set "VENV_DIR=%SCRIPT_DIR%venv"
    )
)

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Virtual environment not found: %VENV_DIR% 1>&2
    exit /b 1
)

call "%VENV_DIR%\Scripts\activate.bat"
cd /d "%SCRIPT_DIR%"
python -u app.py
