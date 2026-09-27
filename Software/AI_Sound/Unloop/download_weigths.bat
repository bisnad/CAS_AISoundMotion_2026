@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

if not exist "pyproject.toml" (
    echo Error: pyproject.toml not found in %SCRIPT_DIR% 1>&2
    echo This script must live in, and be run from, the project root. 1>&2
    exit /b 1
)

set "MODEL_DIR=models\unloop"
set "REPO_ID=hugggof/vampnet"
set "FILES=codec.pth coarse.pth c2f.pth wavebeat.pth"

where ffprobe >nul 2>nul
if errorlevel 1 (
    echo ffprobe not found. Install FFmpeg first ^(e.g. via the project's 1>&2
    echo static-ffmpeg dependency, or download a build from ffmpeg.org^) 1>&2
    exit /b 1
)

if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"

uv sync
if errorlevel 1 exit /b 1

for %%F in (%FILES%) do (
    if exist "%MODEL_DIR%\%%F" (
        echo skip: %%F already present
    ) else (
        echo fetching: %%F
        uv run hf download "%REPO_ID%" "%%F" --local-dir "%MODEL_DIR%"
        if errorlevel 1 exit /b 1
    )
)

echo Environment and model weights ready in %MODEL_DIR%.
endlocal
