@echo off
setlocal EnableDelayedExpansion

REM ===========================================================
REM copy_pyproject_files.bat
REM
REM Copies a set of pyproject.toml source files from the folder
REM this script lives in (the "Installers" folder) into their
REM respective destination folders, renaming them in the process.
REM Prints the source file, target folder, and target file name
REM for each copy.
REM
REM Works both when run from a command prompt (cd + the script)
REM and when double-clicked from File Explorer, because it
REM resolves its own location first and switches into it.
REM ===========================================================

REM --- Resolve the directory this script lives in and switch into it ---
set "SCRIPT_DIR=%~dp0"
REM Strip trailing backslash for cleaner display
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
cd /d "%SCRIPT_DIR%"

echo Running from: %SCRIPT_DIR%
echo.

REM --- Each entry: source_file^|target_folder^|target_filename ---
set "FILES[0]=main_pyproject.windows-cu126.toml|..\Software|pyproject.toml"
set "FILES[1]=aiforsound_pyproject.toml|..\Software\AI_Sound|pyproject.toml"
set "FILES[2]=aiformotion_pyproject.toml|..\Software\AI_Motion|pyproject.toml"
set "FILES[3]=stableaudio3_pyproject.windows-cu126.toml|..\Software\AI_Sound\StableAudio3|pyproject.toml"
set "FILES[4]=mediapipe_pyproject.windows-cpu.toml|..\Software\AI_Motion\MotionTracking\Mediapipe|pyproject.toml"
set "FILES[5]=yolo_pyproject.windows-cu126.toml|..\Software\AI_Motion\MotionTracking\Yolo|pyproject.toml"
set "FILES[6]=bark_tts_pyproject.windows_cu118.toml|..\Software\AI_Sound\BarkTTS|pyproject.toml"
set "FILES[7]=dancediffusion_pyproject_windows_cu118.toml|..\Software\AI_Sound\DanceDiffusion|pyproject.toml"

for /L %%i in (0,1,7) do (
    for /f "tokens=1,2,3 delims=|" %%a in ("!FILES[%%i]!") do (
        set "SRC_FILE=%%a"
        set "TARGET_FOLDER=%%b"
        set "TARGET_FILE=%%c"

        echo Copying:
        echo   Source file:   !SRC_FILE!
        echo   Target folder: !TARGET_FOLDER!
        echo   Target file:   !TARGET_FILE!

        if not exist "!SRC_FILE!" (
            echo   WARNING: Source file "!SRC_FILE!" not found. Skipping. 1>&2
            echo.
        ) else (
            if not exist "!TARGET_FOLDER!" mkdir "!TARGET_FOLDER!"
            copy /Y "!SRC_FILE!" "!TARGET_FOLDER!\!TARGET_FILE!" >nul

            echo   Done.
            echo.
        )
    )
)

echo All copy operations completed.

REM Keep the window open when launched by double-clicking,
REM so you can read the output before it closes automatically.
pause >nul
