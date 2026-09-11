#!/usr/bin/env bash
#
# copy_pyproject_files.sh
#
# Copies a set of pyproject.toml source files from the folder this
# script lives in (the "Installers" folder) into their respective
# destination folders, renaming them in the process. Prints the
# source file, target folder, and target file name for each copy.
#
# Works both when run from a Terminal (cd + ./copy_pyproject_files.sh)
# and when double-clicked from a file manager, because it resolves
# its own location first and switches into it.

set -euo pipefail

# --- Resolve the real directory this script lives in, following symlinks ---
resolve_script_dir() {
    local source="${BASH_SOURCE[0]}"
    while [[ -h "$source" ]]; do
        local dir
        dir="$(cd -P "$(dirname "$source")" >/dev/null 2>&1 && pwd)"
        source="$(readlink "$source")"
        [[ "$source" != /* ]] && source="$dir/$source"
    done
    cd -P "$(dirname "$source")" >/dev/null 2>&1 && pwd
}

SCRIPT_DIR="$(resolve_script_dir)"
cd "$SCRIPT_DIR"

echo "Running from: $SCRIPT_DIR"
echo

# Each entry: "source_file|target_folder|target_filename"
FILES=(
    "main_pyproject.macos-cpu.toml|../Software|pyproject.toml"
    "aiforsound_pyproject.toml|../Software/AI_Sound|pyproject.toml"
    "aiformotion_pyproject.toml|../Software/AI_Motion|pyproject.toml"
    "stableaudio3_pyproject.macos-cpu.toml|../Software/AI_Sound/StableAudio3|pyproject.toml"
    "mediapipe_pyproject.macos-cpu.toml|../Software/AI_Motion/MotionTracking/Mediapipe|pyproject.toml"
    "yolo_pyproject.macos-cpu.toml|../Software/AI_Motion/MotionTracking/Yolo|pyproject.toml"
)

for entry in "${FILES[@]}"; do
    IFS='|' read -r src_file target_folder target_file <<< "$entry"

    echo "Copying:"
    echo "  Source file:   $src_file"
    echo "  Target folder: $target_folder"
    echo "  Target file:   $target_file"

    if [[ ! -f "$src_file" ]]; then
        echo "  WARNING: Source file '$src_file' not found. Skipping." >&2
        echo
        continue
    fi

    mkdir -p "$target_folder"
    cp -- "$src_file" "$target_folder/$target_file"

    echo "  Done."
    echo
done

echo "All copy operations completed."

# Keep the terminal window open when launched by double-clicking,
# so you can read the output before it closes automatically.
read -n 1 -s -r -p "Press any key to close this window..."
echo
