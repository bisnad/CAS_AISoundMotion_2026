#!/usr/bin/env bash
#
# copy_pyproject_files.sh
#
# Copies a set of pyproject.toml source files from the current folder
# into their respective destination folders, renaming them in the
# process. Prints the source file, target folder, and target file
# name for each copy operation.

set -euo pipefail

# Each entry: "source_file|target_folder|target_filename"
FILES=(
    "main_pyproject.macos-cpu.toml|../Software|pyproject.toml"
    "aiforsound_pyproject.toml|../Software/AI_Sound|pyproject.toml"
    "aiformotion_pyproject.toml|../Software/AI_Motion|pyproject.toml"
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
