#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f "pyproject.toml" ]]; then
    echo "Error: pyproject.toml not found in $SCRIPT_DIR" >&2
    echo "This script must live in, and be run from, the project root." >&2
    exit 1
fi

MODEL_DIR="models/unloop"
REPO_ID="hugggof/vampnet"
FILES=("codec.pth" "coarse.pth" "c2f.pth" "wavebeat.pth")

if ! command -v ffprobe &> /dev/null; then
    echo "ffprobe not found. Install FFmpeg first, e.g.: brew install ffmpeg" >&2
    exit 1
fi

mkdir -p "$MODEL_DIR"
uv sync

for f in "${FILES[@]}"; do
    if [[ -f "$MODEL_DIR/$f" ]]; then
        echo "skip: $f already present"
    else
        echo "fetching: $f"
        uv run hf download "$REPO_ID" "$f" --local-dir "$MODEL_DIR"
    fi
done

echo "Environment and model weights ready in $MODEL_DIR."
