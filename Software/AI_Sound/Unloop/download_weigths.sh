#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="models/unloop"
REPO_ID="hugggof/vampnet"
FILES=("codec.pth" "coarse.pth" "c2f.pth" "wavebeat.pth")

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
