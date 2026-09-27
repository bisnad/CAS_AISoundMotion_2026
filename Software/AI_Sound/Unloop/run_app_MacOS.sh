#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv_dir="${1:-$script_dir/venv_unloop}"

if [[ ! -d "$venv_dir" && -d "$script_dir/venv" ]]; then
  venv_dir="$script_dir/venv"
fi

if [[ ! -d "$venv_dir" ]]; then
  echo "Virtual environment not found: $venv_dir" >&2
  exit 1
fi

if [[ -f "$venv_dir/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "$venv_dir/bin/activate"
elif [[ -f "$venv_dir/Scripts/activate" ]]; then
  # shellcheck disable=SC1091
  source "$venv_dir/Scripts/activate"
else
  echo "Could not find an activation script in $venv_dir" >&2
  exit 1
fi

cd "$script_dir"
python -u app.py