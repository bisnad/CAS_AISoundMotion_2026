#!/usr/bin/env bash
set -euo pipefail

venv_dir="${1:-venv_unloop}"

python -m venv "$venv_dir"
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
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .

echo "Virtual environment ready in $venv_dir"