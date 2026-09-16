#!/bin/bash

# Set the path of the directory that contains the python script
SCRIPT_PATH=$(cd "$(dirname "$0")"; pwd -P)

# Change into the directory that contains the python script
cd "$SCRIPT_PATH"

# Run the Python script
uv run mocap_classifier_interactive_lstm.py

# Exit shell
exit 0