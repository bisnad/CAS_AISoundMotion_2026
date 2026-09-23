#!/bin/bash

# Set the path of the directory that contains the python script
SCRIPT_PATH=$(cd "$(dirname "$0")"; pwd -P)

# Change into the directory that contains the python script
cd "$SCRIPT_PATH"

# Run the Python script
uv run stable_audio_3_hello_world.py

# Exit shell
exit 0