#!/bin/bash

# Detect if running in Docker or locally
if [ -f "/app/src/cli.py" ]; then
    # Running in Docker
    python3 -W ignore /app/src/cli.py
else
    # Running locally
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    python3 -W ignore "$SCRIPT_DIR/src/cli.py"
fi