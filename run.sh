#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d .venv ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    .venv/bin/pip install -q -r requirements.txt
fi

if [ -z "${TEAM_ID:-}" ]; then
    echo "Error: TEAM_ID is not set."
    echo "  export TEAM_ID=your-team-name"
    exit 1
fi

exec .venv/bin/python main.py
