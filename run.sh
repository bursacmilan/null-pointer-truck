#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
    for cand in python3.13 python3.12 python3.11 python3.14 python3; do
        if command -v "$cand" >/dev/null 2>&1; then
            ver=$("$cand" -c 'import sys; print(sys.version_info[:2] >= (3,10))')
            if [ "$ver" = "True" ]; then
                PYTHON_BIN="$cand"
                break
            fi
        fi
    done
fi
if [ -z "$PYTHON_BIN" ]; then
    echo "Error: need Python >= 3.10 (pywhispercpp requirement). Install with: brew install python@3.13"
    exit 1
fi

if [ ! -d .venv ]; then
    echo "Creating virtual environment ($PYTHON_BIN)..."
    "$PYTHON_BIN" -m venv .venv
    .venv/bin/pip install -q --upgrade pip
fi

# Always sync deps — fast when already satisfied, picks up changes to requirements.txt.
if [ requirements.txt -nt .venv/.deps-installed ] || [ ! -f .venv/.deps-installed ]; then
    echo "Installing/updating Python dependencies..."
    .venv/bin/pip install -q -r requirements.txt
    touch .venv/.deps-installed
fi

if [ -z "${TEAM_ID:-}" ]; then
    echo "Error: TEAM_ID is not set."
    echo "  export TEAM_ID=your-team-name"
    exit 1
fi

if [ "${STRATEGY:-llm}" != "dummy" ] && [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "Error: ANTHROPIC_API_KEY is not set (required for the LLM strategy)."
    echo "  export ANTHROPIC_API_KEY=sk-ant-..."
    echo "  (or run with STRATEGY=dummy to bypass the LLM for plumbing tests)"
    exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Warning: ffmpeg not found on PATH. whisper.cpp needs it to decode mp3 audio."
    echo "  Install with: brew install ffmpeg"
fi

exec .venv/bin/python main.py
