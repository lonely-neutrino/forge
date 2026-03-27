#!/bin/bash
# Step 2: Collect trajectory data + preprocess with live dashboard
# Usage: ./02_collect_data.sh [games] [--clean] [--deck "Deck Name.dck"...]
#   --clean: delete old trajectories and preprocessed data first
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_DIR="$PROJECT_ROOT/forge-ai-rl/src/main/python"
VENV_PY="$PROJECT_ROOT/forge-ai-rl/venv/bin/python3"
VENV_PY_WIN="$PROJECT_ROOT/forge-ai-rl/venv/Scripts/python.exe"
if command -v cygpath >/dev/null 2>&1; then
    USERPROFILE_UNIX="$(cygpath "$USERPROFILE" 2>/dev/null || printf '%s' "$USERPROFILE")"
    CONDA_PREFIX_UNIX="$(cygpath "$CONDA_PREFIX" 2>/dev/null || printf '%s' "$CONDA_PREFIX")"
else
    USERPROFILE_UNIX="$USERPROFILE"
    CONDA_PREFIX_UNIX="$CONDA_PREFIX"
fi
CONDA_PY="$CONDA_PREFIX_UNIX/python"
CONDA_PY_WIN="$CONDA_PREFIX_UNIX/python.exe"
FORGE_RL_CONDA_WIN="$USERPROFILE_UNIX/anaconda3/envs/forge_rl/python.exe"

if [ -x "$VENV_PY" ]; then
    PYTHON="$VENV_PY"
elif [ -f "$VENV_PY_WIN" ]; then
    PYTHON="$VENV_PY_WIN"
elif [ -n "$CONDA_PREFIX" ] && [ -x "$CONDA_PY" ]; then
    PYTHON="$CONDA_PY"
elif [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PY_WIN" ]; then
    PYTHON="$CONDA_PY_WIN"
elif [ -n "$USERPROFILE" ] && [ -f "$FORGE_RL_CONDA_WIN" ]; then
    PYTHON="$FORGE_RL_CONDA_WIN"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
else
    echo "No Python interpreter found."
    exit 1
fi

cd "$PYTHON_DIR"

GAMES=1000
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
    GAMES="$1"
    shift
fi

EXTRA_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --deck)
            if [ -z "$2" ]; then
                echo "Missing deck name after --deck"
                exit 1
            fi
            EXTRA_ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "Launching collection dashboard for $GAMES games..."
"$PYTHON" training/collect_ui.py --games "$GAMES" "${EXTRA_ARGS[@]}"
