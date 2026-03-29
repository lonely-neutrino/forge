#!/bin/bash
# Step 10: Open the side-by-side replay diff viewer
# Usage:
#   sh 10_view_replay_diff.sh [replay_id] [trace_dir] [seat]
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

normalize_path() {
    local path="$1"
    if [ -z "$path" ]; then
        printf '%s' "$path"
        return
    fi
    if [[ "$PYTHON" == *.exe ]] && command -v cygpath >/dev/null 2>&1; then
        cygpath -m "$path" 2>/dev/null || printf '%s' "$path"
    else
        printf '%s' "$path"
    fi
}

REPLAY_ID="${1:-demo_seed_12345}"
TRACE_DIR="${2:-$PROJECT_ROOT/rl_data/replays}"
SEAT="${3:-1}"

TRACE_DIR_PY="$(normalize_path "$TRACE_DIR")"

echo "Opening replay diff viewer..."
echo "  Replay ID: $REPLAY_ID"
echo "  Trace dir: $TRACE_DIR"
echo "  Seat: $SEAT"

cd "$PYTHON_DIR"
"$PYTHON" training/visualize_game_state.py \
  --replay-trace-dir "$TRACE_DIR_PY" \
  --replay-id "$REPLAY_ID" \
  --seat "$SEAT"
