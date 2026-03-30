#!/bin/bash
# Step 7: AWR (Advantage-Weighted Regression) offline RL training
# Alternative to PPO — collects data under argmax (full strength play)
# Usage: 07_awr_train.sh [model] [rounds] [games] [device]
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

SAVE_DIR="$PROJECT_ROOT/rl_data/checkpoints"

if [ -n "$1" ]; then
    MODEL="$1"
elif [ -f "$SAVE_DIR/awr_model_latest.pt" ]; then
    MODEL="$SAVE_DIR/awr_model_latest.pt"
    echo "Resuming from latest AWR checkpoint"
elif [ -f "$SAVE_DIR/model_with_decisions.pt" ]; then
    MODEL="$SAVE_DIR/model_with_decisions.pt"
    echo "Starting from imitation-learned model"
else
    echo "No model found"
    exit 1
fi

ROUNDS=${2:-50}
GAMES=${3:-100}
DEVICE_ARG=${4:-}
if [ -n "$DEVICE_ARG" ]; then
    DEVICE="$DEVICE_ARG"
else
    DEVICE=$("$PYTHON" -c "import sys; sys.path.insert(0, r'$PYTHON_DIR'); from model.backend import auto_device_name; print(auto_device_name())" 2>/dev/null || echo cpu)
fi

if [[ "$PYTHON" == *.exe ]] && command -v cygpath >/dev/null 2>&1; then
    MODEL_PY="$(cygpath -m "$MODEL")"
    SAVE_DIR_PY="$(cygpath -m "$SAVE_DIR")"
else
    MODEL_PY="$MODEL"
    SAVE_DIR_PY="$SAVE_DIR"
fi

echo "AWR Training: $ROUNDS rounds, $GAMES games/round (ARGMAX collection)"
echo "Model: $MODEL"
echo "Device: $DEVICE"
echo ""

"$PYTHON" training/awr_trainer.py \
    --checkpoint "$MODEL_PY" \
    --save-dir "$SAVE_DIR_PY" \
    --device "$DEVICE" \
    --rounds "$ROUNDS" \
    --games-per-round "$GAMES" \
    --eval-games 50 \
    --awr-epochs 4 \
    --batch-size 64 \
    --lr 1e-4 \
    --temperature 2.0 \
    --port 0
