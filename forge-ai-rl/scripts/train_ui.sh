#!/bin/bash
# Launch training with visual dashboard.
# Usage: ./train_ui.sh [epochs] [max_files]
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON="$PROJECT_ROOT/forge-ai-rl/venv/bin/python3"
PYTHON_DIR="$PROJECT_ROOT/forge-ai-rl/src/main/python"
DATA_DIR="$PROJECT_ROOT/rl_data/trajectories"
SAVE_DIR="$PROJECT_ROOT/rl_data/checkpoints"
EPOCHS="${1:-50}"
MAX_FILES="${2:-}"

NUM_FILES=$(ls "$DATA_DIR"/traj_*.jsonl 2>/dev/null | wc -l)
if [ "$NUM_FILES" -eq 0 ]; then
    echo "No trajectory data in $DATA_DIR. Run collect_data.sh first."
    exit 1
fi
echo "Found $NUM_FILES trajectory files"

EXTRA_ARGS=""
if [ -n "$MAX_FILES" ]; then
    EXTRA_ARGS="--max-files $MAX_FILES"
fi

DEVICE=$("$PYTHON" -c "import sys; sys.path.insert(0, r'$PYTHON_DIR'); from model.backend import auto_device_name; print(auto_device_name())" 2>/dev/null || echo cpu)

cd "$PYTHON_DIR"
"$PYTHON" training/training_ui.py \
    --data-dir "$DATA_DIR" \
    --save-dir "$SAVE_DIR" \
    --epochs "$EPOCHS" \
    --device "$DEVICE" \
    $EXTRA_ARGS
