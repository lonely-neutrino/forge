#!/bin/bash
# Train decision heads with visual dashboard.
# Usage: ./train_decisions_ui.sh [epochs]
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON="$PROJECT_ROOT/forge-ai-rl/venv/bin/python3"
PYTHON_DIR="$PROJECT_ROOT/forge-ai-rl/src/main/python"
DATA_DIR="$PROJECT_ROOT/rl_data/trajectories"
SAVE_DIR="$PROJECT_ROOT/rl_data/checkpoints"
ENCODER="$SAVE_DIR/best_value_model.pt"
EPOCHS="${1:-50}"

if [ ! -f "$ENCODER" ]; then
    echo "No encoder checkpoint at $ENCODER"
    echo "Train the value network first: ./train_ui.sh"
    exit 1
fi

DEVICE=$("$PYTHON" -c "import sys; sys.path.insert(0, r'$PYTHON_DIR'); from model.backend import auto_device_name; print(auto_device_name())" 2>/dev/null || echo cpu)

cd "$PYTHON_DIR"
"$PYTHON" training/train_decisions_ui.py \
    --data-dir "$DATA_DIR" \
    --save-dir "$SAVE_DIR" \
    --encoder-checkpoint "$ENCODER" \
    --epochs "$EPOCHS" \
    --device "$DEVICE"
