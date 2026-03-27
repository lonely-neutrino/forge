#!/bin/bash
# PPO self-play training with visual dashboard.
# Usage: ./ppo_train.sh [rounds] [games_per_round]
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON="$PROJECT_ROOT/forge-ai-rl/venv/bin/python3"
PYTHON_DIR="$PROJECT_ROOT/forge-ai-rl/src/main/python"
SAVE_DIR="$PROJECT_ROOT/rl_data/checkpoints"
CHECKPOINT="$SAVE_DIR/model_with_decisions.pt"
ROUNDS="${1:-20}"
GAMES="${2:-200}"

if [ ! -f "$CHECKPOINT" ]; then
    echo "No model checkpoint at $CHECKPOINT"
    echo "Train decision heads first."
    exit 1
fi

DEVICE=$("$PYTHON" -c "import sys; sys.path.insert(0, r'$PYTHON_DIR'); from model.backend import auto_device_name; print(auto_device_name())" 2>/dev/null || echo cpu)

cd "$PYTHON_DIR"
"$PYTHON" training/ppo_ui.py \
    --checkpoint "$CHECKPOINT" \
    --save-dir "$SAVE_DIR" \
    --device "$DEVICE" \
    --rounds "$ROUNDS" \
    --games-per-round "$GAMES" \
    --eval-games 50
