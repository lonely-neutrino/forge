#!/bin/bash
# Step 7: AWR (Advantage-Weighted Regression) offline RL training
# Alternative to PPO - collects data under argmax (full strength play)
# Usage: 07_awr_train.sh [model] [rounds] [games] [device]
set -e

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/rl_common.sh"
PYTHON="$(rl_find_python)"
cd "$PYTHON_DIR"

SAVE_DIR="$FORGE_RL_CHECKPOINT_DIR"

if [ -n "${1:-}" ]; then
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
DEVICE=${4:-$(rl_auto_device)}

echo "AWR Training: $ROUNDS rounds, $GAMES games/round (ARGMAX collection)"
echo "Model: $MODEL"
echo ""

"$PYTHON" training/awr_trainer.py \
    --checkpoint "$(rl_to_python_path "$MODEL")" \
    --save-dir "$(rl_to_python_path "$SAVE_DIR")" \
    --device "$DEVICE" \
    --rounds "$ROUNDS" \
    --games-per-round "$GAMES" \
    --eval-games 50 \
    --awr-epochs 4 \
    --batch-size 64 \
    --lr 1e-4 \
    --temperature 2.0 \
    --port 0
