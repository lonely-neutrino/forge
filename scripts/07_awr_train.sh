#!/bin/bash
# Step 7: AWR (Advantage-Weighted Regression) offline RL training
# Alternative to PPO — collects data under argmax (full strength play)
# Usage: 07_awr_train.sh [model] [rounds] [games] [device]
set -e
cd /home/maustin/forge/forge-ai-rl/src/main/python
source /home/maustin/forge/forge-ai-rl/venv/bin/activate

SAVE_DIR=/home/maustin/forge/rl_data/checkpoints

# Auto-select checkpoint
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
DEVICE=${4:-dml}

echo "AWR Training: $ROUNDS rounds, $GAMES games/round (ARGMAX collection)"
echo "Model: $MODEL"
echo ""

python training/awr_trainer.py \
    --checkpoint "$MODEL" \
    --save-dir "$SAVE_DIR" \
    --device "$DEVICE" \
    --rounds "$ROUNDS" \
    --games-per-round "$GAMES" \
    --eval-games 50 \
    --awr-epochs 4 \
    --batch-size 64 \
    --lr 1e-4 \
    --temperature 2.0 \
    --port 0
