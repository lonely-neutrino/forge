#!/bin/bash
# Step 9: Self-play PPO training with Elo tracking
# Usage: 09_selfplay_train.sh [model] [rounds] [games] [device]
#
# Both players are the RL model — guarantees 50% win rate for balanced signal.
# Periodically evaluates vs heuristic to measure absolute strength.
# Kill anytime (Ctrl+C) — progress is saved after each round.
set -e
cd /home/maustin/forge/forge-ai-rl/src/main/python
source /home/maustin/forge/forge-ai-rl/venv/bin/activate

pkill -f "model_server\|ppo_ui" 2>/dev/null || true
sleep 1

SAVE_DIR=/home/maustin/forge/rl_data/checkpoints
LATEST_PPO="$SAVE_DIR/ppo_model_latest.pt"
BEST_PPO="$SAVE_DIR/best_ppo_model.pt"
IMITATION="$SAVE_DIR/model_with_decisions.pt"

# Auto-resume: latest > best > imitation
if [ -n "$1" ]; then
    MODEL="$1"
elif [ -f "$LATEST_PPO" ]; then
    MODEL="$LATEST_PPO"
    echo "Resuming from latest PPO checkpoint"
elif [ -f "$BEST_PPO" ]; then
    MODEL="$BEST_PPO"
    echo "Resuming from best PPO checkpoint"
else
    MODEL="$IMITATION"
    echo "Starting from imitation-learned model"
fi

ROUNDS=${2:-100}
GAMES=${3:-400}
DEVICE=${4:-dml}

echo "Self-Play PPO: $ROUNDS rounds, $GAMES games/round"
echo "Model: $MODEL"
echo "Kill anytime — progress saved after each round"
echo ""

python training/ppo_ui.py \
    --checkpoint "$MODEL" \
    --save-dir "$SAVE_DIR" \
    --device "$DEVICE" \
    --rounds "$ROUNDS" \
    --games-per-round "$GAMES" \
    --eval-games 100 \
    --ppo-epochs 4 \
    --batch-size 64 \
    --lr 1e-5 \
    --port 0 \
    --threads 32 \
    --servers 2 \
    --java-procs 2 \
    --collect-mode selfplay \
    --eval-interval 1
