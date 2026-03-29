#!/bin/bash
# Step 6b: Self-play PPO training with Elo tracking
# Usage: 06b_selfplay_train.sh [model] [rounds] [games] [device]
set -e

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/rl_common.sh"
PYTHON="$(rl_find_python)"
cd "$PYTHON_DIR"

pkill -f "model_server\|ppo_ui\|ppo_headless" 2>/dev/null || true
sleep 1

SAVE_DIR="$FORGE_RL_CHECKPOINT_DIR"
LATEST_PPO="$SAVE_DIR/ppo_model_latest.pt"
BEST_PPO="$SAVE_DIR/best_ppo_model.pt"
IMITATION="$SAVE_DIR/model_with_decisions.pt"

if [ -n "${1:-}" ]; then
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
DEVICE=${4:-$(rl_auto_device)}

echo "Self-Play PPO: $ROUNDS rounds, $GAMES games/round"
echo "Model: $MODEL"
echo "Kill anytime - progress saved after each round"
echo ""

"$PYTHON" training/ppo_headless.py \
    --checkpoint "$(rl_to_python_path "$MODEL")" \
    --save-dir "$(rl_to_python_path "$SAVE_DIR")" \
    --traj-dir "$(rl_to_python_path "$FORGE_RL_PPO_TRAJ_DIR")" \
    --eval-dir "$(rl_to_python_path "$FORGE_RL_EVAL_DIR")" \
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
