#!/bin/bash
# Step 6: PPO training (headless, QUEST-friendly)
# Usage: 06_ppo_train.sh [model] [rounds] [games] [device] [--deck "Deck Name.dck"...]
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
SAVE_DIR_PY="$(rl_to_python_path "$SAVE_DIR")"
PPO_TRAJ_DIR_PY="$(rl_to_python_path "$FORGE_RL_PPO_TRAJ_DIR")"
EVAL_DIR_PY="$(rl_to_python_path "$FORGE_RL_EVAL_DIR")"

POSITIONAL_ARGS=()
DECK_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --deck)
            if [ -z "$2" ]; then
                echo "Missing deck name after --deck"
                exit 1
            fi
            DECK_ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ -n "${POSITIONAL_ARGS[0]:-}" ]; then
    MODEL="${POSITIONAL_ARGS[0]}"
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
MODEL_PY="$(rl_to_python_path "$MODEL")"

ROUNDS=${POSITIONAL_ARGS[1]:-50}
GAMES=${POSITIONAL_ARGS[2]:-800}
DEVICE_ARG=${POSITIONAL_ARGS[3]:-}
if [ -n "$DEVICE_ARG" ]; then
    DEVICE="$DEVICE_ARG"
else
    DEVICE="$(rl_auto_device)"
fi

echo "PPO Training: $ROUNDS rounds, $GAMES games/round"
echo "Model: $MODEL"
echo "Device: $DEVICE"
echo "Kill anytime - progress saved after each round"
echo ""

"$PYTHON" training/ppo_headless.py \
    --checkpoint "$MODEL_PY" \
    --save-dir "$SAVE_DIR_PY" \
    --traj-dir "$PPO_TRAJ_DIR_PY" \
    --eval-dir "$EVAL_DIR_PY" \
    --device "$DEVICE" \
    --rounds "$ROUNDS" \
    --games-per-round "$GAMES" \
    --eval-games 100 \
    --ppo-epochs 4 \
    --batch-size 64 \
    --lr 1e-5 \
    --port 0 \
    --threads 32 \
    --servers 4 \
    --java-procs 4 \
    --reward-shaping-coeff 1.0 \
    --reward-shaping-decay 0.95 \
    --league \
    --snapshot-interval 5 \
    --max-opponents 3 \
    "${DECK_ARGS[@]}"
