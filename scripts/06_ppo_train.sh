#!/bin/bash
# Step 6: PPO training with dashboard
# Usage: 06_ppo_train.sh [model] [rounds] [games] [device] [--deck "Deck Name.dck"...]
#
# Auto-resumes from best_ppo_model.pt if it exists (unless a model is specified).
# Kill anytime (Ctrl+C) — progress is saved after each round.
# Restart with no args to continue from where you left off.
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

pkill -f "model_server\|ppo_ui" 2>/dev/null || true
sleep 1

SAVE_DIR="$PROJECT_ROOT/rl_data/checkpoints"
LATEST_PPO="$SAVE_DIR/ppo_model_latest.pt"
BEST_PPO="$SAVE_DIR/best_ppo_model.pt"
IMITATION="$SAVE_DIR/model_with_decisions.pt"
if [[ "$PYTHON" == *.exe ]] && command -v cygpath >/dev/null 2>&1; then
    SAVE_DIR_PY="$(cygpath -m "$SAVE_DIR")"
else
    SAVE_DIR_PY="$SAVE_DIR"
fi

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

# Auto-resume: latest > best > imitation
if [ -n "${POSITIONAL_ARGS[0]}" ]; then
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
if [[ "$PYTHON" == *.exe ]] && command -v cygpath >/dev/null 2>&1; then
    MODEL_PY="$(cygpath -m "$MODEL")"
else
    MODEL_PY="$MODEL"
fi

ROUNDS=${POSITIONAL_ARGS[1]:-50}
GAMES=${POSITIONAL_ARGS[2]:-800}
DEVICE_ARG=${POSITIONAL_ARGS[3]:-}
if [ -n "$DEVICE_ARG" ]; then
    DEVICE="$DEVICE_ARG"
else
    DEVICE=$("$PYTHON" -c "import sys; sys.path.insert(0, r'$PYTHON_DIR'); from model.backend import auto_device_name; print(auto_device_name())" 2>/dev/null || echo cpu)
fi

echo "PPO Training: $ROUNDS rounds, $GAMES games/round"
echo "Model: $MODEL"
echo "Device: $DEVICE"
echo "Kill anytime — progress saved after each round"
echo ""

"$PYTHON" training/ppo_ui.py \
    --checkpoint "$MODEL_PY" \
    --save-dir "$SAVE_DIR_PY" \
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
    "${DECK_ARGS[@]}"
