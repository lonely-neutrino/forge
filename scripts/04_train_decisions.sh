#!/bin/bash
# Step 4: Train attack/block/priority decision heads (imitation learning)
# Usage: 04_train_decisions.sh [epochs] [batch_size] [encoder] [heads] [device] [--joint]
# heads: "all" (default), or comma-separated: "priority", "attack,block", etc.
# --joint: train all heads simultaneously with unfrozen encoder
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

EPOCHS=${1:-10}
BATCH=${2:-256}
CKPT_DIR="$PROJECT_ROOT/rl_data/checkpoints"
HEADS=${4:-all}
DEVICE_ARG=${5:-}
DATA_DIR="$PROJECT_ROOT/rl_data/trajectories"
if [ -n "$DEVICE_ARG" ]; then
    DEVICE="$DEVICE_ARG"
else
    DEVICE=$("$PYTHON" -c "import sys; sys.path.insert(0, r'$PYTHON_DIR'); from model.backend import auto_device_name; print(auto_device_name())" 2>/dev/null || echo cpu)
fi
if [[ "$PYTHON" == *.exe ]] && command -v cygpath >/dev/null 2>&1; then
    DATA_DIR_PY="$(cygpath -m "$DATA_DIR")"
    CKPT_DIR_PY="$(cygpath -m "$CKPT_DIR")"
else
    DATA_DIR_PY="$DATA_DIR"
    CKPT_DIR_PY="$CKPT_DIR"
fi
JOINT_FLAG=""
if [[ "$*" == *"--joint"* ]]; then
    JOINT_FLAG="--joint"
    echo "JOINT MODE: training all heads with unfrozen encoder"
fi


# Auto-select best available checkpoint if not explicitly provided
if [ -z "$3" ]; then
    # Chain order: model_with_decisions > best_block > best_attack > best_priority > best_value
    if [ -f "$CKPT_DIR/model_with_decisions.pt" ]; then
        ENCODER="$CKPT_DIR/model_with_decisions.pt"
    elif [ -f "$CKPT_DIR/best_block_model.pt" ]; then
        ENCODER="$CKPT_DIR/best_block_model.pt"
    elif [ -f "$CKPT_DIR/best_attack_model.pt" ]; then
        ENCODER="$CKPT_DIR/best_attack_model.pt"
    elif [ -f "$CKPT_DIR/best_priority_model.pt" ]; then
        ENCODER="$CKPT_DIR/best_priority_model.pt"
    else
        ENCODER="$CKPT_DIR/best_value_model.pt"
    fi
    echo "Auto-selected encoder: $(basename $ENCODER)"
else
    ENCODER="$3"
fi
if [[ "$PYTHON" == *.exe ]] && command -v cygpath >/dev/null 2>&1; then
    ENCODER_PY="$(cygpath -m "$ENCODER")"
else
    ENCODER_PY="$ENCODER"
fi

# Warn if using base value model when trained heads exist
if [[ "$ENCODER" == *"best_value_model"* ]]; then
    for f in best_priority_model.pt best_attack_model.pt best_block_model.pt model_with_decisions.pt; do
        if [ -f "$CKPT_DIR/$f" ]; then
            echo "WARNING: Using best_value_model.pt but $f exists!"
            echo "  This will DISCARD trained head weights. Use $f instead?"
            echo "  Press Ctrl+C to abort, or Enter to continue anyway."
            read
            break
        fi
    done
fi

echo "Training decision heads for $EPOCHS epochs, batch=$BATCH, heads=$HEADS..."
echo "Encoder: $ENCODER"
echo "Device: $DEVICE"
"$PYTHON" training/train_decisions_ui.py \
    --data-dir "$DATA_DIR_PY" \
    --encoder-checkpoint "$ENCODER_PY" \
    --save-dir "$CKPT_DIR_PY" \
    --device "$DEVICE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH" \
    --heads "$HEADS" \
    $JOINT_FLAG
