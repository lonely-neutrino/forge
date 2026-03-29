#!/bin/bash
# Step 4: Train attack/block/priority decision heads (imitation learning)
# Usage: 04_train_decisions.sh [epochs] [batch_size] [encoder] [heads] [device] [--joint]
set -e

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/rl_common.sh"
PYTHON="$(rl_find_python)"

cd "$PYTHON_DIR"

EPOCHS=${1:-10}
BATCH=${2:-256}
CKPT_DIR="$FORGE_RL_CHECKPOINT_DIR"
HEADS=${4:-all}
DEVICE_ARG=${5:-}
DATA_DIR="$FORGE_RL_DATA_DIR"
if [ -n "$DEVICE_ARG" ]; then
    DEVICE="$DEVICE_ARG"
else
    DEVICE="$(rl_auto_device)"
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
"$PYTHON" training/train_decisions.py \
    --data-dir "$DATA_DIR_PY" \
    --encoder-checkpoint "$ENCODER_PY" \
    --save-dir "$CKPT_DIR_PY" \
    --device "$DEVICE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH" \
    --heads "$HEADS" \
    $JOINT_FLAG
