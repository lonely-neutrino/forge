#!/bin/bash
# Step 3: Train value network (game state encoder)
# Usage: 03_train_value.sh [epochs] [batch_size] [device]
set -e

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/rl_common.sh"
PYTHON="$(rl_find_python)"

cd "$PYTHON_DIR"

EPOCHS=${1:-100}
BATCH=${2:-256}
DEVICE_ARG=${3:-}
DATA_DIR="$FORGE_RL_DATA_DIR"
SAVE_DIR="$FORGE_RL_CHECKPOINT_DIR"
if [[ "$PYTHON" == *.exe ]] && command -v cygpath >/dev/null 2>&1; then
    DATA_DIR_PY="$(cygpath -m "$DATA_DIR")"
    SAVE_DIR_PY="$(cygpath -m "$SAVE_DIR")"
else
    DATA_DIR_PY="$DATA_DIR"
    SAVE_DIR_PY="$SAVE_DIR"
fi
if [ -n "$DEVICE_ARG" ]; then
    DEVICE="$DEVICE_ARG"
else
    DEVICE="$(rl_auto_device)"
fi

echo "Training value network for $EPOCHS epochs, batch=$BATCH (chunked loading)..."
echo "Device: $DEVICE"
"$PYTHON" training/train_value.py \
    --data-dir "$DATA_DIR_PY" \
    --save-dir "$SAVE_DIR_PY" \
    --device "$DEVICE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH"
