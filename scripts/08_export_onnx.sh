#!/bin/bash
# Step 8: Export trained model to ONNX for Java inference
# Usage: 08_export_onnx.sh [checkpoint] [output_dir]
set -e

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/rl_common.sh"
PYTHON="$(rl_find_python)"
cd "$PYTHON_DIR"

CKPT_DIR="$FORGE_RL_CHECKPOINT_DIR"
if [ -z "${1:-}" ]; then
    if [ -f "$CKPT_DIR/best_ppo_model.pt" ]; then
        CKPT="$CKPT_DIR/best_ppo_model.pt"
    else
        CKPT="$CKPT_DIR/model_with_decisions.pt"
    fi
else
    CKPT="$1"
fi
OUTPUT=${2:-"$PROJECT_ROOT/rl_data/models"}
CKPT_PY="$(rl_to_python_path "$CKPT")"
OUTPUT_PY="$(rl_to_python_path "$OUTPUT")"

echo "Exporting ONNX models..."
echo "  Checkpoint: $CKPT"
echo "  Output: $OUTPUT"

"$PYTHON" tools/export_onnx.py \
    --checkpoint "$CKPT_PY" \
    --output-dir "$OUTPUT_PY" \
    --device cpu

FORGE_DIR="$HOME/.forge/res/rl/models"
mkdir -p "$FORGE_DIR"
cp "$OUTPUT"/*.onnx "$OUTPUT"/*.data "$FORGE_DIR/" 2>/dev/null || true
echo "Copied ONNX files to $FORGE_DIR"
