#!/bin/bash
# Step 8: Export trained model to ONNX for Java inference
# Usage: 08_export_onnx.sh [checkpoint] [output_dir]
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

CKPT_DIR="$PROJECT_ROOT/rl_data/checkpoints"
if [ -z "$1" ]; then
    if [ -f "$CKPT_DIR/best_ppo_model.pt" ]; then
        CKPT="$CKPT_DIR/best_ppo_model.pt"
    else
        CKPT="$CKPT_DIR/model_with_decisions.pt"
    fi
else
    CKPT="$1"
fi
OUTPUT=${2:-"$PROJECT_ROOT/rl_data/models"}
if [[ "$PYTHON" == *.exe ]] && command -v cygpath >/dev/null 2>&1; then
    CKPT_PY="$(cygpath -m "$CKPT")"
    OUTPUT_PY="$(cygpath -m "$OUTPUT")"
else
    CKPT_PY="$CKPT"
    OUTPUT_PY="$OUTPUT"
fi

echo "Exporting ONNX models..."
echo "  Checkpoint: $CKPT"
echo "  Output: $OUTPUT"

if command -v pip >/dev/null 2>&1; then
    pip install onnxruntime -q 2>/dev/null || true
fi

"$PYTHON" tools/export_onnx.py \
    --checkpoint "$CKPT_PY" \
    --output-dir "$OUTPUT_PY" \
    --device cpu

# Copy to Forge data directory for GUI access
FORGE_DIR="$HOME/.forge/res/rl/models"
mkdir -p "$FORGE_DIR"
cp "$OUTPUT"/*.onnx "$OUTPUT"/*.data "$FORGE_DIR/" 2>/dev/null
echo "Copied ONNX files to $FORGE_DIR"
