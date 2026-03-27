#!/bin/bash
# Setup script for the Forge RL AI Python environment
# Usage: source forge-ai-rl/setup.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
PYTHON_DIR="$SCRIPT_DIR/src/main/python"

echo "=== Forge RL AI Setup ==="

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate venv
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r "$PYTHON_DIR/requirements.txt"

# Verify CUDA
echo ""
echo "=== Environment Info ==="
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print()

# Estimate memory usage
import sys
sys.path.insert(0, '$PYTHON_DIR')
from model.backend import resolve_backend
from model.gpu_config import auto_detect_profile, estimate_memory_usage
backend = resolve_backend()
profile = auto_detect_profile()
print(f'Backend: {backend.name} ({backend.display_name})')
print(f'GPU Profile: {profile.name}')
print(f'Recommended batch size: {profile.batch_size}')
print(f'Mixed precision (AMP): {backend.use_amp and profile.use_amp}')
mem = estimate_memory_usage(profile.batch_size)
print(f'Estimated VRAM usage: {mem[\"total_gb\"]:.2f} GB')
"

echo ""
echo "=== Setup Complete ==="
echo "Virtual environment: $VENV_DIR"
echo "To activate: source $VENV_DIR/bin/activate"
echo ""
echo "Quick start:"
echo "  1. Start model server:  python3 $PYTHON_DIR/serving/model_server.py --device dml"
echo "  2. Run training:        python3 $PYTHON_DIR/training/trainer.py --device dml --mode imitation"
