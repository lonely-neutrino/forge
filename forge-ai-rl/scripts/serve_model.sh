#!/bin/bash
# Start the model server for RL inference.
# Usage: ./serve_model.sh [model_path]
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON="$PROJECT_ROOT/forge-ai-rl/venv/bin/python3"
PYTHON_DIR="$PROJECT_ROOT/forge-ai-rl/src/main/python"
MODEL="${1:-$PROJECT_ROOT/rl_data/checkpoints/best_value_model.pt}"

if [ ! -f "$PYTHON" ]; then
    echo "Python venv not found. Run setup.sh first."
    exit 1
fi

DEVICE=$("$PYTHON" -c "import sys; sys.path.insert(0, r'$PYTHON_DIR'); from model.backend import auto_device_name; print(auto_device_name())" 2>/dev/null || echo cpu)

echo "=== Starting Model Server ==="
echo "Model: $MODEL"
echo "Device: $DEVICE"
echo "Port: 50051"
echo ""

cd "$PYTHON_DIR"
"$PYTHON" serving/model_server.py \
    --host localhost \
    --port 50051 \
    --device "$DEVICE" \
    --model "$MODEL"
