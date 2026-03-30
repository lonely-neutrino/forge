#!/bin/bash
# Step 5: Evaluate RL model vs heuristic AI
# Usage: 05_eval.sh [checkpoint] [games] [device]
# Uses the same server+Java approach as PPO training
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
    elif [ -f "$CKPT_DIR/model_with_decisions.pt" ]; then
        CKPT="$CKPT_DIR/model_with_decisions.pt"
    else
        CKPT="$CKPT_DIR/best_value_model.pt"
    fi
else
    CKPT="$1"
fi
GAMES=${2:-100}
DEVICE_ARG=${3:-}
if [ -n "$DEVICE_ARG" ]; then
    DEVICE="$DEVICE_ARG"
else
    DEVICE=$("$PYTHON" -c "import sys; sys.path.insert(0, r'$PYTHON_DIR'); from model.backend import auto_device_name; print(auto_device_name())" 2>/dev/null || echo cpu)
fi

EVAL_DIR="${TMPDIR:-/tmp}/rl_eval"
if [[ "$PYTHON" == *.exe ]] && command -v cygpath >/dev/null 2>&1; then
    CKPT_PY="$(cygpath -m "$CKPT")"
    EVAL_DIR_PY="$(cygpath -m "$EVAL_DIR")"
else
    CKPT_PY="$CKPT"
    EVAL_DIR_PY="$EVAL_DIR"
fi

echo "Evaluating RL model vs heuristic..."
echo "  Checkpoint: $CKPT"
echo "  Games: $GAMES"
echo "  Device: $DEVICE"

"$PYTHON" -c "
import json
import os
import socket
import sys
import threading
import time

sys.path.insert(0, r'$PYTHON_DIR')

from training.ppo_trainer import run_games, ModelServerError
from serving.model_server import ModelServer
from model.mtg_model import MTGModel

checkpoint = r'$CKPT_PY'
n_games = int($GAMES)
eval_dir = r'$EVAL_DIR_PY'
device = '$DEVICE'

os.makedirs(eval_dir, exist_ok=True)

model = MTGModel.load(checkpoint, device=device)
model.eval()
server = ModelServer(model, host='0.0.0.0', port=0, device=device)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(('', 0))
port = sock.getsockname()[1]
sock.close()
server.port = port

t = threading.Thread(target=server.start, daemon=True)
t.start()
time.sleep(2)
print(f'Server started on port {port}')

try:
    win_rate, results = run_games(
        n_games, eval_dir, mode='evaluate',
        port=str(port), threads=16, java_procs=2)
    print(f'\\n=== Result: {win_rate:.1%} win rate ({int(win_rate*n_games)}/{n_games}) ===')
except ModelServerError as e:
    print(f'FATAL: {e}')
    sys.exit(1)

total = 0
fallback = 0
for name in os.listdir(eval_dir):
    if not name.endswith('.jsonl'):
        continue
    with open(os.path.join(eval_dir, name), encoding='utf-8') as fh:
        lines = fh.readlines()
    for line in lines[1:]:
        rec = json.loads(line)
        total += 1
        if rec.get('usedFallback', False):
            fallback += 1

if total > 0:
    print(f'Decisions: {total}, Fallback: {fallback} ({fallback/total:.1%})')
else:
    print('Decisions: 0, Fallback: 0 (0.0%)')

if fallback > 0:
    print('WARNING: Some decisions used heuristic fallback!')
else:
    print('All decisions were real RL model decisions.')
"
