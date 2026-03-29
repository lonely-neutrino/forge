#!/bin/bash
# Step 5: Evaluate RL model vs heuristic AI
# Usage: 05_eval.sh [checkpoint] [games] [device]
set -e

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/rl_common.sh"
PYTHON="$(rl_find_python)"
cd "$PYTHON_DIR"

CKPT_DIR="$FORGE_RL_CHECKPOINT_DIR"
if [ -z "${1:-}" ]; then
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
DEVICE=${3:-$(rl_auto_device)}
EVAL_DIR="${FORGE_RL_EVAL_DIR:-${TMPDIR:-/tmp}/rl_eval}"

echo "Evaluating RL model vs heuristic..."
echo "  Checkpoint: $CKPT"
echo "  Games: $GAMES"
echo "  Device: $DEVICE"

"$PYTHON" -c "
import sys, os
sys.path.insert(0, '.')
from training.ppo_trainer import run_games, ModelServerError
from serving.model_server import ModelServer
from model.mtg_model import MTGModel
import threading, json, time

checkpoint = r'$CKPT'
n_games = $GAMES
eval_dir = r'$EVAL_DIR'
os.makedirs(eval_dir, exist_ok=True)

model = MTGModel.load(checkpoint, device='$DEVICE')
model.eval()
server = ModelServer(model, host='0.0.0.0', port=0, device='$DEVICE')

import socket
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
    win_rate, _ = run_games(
        n_games, eval_dir, mode='evaluate',
        port=str(port), threads=16, java_procs=2)
    print(f'\n=== Result: {win_rate:.1%} win rate ({int(win_rate*n_games)}/{n_games}) ===')
except ModelServerError as e:
    print(f'FATAL: {e}')
    sys.exit(1)

total = 0
fallback = 0
for f in os.listdir(eval_dir):
    if not f.endswith('.jsonl'):
        continue
    with open(os.path.join(eval_dir, f)) as fh:
        lines = fh.readlines()
    for line in lines[1:]:
        rec = json.loads(line)
        total += 1
        if rec.get('usedFallback', False):
            fallback += 1
print(f'Decisions: {total}, Fallback: {fallback} ({fallback/total:.1%})')
if fallback > 0:
    print('WARNING: Some decisions used heuristic fallback!')
else:
    print('All decisions were real RL model decisions.')
"
