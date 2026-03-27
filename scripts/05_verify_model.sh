#!/bin/bash
# Step 5: Verify model works in live games
# Checks: server alive, RL decisions real, creatures played, attack probs vary
# Usage: ./05_verify_model.sh [model] [games] [device] [--deck "Deck Name.dck"...]
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

cd "$PROJECT_ROOT"

POSITIONAL_ARGS=()
DECK_OVERRIDE_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --deck)
            if [ -z "$2" ]; then
                echo "Missing deck name after --deck"
                exit 1
            fi
            DECK_OVERRIDE_ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

MODEL=${POSITIONAL_ARGS[0]:-rl_data/checkpoints/model_with_decisions.pt}
GAMES=${POSITIONAL_ARGS[1]:-10}
DEVICE_ARG=${POSITIONAL_ARGS[2]:-}
PORT=50051
MODEL_PATH="$PROJECT_ROOT/$MODEL"
VERIFY_DIR="${TMPDIR:-/tmp}/rl_verify"
VERIFY_PROBS_DIR="${TMPDIR:-/tmp}/rl_verify_probs"
if [[ "$PYTHON" == *.exe ]] && command -v cygpath >/dev/null 2>&1; then
    MODEL_PATH_PY="$(cygpath -m "$MODEL_PATH")"
else
    MODEL_PATH_PY="$MODEL_PATH"
fi
if [ -n "$DEVICE_ARG" ]; then
    DEVICE="$DEVICE_ARG"
else
    DEVICE=$("$PYTHON" -c "import sys; sys.path.insert(0, r'$PYTHON_DIR'); from model.backend import auto_device_name; print(auto_device_name())" 2>/dev/null || echo cpu)
fi

DECK_OUTPUT="$(
    cd "$PYTHON_DIR" &&
    "$PYTHON" training/deck_config.py "${DECK_OVERRIDE_ARGS[@]}"
)"
if [ -z "$DECK_OUTPUT" ]; then
    echo "No RL decks configured."
    exit 1
fi
mapfile -t DECKS <<<"$DECK_OUTPUT"
DECK_ARGS=()
for deck in "${DECKS[@]}"; do
    DECK_ARGS+=(-d "$deck")
done

# Kill any existing server
pkill -f "model_server" 2>/dev/null || true
sleep 1

check_server() {
    "$PYTHON" -c "
import socket, sys
s = socket.socket()
s.settimeout(1.0)
try:
    s.connect(('127.0.0.1', $PORT))
except OSError:
    sys.exit(1)
finally:
    s.close()
"
}

# Start model server in a subshell
echo "Starting model server with $MODEL..."
echo "Device: $DEVICE"
echo "Decks: ${DECKS[*]}"
(
    cd "$PYTHON_DIR"
    exec "$PYTHON" -c "
from serving.model_server import ModelServer
from model.mtg_model import MTGModel
import threading, time
model = MTGModel.load(r'$MODEL_PATH_PY', device='$DEVICE')
server = ModelServer(model, port=$PORT, device='$DEVICE')
t = threading.Thread(target=server.start, daemon=True)
t.start()
print('Server started', flush=True)
while True: time.sleep(1)
"
) &
SERVER_PID=$!
sleep 5

# Verify alive
echo "=== SERVER CHECK ==="
if check_server; then
    echo "SERVER: ALIVE"
else
    echo "SERVER: DEAD"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Run eval
echo "Running $GAMES games..."
cd "$PROJECT_ROOT/forge-gui-desktop"
java -Xmx8192m \
    --add-opens java.base/java.lang=ALL-UNNAMED \
    --add-opens java.base/java.util=ALL-UNNAMED \
    --add-opens java.base/java.text=ALL-UNNAMED \
    --add-opens java.base/java.lang.reflect=ALL-UNNAMED \
    --add-opens java.desktop/javax.imageio.spi=ALL-UNNAMED \
    -jar target/forge-gui-desktop-2.0.12-SNAPSHOT-jar-with-dependencies.jar \
    rltrain evaluate \
    "${DECK_ARGS[@]}" \
    -n "$GAMES" -o "$VERIFY_DIR" \
    -host localhost -port $PORT -q 2>&1 | grep -E "RL Wins|Complete"

echo ""
echo "=== SERVER STILL ALIVE? ==="
if check_server; then
    echo "YES"
else
    echo "NO"
fi

echo ""
echo "=== DECISION TYPES ==="
grep -oh '"decisionType":"[^"]*"' "$VERIFY_DIR"/traj_*.jsonl 2>/dev/null | sort | uniq -c | sort -rn

echo ""
echo "=== CREATURE COUNTS ==="
grep "main1_turn" "$VERIFY_DIR"/traj_*.jsonl 2>/dev/null | head -5 | while IFS= read -r line; do
echo "$line" | cut -d: -f2- | "$PYTHON" -c "
import sys,json
r=json.loads(sys.stdin.read())
gf=r.get('globalFeatures',[])
mc=gf[23] if len(gf)>23 else -1
print(f'  {r.get(\"contextInfo\",\"\"):20s} myCreatures={mc*20:.0f}')
" 2>/dev/null
done

echo ""
echo "=== ATTACK PROBS (should vary, not all 0.01) ==="
grep "RL_MODEL_ATTACK" "$VERIFY_DIR"/traj_*.jsonl 2>/dev/null | head -5 || \
  { cd "$PROJECT_ROOT/forge-gui-desktop" && java -Xmx8192m \
    --add-opens java.base/java.lang=ALL-UNNAMED \
    --add-opens java.base/java.util=ALL-UNNAMED \
    --add-opens java.base/java.text=ALL-UNNAMED \
    --add-opens java.base/java.lang.reflect=ALL-UNNAMED \
    --add-opens java.desktop/javax.imageio.spi=ALL-UNNAMED \
    -jar target/forge-gui-desktop-2.0.12-SNAPSHOT-jar-with-dependencies.jar \
    rltrain evaluate \
    "${DECK_ARGS[@]}" \
    -n 1 -o "$VERIFY_PROBS_DIR" \
    -host localhost -port $PORT 2>&1 | grep "RL_MODEL_ATTACK" | head -5; }

# Cleanup
kill $SERVER_PID 2>/dev/null
echo ""
echo "=== DONE ==="
