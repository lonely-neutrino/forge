#!/bin/bash
# Step 9: Run a deterministic paired replay diff
# Usage:
#   sh 09_replay_diff.sh "<deck1>" ["<deck2>" [seed [replay_id [baseline [compare [model_dir]]]]]]
# Examples:
#   sh 09_replay_diff.sh "C:\Users\tiger\AppData\Roaming\Forge\decks\Red Aggro.dck"
#   sh 09_replay_diff.sh "C:\Decks\A.dck" "C:\Decks\B.dck" 12345 test_run heuristic rl
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GUI_DIR="$PROJECT_ROOT/forge-gui-desktop"
JAR_PATH="$GUI_DIR/target/forge-gui-desktop-2.0.12-SNAPSHOT-jar-with-dependencies.jar"
TRACE_OUT="$PROJECT_ROOT/rl_data/replays"

normalize_path() {
    local path="$1"
    if [ -z "$path" ]; then
        printf '%s' "$path"
        return
    fi
    if command -v cygpath >/dev/null 2>&1; then
        cygpath -m "$path" 2>/dev/null || printf '%s' "$path"
    else
        printf '%s' "$path"
    fi
}

if [ -z "$1" ]; then
    echo "Usage: sh 09_replay_diff.sh \"<deck1>\" [\"<deck2>\" [seed [replay_id [baseline [compare [model_dir]]]]]]"
    exit 1
fi

DECK1="$(normalize_path "$1")"
DECK2_RAW="${2:-$1}"
DECK2="$(normalize_path "$DECK2_RAW")"
SEED="${3:-12345}"
REPLAY_ID="${4:-demo_seed_${SEED}}"
BASELINE="${5:-heuristic}"
COMPARE="${6:-rl}"
MODEL_DIR="$(normalize_path "${7:-$PROJECT_ROOT/rl_data/models}")"

if [ ! -f "$JAR_PATH" ]; then
    echo "Forge fat jar not found. Building first..."
    sh "$SCRIPT_DIR/01_build.sh"
fi

mkdir -p "$TRACE_OUT"

echo "Running deterministic replay diff..."
echo "  Deck 1: $DECK1"
echo "  Deck 2: $DECK2"
echo "  Seed: $SEED"
echo "  Replay ID: $REPLAY_ID"
echo "  Baseline: $BASELINE"
echo "  Compare: $COMPARE"
echo "  Model dir: $MODEL_DIR"
echo "  Trace out: $TRACE_OUT"

cd "$GUI_DIR"
java -Xmx4096m -Dio.netty.tryReflectionSetAccessible=true -Dfile.encoding=UTF-8 \
  -jar "$JAR_PATH" \
  rltrain replaydiff \
  -d "$DECK1" \
  -d "$DECK2" \
  -seed "$SEED" \
  -replay-id "$REPLAY_ID" \
  -trace-out "$(normalize_path "$TRACE_OUT")" \
  -deterministic \
  -baseline "$BASELINE" \
  -compare "$COMPARE" \
  -onnx \
  -m1 "$MODEL_DIR"

echo
echo "Replay traces written to:"
echo "  $TRACE_OUT"
echo
echo "Open the viewer with:"
echo "  sh scripts/10_view_replay_diff.sh \"$REPLAY_ID\""
