#!/bin/bash
# Collect imitation learning data by running heuristic AI vs AI games.
# Usage: ./collect_data.sh [num_games] [--deck "Deck Name.dck"...]
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
FORGE_JAR="$PROJECT_ROOT/forge-gui-desktop/target/forge-gui-desktop-2.0.12-SNAPSHOT-jar-with-dependencies.jar"
PYTHON_DIR="$PROJECT_ROOT/forge-ai-rl/src/main/python"
DATA_DIR="$PROJECT_ROOT/rl_data/trajectories"

if command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
else
    echo "No Python interpreter found."
    exit 1
fi

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

NUM_GAMES="${POSITIONAL_ARGS[0]:-1000}"
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

if [ ! -f "$FORGE_JAR" ]; then
    echo "Forge jar not found. Building..."
    cd "$PROJECT_ROOT"
    mvn package -pl forge-gui-desktop -am \
        -Denforcer.skip=true -Dcheckstyle.skip=true \
        -DskipTests -q
fi

mkdir -p "$DATA_DIR"

echo "=== Collecting $NUM_GAMES games ==="
echo "Output: $DATA_DIR"
echo ""

cd "$PROJECT_ROOT/forge-gui-desktop"
java -Xmx4096m \
    --add-opens java.base/java.lang=ALL-UNNAMED \
    --add-opens java.base/java.util=ALL-UNNAMED \
    --add-opens java.base/java.text=ALL-UNNAMED \
    --add-opens java.base/java.lang.reflect=ALL-UNNAMED \
    --add-opens java.desktop/javax.imageio.spi=ALL-UNNAMED \
    -jar "$FORGE_JAR" \
    rltrain collect \
    "${DECK_ARGS[@]}" \
    -n "$NUM_GAMES" \
    -o "$DATA_DIR" \
    -q

echo ""
echo "Files collected:"
ls -lh "$DATA_DIR" | tail -5
echo "Total: $(ls "$DATA_DIR"/*.jsonl 2>/dev/null | wc -l) trajectory files"
