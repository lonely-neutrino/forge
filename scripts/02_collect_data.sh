#!/bin/bash
# Step 2: Collect trajectory data + preprocess headlessly.
# Usage: ./02_collect_data.sh [games] [--clean] [--zero-intermediate-reward] [--deck "Deck Name.dck"...]
set -e

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/rl_common.sh"
PYTHON="$(rl_find_python)"

cd "$PYTHON_DIR"

GAMES=1000
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
    GAMES="$1"
    shift
fi

EXTRA_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --deck)
            if [ -z "$2" ]; then
                echo "Missing deck name after --deck"
                exit 1
            fi
            EXTRA_ARGS+=("$1" "$2")
            shift 2
            ;;
        --zero-intermediate-reward)
            EXTRA_ARGS+=("$1")
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "Launching headless collection for $GAMES games..."
JAR_PATH="$(rl_resolve_jar)"
"$PYTHON" training/collect_headless.py \
    --games "$GAMES" \
    --output-dir "$(rl_to_python_path "$FORGE_RL_DATA_DIR")" \
    --preprocessed-dir "$(rl_to_python_path "$FORGE_RL_PREPROCESSED_DIR")" \
    --jar-path "$(rl_to_python_path "$JAR_PATH")" \
    "${EXTRA_ARGS[@]}"
