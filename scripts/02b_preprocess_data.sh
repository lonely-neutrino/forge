#!/bin/bash
# Step 2b: Preprocess trajectory data (JSONL -> memory-mapped numpy)
# Run after data collection (02), before training (03/04).
# Usage: 02b_preprocess_data.sh [data_dir] [output_dir]
set -e

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/rl_common.sh"
PYTHON="$(rl_find_python)"

cd "$PYTHON_DIR"

DATA_DIR=${1:-"$FORGE_RL_DATA_DIR"}
OUTPUT_DIR=${2:-"$FORGE_RL_PREPROCESSED_DIR"}

echo "Preprocessing trajectories: $DATA_DIR -> $OUTPUT_DIR"
"$PYTHON" training/preprocess_trajectories.py \
    --data-dir "$(rl_to_python_path "$DATA_DIR")" \
    --output-dir "$(rl_to_python_path "$OUTPUT_DIR")"
