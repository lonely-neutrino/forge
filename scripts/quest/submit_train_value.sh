#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/quest_env.sh"

sbatch_args=()
if [ -n "$QUEST_ACCOUNT" ]; then
    sbatch_args+=(--account "$QUEST_ACCOUNT")
fi
sbatch_args+=(
    --partition "$QUEST_GPU_PARTITION"
    --time "$QUEST_GPU_TIME"
    --cpus-per-task "$QUEST_GPU_CPUS"
    --mem "$QUEST_GPU_MEM"
    --gres "gpu:$QUEST_GPU_COUNT"
)

sbatch "${sbatch_args[@]}" "$SCRIPT_DIR/train_value_gpu.sbatch"
