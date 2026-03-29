#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/quest_env.sh"

sbatch_args=()
if [ -n "$QUEST_ACCOUNT" ]; then
    sbatch_args+=(--account "$QUEST_ACCOUNT")
fi
sbatch_args+=(
    --partition "$QUEST_CPU_PARTITION"
    --time "$QUEST_CPU_TIME"
    --cpus-per-task "$QUEST_CPU_CPUS"
    --mem "$QUEST_CPU_MEM"
)

sbatch "${sbatch_args[@]}" "$SCRIPT_DIR/evaluate_cpu.sbatch"
