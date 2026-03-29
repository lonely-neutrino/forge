#!/bin/bash
#SBATCH --job-name=forge-eval
#SBATCH --partition=genacc_q
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/quest_env.sh"

quest_load_modules
quest_prepare_dirs

cd "$PROJECT_ROOT"
bash scripts/05_eval.sh "${1:-}" "${2:-50}" "${3:-cpu}"
