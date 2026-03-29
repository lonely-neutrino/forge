#!/bin/bash
#SBATCH --job-name=forge-build
#SBATCH --partition=genacc_q
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/quest_env.sh"

quest_load_modules
quest_prepare_dirs

cd "$PROJECT_ROOT"
bash scripts/01_build.sh
bash scripts/05_eval.sh "" 2 cpu
