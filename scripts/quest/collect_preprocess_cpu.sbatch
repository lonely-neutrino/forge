#!/bin/bash
#SBATCH --job-name=forge-collect
#SBATCH --partition=genacc_q
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/quest_env.sh"

quest_load_modules
quest_prepare_dirs
quest_activate_python_env

cd "$PROJECT_ROOT"
bash scripts/01_build.sh
bash scripts/02_collect_data.sh "${1:-500}" --clean
