#!/bin/bash
#SBATCH --job-name=forge-ppo
#SBATCH --partition=gengpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/quest_env.sh"

quest_load_modules
quest_prepare_dirs

cd "$PROJECT_ROOT"
bash scripts/06_ppo_train.sh "${1:-}" "${2:-10}" "${3:-200}" "${4:-cuda}"
