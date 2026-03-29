#!/bin/bash
# Create or refresh the Forge RL Python environment on QUEST.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$SCRIPT_DIR/quest_env.sh"
source "$PROJECT_ROOT/scripts/rl_common.sh"

quest_load_modules
quest_prepare_dirs

cd "$PROJECT_ROOT/forge-ai-rl"

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r src/main/python/requirements.txt

echo "QUEST environment ready."
echo "Project root: $PROJECT_ROOT"
echo "Data dir: $FORGE_RL_DATA_DIR"
echo "Checkpoint dir: $FORGE_RL_CHECKPOINT_DIR"
