#!/bin/bash
# Create or refresh a mamba environment for Forge RL on QUEST.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$SCRIPT_DIR/quest_env.sh"
source "$PROJECT_ROOT/scripts/rl_common.sh"

quest_load_modules
quest_prepare_dirs

if [ -n "$QUEST_MAMBA_INIT" ]; then
    # shellcheck disable=SC1090
    source "$QUEST_MAMBA_INIT"
fi

if command -v micromamba >/dev/null 2>&1; then
    eval "$(micromamba shell hook --shell bash)"
    micromamba create -y -n "$QUEST_MAMBA_ENV" python=3.11
    micromamba activate "$QUEST_MAMBA_ENV"
elif command -v mamba >/dev/null 2>&1; then
    eval "$(mamba shell hook --shell bash)"
    mamba create -y -n "$QUEST_MAMBA_ENV" python=3.11
    mamba activate "$QUEST_MAMBA_ENV"
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda create -y -n "$QUEST_MAMBA_ENV" python=3.11
    conda activate "$QUEST_MAMBA_ENV"
else
    echo "No mamba/conda executable found. Set QUEST_MAMBA_MODULE or QUEST_MAMBA_INIT." >&2
    exit 1
fi

cd "$PROJECT_ROOT"
python -m pip install --upgrade pip
python -m pip install -r forge-ai-rl/src/main/python/requirements.txt

echo "QUEST mamba environment ready."
echo "Env type: mamba"
echo "Env name: $QUEST_MAMBA_ENV"
echo "Project root: $PROJECT_ROOT"
echo "Data dir: $FORGE_RL_DATA_DIR"
echo "Checkpoint dir: $FORGE_RL_CHECKPOINT_DIR"
