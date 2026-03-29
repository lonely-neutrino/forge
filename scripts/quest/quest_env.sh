#!/bin/bash
# Central QUEST-specific defaults for Forge RL jobs.

set -euo pipefail

QUEST_ACCOUNT="${QUEST_ACCOUNT:-}"
QUEST_CPU_PARTITION="${QUEST_CPU_PARTITION:-genacc_q}"
QUEST_GPU_PARTITION="${QUEST_GPU_PARTITION:-gengpu}"

QUEST_CPU_TIME="${QUEST_CPU_TIME:-04:00:00}"
QUEST_GPU_TIME="${QUEST_GPU_TIME:-08:00:00}"

QUEST_CPU_CPUS="${QUEST_CPU_CPUS:-8}"
QUEST_GPU_CPUS="${QUEST_GPU_CPUS:-16}"
QUEST_CPU_MEM="${QUEST_CPU_MEM:-32G}"
QUEST_GPU_MEM="${QUEST_GPU_MEM:-64G}"
QUEST_GPU_COUNT="${QUEST_GPU_COUNT:-1}"

QUEST_JAVA_MODULE="${QUEST_JAVA_MODULE:-}"
QUEST_PYTHON_MODULE="${QUEST_PYTHON_MODULE:-}"
QUEST_CUDA_MODULE="${QUEST_CUDA_MODULE:-}"

QUEST_PROJECT_ROOT="${QUEST_PROJECT_ROOT:-$HOME/forge}"
QUEST_RUN_ROOT="${QUEST_RUN_ROOT:-$HOME/scratch/forge_rl}"

export FORGE_RL_DATA_DIR="${FORGE_RL_DATA_DIR:-$QUEST_RUN_ROOT/trajectories}"
export FORGE_RL_PREPROCESSED_DIR="${FORGE_RL_PREPROCESSED_DIR:-$QUEST_RUN_ROOT/preprocessed}"
export FORGE_RL_CHECKPOINT_DIR="${FORGE_RL_CHECKPOINT_DIR:-$QUEST_RUN_ROOT/checkpoints}"
export FORGE_RL_LOG_DIR="${FORGE_RL_LOG_DIR:-$QUEST_RUN_ROOT/logs}"
export FORGE_RL_PPO_TRAJ_DIR="${FORGE_RL_PPO_TRAJ_DIR:-$QUEST_RUN_ROOT/ppo_trajectories}"
export FORGE_RL_EVAL_DIR="${FORGE_RL_EVAL_DIR:-$QUEST_RUN_ROOT/ppo_eval}"
export FORGE_RL_DEVICE="${FORGE_RL_DEVICE:-cuda}"

quest_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$quest_script_dir/../.." && pwd)}"

quest_load_modules() {
    module purge
    if [ -n "$QUEST_JAVA_MODULE" ]; then
        module load "$QUEST_JAVA_MODULE"
    fi
    if [ -n "$QUEST_PYTHON_MODULE" ]; then
        module load "$QUEST_PYTHON_MODULE"
    fi
    if [ -n "$QUEST_CUDA_MODULE" ]; then
        module load "$QUEST_CUDA_MODULE"
    fi
}

quest_prepare_dirs() {
    mkdir -p \
        "$FORGE_RL_DATA_DIR" \
        "$FORGE_RL_PREPROCESSED_DIR" \
        "$FORGE_RL_CHECKPOINT_DIR" \
        "$FORGE_RL_LOG_DIR" \
        "$FORGE_RL_PPO_TRAJ_DIR" \
        "$FORGE_RL_EVAL_DIR"
}
