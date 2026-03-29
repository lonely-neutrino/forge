#!/bin/bash
# Shared helpers for local and QUEST RL scripts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_DIR="$PROJECT_ROOT/forge-ai-rl/src/main/python"

rl_default_path() {
    local env_name="$1"
    local fallback="$2"
    local value="${!env_name:-}"
    if [ -n "$value" ]; then
        printf '%s\n' "$value"
    else
        printf '%s\n' "$fallback"
    fi
}

FORGE_RL_DATA_DIR="$(rl_default_path FORGE_RL_DATA_DIR "$PROJECT_ROOT/rl_data/trajectories")"
FORGE_RL_PREPROCESSED_DIR="$(rl_default_path FORGE_RL_PREPROCESSED_DIR "$(dirname "$FORGE_RL_DATA_DIR")/preprocessed")"
FORGE_RL_CHECKPOINT_DIR="$(rl_default_path FORGE_RL_CHECKPOINT_DIR "$PROJECT_ROOT/rl_data/checkpoints")"
FORGE_RL_LOG_DIR="$(rl_default_path FORGE_RL_LOG_DIR "$PROJECT_ROOT/rl_data/logs")"
FORGE_RL_PPO_TRAJ_DIR="$(rl_default_path FORGE_RL_PPO_TRAJ_DIR "$PROJECT_ROOT/rl_data/ppo_trajectories")"
FORGE_RL_EVAL_DIR="$(rl_default_path FORGE_RL_EVAL_DIR "${FORGE_RL_PPO_TRAJ_DIR}_eval")"

export PROJECT_ROOT PYTHON_DIR
export FORGE_RL_DATA_DIR FORGE_RL_PREPROCESSED_DIR FORGE_RL_CHECKPOINT_DIR
export FORGE_RL_LOG_DIR FORGE_RL_PPO_TRAJ_DIR FORGE_RL_EVAL_DIR

rl_find_python() {
    local venv_py="$PROJECT_ROOT/forge-ai-rl/venv/bin/python3"
    local venv_py_win="$PROJECT_ROOT/forge-ai-rl/venv/Scripts/python.exe"
    local userprofile_unix="${USERPROFILE:-}"
    local conda_prefix_unix="${CONDA_PREFIX:-}"

    if command -v cygpath >/dev/null 2>&1; then
        if [ -n "${USERPROFILE:-}" ]; then
            userprofile_unix="$(cygpath "$USERPROFILE" 2>/dev/null || printf '%s' "$USERPROFILE")"
        fi
        if [ -n "${CONDA_PREFIX:-}" ]; then
            conda_prefix_unix="$(cygpath "$CONDA_PREFIX" 2>/dev/null || printf '%s' "$CONDA_PREFIX")"
        fi
    fi

    local conda_py="$conda_prefix_unix/python"
    local conda_py_win="$conda_prefix_unix/python.exe"
    local forge_rl_conda_win="$userprofile_unix/anaconda3/envs/forge_rl/python.exe"

    if [ -x "$venv_py" ]; then
        printf '%s\n' "$venv_py"
    elif [ -f "$venv_py_win" ]; then
        printf '%s\n' "$venv_py_win"
    elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "$conda_py" ]; then
        printf '%s\n' "$conda_py"
    elif [ -n "${CONDA_PREFIX:-}" ] && [ -f "$conda_py_win" ]; then
        printf '%s\n' "$conda_py_win"
    elif [ -n "${USERPROFILE:-}" ] && [ -f "$forge_rl_conda_win" ]; then
        printf '%s\n' "$forge_rl_conda_win"
    elif command -v python3 >/dev/null 2>&1; then
        printf 'python3\n'
    elif command -v python >/dev/null 2>&1; then
        printf 'python\n'
    else
        echo "No Python interpreter found." >&2
        exit 1
    fi
}

rl_to_python_path() {
    local path_value="$1"
    if [[ "$PYTHON" == *.exe ]] && command -v cygpath >/dev/null 2>&1; then
        cygpath -m "$path_value"
    else
        printf '%s\n' "$path_value"
    fi
}

rl_auto_device() {
    if [ -n "${FORGE_RL_DEVICE:-}" ]; then
        printf '%s\n' "$FORGE_RL_DEVICE"
        return
    fi
    "$PYTHON" -c "import sys; sys.path.insert(0, r'$PYTHON_DIR'); from model.backend import auto_device_name; print(auto_device_name())" 2>/dev/null || echo cpu
}

rl_resolve_jar() {
    if [ -n "${FORGE_JAR_PATH:-}" ]; then
        if [ ! -f "$FORGE_JAR_PATH" ]; then
            echo "FORGE_JAR_PATH does not exist: $FORGE_JAR_PATH" >&2
            exit 1
        fi
        printf '%s\n' "$FORGE_JAR_PATH"
        return
    fi

    local matches=("$PROJECT_ROOT"/forge-gui-desktop/target/*-jar-with-dependencies.jar)
    if [ ! -f "${matches[0]}" ]; then
        echo "Forge fat jar not found. Run scripts/01_build.sh or set FORGE_JAR_PATH." >&2
        exit 1
    fi
    printf '%s\n' "${matches[0]}"
}
