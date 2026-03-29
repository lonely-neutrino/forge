#!/usr/bin/env python3
"""Shared runtime path helpers for local and QUEST workflows."""

from __future__ import annotations

import glob
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[5]


def _path_from_env(var_name: str, default: Path) -> str:
    value = os.environ.get(var_name)
    if value:
        return os.path.abspath(os.path.expanduser(value))
    return str(default)


def get_data_dir() -> str:
    return _path_from_env(
        'FORGE_RL_DATA_DIR',
        PROJECT_ROOT / 'rl_data' / 'trajectories',
    )


def get_preprocessed_dir() -> str:
    explicit = os.environ.get('FORGE_RL_PREPROCESSED_DIR')
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))
    return os.path.join(os.path.dirname(get_data_dir()), 'preprocessed')


def get_checkpoint_dir() -> str:
    return _path_from_env(
        'FORGE_RL_CHECKPOINT_DIR',
        PROJECT_ROOT / 'rl_data' / 'checkpoints',
    )


def get_log_dir() -> str:
    return _path_from_env(
        'FORGE_RL_LOG_DIR',
        PROJECT_ROOT / 'rl_data' / 'logs',
    )


def get_ppo_traj_dir() -> str:
    return _path_from_env(
        'FORGE_RL_PPO_TRAJ_DIR',
        PROJECT_ROOT / 'rl_data' / 'ppo_trajectories',
    )


def get_eval_dir() -> str:
    explicit = os.environ.get('FORGE_RL_EVAL_DIR')
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))
    return get_ppo_traj_dir() + '_eval'


def get_device_override() -> str | None:
    value = os.environ.get('FORGE_RL_DEVICE')
    return value.strip() if value else None


def resolve_forge_jar() -> str:
    explicit = os.environ.get('FORGE_JAR_PATH')
    if explicit:
        jar_path = os.path.abspath(os.path.expanduser(explicit))
        if not os.path.exists(jar_path):
            raise FileNotFoundError(
                f"FORGE_JAR_PATH does not exist: {jar_path}"
            )
        return jar_path

    jar_glob = str(
        PROJECT_ROOT / 'forge-gui-desktop' / 'target' / '*-jar-with-dependencies.jar'
    )
    matches = sorted(glob.glob(jar_glob))
    if not matches:
        raise FileNotFoundError(
            'Forge fat jar not found. Build it first with scripts/01_build.sh '
            'or set FORGE_JAR_PATH.'
        )
    return matches[-1]
