#!/usr/bin/env python3
"""Headless PPO entrypoint that reuses the existing PPO worker logic."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.ppo_ui import PPOState, ppo_thread
from training.runtime_paths import (
    get_checkpoint_dir,
    get_device_override,
    get_eval_dir,
    get_ppo_traj_dir,
)


def main():
    default_checkpoint = os.path.join(
        get_checkpoint_dir(),
        'model_with_decisions.pt',
    )
    parser = argparse.ArgumentParser(
        description='MTG RL PPO training (headless)'
    )
    parser.add_argument('--checkpoint', default=default_checkpoint)
    parser.add_argument('--save-dir', default=get_checkpoint_dir())
    parser.add_argument('--traj-dir', default=get_ppo_traj_dir())
    parser.add_argument('--eval-dir', default=get_eval_dir())
    parser.add_argument('--device', default=get_device_override())
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--games-per-round', type=int, default=200)
    parser.add_argument('--ppo-epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--eval-games', type=int, default=50)
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--servers', type=int, default=1)
    parser.add_argument('--java-procs', type=int, default=1)
    parser.add_argument(
        '--collect-mode',
        default='evaluate',
        choices=['evaluate', 'selfplay'],
    )
    parser.add_argument('--eval-interval', type=int, default=1)
    parser.add_argument('--reward-shaping-coeff', type=float, default=0.0)
    parser.add_argument('--reward-shaping-decay', type=float, default=0.95)
    parser.add_argument('--league', action='store_true', default=False)
    parser.add_argument('--snapshot-interval', type=int, default=5)
    parser.add_argument('--max-opponents', type=int, default=3)
    parser.add_argument('--deck', action='append')
    args = parser.parse_args()

    state = PPOState()
    ppo_thread(state, args)
    if state.status.startswith('ERROR'):
        sys.exit(1)


if __name__ == '__main__':
    main()
