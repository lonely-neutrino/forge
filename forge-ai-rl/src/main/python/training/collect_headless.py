#!/usr/bin/env python3
"""Headless data collection entrypoint for local and QUEST jobs."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.collect_ui import CollectState, collect_thread
from training.runtime_paths import get_data_dir, get_preprocessed_dir


def main():
    parser = argparse.ArgumentParser(
        description='MTG RL data collection (headless)'
    )
    parser.add_argument('--games', type=int, default=1000)
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--deck', action='append')
    parser.add_argument('--zero-intermediate-reward', action='store_true')
    parser.add_argument('--output-dir', default=get_data_dir())
    parser.add_argument('--preprocessed-dir', default=get_preprocessed_dir())
    parser.add_argument('--skip-preprocess', action='store_true')
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--heap-mb', type=int, default=8192)
    parser.add_argument('--jar-path', default=None)
    args = parser.parse_args()

    state = CollectState()
    collect_thread(state, args)
    if state.status.startswith('ERROR'):
        sys.exit(1)


if __name__ == '__main__':
    main()
