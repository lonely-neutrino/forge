#!/usr/bin/env python3
"""Shared RL deck configuration."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, Iterable, List, Optional


PROJECT_ROOT = str(Path(__file__).resolve().parents[5])
RL_DECK_CONFIG = os.path.join(PROJECT_ROOT, 'rl_data', 'rl_decks.json')
DEFAULT_RL_DECKS = [
    'Green Stompy.dck',
    'White Weenie.dck',
    'Blue Tempo.dck',
    'Red Aggro.dck',
]


def _normalize_decks(decks: Iterable[str], source: str) -> List[str]:
    out: List[str] = []
    for raw in decks:
        if not isinstance(raw, str):
            raise ValueError(
                f'{source} must contain only deck filename strings')
        deck = raw.strip()
        if not deck:
            raise ValueError(
                f'{source} contains an empty deck name')
        out.append(deck)
    if not out:
        raise ValueError(
            f'{source} contains no decks')
    return out


def load_rl_decks(
        overrides: Optional[Iterable[str]] = None,
        config_path: Optional[str] = None,
        warn: Optional[Callable[[str], None]] = None) -> List[str]:
    """Load the shared RL deck list.

    Precedence:
    1. CLI overrides, if provided
    2. rl_data/rl_decks.json
    3. hardcoded defaults only if config is missing/unreadable
    """
    if overrides:
        return _normalize_decks(list(overrides), 'deck overrides')

    config_path = config_path or RL_DECK_CONFIG
    warn = warn or (lambda msg: print(msg, file=sys.stderr))

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
    except FileNotFoundError:
        warn(f'[RL] Deck config not found at {config_path}; '
             'falling back to built-in defaults.')
        return list(DEFAULT_RL_DECKS)
    except json.JSONDecodeError as exc:
        warn(f'[RL] Failed to parse deck config at {config_path}: {exc}; '
             'falling back to built-in defaults.')
        return list(DEFAULT_RL_DECKS)
    except OSError as exc:
        warn(f'[RL] Failed to read deck config at {config_path}: {exc}; '
             'falling back to built-in defaults.')
        return list(DEFAULT_RL_DECKS)

    decks = payload.get('decks')
    if not isinstance(decks, list):
        warn(f"[RL] Deck config at {config_path} is missing a valid 'decks' list; "
             'falling back to built-in defaults.')
        return list(DEFAULT_RL_DECKS)

    return _normalize_decks(decks, f'deck config {config_path}')


def deck_display_name(deck_filename: str) -> str:
    """Convert 'Green Stompy.dck' to 'Green Stompy'."""
    if deck_filename.lower().endswith('.dck'):
        return deck_filename[:-4]
    return deck_filename


def main():
    parser = argparse.ArgumentParser(
        description='Print the shared RL deck list.')
    parser.add_argument('--deck', action='append',
                        help='Override the configured deck list '
                             '(may be repeated)')
    args = parser.parse_args()

    decks = load_rl_decks(overrides=args.deck)
    for deck in decks:
        print(deck)


if __name__ == '__main__':
    main()
