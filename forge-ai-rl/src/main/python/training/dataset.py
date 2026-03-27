"""
Dataset — PyTorch Dataset for loading trajectory files.
Converts JSONL trajectory data into tensors for training.
"""

import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Optional
import logging

from model.backend import resolve_backend

logger = logging.getLogger(__name__)


class TrajectoryDataset(Dataset):
    """
    PyTorch Dataset that loads trajectory JSONL files and produces
    training samples for the value network and decision heads.

    Each sample contains:
    - game_state_flat: flattened game state feature vector
    - global_features: global game state features (64 floats)
    - won: binary label (1 = win, 0 = loss)
    - decision_type: string identifying the decision type
    - selected_indices: what the heuristic AI chose
    """

    def __init__(self, data_dir: str, max_files: int = None,
                 global_dim: int = 96, card_dim: int = 256,
                 max_board: int = 40, max_hand: int = 15,
                 max_gy: int = 20, max_stack: int = 10):
        self.data_dir = data_dir
        self.global_dim = global_dim
        self.card_dim = card_dim
        self.max_board = max_board
        self.max_hand = max_hand
        self.max_gy = max_gy
        self.max_stack = max_stack

        # Expected total flat size per game state
        # global + (my_board + opp_board)*card + hand*card +
        # (my_gy + opp_gy)*card + stack*card
        self.zones_config = [
            ('my_board', max_board, card_dim),
            ('opp_board', max_board, card_dim),
            ('hand', max_hand, card_dim),
            ('my_gy', max_gy, card_dim),
            ('opp_gy', max_gy, card_dim),
            ('stack', max_stack, card_dim),
        ]
        self.flat_size = global_dim
        for _, count, dim in self.zones_config:
            self.flat_size += count * dim

        self.samples = []
        self._load_files(max_files)

    def _load_files(self, max_files: Optional[int]):
        """Load all trajectory files from the data directory."""
        path = Path(self.data_dir)
        if not path.exists():
            logger.warning(f"Data directory not found: {self.data_dir}")
            return

        files = sorted(path.glob('traj_*.jsonl'))
        if max_files:
            files = files[:max_files]

        wins = 0
        losses = 0
        for filepath in files:
            try:
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                if len(lines) < 2:
                    continue

                header = json.loads(lines[0])
                won = header.get('won', False)
                if won:
                    wins += 1
                else:
                    losses += 1

                # Each subsequent line is a decision record
                for line in lines[1:]:
                    record = json.loads(line)
                    self.samples.append({
                        'game_state_flat': np.array(
                            record.get('gameStateFlat', []),
                            dtype=np.float32),
                        'global_features': np.array(
                            record.get('globalFeatures', []),
                            dtype=np.float32),
                        'won': 1.0 if won else 0.0,
                        'terminal_reward': record.get(
                            'terminalReward', 0.0),
                        'intermediate_reward': record.get(
                            'intermediateReward', 0.0),
                        'decision_type': record.get(
                            'decisionType', ''),
                        'value_estimate': record.get(
                            'valueEstimate', 0.0),
                    })
            except Exception as e:
                logger.debug(f"Error loading {filepath}: {e}")

        logger.info(
            f"Loaded {len(self.samples)} samples from "
            f"{len(files)} files "
            f"(wins: {wins}, losses: {losses})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Parse flat game state into structured tensors
        flat = sample['game_state_flat'].copy()
        global_feats = sample['global_features'].copy()

        # Clamp extreme values (from card hash encoding bug)
        np.clip(flat, -10.0, 10.0, out=flat)
        flat = np.nan_to_num(flat, nan=0.0, posinf=1.0,
                             neginf=-1.0)
        np.clip(global_feats, -10.0, 10.0, out=global_feats)
        global_feats = np.nan_to_num(
            global_feats, nan=0.0, posinf=1.0, neginf=-1.0)

        # Pad or truncate global features
        g = np.zeros(self.global_dim, dtype=np.float32)
        g_len = min(len(global_feats), self.global_dim)
        if g_len > 0:
            g[:g_len] = global_feats[:g_len]

        # Extract zone features from flat array
        zones = {}
        masks = {}
        offset = self.global_dim  # skip global features
        for name, count, dim in self.zones_config:
            zone_size = count * dim
            zone_data = np.zeros(
                (count, dim), dtype=np.float32)
            zone_mask = np.zeros(count, dtype=np.bool_)

            if offset + zone_size <= len(flat):
                raw = flat[offset:offset + zone_size].reshape(
                    count, dim)
                # A card slot is "real" if it has any nonzero
                for i in range(count):
                    if np.any(raw[i] != 0):
                        zone_data[i] = raw[i]
                        zone_mask[i] = True
            offset += zone_size

            zones[name] = zone_data
            masks[name + '_mask'] = zone_mask

        return {
            'global_features': torch.from_numpy(g),
            'my_board': torch.from_numpy(
                zones['my_board']),
            'my_board_mask': torch.from_numpy(
                masks['my_board_mask']),
            'opp_board': torch.from_numpy(
                zones['opp_board']),
            'opp_board_mask': torch.from_numpy(
                masks['opp_board_mask']),
            'hand': torch.from_numpy(zones['hand']),
            'hand_mask': torch.from_numpy(
                masks['hand_mask']),
            'my_gy': torch.from_numpy(zones['my_gy']),
            'my_gy_mask': torch.from_numpy(
                masks['my_gy_mask']),
            'opp_gy': torch.from_numpy(zones['opp_gy']),
            'opp_gy_mask': torch.from_numpy(
                masks['opp_gy_mask']),
            'stack': torch.from_numpy(zones['stack']),
            'stack_mask': torch.from_numpy(
                masks['stack_mask']),
            'won': torch.tensor(
                sample['won'], dtype=torch.float32),
            'value_target': torch.tensor(
                1.0 if sample['won'] > 0.5 else -1.0,
                dtype=torch.float32),
        }


def create_dataloader(data_dir: str, batch_size: int = 64,
                      shuffle: bool = True,
                      num_workers: int = 4,
                      max_files: int = None) -> DataLoader:
    """Create a DataLoader from trajectory files."""
    dataset = TrajectoryDataset(data_dir, max_files=max_files)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=resolve_backend().pin_memory,
        drop_last=True,
    )
