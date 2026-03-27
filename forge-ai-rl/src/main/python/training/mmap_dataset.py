"""
Memory-mapped datasets for training.

Loads preprocessed numpy arrays with mmap_mode='r', keeping RAM
usage at O(batch_size) regardless of dataset size.

Game state is stored once in shared/ and accessed via gs_index
indirection from each decision type's dataset.

Usage:
    ds = MmapValueDataset('preprocessed/', train=True)
    loader = DataLoader(ds, batch_size=256, shuffle=True,
                        num_workers=0, collate_fn=lambda x: x)
"""

import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset


def _split_file_ids(file_ids, val_fraction=0.1, seed=42):
    """Split file IDs into train/val sets."""
    unique_ids = sorted(set(int(x) for x in file_ids))
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_ids)
    n_val = max(1, int(len(unique_ids) * val_fraction))
    val_ids = set(unique_ids[:n_val])
    train_ids = set(unique_ids[n_val:])
    return train_ids, val_ids


GLOBAL_DIM = 96
CARD_DIM = 256
ZONES_CONFIG = [('my_board', 40), ('opp_board', 40),
                ('hand', 15), ('my_gy', 20),
                ('opp_gy', 20), ('stack', 10)]
TOTAL_CARDS = sum(c for _, c in ZONES_CONFIG)  # 145
GAME_STATE_DIM = GLOBAL_DIM + TOTAL_CARDS * CARD_DIM  # 37,216


def parse_game_state(flat, global_feats):
    """Parse flat game state into zone tensors.
    Canonical implementation — all other files should import
    this function rather than defining their own copy."""
    g = np.zeros(GLOBAL_DIM, dtype=np.float32)
    gl = min(len(global_feats), GLOBAL_DIM)
    if gl > 0:
        g[:gl] = global_feats[:gl]
    zdata = {}
    zmask = {}
    offset = GLOBAL_DIM
    for name, count in ZONES_CONFIG:
        zs = count * CARD_DIM
        zd = np.zeros((count, CARD_DIM), dtype=np.float32)
        zm = np.zeros(count, dtype=np.bool_)
        if offset + zs <= len(flat):
            raw = flat[offset:offset + zs].reshape(
                count, CARD_DIM)
            for j in range(count):
                if np.any(raw[j] != 0):
                    zd[j] = raw[j]
                    zm[j] = True
        offset += zs
        zdata[name] = zd
        zmask[name + '_mask'] = zm
    return g, zdata, zmask


class SharedState:
    """Shared mmap arrays for game state.
    Loaded once, referenced by all datasets."""

    def __init__(self, preprocessed_dir):
        sh = os.path.join(preprocessed_dir, 'shared')
        self.game_state = np.load(
            os.path.join(sh, 'game_state.npy'),
            mmap_mode='r')
        self.global_features = np.load(
            os.path.join(sh, 'global_features.npy'),
            mmap_mode='r')
        self.outcome = np.load(
            os.path.join(sh, 'outcome.npy'),
            mmap_mode='r')
        self.file_id = np.load(
            os.path.join(sh, 'file_id.npy'),
            mmap_mode='r')


class MmapValueDataset(Dataset):
    """Memory-mapped dataset for value network training.
    Uses shared game state directly (every record is a
    value training sample)."""

    def __init__(self, preprocessed_dir, train=True,
                 val_fraction=0.1, shared=None):
        self.shared = shared or SharedState(
            preprocessed_dir)
        s = self.shared

        train_ids, val_ids = _split_file_ids(
            s.file_id, val_fraction)
        target_ids = train_ids if train else val_ids

        self.indices = np.array([
            i for i in range(len(s.file_id))
            if int(s.file_id[i]) in target_ids
        ], dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        s = self.shared
        flat = np.array(s.game_state[i])
        gf = np.array(s.global_features[i])

        g, zones, masks = parse_game_state(flat, gf)

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
            'value_target': torch.tensor(
                float(s.outcome[i]),
                dtype=torch.float32),
        }


class MmapPriorityDataset(Dataset):
    """Memory-mapped dataset for priority head.
    References shared game state via gs_index."""

    def __init__(self, preprocessed_dir, train=True,
                 val_fraction=0.1, shared=None):
        self.shared = shared or SharedState(
            preprocessed_dir)
        pd = os.path.join(preprocessed_dir, 'priority')
        self.gs_index = np.load(
            os.path.join(pd, 'gs_index.npy'),
            mmap_mode='r')
        self.candidates = np.load(
            os.path.join(pd, 'candidates.npy'),
            mmap_mode='r')
        self.candidate_mask = np.load(
            os.path.join(pd, 'candidate_mask.npy'),
            mmap_mode='r')
        self.selected_idx = np.load(
            os.path.join(pd, 'selected_idx.npy'),
            mmap_mode='r')
        self.action_probs = np.load(
            os.path.join(pd, 'action_probs.npy'),
            mmap_mode='r')

        # Train/val split using shared file_id
        s = self.shared
        train_ids, val_ids = _split_file_ids(
            s.file_id, val_fraction)
        target_ids = train_ids if train else val_ids

        self.indices = np.array([
            i for i in range(len(self.gs_index))
            if int(s.file_id[self.gs_index[i]])
            in target_ids
        ], dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        si = int(self.gs_index[i])
        s = self.shared

        mask = np.array(self.candidate_mask[i])
        n_actions = int(mask.sum())

        return {
            'global_features': np.array(
                s.global_features[si]),
            'game_state_flat': np.array(
                s.game_state[si]),
            'action_features': np.array(
                self.candidates[i, :n_actions]),
            'selected_idx': int(self.selected_idx[i]),
            'n_actions': n_actions,
            'won': float(s.outcome[si] > 0),
        }


class MmapAttackDataset(Dataset):
    """Memory-mapped dataset for attack head."""

    def __init__(self, preprocessed_dir, train=True,
                 val_fraction=0.1, shared=None):
        self.shared = shared or SharedState(
            preprocessed_dir)
        ad = os.path.join(preprocessed_dir, 'attack')
        self.gs_index = np.load(
            os.path.join(ad, 'gs_index.npy'),
            mmap_mode='r')
        self.creatures = np.load(
            os.path.join(ad, 'creatures.npy'),
            mmap_mode='r')
        self.creature_mask = np.load(
            os.path.join(ad, 'creature_mask.npy'),
            mmap_mode='r')
        self.action_mask = np.load(
            os.path.join(ad, 'action_mask.npy'),
            mmap_mode='r')
        self.action_probs = np.load(
            os.path.join(ad, 'action_probs.npy'),
            mmap_mode='r')

        s = self.shared
        train_ids, val_ids = _split_file_ids(
            s.file_id, val_fraction)
        target_ids = train_ids if train else val_ids

        self.indices = np.array([
            i for i in range(len(self.gs_index))
            if int(s.file_id[self.gs_index[i]])
            in target_ids
        ], dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        si = int(self.gs_index[i])
        s = self.shared

        cmask = np.array(self.creature_mask[i])
        n = int(cmask.sum())

        return {
            'global_features': np.array(
                s.global_features[si]),
            'game_state_flat': np.array(
                s.game_state[si]),
            'creature_features': np.array(
                self.creatures[i, :n]),
            'action_mask': np.array(
                self.action_mask[i, :n]),
            'n_creatures': n,
            'won': float(s.outcome[si] > 0),
        }


class MmapBlockDataset(Dataset):
    """Memory-mapped dataset for block head."""

    def __init__(self, preprocessed_dir, train=True,
                 val_fraction=0.1, shared=None):
        self.shared = shared or SharedState(
            preprocessed_dir)
        bd = os.path.join(preprocessed_dir, 'block')
        self.gs_index = np.load(
            os.path.join(bd, 'gs_index.npy'),
            mmap_mode='r')
        self.pairs = np.load(
            os.path.join(bd, 'pairs.npy'),
            mmap_mode='r')
        self.pair_mask = np.load(
            os.path.join(bd, 'pair_mask.npy'),
            mmap_mode='r')
        self.action_mask = np.load(
            os.path.join(bd, 'action_mask.npy'),
            mmap_mode='r')
        self.action_probs = np.load(
            os.path.join(bd, 'action_probs.npy'),
            mmap_mode='r')

        s = self.shared
        train_ids, val_ids = _split_file_ids(
            s.file_id, val_fraction)
        target_ids = train_ids if train else val_ids

        self.indices = np.array([
            i for i in range(len(self.gs_index))
            if int(s.file_id[self.gs_index[i]])
            in target_ids
        ], dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        si = int(self.gs_index[i])
        s = self.shared

        pmask = np.array(self.pair_mask[i])
        n = int(pmask.sum())

        return {
            'global_features': np.array(
                s.global_features[si]),
            'game_state_flat': np.array(
                s.game_state[si]),
            'pair_features': np.array(
                self.pairs[i, :n]),
            'action_mask': np.array(
                self.action_mask[i, :n]),
            'n_pairs': n,
            'won': float(s.outcome[si] > 0),
        }


class MmapTargetDataset(Dataset):
    """Memory-mapped dataset for target head.
    Single-select from 256-dim candidate features."""

    def __init__(self, preprocessed_dir, train=True,
                 val_fraction=0.1, shared=None):
        self.shared = shared or SharedState(
            preprocessed_dir)
        td = os.path.join(preprocessed_dir, 'target')
        if not os.path.isdir(td):
            self.indices = np.array([], dtype=np.int64)
            return
        self.gs_index = np.load(
            os.path.join(td, 'gs_index.npy'),
            mmap_mode='r')
        self.candidates = np.load(
            os.path.join(td, 'candidates.npy'),
            mmap_mode='r')
        self.candidate_mask = np.load(
            os.path.join(td, 'candidate_mask.npy'),
            mmap_mode='r')
        self.selected_idx = np.load(
            os.path.join(td, 'selected_idx.npy'),
            mmap_mode='r')
        spell_path = os.path.join(td, 'spell_features.npy')
        self.spell_features = np.load(
            spell_path, mmap_mode='r') \
            if os.path.exists(spell_path) else None

        s = self.shared
        train_ids, val_ids = _split_file_ids(
            s.file_id, val_fraction)
        target_ids = train_ids if train else val_ids

        self.indices = np.array([
            i for i in range(len(self.gs_index))
            if int(s.file_id[self.gs_index[i]])
            in target_ids
        ], dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        si = int(self.gs_index[i])
        s = self.shared

        mask = np.array(self.candidate_mask[i])
        n = int(mask.sum())

        return {
            'global_features': np.array(
                s.global_features[si]),
            'game_state_flat': np.array(
                s.game_state[si]),
            'candidate_features': np.array(
                self.candidates[i, :n]),
            'selected_idx': int(self.selected_idx[i]),
            'n_candidates': n,
            'won': float(s.outcome[si] > 0),
            'spell_features': np.array(
                self.spell_features[i]) \
                if self.spell_features is not None \
                else np.zeros(64, dtype=np.float32),
        }


class MmapCardSelectDataset(Dataset):
    """Memory-mapped dataset for card select head.
    Multi-select from 256-dim candidate features."""

    def __init__(self, preprocessed_dir, train=True,
                 val_fraction=0.1, shared=None):
        self.shared = shared or SharedState(
            preprocessed_dir)
        csd = os.path.join(preprocessed_dir, 'card_select')
        if not os.path.isdir(csd):
            self.indices = np.array([], dtype=np.int64)
            return
        self.gs_index = np.load(
            os.path.join(csd, 'gs_index.npy'),
            mmap_mode='r')
        self.candidates = np.load(
            os.path.join(csd, 'candidates.npy'),
            mmap_mode='r')
        self.candidate_mask = np.load(
            os.path.join(csd, 'candidate_mask.npy'),
            mmap_mode='r')
        self.action_mask = np.load(
            os.path.join(csd, 'action_mask.npy'),
            mmap_mode='r')

        s = self.shared
        train_ids, val_ids = _split_file_ids(
            s.file_id, val_fraction)
        target_ids = train_ids if train else val_ids

        self.indices = np.array([
            i for i in range(len(self.gs_index))
            if int(s.file_id[self.gs_index[i]])
            in target_ids
        ], dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        si = int(self.gs_index[i])
        s = self.shared

        cmask = np.array(self.candidate_mask[i])
        n = int(cmask.sum())

        return {
            'global_features': np.array(
                s.global_features[si]),
            'game_state_flat': np.array(
                s.game_state[si]),
            'candidate_features': np.array(
                self.candidates[i, :n]),
            'action_mask': np.array(
                self.action_mask[i, :n]),
            'n_candidates': n,
            'won': float(s.outcome[si] > 0),
        }


class MmapMulliganDataset(Dataset):
    """Memory-mapped dataset for mulligan head.
    Binary keep/mulligan + hand card features."""

    def __init__(self, preprocessed_dir, train=True,
                 val_fraction=0.1, shared=None):
        self.shared = shared or SharedState(
            preprocessed_dir)
        muld = os.path.join(preprocessed_dir, 'mulligan')
        if not os.path.isdir(muld):
            self.indices = np.array([], dtype=np.int64)
            return
        self.gs_index = np.load(
            os.path.join(muld, 'gs_index.npy'),
            mmap_mode='r')
        self.hand_features = np.load(
            os.path.join(muld, 'hand_features.npy'),
            mmap_mode='r')
        self.hand_mask = np.load(
            os.path.join(muld, 'hand_mask.npy'),
            mmap_mode='r')
        self.keep_decision = np.load(
            os.path.join(muld, 'keep_decision.npy'),
            mmap_mode='r')

        s = self.shared
        train_ids, val_ids = _split_file_ids(
            s.file_id, val_fraction)
        target_ids = train_ids if train else val_ids

        self.indices = np.array([
            i for i in range(len(self.gs_index))
            if int(s.file_id[self.gs_index[i]])
            in target_ids
        ], dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        si = int(self.gs_index[i])
        s = self.shared

        hmask = np.array(self.hand_mask[i])
        n = int(hmask.sum())

        return {
            'global_features': np.array(
                s.global_features[si]),
            'game_state_flat': np.array(
                s.game_state[si]),
            'hand_features': np.array(
                self.hand_features[i, :n]),
            'n_cards': n,
            'keep': float(self.keep_decision[i]),
            'won': float(s.outcome[si] > 0),
        }


class MmapBinaryDataset(Dataset):
    """Memory-mapped dataset for binary head.
    Just game state + yes/no decision."""

    def __init__(self, preprocessed_dir, train=True,
                 val_fraction=0.1, shared=None):
        self.shared = shared or SharedState(
            preprocessed_dir)
        bind = os.path.join(preprocessed_dir, 'binary')
        if not os.path.isdir(bind):
            self.indices = np.array([], dtype=np.int64)
            return
        self.gs_index = np.load(
            os.path.join(bind, 'gs_index.npy'),
            mmap_mode='r')
        self.decision = np.load(
            os.path.join(bind, 'decision.npy'),
            mmap_mode='r')

        s = self.shared
        train_ids, val_ids = _split_file_ids(
            s.file_id, val_fraction)
        target_ids = train_ids if train else val_ids

        self.indices = np.array([
            i for i in range(len(self.gs_index))
            if int(s.file_id[self.gs_index[i]])
            in target_ids
        ], dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        si = int(self.gs_index[i])
        s = self.shared

        return {
            'global_features': np.array(
                s.global_features[si]),
            'game_state_flat': np.array(
                s.game_state[si]),
            'decision': float(self.decision[i]),
            'won': float(s.outcome[si] > 0),
        }
