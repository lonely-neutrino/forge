#!/usr/bin/env python3
"""
Decision Head Training Dashboard — Tkinter GUI for monitoring
attack and block head imitation learning.

Launch:
    python training/train_decisions_ui.py \
        --data-dir /path/to/trajectories \
        --device cuda --epochs 50
"""

import argparse
import json
import os
import sys
import threading
import time
import random
from dataclasses import dataclass, field
from contextlib import nullcontext
from typing import List
from pathlib import Path

import tkinter as tk
from tkinter import ttk

os.environ['PYTHONUNBUFFERED'] = '1'

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg)
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from model.mtg_model import MTGModel
from model.backend import is_cuda_device, resolve_backend
from model.gpu_config import auto_detect_profile
from training.mmap_dataset import parse_game_state, GAME_STATE_DIM, CARD_DIM, GLOBAL_DIM, ZONES_CONFIG


# ── Shared state ─────────────────────────────────────

@dataclass
class TrainingState:
    status: str = "Idle"
    phase: str = ""
    chart_dirty: bool = False
    log_lines: List[str] = field(default_factory=list)
    log_dirty: bool = False

    # Loading
    files_total: int = 0
    files_loaded: int = 0
    attack_samples: int = 0
    block_samples: int = 0
    priority_samples: int = 0

    # Config
    device: str = "cpu"
    gpu_name: str = ""
    encoder_params: int = 0
    head_params: int = 0

    # Current head being trained
    current_head: str = ""
    epoch: int = 0
    total_epochs: int = 50
    epoch_progress: float = 0.0

    # Metrics per head
    attack_train_losses: List[float] = field(
        default_factory=list)
    attack_train_accs: List[float] = field(
        default_factory=list)
    attack_val_losses: List[float] = field(
        default_factory=list)
    attack_val_accs: List[float] = field(
        default_factory=list)
    attack_best_acc: float = 0.0

    block_train_losses: List[float] = field(
        default_factory=list)
    block_train_accs: List[float] = field(
        default_factory=list)
    block_val_losses: List[float] = field(
        default_factory=list)
    block_val_accs: List[float] = field(
        default_factory=list)
    block_best_acc: float = 0.0

    priority_train_losses: List[float] = field(
        default_factory=list)
    priority_train_accs: List[float] = field(
        default_factory=list)
    priority_val_losses: List[float] = field(
        default_factory=list)
    priority_val_accs: List[float] = field(
        default_factory=list)
    priority_best_acc: float = 0.0

    target_train_losses: List[float] = field(
        default_factory=list)
    target_train_accs: List[float] = field(
        default_factory=list)
    target_val_losses: List[float] = field(
        default_factory=list)
    target_val_accs: List[float] = field(
        default_factory=list)
    target_best_acc: float = 0.0

    card_select_train_losses: List[float] = field(
        default_factory=list)
    card_select_train_accs: List[float] = field(
        default_factory=list)
    card_select_val_losses: List[float] = field(
        default_factory=list)
    card_select_val_accs: List[float] = field(
        default_factory=list)
    card_select_best_acc: float = 0.0

    mulligan_train_losses: List[float] = field(
        default_factory=list)
    mulligan_train_accs: List[float] = field(
        default_factory=list)
    mulligan_val_losses: List[float] = field(
        default_factory=list)
    mulligan_val_accs: List[float] = field(
        default_factory=list)
    mulligan_best_acc: float = 0.0

    binary_train_losses: List[float] = field(
        default_factory=list)
    binary_train_accs: List[float] = field(
        default_factory=list)
    binary_val_losses: List[float] = field(
        default_factory=list)
    binary_val_accs: List[float] = field(
        default_factory=list)
    binary_best_acc: float = 0.0

    current_train_loss: float = 0.0
    current_train_acc: float = 0.0
    current_val_loss: float = 0.0
    current_val_acc: float = 0.0
    epoch_time: float = 0.0
    elapsed: float = 0.0
    eta: float = 0.0

    gpu_mem_used_mb: float = 0.0
    gpu_mem_total_mb: float = 0.0


def log(state, msg):
    """Thread-safe logging to console and state."""
    print(msg, flush=True)
    state.log_lines.append(msg)
    if len(state.log_lines) > 200:
        state.log_lines = state.log_lines[-200:]
    state.log_dirty = True


# ── Data loading ─────────────────────────────────────

## parse_game_state imported from mmap_dataset


def load_decisions(data_dir, state, max_files=None,
                   heads=None, chunk_size=200):
    """Load decision samples in chunks to bound memory.

    For priority (140K+ samples), loads chunk_size files
    at a time. For attack/block (small), loads all at once.
    """
    from training.chunked_loader import load_decision_samples

    state.phase = "loading"
    state.status = "Loading trajectory files..."

    path = Path(data_dir)
    files = sorted(path.glob('traj_*.jsonl'))
    if max_files:
        files = files[:max_files]
    state.files_total = len(files)

    attack_samples = []
    block_samples = []
    priority_samples = []

    # Load in chunks to bound memory
    for ci in range(0, len(files), chunk_size):
        chunk_files = files[ci:ci + chunk_size]
        a, b, p = load_decision_samples(
            chunk_files, heads=
            list(heads) if heads else None)
        attack_samples.extend(a)
        block_samples.extend(b)
        priority_samples.extend(p)
        state.files_loaded = min(ci + chunk_size,
                                 len(files))
        state.attack_samples = len(attack_samples)
        state.block_samples = len(block_samples)
        state.priority_samples = len(priority_samples)

    state.status = (
        f"Loaded {len(attack_samples)} attack, "
        f"{len(block_samples)} block, "
        f"{len(priority_samples)} priority decisions")
    return attack_samples, block_samples, priority_samples


def load_decisions_old(data_dir, state, max_files=None,
                   heads=None):
    """Original load function — kept as fallback."""
    state.phase = "loading"
    state.status = "Loading trajectory files..."

    path = Path(data_dir)
    files = sorted(path.glob('traj_*.jsonl'))
    if max_files:
        files = files[:max_files]
    state.files_total = len(files)

    attack_samples = []
    block_samples = []
    priority_samples = []

    for i, filepath in enumerate(files):
        state.files_loaded = i + 1
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            if len(lines) < 2:
                continue
            header = json.loads(lines[0])
            won = header.get('won', False)

            for line in lines[1:]:
                rec = json.loads(line)
                dt = rec.get('decisionType', '')

                # Skip decision types we're not training
                if heads:
                    if dt == 'PRIORITY_ACTION' and \
                            'priority' not in heads:
                        continue
                    if dt == 'DECLARE_ATTACKERS' and \
                            'attack' not in heads:
                        continue
                    if dt == 'DECLARE_BLOCKERS' and \
                            'block' not in heads:
                        continue

                cand = rec.get('candidateFeatures', [])
                sel = rec.get('selectedIndices', [])

                if len(cand) < 1:
                    continue

                gf = np.array(
                    rec.get('globalFeatures', []),
                    dtype=np.float32)
                np.clip(gf, -10, 10, out=gf)
                gf = np.nan_to_num(gf)
                g = np.zeros(64, dtype=np.float32)
                gl = min(len(gf), 64)
                if gl > 0:
                    g[:gl] = gf[:gl]

                flat = np.array(
                    rec.get('gameStateFlat', []),
                    dtype=np.float32)
                np.clip(flat, -10, 10, out=flat)
                flat = np.nan_to_num(flat)

                if dt == 'PRIORITY_ACTION':
                    # Priority: 64-dim action features,
                    # single-select (softmax CE)
                    n = len(cand)
                    actions = np.zeros(
                        (n, 64), dtype=np.float32)
                    for j, cf in enumerate(cand):
                        cl = min(len(cf), 64)
                        actions[j, :cl] = np.array(
                            cf[:cl], dtype=np.float32)
                    np.clip(actions, -10, 10, out=actions)
                    actions = np.nan_to_num(actions)

                    # Selected index (single-select)
                    selected_idx = sel[0] if sel else n - 1
                    if selected_idx >= n:
                        selected_idx = n - 1

                    priority_samples.append({
                        'global_features': g,
                        'game_state_flat': flat,
                        'action_features': actions,
                        'selected_idx': selected_idx,
                        'n_actions': n,
                        'won': 1.0 if won else 0.0,
                    })
                    state.priority_samples = len(
                        priority_samples)
                    continue

                if dt == 'DECLARE_ATTACKERS':
                    # Attack: card features,
                    # multi-select (binary BCE)
                    n = len(cand)
                    creatures = np.zeros(
                        (n, CARD_DIM), dtype=np.float32)
                    for j, cf in enumerate(cand):
                        cl = min(len(cf), CARD_DIM)
                        creatures[j, :cl] = np.array(
                            cf[:cl], dtype=np.float32)
                    np.clip(creatures, -10, 10,
                            out=creatures)
                    creatures = np.nan_to_num(creatures)

                    mask = np.zeros(n, dtype=np.float32)
                    for idx in sel:
                        if 0 <= idx < n:
                            mask[idx] = 1.0

                    attack_samples.append({
                        'global_features': g,
                        'game_state_flat': flat,
                        'creature_features': creatures,
                        'action_mask': mask,
                        'n_creatures': n,
                        'won': 1.0 if won else 0.0,
                    })
                    state.attack_samples = len(
                        attack_samples)

                elif dt == 'DECLARE_BLOCKERS':
                    # Block: 256-dim pair features
                    # (blocker+attacker concat),
                    # multi-select assignment
                    n = len(cand)
                    feat_dim = len(cand[0]) if cand else 0
                    if feat_dim < 200:
                        continue  # skip old-format data

                    pairs = np.zeros(
                        (n, 256), dtype=np.float32)
                    for j, cf in enumerate(cand):
                        cl = min(len(cf), 256)
                        pairs[j, :cl] = np.array(
                            cf[:cl], dtype=np.float32)
                    np.clip(pairs, -10, 10, out=pairs)
                    pairs = np.nan_to_num(pairs)

                    mask = np.zeros(n, dtype=np.float32)
                    for idx in sel:
                        if 0 <= idx < n:
                            mask[idx] = 1.0

                    block_samples.append({
                        'global_features': g,
                        'game_state_flat': flat,
                        'pair_features': pairs,
                        'action_mask': mask,
                        'n_pairs': n,
                        'won': 1.0 if won else 0.0,
                    })
                    state.block_samples = len(
                        block_samples)
        except Exception:
            pass

    state.status = (
        f"Loaded {len(attack_samples)} attack, "
        f"{len(block_samples)} block, "
        f"{len(priority_samples)} priority decisions")
    return attack_samples, block_samples, priority_samples


# ── Batch processing ─────────────────────────────────

def _get_n_items(s):
    """Get item count from sample — handles different field names."""
    for key in ('n_creatures', 'n_candidates', 'n_cards'):
        if key in s:
            return s[key]
    return 1

def _get_features(s):
    """Get feature array from sample — handles different field names."""
    for key in ('creature_features', 'candidate_features', 'hand_features'):
        if key in s:
            return s[key]
    return None

def make_batch(model, batch, device, use_amp, head,
               encoder_grad=False):
    max_c = max(_get_n_items(s) for s in batch)
    max_c = max(max_c, 1)
    bs = len(batch)
    cd = CARD_DIM

    cf = torch.zeros(bs, max_c, cd, device=device)
    cm = torch.zeros(bs, max_c, dtype=torch.bool,
                      device=device)
    tgt = torch.zeros(bs, max_c, device=device)
    gf = torch.zeros(bs, GLOBAL_DIM, device=device)

    mb = torch.zeros(bs, 40, cd, device=device)
    mbm = torch.zeros(bs, 40, dtype=torch.bool,
                       device=device)
    ob = torch.zeros(bs, 40, cd, device=device)
    obm = torch.zeros(bs, 40, dtype=torch.bool,
                       device=device)
    h = torch.zeros(bs, 15, cd, device=device)
    hm = torch.zeros(bs, 15, dtype=torch.bool,
                      device=device)
    mg = torch.zeros(bs, 20, cd, device=device)
    mgm = torch.zeros(bs, 20, dtype=torch.bool,
                       device=device)
    og = torch.zeros(bs, 20, cd, device=device)
    ogm = torch.zeros(bs, 20, dtype=torch.bool,
                       device=device)
    st = torch.zeros(bs, 10, cd, device=device)
    stm = torch.zeros(bs, 10, dtype=torch.bool,
                       device=device)

    for i, s in enumerate(batch):
        nc = _get_n_items(s)
        feats = _get_features(s)
        if feats is not None:
            cf[i, :nc] = torch.from_numpy(feats)
        cm[i, :nc] = True
        amask = s.get('action_mask')
        if amask is not None:
            tgt[i, :nc] = torch.from_numpy(amask)
        else:
            # Mulligan: keep=1.0 means all cards selected
            keep = s.get('keep', 0.0)
            tgt[i, :nc] = keep

        g, zones, masks = parse_game_state(
            s['game_state_flat'], s['global_features'])
        gf[i] = torch.from_numpy(g)
        mb[i] = torch.from_numpy(zones['my_board'])
        mbm[i] = torch.from_numpy(masks['my_board_mask'])
        ob[i] = torch.from_numpy(zones['opp_board'])
        obm[i] = torch.from_numpy(
            masks['opp_board_mask'])
        h[i] = torch.from_numpy(zones['hand'])
        hm[i] = torch.from_numpy(masks['hand_mask'])
        mg[i] = torch.from_numpy(zones['my_gy'])
        mgm[i] = torch.from_numpy(masks['my_gy_mask'])
        og[i] = torch.from_numpy(zones['opp_gy'])
        ogm[i] = torch.from_numpy(masks['opp_gy_mask'])
        st[i] = torch.from_numpy(zones['stack'])
        stm[i] = torch.from_numpy(masks['stack_mask'])

    with torch.amp.autocast('cuda', enabled=use_amp):
        ctx = torch.no_grad() if not encoder_grad \
            else nullcontext()
        with ctx:
            state_emb = model.encode_state(
                gf, mb, mbm, ob, obm, h, hm,
                mg, mgm, og, ogm, st, stm)

        logits = head(state_emb, cf, cm)

        # Only compute loss on real creatures (mask=True)
        # Replace padding logits with 0 to avoid NaN in BCE
        safe_logits = logits.clone()
        safe_logits[~cm] = 0.0
        safe_tgt = tgt.clone()
        safe_tgt[~cm] = 0.0

        loss = nn.functional.binary_cross_entropy_with_logits(
            safe_logits, safe_tgt, reduction='none')
        loss = (loss * cm.float()).sum() / \
               cm.float().sum().clamp(min=1)

    with torch.no_grad():
        preds = (logits > 0).float()
        correct = ((preds == tgt) *
                   cm.float()).sum().item()
        total = cm.float().sum().item()

    return loss, correct, total


def make_priority_batch(model, batch, device, use_amp,
                        head, encoder_grad=False):
    """Build a priority batch with CrossEntropyLoss
    (single-select softmax)."""
    max_a = max(s['n_actions'] for s in batch)
    max_a = max(max_a, 1)
    bs = len(batch)

    cd = CARD_DIM
    af = torch.zeros(bs, max_a, 64, device=device)
    am = torch.zeros(bs, max_a, dtype=torch.bool,
                      device=device)
    tgt = torch.zeros(bs, dtype=torch.long, device=device)
    gf = torch.zeros(bs, GLOBAL_DIM, device=device)

    mb = torch.zeros(bs, 40, cd, device=device)
    mbm = torch.zeros(bs, 40, dtype=torch.bool,
                       device=device)
    ob = torch.zeros(bs, 40, cd, device=device)
    obm = torch.zeros(bs, 40, dtype=torch.bool,
                       device=device)
    h = torch.zeros(bs, 15, cd, device=device)
    hm = torch.zeros(bs, 15, dtype=torch.bool,
                      device=device)
    mg = torch.zeros(bs, 20, cd, device=device)
    mgm = torch.zeros(bs, 20, dtype=torch.bool,
                       device=device)
    og = torch.zeros(bs, 20, cd, device=device)
    ogm = torch.zeros(bs, 20, dtype=torch.bool,
                       device=device)
    st = torch.zeros(bs, 10, cd, device=device)
    stm = torch.zeros(bs, 10, dtype=torch.bool,
                       device=device)

    for i, s in enumerate(batch):
        na = s['n_actions']
        af[i, :na] = torch.from_numpy(
            s['action_features'])
        am[i, :na] = True
        tgt[i] = s['selected_idx']

        g, zones, masks = parse_game_state(
            s['game_state_flat'], s['global_features'])
        gf[i] = torch.from_numpy(g)
        mb[i] = torch.from_numpy(zones['my_board'])
        mbm[i] = torch.from_numpy(masks['my_board_mask'])
        ob[i] = torch.from_numpy(zones['opp_board'])
        obm[i] = torch.from_numpy(
            masks['opp_board_mask'])
        h[i] = torch.from_numpy(zones['hand'])
        hm[i] = torch.from_numpy(masks['hand_mask'])
        mg[i] = torch.from_numpy(zones['my_gy'])
        mgm[i] = torch.from_numpy(masks['my_gy_mask'])
        og[i] = torch.from_numpy(zones['opp_gy'])
        ogm[i] = torch.from_numpy(masks['opp_gy_mask'])
        st[i] = torch.from_numpy(zones['stack'])
        stm[i] = torch.from_numpy(masks['stack_mask'])

    # Class weighting: upweight play decisions vs pass
    # Pass is always the last action (index n_actions - 1)
    is_pass = torch.tensor(
        [s['selected_idx'] == s['n_actions'] - 1
         for s in batch], device=device)
    n_pass = is_pass.sum().item()
    n_play = bs - n_pass
    if n_pass > 0 and n_play > 0:
        # Inverse frequency: play gets weight n_pass/n_total,
        # pass gets weight n_play/n_total
        play_weight = n_pass / bs
        pass_weight = n_play / bs
        sample_weights = torch.where(
            is_pass, pass_weight, play_weight)
    else:
        sample_weights = torch.ones(bs, device=device)

    with torch.amp.autocast('cuda', enabled=use_amp):
        ctx = torch.no_grad() if not encoder_grad \
            else nullcontext()
        with ctx:
            state_emb = model.encode_state(
                gf, mb, mbm, ob, obm, h, hm,
                mg, mgm, og, ogm, st, stm)

        logits = head(state_emb, af, am)

        loss = nn.functional.cross_entropy(
            logits, tgt, reduction='none')
        loss = (loss * sample_weights).mean()

    with torch.no_grad():
        preds = logits.argmax(dim=1)
        correct = (preds == tgt).sum().item()
        total = bs

    return loss, correct, total


def make_target_batch(model, batch, device, use_amp,
                      head, encoder_grad=False):
    """Build a target batch with CrossEntropyLoss
    (single-select softmax, 256-dim candidate features)."""
    max_c = max(s['n_candidates'] for s in batch)
    max_c = max(max_c, 1)
    bs = len(batch)
    cd = CARD_DIM

    cf = torch.zeros(bs, max_c, cd, device=device)
    cm = torch.zeros(bs, max_c, dtype=torch.bool,
                      device=device)
    tgt = torch.zeros(bs, dtype=torch.long, device=device)
    gf = torch.zeros(bs, GLOBAL_DIM, device=device)

    mb = torch.zeros(bs, 40, cd, device=device)
    mbm = torch.zeros(bs, 40, dtype=torch.bool, device=device)
    ob = torch.zeros(bs, 40, cd, device=device)
    obm = torch.zeros(bs, 40, dtype=torch.bool, device=device)
    h = torch.zeros(bs, 15, cd, device=device)
    hm = torch.zeros(bs, 15, dtype=torch.bool, device=device)
    mg = torch.zeros(bs, 20, cd, device=device)
    mgm = torch.zeros(bs, 20, dtype=torch.bool, device=device)
    og = torch.zeros(bs, 20, cd, device=device)
    ogm = torch.zeros(bs, 20, dtype=torch.bool, device=device)
    st = torch.zeros(bs, 10, cd, device=device)
    stm = torch.zeros(bs, 10, dtype=torch.bool, device=device)

    for i, s in enumerate(batch):
        nc = s['n_candidates']
        cf[i, :nc] = torch.from_numpy(s['candidate_features'])
        cm[i, :nc] = True
        tgt[i] = s['selected_idx']

        g, zones, masks = parse_game_state(
            s['game_state_flat'], s['global_features'])
        gf[i] = torch.from_numpy(g)
        mb[i] = torch.from_numpy(zones['my_board'])
        mbm[i] = torch.from_numpy(masks['my_board_mask'])
        ob[i] = torch.from_numpy(zones['opp_board'])
        obm[i] = torch.from_numpy(masks['opp_board_mask'])
        h[i] = torch.from_numpy(zones['hand'])
        hm[i] = torch.from_numpy(masks['hand_mask'])
        mg[i] = torch.from_numpy(zones['my_gy'])
        mgm[i] = torch.from_numpy(masks['my_gy_mask'])
        og[i] = torch.from_numpy(zones['opp_gy'])
        ogm[i] = torch.from_numpy(masks['opp_gy_mask'])
        st[i] = torch.from_numpy(zones['stack'])
        stm[i] = torch.from_numpy(masks['stack_mask'])

    with torch.amp.autocast('cuda', enabled=use_amp):
        ctx = torch.no_grad() if not encoder_grad \
            else nullcontext()
        with ctx:
            state_emb = model.encode_state(
                gf, mb, mbm, ob, obm, h, hm,
                mg, mgm, og, ogm, st, stm)

        logits = head(state_emb, cf, cm)
        loss = nn.functional.cross_entropy(logits, tgt)

    with torch.no_grad():
        preds = logits.argmax(dim=1)
        correct = (preds == tgt).sum().item()

    return loss, correct, bs


def make_mulligan_batch(model, batch, device, use_amp, head,
                        encoder_grad=False):
    """Build a mulligan batch — binary keep/mull from hand features."""
    max_c = max(s.get('n_cards', 7) for s in batch)
    max_c = max(max_c, 1)
    bs = len(batch)
    cd = CARD_DIM

    hf = torch.zeros(bs, max_c, cd, device=device)
    hm = torch.zeros(bs, max_c, dtype=torch.bool, device=device)
    tgt = torch.zeros(bs, device=device)
    gf = torch.zeros(bs, GLOBAL_DIM, device=device)

    mb = torch.zeros(bs, 40, cd, device=device)
    mbm = torch.zeros(bs, 40, dtype=torch.bool, device=device)
    ob = torch.zeros(bs, 40, cd, device=device)
    obm = torch.zeros(bs, 40, dtype=torch.bool, device=device)
    h = torch.zeros(bs, 15, cd, device=device)
    hmask = torch.zeros(bs, 15, dtype=torch.bool, device=device)
    mg = torch.zeros(bs, 20, cd, device=device)
    mgm = torch.zeros(bs, 20, dtype=torch.bool, device=device)
    og = torch.zeros(bs, 20, cd, device=device)
    ogm = torch.zeros(bs, 20, dtype=torch.bool, device=device)
    st = torch.zeros(bs, 10, cd, device=device)
    stm = torch.zeros(bs, 10, dtype=torch.bool, device=device)

    for i, s in enumerate(batch):
        nc = s.get('n_cards', 0)
        feats = s.get('hand_features')
        if feats is not None and nc > 0:
            hf[i, :nc] = torch.from_numpy(feats)
        hm[i, :nc] = True
        tgt[i] = s.get('keep', 0.0)

        g, zones, masks = parse_game_state(
            s['game_state_flat'], s['global_features'])
        gf[i] = torch.from_numpy(g)
        mb[i] = torch.from_numpy(zones['my_board'])
        mbm[i] = torch.from_numpy(masks['my_board_mask'])
        ob[i] = torch.from_numpy(zones['opp_board'])
        obm[i] = torch.from_numpy(masks['opp_board_mask'])
        h[i] = torch.from_numpy(zones['hand'])
        hmask[i] = torch.from_numpy(masks['hand_mask'])
        mg[i] = torch.from_numpy(zones['my_gy'])
        mgm[i] = torch.from_numpy(masks['my_gy_mask'])
        og[i] = torch.from_numpy(zones['opp_gy'])
        ogm[i] = torch.from_numpy(masks['opp_gy_mask'])
        st[i] = torch.from_numpy(zones['stack'])
        stm[i] = torch.from_numpy(masks['stack_mask'])

    with torch.amp.autocast('cuda', enabled=use_amp):
        ctx = torch.no_grad() if not encoder_grad \
            else nullcontext()
        with ctx:
            state_emb = model.encode_state(
                gf, mb, mbm, ob, obm, h, hmask,
                mg, mgm, og, ogm, st, stm)

        logits = head(state_emb, hf, hm)  # (bs,) keep logit
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, tgt)

    with torch.no_grad():
        preds = (logits > 0).float()
        correct = (preds == tgt).sum().item()

    return loss, correct, bs


def make_binary_batch(model, batch, device, use_amp, head,
                      encoder_grad=False):
    """Build a binary batch — yes/no from game state only."""
    bs = len(batch)
    cd = CARD_DIM

    tgt = torch.zeros(bs, device=device)
    gf = torch.zeros(bs, GLOBAL_DIM, device=device)

    mb = torch.zeros(bs, 40, cd, device=device)
    mbm = torch.zeros(bs, 40, dtype=torch.bool, device=device)
    ob = torch.zeros(bs, 40, cd, device=device)
    obm = torch.zeros(bs, 40, dtype=torch.bool, device=device)
    h = torch.zeros(bs, 15, cd, device=device)
    hm = torch.zeros(bs, 15, dtype=torch.bool, device=device)
    mg = torch.zeros(bs, 20, cd, device=device)
    mgm = torch.zeros(bs, 20, dtype=torch.bool, device=device)
    og = torch.zeros(bs, 20, cd, device=device)
    ogm = torch.zeros(bs, 20, dtype=torch.bool, device=device)
    st = torch.zeros(bs, 10, cd, device=device)
    stm = torch.zeros(bs, 10, dtype=torch.bool, device=device)

    for i, s in enumerate(batch):
        tgt[i] = s.get('decision', 0.0)

        g, zones, masks = parse_game_state(
            s['game_state_flat'], s['global_features'])
        gf[i] = torch.from_numpy(g)
        mb[i] = torch.from_numpy(zones['my_board'])
        mbm[i] = torch.from_numpy(masks['my_board_mask'])
        ob[i] = torch.from_numpy(zones['opp_board'])
        obm[i] = torch.from_numpy(masks['opp_board_mask'])
        h[i] = torch.from_numpy(zones['hand'])
        hm[i] = torch.from_numpy(masks['hand_mask'])
        mg[i] = torch.from_numpy(zones['my_gy'])
        mgm[i] = torch.from_numpy(masks['my_gy_mask'])
        og[i] = torch.from_numpy(zones['opp_gy'])
        ogm[i] = torch.from_numpy(masks['opp_gy_mask'])
        st[i] = torch.from_numpy(zones['stack'])
        stm[i] = torch.from_numpy(masks['stack_mask'])

    with torch.amp.autocast('cuda', enabled=use_amp):
        ctx = torch.no_grad() if not encoder_grad \
            else nullcontext()
        with ctx:
            state_emb = model.encode_state(
                gf, mb, mbm, ob, obm, h, hm,
                mg, mgm, og, ogm, st, stm)

        logits = head(state_emb)  # (bs,) binary logit
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, tgt)

    with torch.no_grad():
        preds = (logits > 0).float()
        correct = (preds == tgt).sum().item()

    return loss, correct, bs


def make_value_batch(model, batch, device, use_amp,
                     head=None, encoder_grad=False):
    """Build a value batch — MSE on game outcome.
    batch is a list of dicts from MmapValueDataset."""
    import torch.utils.data as tud

    # Stack tensors from list of dicts
    gf = torch.stack([s['global_features'] for s in batch]
                     ).to(device)
    mb = torch.stack([s['my_board'] for s in batch]
                     ).to(device)
    mbm = torch.stack([s['my_board_mask'] for s in batch]
                      ).to(device)
    ob = torch.stack([s['opp_board'] for s in batch]
                     ).to(device)
    obm = torch.stack([s['opp_board_mask'] for s in batch]
                      ).to(device)
    h = torch.stack([s['hand'] for s in batch]).to(device)
    hm = torch.stack([s['hand_mask'] for s in batch]
                     ).to(device)
    mg = torch.stack([s['my_gy'] for s in batch]
                     ).to(device)
    mgm = torch.stack([s['my_gy_mask'] for s in batch]
                      ).to(device)
    og = torch.stack([s['opp_gy'] for s in batch]
                     ).to(device)
    ogm = torch.stack([s['opp_gy_mask'] for s in batch]
                      ).to(device)
    st = torch.stack([s['stack'] for s in batch]
                     ).to(device)
    stm = torch.stack([s['stack_mask'] for s in batch]
                      ).to(device)
    tgt = torch.stack([s['value_target'] for s in batch]
                      ).to(device)

    bs = len(batch)

    with torch.amp.autocast('cuda', enabled=use_amp):
        ctx = torch.no_grad() if not encoder_grad \
            else nullcontext()
        with ctx:
            state_emb = model.encode_state(
                gf, mb, mbm, ob, obm, h, hm,
                mg, mgm, og, ogm, st, stm)

        pred = model.get_value(state_emb).squeeze(-1)
        loss = nn.functional.mse_loss(pred, tgt)

    with torch.no_grad():
        correct = ((pred > 0) == (tgt > 0)).sum().item()

    return loss, correct, bs


def make_block_batch(model, batch, device, use_amp,
                     head, encoder_grad=False):
    """Build a block batch for the BlockHead.

    Block data has (blocker,attacker) pair features (256-dim).
    BlockHead expects separate blocker/attacker tensors.
    We reconstruct these from the pairs: for each sample,
    extract unique blockers and attackers from the pairs.
    """
    bs = len(batch)

    # Find max blockers and attackers across batch
    # Each pair is (blocker_i, attacker_j) — we need to
    # figure out how many unique blockers/attackers
    # Pairs are ordered: b0a0, b0a1, ..., b1a0, b1a1, ...
    # Last pair is the "no block" zero vector
    max_b, max_a = 0, 0
    for s in batch:
        n = s['n_pairs']
        pf = s['pair_features']
        # Count non-zero unique blocker features
        # Pairs are b*a + no_block, so n_pairs = nb*na + 1
        # We detect dimensions from feature patterns
        # Simpler: extract from contextInfo if available,
        # or infer from pair count
        # Last pair is zero (no-block), so real pairs = n-1
        real = n - 1
        if real <= 0:
            continue
        # Find number of attackers: first blocker's pairs
        # end when blocker features change
        first_blocker = pf[0, :CARD_DIM]
        na = 1
        for j in range(1, real):
            if np.allclose(pf[j, :CARD_DIM], first_blocker,
                           atol=0.01):
                na += 1
            else:
                break
        nb = real // max(na, 1)
        max_b = max(max_b, nb)
        max_a = max(max_a, na)

    max_b = max(max_b, 1)
    max_a = max(max_a, 1)

    cd = CARD_DIM
    bf = torch.zeros(bs, max_b, cd, device=device)
    bm = torch.zeros(bs, max_b, dtype=torch.bool,
                      device=device)
    af = torch.zeros(bs, max_a, cd, device=device)
    am_t = torch.zeros(bs, max_a, dtype=torch.bool,
                       device=device)
    # Target: for each blocker, which attacker (or no-block)
    tgt = torch.full((bs, max_b), max_a, dtype=torch.long,
                     device=device)  # default = no block
    gf = torch.zeros(bs, GLOBAL_DIM, device=device)

    mb = torch.zeros(bs, 40, cd, device=device)
    mbm = torch.zeros(bs, 40, dtype=torch.bool,
                       device=device)
    ob = torch.zeros(bs, 40, cd, device=device)
    obm = torch.zeros(bs, 40, dtype=torch.bool,
                       device=device)
    h = torch.zeros(bs, 15, cd, device=device)
    hm = torch.zeros(bs, 15, dtype=torch.bool,
                      device=device)
    mg = torch.zeros(bs, 20, cd, device=device)
    mgm = torch.zeros(bs, 20, dtype=torch.bool,
                       device=device)
    og = torch.zeros(bs, 20, cd, device=device)
    ogm = torch.zeros(bs, 20, dtype=torch.bool,
                       device=device)
    st = torch.zeros(bs, 10, cd, device=device)
    stm = torch.zeros(bs, 10, dtype=torch.bool,
                       device=device)

    for i, s in enumerate(batch):
        n = s['n_pairs']
        pf_np = s['pair_features']
        sel = s['action_mask']
        real = n - 1
        if real <= 0:
            continue

        # Infer nb, na from pair structure
        first_blocker = pf_np[0, :CARD_DIM]
        na = 1
        for j in range(1, real):
            if np.allclose(pf_np[j, :CARD_DIM], first_blocker,
                           atol=0.01):
                na += 1
            else:
                break
        nb = real // max(na, 1)

        # Extract unique blockers and attackers
        for b_idx in range(min(nb, max_b)):
            bf[i, b_idx] = torch.from_numpy(
                pf_np[b_idx * na, :CARD_DIM])
            bm[i, b_idx] = True
        for a_idx in range(min(na, max_a)):
            af[i, a_idx] = torch.from_numpy(
                pf_np[a_idx, CARD_DIM:CARD_DIM*2])
            am_t[i, a_idx] = True

        # Build assignment target from selected pairs
        for pair_idx in range(real):
            if sel[pair_idx] > 0.5:
                b_idx = pair_idx // na
                a_idx = pair_idx % na
                if b_idx < max_b and a_idx < max_a:
                    tgt[i, b_idx] = a_idx

        g, zones, masks = parse_game_state(
            s['game_state_flat'], s['global_features'])
        gf[i] = torch.from_numpy(g)
        mb[i] = torch.from_numpy(zones['my_board'])
        mbm[i] = torch.from_numpy(masks['my_board_mask'])
        ob[i] = torch.from_numpy(zones['opp_board'])
        obm[i] = torch.from_numpy(
            masks['opp_board_mask'])
        h[i] = torch.from_numpy(zones['hand'])
        hm[i] = torch.from_numpy(masks['hand_mask'])
        mg[i] = torch.from_numpy(zones['my_gy'])
        mgm[i] = torch.from_numpy(masks['my_gy_mask'])
        og[i] = torch.from_numpy(zones['opp_gy'])
        ogm[i] = torch.from_numpy(masks['opp_gy_mask'])
        st[i] = torch.from_numpy(zones['stack'])
        stm[i] = torch.from_numpy(masks['stack_mask'])

    with torch.amp.autocast('cuda', enabled=use_amp):
        ctx = torch.no_grad() if not encoder_grad \
            else nullcontext()
        with ctx:
            state_emb = model.encode_state(
                gf, mb, mbm, ob, obm, h, hm,
                mg, mgm, og, ogm, st, stm)

        # BlockHead: (batch, max_b, max_a+1) logits
        logits = head(state_emb, bf, bm, af, am_t)

        # CE loss per blocker
        # logits: (bs, max_b, max_a+1), tgt: (bs, max_b)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            tgt.reshape(-1),
            reduction='none')
        # Mask to real blockers only
        loss = (loss.reshape(bs, max_b)
                * bm.float()).sum() / \
               bm.float().sum().clamp(min=1)

    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        correct = ((preds == tgt)
                   * bm).sum().item()
        total = bm.sum().item()

    return loss, correct, total


def train_block_head(model, head, samples, args,
                     state, device, use_amp):
    """Train the block head with per-blocker CE loss."""
    state.current_head = 'block'
    state.status = "Training block head..."
    log(state, f"\n=== Training block head ===")
    log(state, f"Samples: {len(samples)}")

    for p in model.state_encoder.parameters():
        p.requires_grad = False
    for p in model.value_network.parameters():
        p.requires_grad = False
    for p in head.parameters():
        p.requires_grad = True

    state.head_params = sum(
        p.numel() for p in head.parameters())
    log(state, f"Head params: {state.head_params:,}")

    optimizer = optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda') if use_amp \
        else None

    # Split by game to prevent data leakage
    from training.chunked_loader import split_by_game
    train_s, val_s = split_by_game(samples)

    best_acc = 0
    save_path = os.path.join(
        args.save_dir, 'best_block_model.pt')

    tl_list = state.block_train_losses
    ta_list = state.block_train_accs
    vl_list = state.block_val_losses
    va_list = state.block_val_accs

    start = time.time()

    for epoch in range(1, args.epochs + 1):
        state.epoch = epoch
        state.status = (
            f"Training block head: "
            f"epoch {epoch}/{args.epochs}")
        t0 = time.time()

        model.train()
        random.shuffle(train_s)
        tloss, tc, tt = 0, 0, 0

        for bi in range(0, len(train_s), args.batch_size):
            batch = train_s[bi:bi + args.batch_size]
            if len(batch) < 2:
                continue
            state.epoch_progress = bi / max(
                len(train_s), 1)

            loss, correct, total = make_block_batch(
                model, batch, device, use_amp, head)

            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    head.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    head.parameters(), 1.0)
                optimizer.step()

            tloss += loss.item() * max(total, 1)
            tc += correct
            tt += total

        scheduler.step()

        model.eval()
        vloss, vc, vt = 0, 0, 0
        with torch.no_grad():
            for bi in range(0, len(val_s), args.batch_size):
                batch = val_s[bi:bi + args.batch_size]
                if len(batch) < 2:
                    continue
                loss, correct, total = make_block_batch(
                    model, batch, device, use_amp, head)
                vloss += loss.item() * max(total, 1)
                vc += correct
                vt += total

        ta = tc / max(tt, 1)
        va = vc / max(vt, 1)
        tl = tloss / max(tt, 1)
        vl = vloss / max(vt, 1)

        state.current_train_loss = tl
        state.current_train_acc = ta
        state.current_val_loss = vl
        state.current_val_acc = va
        tl_list.append(tl)
        ta_list.append(ta)
        vl_list.append(vl)
        va_list.append(va)

        if va > best_acc:
            best_acc = va
            model.save(save_path)

        state.block_best_acc = best_acc

        status = ' *BEST*' if va == best_acc and va > 0 \
            else ''
        log(state,
            f"  Epoch {epoch:>3d}/{args.epochs} | "
            f"TrLoss {tl:.4f} TrAcc {ta:.1%} | "
            f"VlLoss {vl:.4f} VlAcc {va:.1%}{status}")

        if np.isnan(tl) or np.isnan(vl):
            log(state,
                "  NaN detected! Stopping block training.")
            break

        state.epoch_time = time.time() - t0
        state.elapsed = time.time() - start
        state.eta = (args.epochs - epoch) * state.epoch_time
        state.epoch_progress = 1.0
        state.chart_dirty = True

        if is_cuda_device(device):
            torch.cuda.synchronize()
            state.gpu_mem_used_mb = (
                torch.cuda.memory_allocated() / 1024**2)
            state.gpu_mem_total_mb = (
                torch.cuda.get_device_properties(0)
                .total_memory / 1024**2)

        time.sleep(0.05)

    return best_acc


def train_priority_head(model, head, samples, args,
                        state, device, use_amp):
    """Train the priority head with CrossEntropyLoss."""
    state.current_head = 'priority'
    state.status = "Training priority head..."
    log(state, f"\n=== Training priority head ===")
    log(state, f"Samples: {len(samples)}")

    for p in model.state_encoder.parameters():
        p.requires_grad = False
    for p in model.value_network.parameters():
        p.requires_grad = False
    for p in head.parameters():
        p.requires_grad = True

    state.head_params = sum(
        p.numel() for p in head.parameters())
    log(state, f"Head params: {state.head_params:,}")

    optimizer = optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda') if use_amp \
        else None

    # Split by game to prevent data leakage
    from training.chunked_loader import split_by_game
    train_s, val_s = split_by_game(samples)

    best_acc = 0
    save_path = os.path.join(
        args.save_dir, 'best_priority_model.pt')

    tl_list = state.priority_train_losses
    ta_list = state.priority_train_accs
    vl_list = state.priority_val_losses
    va_list = state.priority_val_accs

    start = time.time()

    for epoch in range(1, args.epochs + 1):
        state.epoch = epoch
        state.status = (
            f"Training priority head: "
            f"epoch {epoch}/{args.epochs}")
        t0 = time.time()

        # Train
        model.train()
        random.shuffle(train_s)
        tloss, tc, tt = 0, 0, 0

        for bi in range(0, len(train_s), args.batch_size):
            batch = train_s[bi:bi + args.batch_size]
            if len(batch) < 2:
                continue
            state.epoch_progress = bi / max(
                len(train_s), 1)

            loss, correct, total = make_priority_batch(
                model, batch, device, use_amp, head)

            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    head.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    head.parameters(), 1.0)
                optimizer.step()

            tloss += loss.item() * total
            tc += correct
            tt += total

        scheduler.step()

        # Val
        model.eval()
        vloss, vc, vt = 0, 0, 0
        with torch.no_grad():
            for bi in range(0, len(val_s), args.batch_size):
                batch = val_s[bi:bi + args.batch_size]
                if len(batch) < 2:
                    continue
                loss, correct, total = make_priority_batch(
                    model, batch, device, use_amp, head)
                vloss += loss.item() * total
                vc += correct
                vt += total

        ta = tc / max(tt, 1)
        va = vc / max(vt, 1)
        tl = tloss / max(tt, 1)
        vl = vloss / max(vt, 1)

        state.current_train_loss = tl
        state.current_train_acc = ta
        state.current_val_loss = vl
        state.current_val_acc = va
        tl_list.append(tl)
        ta_list.append(ta)
        vl_list.append(vl)
        va_list.append(va)

        if va > best_acc:
            best_acc = va
            model.save(save_path)

        state.priority_best_acc = best_acc

        status = ' *BEST*' if va == best_acc and va > 0 \
            else ''
        log(state,
            f"  Epoch {epoch:>3d}/{args.epochs} | "
            f"TrLoss {tl:.4f} TrAcc {ta:.1%} | "
            f"VlLoss {vl:.4f} VlAcc {va:.1%}{status}")

        if np.isnan(tl) or np.isnan(vl):
            log(state,
                "  NaN detected! Stopping priority training.")
            break

        state.epoch_time = time.time() - t0
        state.elapsed = time.time() - start
        state.eta = (args.epochs - epoch) * state.epoch_time
        state.epoch_progress = 1.0
        state.chart_dirty = True

        if is_cuda_device(device):
            torch.cuda.synchronize()
            state.gpu_mem_used_mb = (
                torch.cuda.memory_allocated() / 1024**2)
            state.gpu_mem_total_mb = (
                torch.cuda.get_device_properties(0)
                .total_memory / 1024**2)

        time.sleep(0.05)

    return best_acc


# ── Training thread ──────────────────────────────────

def train_head(model, head, head_name, samples, args,
               state, device, use_amp):
    state.current_head = head_name
    state.status = f"Training {head_name} head..."
    log(state, f"\n=== Training {head_name} head ===")
    log(state, f"Samples: {len(samples)}")

    for p in model.state_encoder.parameters():
        p.requires_grad = False
    for p in model.value_network.parameters():
        p.requires_grad = False
    for p in head.parameters():
        p.requires_grad = True

    state.head_params = sum(
        p.numel() for p in head.parameters())
    log(state, f"Head params: {state.head_params:,}")

    optimizer = optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # Split by game to prevent data leakage
    from training.chunked_loader import split_by_game
    train_s, val_s = split_by_game(samples)

    best_acc = 0
    save_path = os.path.join(
        args.save_dir, f'best_{head_name}_model.pt')

    # Get the right metric lists
    tl_list = getattr(state, f'{head_name}_train_losses',
                      state.block_train_losses)
    ta_list = getattr(state, f'{head_name}_train_accs',
                      state.block_train_accs)
    vl_list = getattr(state, f'{head_name}_val_losses',
                      state.block_val_losses)
    va_list = getattr(state, f'{head_name}_val_accs',
                      state.block_val_accs)

    start = time.time()

    for epoch in range(1, args.epochs + 1):
        state.epoch = epoch
        state.status = (
            f"Training {head_name} head: "
            f"epoch {epoch}/{args.epochs}")
        t0 = time.time()

        # Train
        model.train()
        random.shuffle(train_s)
        tloss, tc, tt = 0, 0, 0
        n_batches = max(
            1, len(train_s) // args.batch_size)

        for bi in range(0, len(train_s), args.batch_size):
            batch = train_s[bi:bi + args.batch_size]
            if len(batch) < 2:
                continue
            state.epoch_progress = bi / max(
                len(train_s), 1)

            loss, correct, total = make_batch(
                model, batch, device, use_amp, head)

            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    head.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    head.parameters(), 1.0)
                optimizer.step()

            tloss += loss.item() * total
            tc += correct
            tt += total

        scheduler.step()

        # Val
        model.eval()
        vloss, vc, vt = 0, 0, 0
        with torch.no_grad():
            for bi in range(0, len(val_s), args.batch_size):
                batch = val_s[bi:bi + args.batch_size]
                if len(batch) < 2:
                    continue
                loss, correct, total = make_batch(
                    model, batch, device, use_amp, head)
                vloss += loss.item() * total
                vc += correct
                vt += total

        ta = tc / max(tt, 1)
        va = vc / max(vt, 1)
        tl = tloss / max(tt, 1)
        vl = vloss / max(vt, 1)

        state.current_train_loss = tl
        state.current_train_acc = ta
        state.current_val_loss = vl
        state.current_val_acc = va
        tl_list.append(tl)
        ta_list.append(ta)
        vl_list.append(vl)
        va_list.append(va)

        if va > best_acc:
            best_acc = va
            model.save(save_path)

        setattr(state, f'{head_name}_best_acc', best_acc)

        status = ' *BEST*' if va == best_acc and va > 0 else ''
        log(state,
            f"  Epoch {epoch:>3d}/{args.epochs} | "
            f"TrLoss {tl:.4f} TrAcc {ta:.1%} | "
            f"VlLoss {vl:.4f} VlAcc {va:.1%}{status}")

        # Early stop on NaN
        if np.isnan(tl) or np.isnan(vl):
            log(state, f"  NaN detected! Stopping {head_name} training.")
            break

        state.epoch_time = time.time() - t0
        state.elapsed = time.time() - start
        state.eta = (args.epochs - epoch) * state.epoch_time
        state.epoch_progress = 1.0
        state.chart_dirty = True

        if is_cuda_device(device):
            torch.cuda.synchronize()
            state.gpu_mem_used_mb = (
                torch.cuda.memory_allocated() / 1024**2)
            state.gpu_mem_total_mb = (
                torch.cuda.get_device_properties(0)
                .total_memory / 1024**2)

        time.sleep(0.05)

    return best_acc


def train_head_mmap(model, head, head_name,
                    train_ds, val_ds, args, state,
                    device, use_amp, batch_fn):
    """Generic mmap-based head training.
    Uses DataLoader directly, no list materialization."""
    import torch.utils.data as tud

    state.current_head = head_name
    state.status = f"Training {head_name} head..."
    log(state, f"\n=== Training {head_name} head (mmap)"
        f" ===")
    log(state, f"Train: {len(train_ds)}, "
        f"Val: {len(val_ds)}")

    for p in model.state_encoder.parameters():
        p.requires_grad = False
    for p in model.value_network.parameters():
        p.requires_grad = False
    for p in head.parameters():
        p.requires_grad = True

    state.head_params = sum(
        p.numel() for p in head.parameters())
    log(state, f"Head params: {state.head_params:,}")

    optimizer = optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda') if use_amp \
        else None

    train_loader = tud.DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=0,
        collate_fn=lambda x: x)
    val_loader = tud.DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=0,
        collate_fn=lambda x: x)

    best_acc = 0
    save_path = os.path.join(
        args.save_dir, f'best_{head_name}_model.pt')

    tl_list = getattr(state, f'{head_name}_train_losses',
                      state.block_train_losses)
    ta_list = getattr(state, f'{head_name}_train_accs',
                      state.block_train_accs)
    vl_list = getattr(state, f'{head_name}_val_losses',
                      state.block_val_losses)
    va_list = getattr(state, f'{head_name}_val_accs',
                      state.block_val_accs)

    start = time.time()

    for epoch in range(1, args.epochs + 1):
        state.epoch = epoch
        state.status = (
            f"Training {head_name} head: "
            f"epoch {epoch}/{args.epochs}")
        t0 = time.time()

        model.train()
        tloss, tc, tt = 0, 0, 0

        for bi, batch in enumerate(train_loader):
            if len(batch) < 2:
                continue
            state.epoch_progress = bi / max(
                len(train_loader), 1)

            loss, correct, total = batch_fn(
                model, batch, device, use_amp, head)

            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    head.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    head.parameters(), 1.0)
                optimizer.step()

            tloss += loss.item() * total
            tc += correct
            tt += total

        scheduler.step()

        model.eval()
        vloss, vc, vt = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) < 2:
                    continue
                loss, correct, total = batch_fn(
                    model, batch, device, use_amp, head)
                vloss += loss.item() * total
                vc += correct
                vt += total

        ta = tc / max(tt, 1)
        va = vc / max(vt, 1)
        tl = tloss / max(tt, 1)
        vl = vloss / max(vt, 1)

        state.current_train_loss = tl
        state.current_train_acc = ta
        state.current_val_loss = vl
        state.current_val_acc = va
        tl_list.append(tl)
        ta_list.append(ta)
        vl_list.append(vl)
        va_list.append(va)

        if va > best_acc:
            best_acc = va
            model.save(save_path)

        setattr(state, f'{head_name}_best_acc', best_acc)

        status = ' *BEST*' if va == best_acc and va > 0 \
            else ''
        log(state,
            f"  Epoch {epoch:>3d}/{args.epochs} | "
            f"TrLoss {tl:.4f} TrAcc {ta:.1%} | "
            f"VlLoss {vl:.4f} VlAcc {va:.1%}{status}")

        if np.isnan(tl) or np.isnan(vl):
            log(state,
                f"  NaN detected! Stopping {head_name}.")
            break

        state.epoch_time = time.time() - t0
        state.elapsed = time.time() - start
        state.eta = (args.epochs - epoch) * state.epoch_time
        state.epoch_progress = 1.0
        state.chart_dirty = True

        if is_cuda_device(device):
            torch.cuda.synchronize()
            state.gpu_mem_used_mb = (
                torch.cuda.memory_allocated() / 1024**2)
            state.gpu_mem_total_mb = (
                torch.cuda.get_device_properties(0)
                .total_memory / 1024**2)

        time.sleep(0.05)

    return best_acc


def train_priority_head_mmap(model, head,
                             train_ds, val_ds,
                             args, state, device,
                             use_amp):
    """Train priority head using mmap datasets directly.
    Avoids materializing 140K samples into RAM."""
    import torch.utils.data as tud

    state.current_head = 'priority'
    state.status = "Training priority head..."
    log(state, f"\n=== Training priority head (mmap) ===")
    log(state, f"Train: {len(train_ds)}, "
        f"Val: {len(val_ds)}")

    for p in model.state_encoder.parameters():
        p.requires_grad = False
    for p in model.value_network.parameters():
        p.requires_grad = False
    for p in head.parameters():
        p.requires_grad = True

    state.head_params = sum(
        p.numel() for p in head.parameters())
    log(state, f"Head params: {state.head_params:,}")

    optimizer = optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda') if use_amp \
        else None

    # DataLoaders with list collation (batch = list of dicts)
    train_loader = tud.DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=0,
        collate_fn=lambda x: x)
    val_loader = tud.DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=0,
        collate_fn=lambda x: x)

    best_acc = 0
    save_path = os.path.join(
        args.save_dir, 'best_priority_model.pt')

    tl_list = state.priority_train_losses
    ta_list = state.priority_train_accs
    vl_list = state.priority_val_losses
    va_list = state.priority_val_accs

    start = time.time()

    for epoch in range(1, args.epochs + 1):
        state.epoch = epoch
        state.status = (
            f"Training priority head: "
            f"epoch {epoch}/{args.epochs}")
        t0 = time.time()

        # Train
        model.train()
        tloss, tc, tt = 0, 0, 0

        for bi, batch in enumerate(train_loader):
            if len(batch) < 2:
                continue
            state.epoch_progress = bi / max(
                len(train_loader), 1)

            loss, correct, total = make_priority_batch(
                model, batch, device, use_amp, head)

            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    head.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    head.parameters(), 1.0)
                optimizer.step()

            tloss += loss.item() * total
            tc += correct
            tt += total

        scheduler.step()

        # Val
        model.eval()
        vloss, vc, vt = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) < 2:
                    continue
                loss, correct, total = make_priority_batch(
                    model, batch, device, use_amp, head)
                vloss += loss.item() * total
                vc += correct
                vt += total

        ta = tc / max(tt, 1)
        va = vc / max(vt, 1)
        tl = tloss / max(tt, 1)
        vl = vloss / max(vt, 1)

        state.current_train_loss = tl
        state.current_train_acc = ta
        state.current_val_loss = vl
        state.current_val_acc = va
        tl_list.append(tl)
        ta_list.append(ta)
        vl_list.append(vl)
        va_list.append(va)

        if va > best_acc:
            best_acc = va
            model.save(save_path)

        state.priority_best_acc = best_acc

        status = ' *BEST*' if va == best_acc and va > 0 \
            else ''
        log(state,
            f"  Epoch {epoch:>3d}/{args.epochs} | "
            f"TrLoss {tl:.4f} TrAcc {ta:.1%} | "
            f"VlLoss {vl:.4f} VlAcc {va:.1%}{status}")

        if np.isnan(tl) or np.isnan(vl):
            log(state,
                "  NaN detected! Stopping priority.")
            break

        state.epoch_time = time.time() - t0
        state.elapsed = time.time() - start
        state.eta = (args.epochs - epoch) * state.epoch_time
        state.epoch_progress = 1.0
        state.chart_dirty = True

        if is_cuda_device(device):
            torch.cuda.synchronize()
            state.gpu_mem_used_mb = (
                torch.cuda.memory_allocated() / 1024**2)
            state.gpu_mem_total_mb = (
                torch.cuda.get_device_properties(0)
                .total_memory / 1024**2)

        time.sleep(0.05)

    return best_acc


def train_joint_mmap(model, head_configs, args, state,
                     device, use_amp):
    """Train all heads jointly with unfrozen encoder.

    head_configs: list of (head_name, head, train_ds,
                           val_ds, batch_fn)
    Encoder gets gradients from all heads (accumulated),
    stepped once per round-robin cycle.
    """
    import torch.utils.data as tud

    state.status = "Joint training (encoder unfrozen)..."
    log(state, "\n=== Joint training: encoder + all heads ===")

    # Unfreeze encoder; value network unfrozen if in configs
    has_value = any(name == 'value' for name, *_ in
                    head_configs)
    for p in model.state_encoder.parameters():
        p.requires_grad = True
    for p in model.value_network.parameters():
        p.requires_grad = has_value

    encoder_params = list(model.state_encoder.parameters())
    encoder_lr = args.lr * 0.1
    encoder_optimizer = optim.AdamW(
        encoder_params, lr=encoder_lr, weight_decay=1e-4)
    encoder_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        encoder_optimizer, T_max=args.epochs)

    log(state, f"Encoder LR: {encoder_lr:.1e}, "
        f"Head LR: {args.lr:.1e}")
    log(state, f"Encoder params: "
        f"{sum(p.numel() for p in encoder_params):,}")

    # Per-head setup
    head_optimizers = {}
    head_schedulers = {}
    head_loaders = {}
    head_val_loaders = {}
    head_best_acc = {}
    scaler = torch.amp.GradScaler('cuda') if use_amp \
        else None

    for name, head, train_ds, val_ds, batch_fn in \
            head_configs:
        for p in head.parameters():
            p.requires_grad = True
        head_optimizers[name] = optim.AdamW(
            head.parameters(), lr=args.lr,
            weight_decay=1e-4)
        head_schedulers[name] = \
            optim.lr_scheduler.CosineAnnealingLR(
                head_optimizers[name], T_max=args.epochs)
        head_loaders[name] = tud.DataLoader(
            train_ds, batch_size=args.batch_size,
            shuffle=True, num_workers=0,
            collate_fn=lambda x: x)
        head_val_loaders[name] = tud.DataLoader(
            val_ds, batch_size=args.batch_size,
            shuffle=False, num_workers=0,
            collate_fn=lambda x: x)
        head_best_acc[name] = 0.0
        n_params = sum(p.numel() for p in head.parameters())
        log(state, f"  {name}: {len(train_ds)} train, "
            f"{len(val_ds)} val, {n_params:,} params")

    start = time.time()
    save_path = os.path.join(
        args.save_dir, 'model_with_decisions.pt')

    for epoch in range(1, args.epochs + 1):
        state.epoch = epoch
        state.status = (
            f"Joint training: epoch {epoch}/{args.epochs}")
        t0 = time.time()
        model.train()

        # Create iterators for all heads
        head_iters = {
            name: iter(head_loaders[name])
            for name in head_loaders}

        # Track per-head stats
        head_stats = {name: {'loss': 0, 'correct': 0,
                             'total': 0, 'batches': 0}
                      for name in head_loaders}

        # Round-robin training
        active_heads = set(head_iters.keys())
        batch_count = 0
        while active_heads:
            encoder_optimizer.zero_grad()
            for hname in head_optimizers:
                head_optimizers[hname].zero_grad()

            # One batch from each active head
            finished = set()
            ran_this_cycle = set()
            for name in list(active_heads):
                try:
                    batch = next(head_iters[name])
                except StopIteration:
                    finished.add(name)
                    continue

                if len(batch) < 2:
                    continue

                # Find the batch_fn and head for this name
                batch_fn = None
                head = None
                for cfg in head_configs:
                    if cfg[0] == name:
                        batch_fn = cfg[4]
                        head = cfg[1]
                        break

                loss, correct, total = batch_fn(
                    model, batch, device, use_amp, head,
                    encoder_grad=True)

                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                ran_this_cycle.add(name)
                head_stats[name]['loss'] += \
                    loss.item() * total
                head_stats[name]['correct'] += correct
                head_stats[name]['total'] += total
                head_stats[name]['batches'] += 1

            active_heads -= finished
            batch_count += 1

            # Skip optimizer step if nothing ran
            if not ran_this_cycle:
                continue

            # Unscale, clip, step — only for optimizers
            # that had gradients this cycle
            if scaler:
                scaler.unscale_(encoder_optimizer)
                for hname in ran_this_cycle:
                    scaler.unscale_(
                        head_optimizers[hname])

            torch.nn.utils.clip_grad_norm_(
                encoder_params, 1.0)
            for hname in ran_this_cycle:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in head_optimizers[hname]
                     .param_groups[0]['params']], 1.0)

            if scaler:
                scaler.step(encoder_optimizer)
                for hname in ran_this_cycle:
                    scaler.step(
                        head_optimizers[hname])
                scaler.update()
            else:
                encoder_optimizer.step()
                for hname in ran_this_cycle:
                    head_optimizers[hname].step()

            state.epoch_progress = min(
                batch_count / max(
                    max(len(l) for l in
                        head_loaders.values()), 1),
                1.0)

        # Step schedulers
        encoder_scheduler.step()
        for name in head_schedulers:
            head_schedulers[name].step()

        # Validation
        model.eval()
        head_val_stats = {name: {'loss': 0, 'correct': 0,
                                  'total': 0}
                          for name in head_val_loaders}
        with torch.no_grad():
            for name in head_val_loaders:
                batch_fn = None
                head = None
                for cfg in head_configs:
                    if cfg[0] == name:
                        batch_fn = cfg[4]
                        head = cfg[1]
                        break
                for batch in head_val_loaders[name]:
                    if len(batch) < 2:
                        continue
                    loss, correct, total = batch_fn(
                        model, batch, device, use_amp, head)
                    head_val_stats[name]['loss'] += \
                        loss.item() * total
                    head_val_stats[name]['correct'] += \
                        correct
                    head_val_stats[name]['total'] += total

        # Log results
        log_parts = [f"Epoch {epoch:>3d}/{args.epochs}"]
        any_improved = False
        for name in sorted(head_stats.keys()):
            ts = head_stats[name]
            vs = head_val_stats[name]
            ta = ts['correct'] / max(ts['total'], 1)
            va = vs['correct'] / max(vs['total'], 1)
            tl = ts['loss'] / max(ts['total'], 1)
            vl = vs['loss'] / max(vs['total'], 1)

            # Track per-head lists for chart
            tl_list = getattr(
                state, f'{name}_train_losses', None)
            if tl_list is not None:
                tl_list.append(tl)
            ta_list = getattr(
                state, f'{name}_train_accs', None)
            if ta_list is not None:
                ta_list.append(ta)
            vl_list = getattr(
                state, f'{name}_val_losses', None)
            if vl_list is not None:
                vl_list.append(vl)
            va_list = getattr(
                state, f'{name}_val_accs', None)
            if va_list is not None:
                va_list.append(va)

            improved = ''
            if va > head_best_acc[name]:
                head_best_acc[name] = va
                improved = '*'
                any_improved = True
            setattr(state, f'{name}_best_acc',
                    head_best_acc[name])

            log_parts.append(
                f"{name}={va:.1%}{improved}")

        log(state, "  " + " | ".join(log_parts))

        # Save if any head improved
        if any_improved:
            model.save(save_path)
            log(state, f"  Saved: {save_path}")

        state.epoch_time = time.time() - t0
        state.elapsed = time.time() - start
        state.eta = (args.epochs - epoch) * state.epoch_time
        state.epoch_progress = 1.0
        state.chart_dirty = True

        if device.startswith('cuda'):
            torch.cuda.synchronize()
            state.gpu_mem_used_mb = (
                torch.cuda.memory_allocated() / 1024**2)
            state.gpu_mem_total_mb = (
                torch.cuda.get_device_properties(0)
                .total_memory / 1024**2)

        time.sleep(0.05)

    log(state, "\n=== Joint training complete ===")
    for name in sorted(head_best_acc.keys()):
        log(state, f"  {name}: best val acc "
            f"{head_best_acc[name]:.1%}")

    return head_best_acc


def trainer_thread(state, args):
    try:
        profile = auto_detect_profile(args.device)
        backend = resolve_backend(args.device)
        device = backend.torch_device
        use_amp = profile.use_amp and backend.use_amp

        state.device = backend.name
        state.gpu_name = f'{backend.name} ({profile.name})'
        state.total_epochs = args.epochs

        # Load data via mmap or fallback to JSONL
        all_heads = ['priority', 'attack', 'block',
                     'target', 'card_select',
                     'mulligan', 'binary']
        heads_list = (args.heads.split(',')
                 if args.heads != 'all'
                 else all_heads)

        preprocessed_dir = os.path.join(
            os.path.dirname(args.data_dir),
            'preprocessed')
        use_mmap = os.path.exists(
            os.path.join(preprocessed_dir,
                         'metadata.json'))

        if use_mmap:
            from training.mmap_dataset import (
                MmapPriorityDataset,
                MmapAttackDataset,
                MmapBlockDataset,
                MmapTargetDataset,
                MmapCardSelectDataset,
                MmapMulliganDataset,
                MmapBinaryDataset)
            import json as json_mod

            state.status = "Loading preprocessed data..."
            with open(os.path.join(preprocessed_dir,
                                   'metadata.json')) as f:
                meta = json_mod.load(f)

            priority_samples = []
            attack_samples = []
            block_samples = []

            if 'priority' in heads_list:
                # Priority uses DataLoader directly
                # (too large to materialize into list)
                priority_train_ds = MmapPriorityDataset(
                    preprocessed_dir, train=True)
                priority_val_ds = MmapPriorityDataset(
                    preprocessed_dir, train=False)
                state.priority_samples = len(
                    priority_train_ds)
                log(state,
                    f"Priority: {len(priority_train_ds)}"
                    f" train, {len(priority_val_ds)}"
                    f" val samples (mmap)")

            if 'attack' in heads_list:
                attack_train_ds = MmapAttackDataset(
                    preprocessed_dir, train=True)
                attack_val_ds = MmapAttackDataset(
                    preprocessed_dir, train=False)
                state.attack_samples = len(
                    attack_train_ds)
                log(state,
                    f"Attack: {len(attack_train_ds)}"
                    f" train, {len(attack_val_ds)}"
                    f" val samples (mmap)")

            if 'block' in heads_list:
                block_train_ds = MmapBlockDataset(
                    preprocessed_dir, train=True)
                block_val_ds = MmapBlockDataset(
                    preprocessed_dir, train=False)
                state.block_samples = len(
                    block_train_ds)
                log(state,
                    f"Block: {len(block_train_ds)}"
                    f" train, {len(block_val_ds)}"
                    f" val samples (mmap)")

            target_train_ds = target_val_ds = None
            if 'target' in heads_list:
                target_train_ds = MmapTargetDataset(
                    preprocessed_dir, train=True)
                target_val_ds = MmapTargetDataset(
                    preprocessed_dir, train=False)
                if len(target_train_ds) > 0:
                    log(state,
                        f"Target: {len(target_train_ds)}"
                        f" train, {len(target_val_ds)}"
                        f" val samples (mmap)")
                else:
                    log(state, "Target: no data")
                    target_train_ds = None

            card_select_train_ds = card_select_val_ds = None
            if 'card_select' in heads_list:
                card_select_train_ds = MmapCardSelectDataset(
                    preprocessed_dir, train=True)
                card_select_val_ds = MmapCardSelectDataset(
                    preprocessed_dir, train=False)
                if len(card_select_train_ds) > 0:
                    log(state,
                        f"Card Select: {len(card_select_train_ds)}"
                        f" train, {len(card_select_val_ds)}"
                        f" val samples (mmap)")
                else:
                    log(state, "Card Select: no data")
                    card_select_train_ds = None

            mulligan_train_ds = mulligan_val_ds = None
            if 'mulligan' in heads_list:
                mulligan_train_ds = MmapMulliganDataset(
                    preprocessed_dir, train=True)
                mulligan_val_ds = MmapMulliganDataset(
                    preprocessed_dir, train=False)
                if len(mulligan_train_ds) > 0:
                    log(state,
                        f"Mulligan: {len(mulligan_train_ds)}"
                        f" train, {len(mulligan_val_ds)}"
                        f" val samples (mmap)")
                else:
                    log(state, "Mulligan: no data")
                    mulligan_train_ds = None

            binary_train_ds = binary_val_ds = None
            if 'binary' in heads_list:
                binary_train_ds = MmapBinaryDataset(
                    preprocessed_dir, train=True)
                binary_val_ds = MmapBinaryDataset(
                    preprocessed_dir, train=False)
                if len(binary_train_ds) > 0:
                    log(state,
                        f"Binary: {len(binary_train_ds)}"
                        f" train, {len(binary_val_ds)}"
                        f" val samples (mmap)")
                else:
                    log(state, "Binary: no data")
                    binary_train_ds = None
        else:
            # Fallback: load from JSONL (chunked)
            log(state,
                "No preprocessed data — loading JSONL")
            heads_filter = (
                list(heads_list)
                if heads_list != ['priority', 'attack',
                                  'block']
                else None)
            attack_samples, block_samples, \
                priority_samples = load_decisions(
                    args.data_dir, state,
                    args.max_files,
                    heads=heads_filter)

        # Load model
        state.phase = "training"
        state.status = "Loading encoder..."
        log(state, f"\nDevice: {state.device} ({state.gpu_name})")
        log(state, f"AMP: {use_amp}")
        if os.path.exists(args.encoder_checkpoint):
            log(state, f"Loading encoder: "
                f"{args.encoder_checkpoint}")
            model = MTGModel.load(
                args.encoder_checkpoint, device=backend.name)
            log(state, "Encoder loaded.")
        else:
            log(state, "No checkpoint — random init")
            model = MTGModel().to(device)

        state.encoder_params = sum(
            p.numel()
            for p in model.state_encoder.parameters())

        os.makedirs(args.save_dir, exist_ok=True)

        log(state, f"Heads to train: {heads_list}")

        # Joint training mode: all heads + value + unfrozen encoder
        if getattr(args, 'joint', False) and use_mmap:
            # Load value dataset for joint training
            from training.mmap_dataset import MmapValueDataset
            value_train_ds = MmapValueDataset(
                preprocessed_dir, train=True)
            value_val_ds = MmapValueDataset(
                preprocessed_dir, train=False)
            log(state, f"Value: {len(value_train_ds)} train, "
                f"{len(value_val_ds)} val (joint)")

            head_configs = []
            # Include value network
            head_configs.append((
                'value', model.value_network,
                value_train_ds, value_val_ds,
                make_value_batch))
            if 'priority' in heads_list and \
                    priority_train_ds:
                head_configs.append((
                    'priority', model.priority_head,
                    priority_train_ds, priority_val_ds,
                    make_priority_batch))
            if 'attack' in heads_list and \
                    attack_train_ds:
                head_configs.append((
                    'attack', model.attack_head,
                    attack_train_ds, attack_val_ds,
                    make_batch))
            if 'block' in heads_list and \
                    block_train_ds:
                head_configs.append((
                    'block', model.block_head,
                    block_train_ds, block_val_ds,
                    make_block_batch))
            if 'target' in heads_list and \
                    target_train_ds:
                head_configs.append((
                    'target', model.target_head,
                    target_train_ds, target_val_ds,
                    make_target_batch))
            if 'card_select' in heads_list and \
                    card_select_train_ds:
                head_configs.append((
                    'card_select', model.card_select_head,
                    card_select_train_ds,
                    card_select_val_ds,
                    make_batch))
            if 'mulligan' in heads_list and \
                    mulligan_train_ds:
                head_configs.append((
                    'mulligan', model.mulligan_head,
                    mulligan_train_ds, mulligan_val_ds,
                    make_mulligan_batch))
            if 'binary' in heads_list and \
                    binary_train_ds:
                head_configs.append((
                    'binary', model.binary_head,
                    binary_train_ds, binary_val_ds,
                    make_binary_batch))

            if head_configs:
                train_joint_mmap(
                    model, head_configs, args, state,
                    device, use_amp)

                # Save combined model
                save_path = os.path.join(
                    args.save_dir,
                    'model_with_decisions.pt')
                model.save(save_path)
                log(state, f"\nModel saved: {save_path}")

                state.status = "Training complete!"
                state.phase = "done"
                state.chart_dirty = True
                return  # skip sequential training

        # Train priority head first (softmax CE, not BCE)
        if 'priority' in heads_list:
            if use_mmap and priority_train_ds:
                train_priority_head_mmap(
                    model, model.priority_head,
                    priority_train_ds, priority_val_ds,
                    args, state, device, use_amp)
            elif priority_samples:
                train_priority_head(
                    model, model.priority_head,
                    priority_samples, args, state,
                    device, use_amp)

        # Train attack head
        if 'attack' in heads_list:
            state.epoch = 0
            state.epoch_progress = 0
            if use_mmap and attack_train_ds:
                train_head_mmap(
                    model, model.attack_head, 'attack',
                    attack_train_ds, attack_val_ds,
                    args, state, device, use_amp,
                    make_batch)
            elif attack_samples:
                train_head(model, model.attack_head,
                           'attack', attack_samples,
                           args, state, device, use_amp)

        # Train block head (pair-based assignment)
        if 'block' in heads_list:
            state.epoch = 0
            state.epoch_progress = 0
            if use_mmap and block_train_ds:
                train_head_mmap(
                    model, model.block_head, 'block',
                    block_train_ds, block_val_ds,
                    args, state, device, use_amp,
                    make_block_batch)
            elif block_samples:
                train_block_head(
                    model, model.block_head,
                    block_samples, args, state,
                    device, use_amp)

        # Train target head (single-select CE, 256-dim candidates)
        if 'target' in heads_list and use_mmap and target_train_ds:
            state.epoch = 0
            state.epoch_progress = 0
            train_head_mmap(
                model, model.target_head, 'target',
                target_train_ds, target_val_ds,
                args, state, device, use_amp,
                make_target_batch)

        # Train card select head (multi-select, like attack)
        if 'card_select' in heads_list and use_mmap and card_select_train_ds:
            state.epoch = 0
            state.epoch_progress = 0
            log(state, f"\n=== Training card_select head (mmap) ===")
            log(state, f"Train: {len(card_select_train_ds)}, Val: {len(card_select_val_ds)}")
            train_head_mmap(
                model, model.card_select_head, 'card_select',
                card_select_train_ds, card_select_val_ds,
                args, state, device, use_amp,
                make_batch)

        # Train mulligan head (binary keep/mull)
        if 'mulligan' in heads_list and use_mmap and mulligan_train_ds:
            state.epoch = 0
            state.epoch_progress = 0
            train_head_mmap(
                model, model.mulligan_head, 'mulligan',
                mulligan_train_ds, mulligan_val_ds,
                args, state, device, use_amp,
                make_mulligan_batch)

        # Train binary head (simple BCE from game state)
        if 'binary' in heads_list and use_mmap and binary_train_ds:
            state.epoch = 0
            state.epoch_progress = 0
            train_head_mmap(
                model, model.binary_head, 'binary',
                binary_train_ds, binary_val_ds,
                args, state, device, use_amp,
                make_binary_batch)

        # Save combined model
        save_path = os.path.join(
            args.save_dir, 'model_with_decisions.pt')
        model.save(save_path)

        log(state, f"\n=== Training Complete ===")
        heads_trained = [h.strip() for h in args.heads.split(',')] if args.heads != 'all' else all_heads
        for h in all_heads:
            acc = getattr(state, f'{h}_best_acc', 0.0)
            if h in heads_trained and acc > 0:
                log(state, f"{h}: best val acc {acc:.1%}")
            elif h in heads_trained:
                log(state, f"{h}: trained (no accuracy metric)")
            elif acc == 0.0:
                log(state, f"{h}: not trained (weights from encoder checkpoint)")
        log(state, f"Model saved: {save_path}")

        state.status = "Training complete!"
        state.phase = "done"
        state.chart_dirty = True

    except Exception as e:
        log(state, f"\nERROR: {e}")
        state.status = f"ERROR: {e}"
        state.phase = "done"
        state.chart_dirty = True
        import traceback
        traceback.print_exc()
        with open('/tmp/rl_decision_error.log', 'w') as f:
            traceback.print_exc(file=f)


# ── Tkinter Dashboard ────────────────────────────────

class DecisionDashboard:
    def __init__(self, root, state):
        self.root = root
        self.state = state
        self.root.title(
            "MTG RL — Decision Head Training")
        self.root.geometry("900x750")
        self.root.configure(bg='#1e1e2e')

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Header.TLabel',
            font=('Helvetica', 16, 'bold'),
            background='#1e1e2e', foreground='#cdd6f4')
        style.configure('Stat.TLabel',
            font=('Consolas', 11),
            background='#1e1e2e', foreground='#a6adc8')
        style.configure('Value.TLabel',
            font=('Consolas', 11, 'bold'),
            background='#1e1e2e', foreground='#89b4fa')
        style.configure('Good.TLabel',
            font=('Consolas', 11, 'bold'),
            background='#1e1e2e', foreground='#a6e3a1')
        style.configure('Status.TLabel',
            font=('Consolas', 10),
            background='#1e1e2e', foreground='#f9e2af')
        style.configure('Dark.TFrame',
            background='#1e1e2e')
        style.configure("blue.Horizontal.TProgressbar",
            troughcolor='#313244', background='#89b4fa')

        self._build_ui()
        self._update_loop()

    def _build_ui(self):
        m = ttk.Frame(self.root, style='Dark.TFrame')
        m.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(m,
            text="MTG RL — Decision Head Training",
            style='Header.TLabel').pack(pady=(0, 10))

        self.status_var = tk.StringVar(
            value="Initializing...")
        ttk.Label(m, textvariable=self.status_var,
            style='Status.TLabel').pack()

        pf = ttk.Frame(m, style='Dark.TFrame')
        pf.pack(fill=tk.X, pady=5)
        self.progress = ttk.Progressbar(
            pf, length=860, mode='determinate',
            style="blue.Horizontal.TProgressbar")
        self.progress.pack(fill=tk.X, pady=2)
        self.prog_label = tk.StringVar(value="")
        ttk.Label(pf, textvariable=self.prog_label,
            style='Stat.TLabel').pack(anchor='w')

        # Stats
        sf = ttk.Frame(m, style='Dark.TFrame')
        sf.pack(fill=tk.X, pady=5)
        self.stat_vars = {}
        stats = [
            ('Head', '—'), ('Epoch', '—'),
            ('Train Loss', '—'), ('Train Acc', '—'),
            ('Val Loss', '—'), ('Val Acc', '—'),
            ('Best Attack', '—'), ('Best Block', '—'),
            ('Best Priority', '—'), ('Epoch Time', '—'),
            ('ETA', '—'), ('GPU Mem', '—'),
            ('Head Params', '—'),
        ]
        for i, (label, default) in enumerate(stats):
            r, c = divmod(i, 4)
            lbl = ttk.Label(sf, text=f"{label}:",
                style='Stat.TLabel')
            lbl.grid(row=r, column=c*2, sticky='w',
                padx=(10, 2), pady=2)
            var = tk.StringVar(value=default)
            val = ttk.Label(sf, textvariable=var,
                style='Value.TLabel')
            val.grid(row=r, column=c*2+1, sticky='w',
                padx=(0, 15), pady=2)
            self.stat_vars[label] = var

        # Charts
        if HAS_MATPLOTLIB:
            cf = ttk.Frame(m, style='Dark.TFrame')
            cf.pack(fill=tk.BOTH, expand=True, pady=5)

            self.fig = Figure(figsize=(8.5, 6), dpi=100,
                facecolor='#1e1e2e')

            # Dynamic chart grid — all possible heads
            self.chart_heads = [
                ('priority', 'Priority'),
                ('attack', 'Attack'),
                ('block', 'Block'),
                ('target', 'Target'),
                ('card_select', 'Card Select'),
                ('mulligan', 'Mulligan'),
                ('binary', 'Binary'),
            ]
            n_heads = len(self.chart_heads)
            cols = min(n_heads, 4)
            rows = 2  # loss + accuracy rows
            self.fig.set_size_inches(2.5 * cols, 3 * rows)

            self.head_axes = {}
            for i, (hname, htitle) in enumerate(self.chart_heads):
                ax_loss = self.fig.add_subplot(rows, cols, i + 1)
                self.head_axes[f'{hname}_loss'] = ax_loss

            for ax_key, ax in self.head_axes.items():
                hname = ax_key.replace('_loss', '')
                htitle = dict(self.chart_heads).get(hname, hname)
                ax.set_facecolor('#313244')
                ax.set_title(f'{htitle}', color='#cdd6f4',
                    fontsize=8)
                ax.tick_params(colors='#6c7086',
                    labelsize=6)
                for spine in ax.spines.values():
                    spine.set_color('#45475a')

            self.fig.tight_layout(pad=2.0)
            self.canvas = FigureCanvasTkAgg(
                self.fig, master=cf)
            self.canvas.get_tk_widget().pack(
                fill=tk.BOTH, expand=True)

    def _update_loop(self):
        s = self.state
        self.status_var.set(s.status)

        if s.phase == "loading":
            pct = s.files_loaded / max(
                s.files_total, 1) * 100
            self.progress['value'] = pct
            self.prog_label.set(
                f"Loading: {s.files_loaded}/"
                f"{s.files_total} files | "
                f"Atk: {s.attack_samples} | "
                f"Blk: {s.block_samples} | "
                f"Pri: {s.priority_samples}")
        elif s.phase == "training":
            # Three heads: priority=0-33, attack=33-66, block=66-100
            head_offsets = {'priority': 0, 'attack': 33,
                            'block': 66}
            head_offset = head_offsets.get(
                s.current_head, 0)
            total_pct = head_offset + (
                (s.epoch - 1 + s.epoch_progress)
                / max(s.total_epochs, 1) * 33)
            self.progress['value'] = total_pct
            self.prog_label.set(
                f"{s.current_head.title()} head: "
                f"epoch {s.epoch}/{s.total_epochs}")
        elif s.phase == "done":
            self.progress['value'] = 100

        self.stat_vars['Head'].set(
            s.current_head.title() or '—')
        self.stat_vars['Epoch'].set(
            f"{s.epoch}/{s.total_epochs}")
        self.stat_vars['Train Loss'].set(
            f"{s.current_train_loss:.4f}")
        self.stat_vars['Train Acc'].set(
            f"{s.current_train_acc:.1%}")
        self.stat_vars['Val Loss'].set(
            f"{s.current_val_loss:.4f}")
        self.stat_vars['Val Acc'].set(
            f"{s.current_val_acc:.1%}")
        self.stat_vars['Best Attack'].set(
            f"{s.attack_best_acc:.1%}")
        self.stat_vars['Best Block'].set(
            f"{s.block_best_acc:.1%}")
        self.stat_vars['Best Priority'].set(
            f"{s.priority_best_acc:.1%}")
        self.stat_vars['Epoch Time'].set(
            f"{s.epoch_time:.1f}s")
        self.stat_vars['ETA'].set(
            f"{s.eta:.0f}s" if s.eta > 0 else "—")
        self.stat_vars['Head Params'].set(
            f"{s.head_params:,}" if s.head_params else "—")
        if s.gpu_mem_total_mb > 0:
            self.stat_vars['GPU Mem'].set(
                f"{s.gpu_mem_used_mb:.0f}/"
                f"{s.gpu_mem_total_mb:.0f} MB")

        if HAS_MATPLOTLIB and s.chart_dirty:
            s.chart_dirty = False
            self._draw_charts(s)

        self.root.after(500, self._update_loop)

    def _draw_charts(self, s):
        # Draw loss + accuracy for each head that has data
        for hname, htitle in self.chart_heads:
            ax_key = f'{hname}_loss'
            if ax_key not in self.head_axes:
                continue
            ax = self.head_axes[ax_key]

            tl = getattr(s, f'{hname}_train_losses', [])
            vl = getattr(s, f'{hname}_val_losses', [])
            ta = getattr(s, f'{hname}_train_accs', [])
            va = getattr(s, f'{hname}_val_accs', [])

            ax.clear()
            ax.set_facecolor('#313244')
            ax.set_title(f'{htitle}', color='#cdd6f4',
                fontsize=8)

            if ta:
                ep = range(1, len(ta) + 1)
                # Plot accuracy (primary metric)
                ax.plot(ep, ta, color='#89b4fa',
                    linewidth=1.5, label='Train Acc')
                if va:
                    ax.plot(ep, va, color='#f38ba8',
                        linewidth=1.5, label='Val Acc')
                ax.set_ylim(0, 1.05)
                ax.axhline(y=0.5, color='#585b70',
                    linestyle='--', linewidth=0.8)
                ax.legend(fontsize=6,
                    facecolor='#313244',
                    edgecolor='#45475a',
                    labelcolor='#cdd6f4')
            ax.tick_params(colors='#6c7086', labelsize=7)
            for spine in ax.spines.values():
                spine.set_color('#45475a')

        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()


# ── Main ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Decision Head Training Dashboard')
    parser.add_argument('--data-dir',
        default='../../rl_data/trajectories')
    parser.add_argument('--save-dir',
        default='../../rl_data/checkpoints')
    parser.add_argument('--encoder-checkpoint',
        default='../../rl_data/checkpoints/'
                'best_value_model.pt')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int,
        default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default=None)
    parser.add_argument('--max-files', type=int,
        default=None)
    parser.add_argument('--heads', default='all',
        help='Comma-separated heads to train: '
             'priority,attack,block or "all"')
    parser.add_argument('--joint', action='store_true',
        help='Train all heads jointly with unfrozen '
             'encoder')
    args = parser.parse_args()

    state = TrainingState()

    t = threading.Thread(target=trainer_thread,
        args=(state, args), daemon=True)
    t.start()

    root = tk.Tk()
    app = DecisionDashboard(root, state)
    root.mainloop()


if __name__ == '__main__':
    main()
