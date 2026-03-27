#!/usr/bin/env python3
"""
Train Decision Heads — imitation learning on heuristic AI choices.

Loads the pre-trained value network (frozen encoder), then trains
the specialized decision heads to predict what the heuristic AI
chose at each decision point.

Heads trained:
- AttackHead: which creatures to attack with (binary per creature)
- BlockHead: which creatures to block with
- PriorityHead: which spell/ability to play (future)

Usage:
    python training/train_decisions.py \
        --data-dir /path/to/trajectories \
        --encoder-checkpoint /path/to/best_value_model.pt \
        --device cuda \
        --epochs 50
"""

import argparse
import json
import os
import sys
import time
import random
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

os.environ['PYTHONUNBUFFERED'] = '1'

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

from model.mtg_model import MTGModel
from model.backend import resolve_backend
from model.gpu_config import auto_detect_profile
from training.mmap_dataset import parse_game_state, GAME_STATE_DIM, CARD_DIM, GLOBAL_DIM, ZONES_CONFIG
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout)
logger = logging.getLogger(__name__)


# ── Data loading ─────────────────────────────────────

def load_attack_decisions(data_dir, max_files=None):
    """
    Load DECLARE_ATTACKERS decisions with candidate features.

    Returns list of dicts with:
    - game_state_flat: full game state
    - global_features: 64 global features
    - creature_features: (N, CARD_DIM) features per creature
    - attack_mask: (N,) bool — which creatures attacked
    - won: whether this player won
    """
    path = Path(data_dir)
    files = sorted(path.glob('traj_*.jsonl'))
    if max_files:
        files = files[:max_files]

    samples = []
    skipped = 0

    print(f'  Loading attack decisions from {len(files)} '
          f'files...', flush=True)

    from training.chunked_loader import extract_game_id

    for i, filepath in enumerate(files):
        if i % 200 == 0:
            print(f'  {i}/{len(files)} files, '
                  f'{len(samples)} attack samples...',
                  end='\r', flush=True)
        try:
            gid = extract_game_id(filepath)
            with open(filepath, 'r') as f:
                lines = f.readlines()
            if len(lines) < 2:
                continue

            header = json.loads(lines[0])
            won = header.get('won', False)

            for line in lines[1:]:
                rec = json.loads(line)
                if rec.get('decisionType') != \
                        'DECLARE_ATTACKERS':
                    continue

                cand_feats = rec.get('candidateFeatures', [])
                selected = rec.get('selectedIndices', [])

                # Need at least 1 creature to be meaningful
                if len(cand_feats) < 1:
                    skipped += 1
                    continue

                # Parse features
                n_creatures = len(cand_feats)
                creatures = np.zeros(
                    (n_creatures, CARD_DIM), dtype=np.float32)
                for j, cf in enumerate(cand_feats):
                    cl = min(len(cf), CARD_DIM)
                    creatures[j, :cl] = np.array(
                        cf[:cl], dtype=np.float32)

                # Clamp
                np.clip(creatures, -10, 10, out=creatures)
                creatures = np.nan_to_num(creatures)

                # Attack mask
                attack_mask = np.zeros(
                    n_creatures, dtype=np.float32)
                for idx in selected:
                    if 0 <= idx < n_creatures:
                        attack_mask[idx] = 1.0

                # Global features
                gf = np.array(
                    rec.get('globalFeatures', []),
                    dtype=np.float32)
                np.clip(gf, -10, 10, out=gf)
                gf = np.nan_to_num(gf)
                g = np.zeros(64, dtype=np.float32)
                gl = min(len(gf), 64)
                if gl > 0:
                    g[:gl] = gf[:gl]

                # Full game state
                flat = np.array(
                    rec.get('gameStateFlat', []),
                    dtype=np.float32)
                np.clip(flat, -10, 10, out=flat)
                flat = np.nan_to_num(flat)

                samples.append({
                    'global_features': g,
                    'game_state_flat': flat,
                    'creature_features': creatures,
                    'attack_mask': attack_mask,
                    'n_creatures': n_creatures,
                    'won': 1.0 if won else 0.0,
                    'game_id': gid,
                })

        except Exception:
            pass

    print(f'  Loaded {len(samples)} attack decisions '
          f'(skipped {skipped} empty)', flush=True)

    # Stats
    if samples:
        avg_creatures = np.mean(
            [s['n_creatures'] for s in samples])
        avg_attackers = np.mean(
            [s['attack_mask'].sum() for s in samples])
        attack_rate = np.mean(
            [s['attack_mask'].mean() for s in samples])
        print(f'  Avg creatures: {avg_creatures:.1f}, '
              f'avg attackers: {avg_attackers:.1f}, '
              f'attack rate: {attack_rate:.1%}', flush=True)

    return samples


def load_block_decisions(data_dir, max_files=None):
    """Load DECLARE_BLOCKERS decisions."""
    path = Path(data_dir)
    files = sorted(path.glob('traj_*.jsonl'))
    if max_files:
        files = files[:max_files]

    samples = []

    from training.chunked_loader import extract_game_id

    print(f'  Loading block decisions...', flush=True)

    for filepath in files:
        try:
            gid = extract_game_id(filepath)
            with open(filepath, 'r') as f:
                lines = f.readlines()
            if len(lines) < 2:
                continue
            header = json.loads(lines[0])
            won = header.get('won', False)

            for line in lines[1:]:
                rec = json.loads(line)
                if rec.get('decisionType') != \
                        'DECLARE_BLOCKERS':
                    continue

                cand_feats = rec.get('candidateFeatures', [])
                selected = rec.get('selectedIndices', [])
                if len(cand_feats) < 1:
                    continue

                n = len(cand_feats)
                pair_dim = CARD_DIM * 2  # blocker + attacker
                creatures = np.zeros(
                    (n, pair_dim), dtype=np.float32)
                for j, cf in enumerate(cand_feats):
                    cl = min(len(cf), pair_dim)
                    creatures[j, :cl] = np.array(
                        cf[:cl], dtype=np.float32)
                np.clip(creatures, -10, 10, out=creatures)
                creatures = np.nan_to_num(creatures)

                block_mask = np.zeros(n, dtype=np.float32)
                for idx in selected:
                    if 0 <= idx < n:
                        block_mask[idx] = 1.0

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

                samples.append({
                    'global_features': g,
                    'game_state_flat': flat,
                    'creature_features': creatures,
                    'block_mask': block_mask,
                    'n_creatures': n,
                    'won': 1.0 if won else 0.0,
                    'game_id': gid,
                })
        except Exception:
            pass

    print(f'  Loaded {len(samples)} block decisions',
          flush=True)
    return samples


## parse_game_state imported from mmap_dataset


# ── Training ─────────────────────────────────────────

def train_attack_head(model, samples, args, device, use_amp):
    """Train the attack head via imitation learning."""
    print('\n  === Training Attack Head ===', flush=True)

    # Freeze encoder, unfreeze attack head
    for param in model.state_encoder.parameters():
        param.requires_grad = False
    for param in model.value_network.parameters():
        param.requires_grad = False
    for param in model.attack_head.parameters():
        param.requires_grad = True

    trainable = sum(
        p.numel() for p in model.attack_head.parameters())
    print(f'  Trainable params: {trainable:,} '
          f'(attack head only)', flush=True)

    optimizer = optim.AdamW(
        model.attack_head.parameters(),
        lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # Split by game to prevent data leakage
    from training.chunked_loader import split_by_game
    train_samples, val_samples = split_by_game(samples)
    print(f'  Train: {len(train_samples)} | '
          f'Val: {len(val_samples)} (by game)', flush=True)

    best_val_acc = 0
    save_path = os.path.join(
        args.save_dir, 'best_attack_model.pt')

    print(flush=True)
    print('  Epoch | Train Loss | Train Acc |'
          ' Val Loss  | Val Acc |  Time', flush=True)
    print('  ──────┼────────────┼───────────┼'
          '───────────┼─────────┼──────', flush=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        random.shuffle(train_samples)
        tloss, tcorrect, ttotal = 0, 0, 0

        for bi in range(0, len(train_samples),
                        args.batch_size):
            batch = train_samples[
                bi:bi + args.batch_size]
            if len(batch) < 2:
                continue

            loss, correct, total = _attack_batch(
                model, batch, device, use_amp)

            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.attack_head.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.attack_head.parameters(), 1.0)
                optimizer.step()

            tloss += loss.item() * total
            tcorrect += correct
            ttotal += total

        scheduler.step()

        # Val
        model.eval()
        vloss, vcorrect, vtotal = 0, 0, 0
        with torch.no_grad():
            for bi in range(0, len(val_samples),
                            args.batch_size):
                batch = val_samples[
                    bi:bi + args.batch_size]
                if len(batch) < 2:
                    continue
                loss, correct, total = _attack_batch(
                    model, batch, device, use_amp)
                vloss += loss.item() * total
                vcorrect += correct
                vtotal += total

        ta = tcorrect / max(ttotal, 1)
        va = vcorrect / max(vtotal, 1)
        tl = tloss / max(ttotal, 1)
        vl = vloss / max(vtotal, 1)
        elapsed = time.time() - t0

        status = ''
        if va > best_val_acc:
            best_val_acc = va
            model.save(save_path)
            status = ' *'

        print(f'  {epoch:>4d}   | {tl:>10.4f} | {ta:>8.1%} |'
              f' {vl:>9.4f} | {va:>6.1%} |'
              f' {elapsed:>4.1f}s{status}', flush=True)

    print(f'\n  Best val accuracy: {best_val_acc:.1%}',
          flush=True)
    return best_val_acc


def _attack_batch(model, batch, device, use_amp):
    """Process one batch of attack decisions."""
    # Pad creatures to max in batch
    max_c = max(s['n_creatures'] for s in batch)
    max_c = max(max_c, 1)
    bs = len(batch)

    cd = CARD_DIM
    creature_feats = torch.zeros(bs, max_c, cd,
                                  device=device)
    creature_mask = torch.zeros(bs, max_c,
                                 dtype=torch.bool,
                                 device=device)
    targets = torch.zeros(bs, max_c, device=device)
    global_feats = torch.zeros(bs, GLOBAL_DIM, device=device)

    # Parse game states for encoder
    my_board = torch.zeros(bs, 40, cd, device=device)
    my_board_m = torch.zeros(bs, 40, dtype=torch.bool,
                              device=device)
    opp_board = torch.zeros(bs, 40, cd, device=device)
    opp_board_m = torch.zeros(bs, 40, dtype=torch.bool,
                               device=device)
    hand = torch.zeros(bs, 15, cd, device=device)
    hand_m = torch.zeros(bs, 15, dtype=torch.bool,
                          device=device)
    my_gy = torch.zeros(bs, 20, cd, device=device)
    my_gy_m = torch.zeros(bs, 20, dtype=torch.bool,
                           device=device)
    opp_gy = torch.zeros(bs, 20, cd, device=device)
    opp_gy_m = torch.zeros(bs, 20, dtype=torch.bool,
                            device=device)
    stack = torch.zeros(bs, 10, cd, device=device)
    stack_m = torch.zeros(bs, 10, dtype=torch.bool,
                           device=device)

    for i, s in enumerate(batch):
        nc = s['n_creatures']
        creature_feats[i, :nc] = torch.from_numpy(
            s['creature_features'])
        creature_mask[i, :nc] = True
        targets[i, :nc] = torch.from_numpy(
            s['attack_mask'])

        g, zones, masks = parse_game_state(
            s['game_state_flat'], s['global_features'])
        global_feats[i] = torch.from_numpy(g)
        my_board[i] = torch.from_numpy(
            zones['my_board'])
        my_board_m[i] = torch.from_numpy(
            masks['my_board_mask'])
        opp_board[i] = torch.from_numpy(
            zones['opp_board'])
        opp_board_m[i] = torch.from_numpy(
            masks['opp_board_mask'])
        hand[i] = torch.from_numpy(zones['hand'])
        hand_m[i] = torch.from_numpy(
            masks['hand_mask'])
        my_gy[i] = torch.from_numpy(zones['my_gy'])
        my_gy_m[i] = torch.from_numpy(
            masks['my_gy_mask'])
        opp_gy[i] = torch.from_numpy(zones['opp_gy'])
        opp_gy_m[i] = torch.from_numpy(
            masks['opp_gy_mask'])
        stack[i] = torch.from_numpy(zones['stack'])
        stack_m[i] = torch.from_numpy(
            masks['stack_mask'])

    with torch.amp.autocast('cuda', enabled=use_amp):
        # Encode game state (frozen)
        with torch.no_grad():
            state = model.encode_state(
                global_feats,
                my_board, my_board_m,
                opp_board, opp_board_m,
                hand, hand_m,
                my_gy, my_gy_m,
                opp_gy, opp_gy_m,
                stack, stack_m)

        # Attack head forward
        logits = model.attack_head(
            state, creature_feats, creature_mask)

        # Binary cross-entropy per creature
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets,
            reduction='none')
        # Only count real creatures
        loss = (loss * creature_mask.float()).sum() / \
               creature_mask.float().sum().clamp(min=1)

    # Accuracy: per-creature binary accuracy
    with torch.no_grad():
        preds = (logits > 0).float()
        correct = ((preds == targets) *
                   creature_mask.float()).sum().item()
        total = creature_mask.float().sum().item()

    return loss, correct, total


def train_block_head(model, samples, args, device, use_amp):
    """Train the block head (same structure as attack)."""
    print('\n  === Training Block Head ===', flush=True)

    for param in model.state_encoder.parameters():
        param.requires_grad = False
    for param in model.attack_head.parameters():
        param.requires_grad = False
    for param in model.block_head.parameters():
        param.requires_grad = True

    # Reuse attack training logic with block_mask
    # Convert block samples to attack format
    for s in samples:
        s['attack_mask'] = s['block_mask']

    trainable = sum(
        p.numel() for p in model.block_head.parameters())
    print(f'  Trainable params: {trainable:,}', flush=True)

    # Use same training loop but with block head
    # For now, use the attack head as a stand-in
    # (same binary per-creature architecture)
    optimizer = optim.AdamW(
        model.block_head.parameters(),
        lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # Split by game to prevent data leakage
    from training.chunked_loader import split_by_game
    train_s, val_s = split_by_game(samples)
    print(f'  Train: {len(train_s)} | Val: {len(val_s)}'
          ' (by game)', flush=True)

    best_val_acc = 0
    save_path = os.path.join(
        args.save_dir, 'best_block_model.pt')

    print(flush=True)
    print('  Epoch | Train Loss | Train Acc |'
          ' Val Loss  | Val Acc |  Time', flush=True)
    print('  ──────┼────────────┼───────────┼'
          '───────────┼─────────┼──────', flush=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        model.train()
        random.shuffle(train_s)
        tl, tc, tt = 0, 0, 0
        for bi in range(0, len(train_s), args.batch_size):
            batch = train_s[bi:bi + args.batch_size]
            if len(batch) < 2:
                continue
            # Use attack_batch but swap the head
            loss, correct, total = _block_batch(
                model, batch, device, use_amp)
            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.block_head.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.block_head.parameters(), 1.0)
                optimizer.step()
            tl += loss.item() * total
            tc += correct
            tt += total

        scheduler.step()

        model.eval()
        vl, vc, vt = 0, 0, 0
        with torch.no_grad():
            for bi in range(0, len(val_s), args.batch_size):
                batch = val_s[bi:bi + args.batch_size]
                if len(batch) < 2:
                    continue
                loss, correct, total = _block_batch(
                    model, batch, device, use_amp)
                vl += loss.item() * total
                vc += correct
                vt += total

        ta = tc / max(tt, 1)
        va = vc / max(vt, 1)
        elapsed = time.time() - t0

        status = ''
        if va > best_val_acc:
            best_val_acc = va
            model.save(save_path)
            status = ' *'

        print(f'  {epoch:>4d}   | {tl/max(tt,1):>10.4f} |'
              f' {ta:>8.1%} |'
              f' {vl/max(vt,1):>9.4f} | {va:>6.1%} |'
              f' {elapsed:>4.1f}s{status}', flush=True)

    print(f'\n  Best val accuracy: {best_val_acc:.1%}',
          flush=True)
    return best_val_acc


def _block_batch(model, batch, device, use_amp):
    """Same as _attack_batch but uses block_head."""
    max_c = max(s['n_creatures'] for s in batch)
    max_c = max(max_c, 1)
    bs = len(batch)
    cd = CARD_DIM

    creature_feats = torch.zeros(bs, max_c, cd,
                                  device=device)
    creature_mask = torch.zeros(bs, max_c,
                                 dtype=torch.bool,
                                 device=device)
    targets = torch.zeros(bs, max_c, device=device)
    global_feats = torch.zeros(bs, GLOBAL_DIM, device=device)

    my_board = torch.zeros(bs, 40, cd, device=device)
    my_board_m = torch.zeros(bs, 40, dtype=torch.bool,
                              device=device)
    opp_board = torch.zeros(bs, 40, cd, device=device)
    opp_board_m = torch.zeros(bs, 40, dtype=torch.bool,
                               device=device)
    hand = torch.zeros(bs, 15, cd, device=device)
    hand_m = torch.zeros(bs, 15, dtype=torch.bool,
                          device=device)
    my_gy = torch.zeros(bs, 20, cd, device=device)
    my_gy_m = torch.zeros(bs, 20, dtype=torch.bool,
                           device=device)
    opp_gy = torch.zeros(bs, 20, cd, device=device)
    opp_gy_m = torch.zeros(bs, 20, dtype=torch.bool,
                            device=device)
    stack = torch.zeros(bs, 10, cd, device=device)
    stack_m = torch.zeros(bs, 10, dtype=torch.bool,
                           device=device)

    for i, s in enumerate(batch):
        nc = s['n_creatures']
        creature_feats[i, :nc] = torch.from_numpy(
            s['creature_features'])
        creature_mask[i, :nc] = True
        targets[i, :nc] = torch.from_numpy(
            s['attack_mask'])  # reused field

        g, zones, masks = parse_game_state(
            s['game_state_flat'], s['global_features'])
        global_feats[i] = torch.from_numpy(g)
        my_board[i] = torch.from_numpy(
            zones['my_board'])
        my_board_m[i] = torch.from_numpy(
            masks['my_board_mask'])
        opp_board[i] = torch.from_numpy(
            zones['opp_board'])
        opp_board_m[i] = torch.from_numpy(
            masks['opp_board_mask'])
        hand[i] = torch.from_numpy(zones['hand'])
        hand_m[i] = torch.from_numpy(
            masks['hand_mask'])
        my_gy[i] = torch.from_numpy(zones['my_gy'])
        my_gy_m[i] = torch.from_numpy(
            masks['my_gy_mask'])
        opp_gy[i] = torch.from_numpy(zones['opp_gy'])
        opp_gy_m[i] = torch.from_numpy(
            masks['opp_gy_mask'])
        stack[i] = torch.from_numpy(zones['stack'])
        stack_m[i] = torch.from_numpy(
            masks['stack_mask'])

    with torch.amp.autocast('cuda', enabled=use_amp):
        with torch.no_grad():
            state = model.encode_state(
                global_feats,
                my_board, my_board_m,
                opp_board, opp_board_m,
                hand, hand_m,
                my_gy, my_gy_m,
                opp_gy, opp_gy_m,
                stack, stack_m)

        # Block head: use same interface as attack head
        # (binary per creature: block or not)
        logits = model.attack_head(
            state, creature_feats, creature_mask)

        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none')
        loss = (loss * creature_mask.float()).sum() / \
               creature_mask.float().sum().clamp(min=1)

    with torch.no_grad():
        preds = (logits > 0).float()
        correct = ((preds == targets) *
                   creature_mask.float()).sum().item()
        total = creature_mask.float().sum().item()

    return loss, correct, total


# ── Main ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train MTG RL Decision Heads')
    parser.add_argument('--data-dir',
        default='../../rl_data/trajectories')
    parser.add_argument('--save-dir',
        default='../../rl_data/checkpoints')
    parser.add_argument('--encoder-checkpoint',
        default='../../rl_data/checkpoints/best_value_model.pt',
        help='Pre-trained value model with encoder')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int,
        default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default=None)
    parser.add_argument('--max-files', type=int,
        default=None)
    parser.add_argument('--heads', default='attack,block',
        help='Comma-separated heads to train')
    args = parser.parse_args()

    profile = auto_detect_profile()
    backend = resolve_backend(args.device)
    device = backend.torch_device
    use_amp = profile.use_amp and backend.use_amp

    os.makedirs(args.save_dir, exist_ok=True)

    print('┌────────────────────────────────────────────┐',
          flush=True)
    print('│    MTG RL — Decision Head Training         │',
          flush=True)
    print('└────────────────────────────────────────────┘',
          flush=True)
    print(f'  Device: {backend.name} ({profile.name})',
          flush=True)
    print(f'  AMP: {use_amp}', flush=True)
    print(f'  Encoder: {args.encoder_checkpoint}',
          flush=True)

    # Load pre-trained model
    if os.path.exists(args.encoder_checkpoint):
        print(f'  Loading pre-trained encoder...',
              flush=True)
        model = MTGModel.load(
            args.encoder_checkpoint, device=backend.name)
        print(f'  Loaded.', flush=True)
    else:
        print(f'  No checkpoint found, using random init',
              flush=True)
        model = MTGModel().to(device)

    heads = args.heads.split(',')

    if 'attack' in heads:
        attack_samples = load_attack_decisions(
            args.data_dir, args.max_files)
        if attack_samples:
            train_attack_head(
                model, attack_samples, args,
                device, use_amp)
        else:
            print('  No attack data found', flush=True)

    if 'block' in heads:
        block_samples = load_block_decisions(
            args.data_dir, args.max_files)
        if block_samples:
            train_block_head(
                model, block_samples, args,
                device, use_amp)
        else:
            print('  No block data found', flush=True)

    # Save final combined model
    model.save(os.path.join(
        args.save_dir, 'model_with_decisions.pt'))
    print(f'\n  Saved to {args.save_dir}/'
          f'model_with_decisions.pt', flush=True)


if __name__ == '__main__':
    main()
