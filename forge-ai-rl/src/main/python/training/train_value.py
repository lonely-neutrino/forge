#!/usr/bin/env python3
"""
Train Value Network — learns to evaluate MTG game states.

Trains the game state encoder + value network to predict
win/loss from mid-game state snapshots.

Features:
- Live terminal dashboard with training metrics
- Progress bars for data loading and epochs
- TensorBoard logging for detailed analysis
- AMP (fp16) for GPU memory efficiency
- Auto-saves best model checkpoint

Usage:
    python training/train_value.py \
        --data-dir /path/to/trajectories \
        --device cuda \
        --epochs 50
"""

import argparse
import os
import sys
import time
import logging
import signal

import torch
import torch.nn as nn
import torch.optim as optim

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

from model.mtg_model import MTGModel
from model.backend import resolve_backend
from model.gpu_config import auto_detect_profile, estimate_memory_usage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout)
logger = logging.getLogger(__name__)


# ── Terminal UI helpers ──────────────────────────────

def clear_line():
    print('\r' + ' ' * 80 + '\r', end='', flush=True)

def progress_bar(current, total, width=40, prefix='',
                 suffix='', fill='█'):
    pct = current / max(total, 1)
    filled = int(width * pct)
    bar = fill * filled + '░' * (width - filled)
    print(f'\r  {prefix} |{bar}| {current}/{total}'
          f' {pct:.0%} {suffix}', end='', flush=True)
    if current >= total:
        print(flush=True)


def print_header(title):
    w = 62
    print(flush=True)
    print('┌' + '─' * w + '┐', flush=True)
    print('│' + title.center(w) + '│', flush=True)
    print('└' + '─' * w + '┘', flush=True)


def print_config(items):
    """Print config as a nice table."""
    print('  ┌──────────────────────┬───────────────────────┐',
          flush=True)
    for k, v in items:
        print(f'  │ {k:<20s} │ {str(v):>21s} │', flush=True)
    print('  └──────────────────────┴───────────────────────┘',
          flush=True)


def print_epoch_header():
    print(flush=True)
    print('  Epoch │ Train Loss │ Train Acc │'
          ' Val Loss │ Val Acc │  Time  │ Status',
          flush=True)
    print('  ──────┼────────────┼───────────┼'
          '──────────┼─────────┼────────┼───────',
          flush=True)


def print_epoch_row(epoch, epochs, tloss, tacc,
                    vloss, vacc, elapsed, status=''):
    print(f'  {epoch:>4d}/{epochs:<1d} │'
          f' {tloss:>10.6f} │'
          f' {tacc:>8.1%} │'
          f' {vloss:>8.6f} │'
          f' {vacc:>6.1%} │'
          f' {elapsed:>5.1f}s │'
          f' {status}', flush=True)


def print_summary(best_acc, best_epoch, total_time, save_path):
    print(flush=True)
    print('  ┌──────────────────────────────────────────┐',
          flush=True)
    print(f'  │ Best val accuracy: {best_acc:>6.1%}'
          f' (epoch {best_epoch})      │', flush=True)
    print(f'  │ Total training time: {total_time:>6.1f}s'
          f'              │', flush=True)
    print(f'  │ Model: {os.path.basename(save_path):<33s}│',
          flush=True)
    print('  └──────────────────────────────────────────┘',
          flush=True)


# ── Dataset with progress ────────────────────────────

def load_dataset_with_progress(data_dir, global_dim=96,
                                card_dim=256, max_board=40,
                                max_hand=15, max_gy=20,
                                max_stack=10, max_files=None):
    """Load trajectory files with a progress bar."""
    import json
    import numpy as np
    from pathlib import Path

    path = Path(data_dir)
    files = sorted(path.glob('traj_*.jsonl'))
    if max_files:
        files = files[:max_files]
    if not files:
        logger.error(f"No trajectory files in {data_dir}")
        return []

    zones_config = [
        ('my_board', max_board, card_dim),
        ('opp_board', max_board, card_dim),
        ('hand', max_hand, card_dim),
        ('my_gy', max_gy, card_dim),
        ('opp_gy', max_gy, card_dim),
        ('stack', max_stack, card_dim),
    ]

    samples = []
    wins = 0
    losses = 0
    total_decisions = 0

    print(f'  Loading {len(files)} trajectory files...',
          flush=True)

    from training.chunked_loader import extract_game_id

    for i, filepath in enumerate(files):
        if i % 50 == 0 or i == len(files) - 1:
            progress_bar(i + 1, len(files),
                         prefix='Loading',
                         suffix=f'{len(samples)} samples')
        try:
            gid = extract_game_id(filepath)
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

            for line in lines[1:]:
                rec = json.loads(line)
                flat = np.array(
                    rec.get('gameStateFlat', []),
                    dtype=np.float32)
                gf = np.array(
                    rec.get('globalFeatures', []),
                    dtype=np.float32)

                # Clamp extreme values
                np.clip(flat, -10.0, 10.0, out=flat)
                flat = np.nan_to_num(flat, nan=0.0,
                                     posinf=1.0, neginf=-1.0)
                np.clip(gf, -10.0, 10.0, out=gf)
                gf = np.nan_to_num(gf, nan=0.0,
                                   posinf=1.0, neginf=-1.0)

                # Parse zones from flat array
                g = np.zeros(global_dim, dtype=np.float32)
                g_len = min(len(gf), global_dim)
                if g_len > 0:
                    g[:g_len] = gf[:g_len]

                zones = {}
                masks = {}
                offset = global_dim
                for name, count, dim in zones_config:
                    zs = count * dim
                    zd = np.zeros((count, dim),
                                  dtype=np.float32)
                    zm = np.zeros(count, dtype=np.bool_)
                    if offset + zs <= len(flat):
                        raw = flat[offset:offset + zs].reshape(
                            count, dim)
                        for j in range(count):
                            if np.any(raw[j] != 0):
                                zd[j] = raw[j]
                                zm[j] = True
                    offset += zs
                    zones[name] = zd
                    masks[name + '_mask'] = zm

                samples.append({
                    'global_features': g,
                    'zones': zones,
                    'masks': masks,
                    'won': 1.0 if won else 0.0,
                    'value_target':
                        1.0 if won else -1.0,
                    'game_id': gid,
                })
                total_decisions += 1

        except Exception:
            pass

    print(flush=True)
    print(f'  Loaded {len(samples)} samples '
          f'({wins}W/{losses}L, '
          f'{total_decisions} decisions)', flush=True)
    return samples


class TensorDataset(torch.utils.data.Dataset):
    """Wraps parsed samples into a PyTorch Dataset."""
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'global_features': torch.from_numpy(
                s['global_features']),
            'my_board': torch.from_numpy(
                s['zones']['my_board']),
            'my_board_mask': torch.from_numpy(
                s['masks']['my_board_mask']),
            'opp_board': torch.from_numpy(
                s['zones']['opp_board']),
            'opp_board_mask': torch.from_numpy(
                s['masks']['opp_board_mask']),
            'hand': torch.from_numpy(
                s['zones']['hand']),
            'hand_mask': torch.from_numpy(
                s['masks']['hand_mask']),
            'my_gy': torch.from_numpy(
                s['zones']['my_gy']),
            'my_gy_mask': torch.from_numpy(
                s['masks']['my_gy_mask']),
            'opp_gy': torch.from_numpy(
                s['zones']['opp_gy']),
            'opp_gy_mask': torch.from_numpy(
                s['masks']['opp_gy_mask']),
            'stack': torch.from_numpy(
                s['zones']['stack']),
            'stack_mask': torch.from_numpy(
                s['masks']['stack_mask']),
            'won': torch.tensor(
                s['won'], dtype=torch.float32),
            'value_target': torch.tensor(
                s['value_target'], dtype=torch.float32),
        }


# ── Training loops ───────────────────────────────────

def train_epoch(model, loader, optimizer, scaler,
                device, use_amp):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in loader:
        g = batch['global_features'].to(device)
        mb = batch['my_board'].to(device)
        mbm = batch['my_board_mask'].to(device)
        ob = batch['opp_board'].to(device)
        obm = batch['opp_board_mask'].to(device)
        h = batch['hand'].to(device)
        hm = batch['hand_mask'].to(device)
        mg = batch['my_gy'].to(device)
        mgm = batch['my_gy_mask'].to(device)
        og = batch['opp_gy'].to(device)
        ogm = batch['opp_gy_mask'].to(device)
        s = batch['stack'].to(device)
        sm = batch['stack_mask'].to(device)
        tgt = batch['value_target'].to(device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=use_amp):
            emb = model.encode_state(
                g, mb, mbm, ob, obm, h, hm,
                mg, mgm, og, ogm, s, sm)
            pred = model.get_value(emb).squeeze(-1)
            loss = nn.functional.mse_loss(pred, tgt)

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0)
            optimizer.step()

        bs = tgt.shape[0]
        total_loss += loss.item() * bs
        correct += ((pred > 0) == (tgt > 0)).sum().item()
        total += bs

    return total_loss / max(1, total), correct / max(1, total)


@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for batch in loader:
        g = batch['global_features'].to(device)
        mb = batch['my_board'].to(device)
        mbm = batch['my_board_mask'].to(device)
        ob = batch['opp_board'].to(device)
        obm = batch['opp_board_mask'].to(device)
        h = batch['hand'].to(device)
        hm = batch['hand_mask'].to(device)
        mg = batch['my_gy'].to(device)
        mgm = batch['my_gy_mask'].to(device)
        og = batch['opp_gy'].to(device)
        ogm = batch['opp_gy_mask'].to(device)
        s = batch['stack'].to(device)
        sm = batch['stack_mask'].to(device)
        tgt = batch['value_target'].to(device)

        with torch.amp.autocast('cuda', enabled=use_amp):
            emb = model.encode_state(
                g, mb, mbm, ob, obm, h, hm,
                mg, mgm, og, ogm, s, sm)
            pred = model.get_value(emb).squeeze(-1)
            loss = nn.functional.mse_loss(pred, tgt)

        bs = tgt.shape[0]
        total_loss += loss.item() * bs
        correct += ((pred > 0) == (tgt > 0)).sum().item()
        total += bs

    return total_loss / max(1, total), correct / max(1, total)


# ── Main ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train MTG RL Value Network')
    parser.add_argument('--data-dir',
        default='../../rl_data/trajectories')
    parser.add_argument('--save-dir',
        default='../../rl_data/checkpoints')
    parser.add_argument('--log-dir',
        default='../../rl_data/runs/value_train')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', default=None)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--max-files', type=int, default=None,
        help='Limit number of files to load (for testing)')
    args = parser.parse_args()

    # Setup
    profile = auto_detect_profile(args.device)
    backend = resolve_backend(args.device)
    device = backend.torch_device
    batch_size = args.batch_size or profile.batch_size
    use_amp = profile.use_amp and backend.use_amp

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Header
    print_header('MTG RL — Value Network Training')

    print_config([
        ('Device', f'{backend.name} ({profile.name})'),
        ('Batch size', str(batch_size)),
        ('Learning rate', f'{args.lr:.0e}'),
        ('Epochs', str(args.epochs)),
        ('Mixed precision', str(use_amp)),
        ('Data', args.data_dir),
    ])

    # Load data
    print(flush=True)
    samples = load_dataset_with_progress(
        args.data_dir, max_files=args.max_files if hasattr(args, 'max_files') else None)
    if not samples:
        return

    # Split by game to prevent data leakage
    from training.chunked_loader import split_by_game
    train_samples, val_samples = split_by_game(
        samples, val_fraction=args.val_split)
    print(f'  Split: {len(train_samples)} train, '
          f'{len(val_samples)} val (by game)', flush=True)
    train_ds = TensorDataset(train_samples)
    val_ds = TensorDataset(val_samples)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0,
        pin_memory=backend.pin_memory,
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0,
        pin_memory=backend.pin_memory)

    print(f'  Split: {n_train} train / {n_val} val',
          flush=True)

    # Model
    model = MTGModel().to(device)
    params = model.count_parameters()
    print(f'  Model: {params["total"]:,} parameters',
          flush=True)

    mem = estimate_memory_usage(batch_size)
    if backend.is_gpu:
        print(f'  Est. VRAM: {mem["total_gb"]:.2f} GB',
              flush=True)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # TensorBoard
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(args.log_dir)
    except ImportError:
        writer = None

    # Training
    print_epoch_header()
    best_val_acc = 0
    best_epoch = 0
    t_total_start = time.time()
    save_path = os.path.join(
        args.save_dir, 'best_value_model.pt')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tloss, tacc = train_epoch(
            model, train_loader, optimizer, scaler,
            device, use_amp)
        vloss, vacc = evaluate(
            model, val_loader, device, use_amp)
        scheduler.step()

        elapsed = time.time() - t0

        # Status indicator
        status = ''
        if vacc > best_val_acc:
            best_val_acc = vacc
            best_epoch = epoch
            model.save(save_path)
            status = '★ best'
        elif epoch % args.save_every == 0:
            model.save(os.path.join(
                args.save_dir,
                f'value_model_epoch_{epoch}.pt'))
            status = 'saved'

        print_epoch_row(epoch, args.epochs,
                        tloss, tacc, vloss, vacc,
                        elapsed, status)

        if writer:
            writer.add_scalar('train/loss', tloss, epoch)
            writer.add_scalar('train/accuracy', tacc, epoch)
            writer.add_scalar('val/loss', vloss, epoch)
            writer.add_scalar('val/accuracy', vacc, epoch)
            writer.add_scalar('train/lr',
                scheduler.get_last_lr()[0], epoch)

    # Final
    model.save(os.path.join(
        args.save_dir, 'value_model_final.pt'))
    total_time = time.time() - t_total_start

    print_summary(best_val_acc, best_epoch,
                  total_time, save_path)

    if writer:
        writer.close()


if __name__ == '__main__':
    main()
