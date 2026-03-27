#!/usr/bin/env python3
"""
Training Dashboard — Tkinter GUI for monitoring RL training.

Shows real-time:
- Loss and accuracy curves
- Training vs validation metrics
- GPU memory usage
- Data loading progress
- Epoch progress bar
- Live stats table

Launch:
    python training/training_ui.py --data-dir /path/to/trajectories --device cuda
"""

import argparse
import os
import sys
import threading
import time
import queue
from dataclasses import dataclass, field
from typing import List

import tkinter as tk
from tkinter import ttk

# Add parent to path
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

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
from model.backend import resolve_backend
from model.gpu_config import auto_detect_profile


# ── Shared state between trainer thread and UI ───────

@dataclass
class TrainingState:
    # Status
    status: str = "Idle"
    phase: str = ""  # "loading", "training", "done"
    chart_dirty: bool = False  # set by trainer, cleared by UI

    # Data loading
    files_total: int = 0
    files_loaded: int = 0
    samples_loaded: int = 0
    wins: int = 0
    losses: int = 0

    # Training config
    device: str = "cpu"
    gpu_name: str = ""
    model_params: int = 0
    batch_size: int = 64
    n_train: int = 0
    n_val: int = 0

    # Current epoch
    epoch: int = 0
    total_epochs: int = 50
    epoch_progress: float = 0.0  # 0-1 within epoch

    # Metrics history
    train_losses: List[float] = field(default_factory=list)
    train_accs: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_accs: List[float] = field(default_factory=list)

    # Current epoch metrics
    current_train_loss: float = 0.0
    current_train_acc: float = 0.0
    current_val_loss: float = 0.0
    current_val_acc: float = 0.0
    best_val_acc: float = 0.0
    best_epoch: int = 0
    epoch_time: float = 0.0

    # GPU
    gpu_mem_used_mb: float = 0.0
    gpu_mem_total_mb: float = 0.0

    # Timing
    start_time: float = 0.0
    elapsed: float = 0.0
    eta: float = 0.0


# ── Dataset loading (same as train_value.py) ─────────

def load_data_threaded(state: TrainingState, data_dir: str,
                       max_files=None):
    import json
    import numpy as np
    from pathlib import Path

    state.phase = "loading"
    state.status = "Loading trajectory files..."

    path = Path(data_dir)
    files = sorted(path.glob('traj_*.jsonl'))
    if max_files:
        files = files[:max_files]
    state.files_total = len(files)

    global_dim = 96
    card_dim = 256
    zones_config = [
        ('my_board', 40, card_dim),
        ('opp_board', 40, card_dim),
        ('hand', 15, card_dim),
        ('my_gy', 20, card_dim),
        ('opp_gy', 20, card_dim),
        ('stack', 10, card_dim),
    ]

    samples = []
    for i, filepath in enumerate(files):
        state.files_loaded = i + 1
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            if len(lines) < 2:
                continue
            header = json.loads(lines[0])
            won = header.get('won', False)
            if won:
                state.wins += 1
            else:
                state.losses += 1

            for line in lines[1:]:
                rec = json.loads(line)
                flat = np.array(
                    rec.get('gameStateFlat', []),
                    dtype=np.float32)
                gf = np.array(
                    rec.get('globalFeatures', []),
                    dtype=np.float32)
                np.clip(flat, -10.0, 10.0, out=flat)
                flat = np.nan_to_num(flat)
                np.clip(gf, -10.0, 10.0, out=gf)
                gf = np.nan_to_num(gf)

                g = np.zeros(global_dim, dtype=np.float32)
                gl = min(len(gf), global_dim)
                if gl > 0:
                    g[:gl] = gf[:gl]

                zones = {}
                masks = {}
                offset = global_dim
                for name, count, dim in zones_config:
                    zs = count * dim
                    zd = np.zeros((count, dim),
                                  dtype=np.float32)
                    zm = np.zeros(count, dtype=np.bool_)
                    if offset + zs <= len(flat):
                        raw = flat[offset:offset+zs].reshape(
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
                    'value_target':
                        1.0 if won else -1.0,
                })
                state.samples_loaded += 1
        except Exception:
            pass

    state.status = f"Loaded {len(samples)} samples"
    return samples


# ── Simple tensor dataset ────────────────────────────

class SimpleDataset(torch.utils.data.Dataset):
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
            'hand': torch.from_numpy(s['zones']['hand']),
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
            'value_target': torch.tensor(
                s['value_target'], dtype=torch.float32),
        }


# ── Training thread ──────────────────────────────────

def trainer_thread(state: TrainingState, args):
    """Runs training in background thread, updating state."""
    try:
        profile = auto_detect_profile(args.device)
        backend = resolve_backend(args.device)
        device = backend.torch_device
        batch_size = args.batch_size or profile.batch_size
        use_amp = profile.use_amp and backend.use_amp

        state.device = backend.display_name
        state.gpu_name = f'{backend.name} ({profile.name})'
        state.batch_size = batch_size
        state.total_epochs = args.epochs

        # Load data via memory-mapped preprocessed files
        from training.mmap_dataset import MmapValueDataset

        state.phase = "loading"

        # Check for preprocessed data
        preprocessed_dir = os.path.join(
            os.path.dirname(args.data_dir),
            'preprocessed')
        if not os.path.exists(
                os.path.join(preprocessed_dir,
                             'metadata.json')):
            state.status = (
                "ERROR: No preprocessed data. "
                "Run scripts/02b_preprocess_data.sh")
            state.phase = "done"
            return

        state.status = "Loading preprocessed data..."
        import json as json_mod
        with open(os.path.join(preprocessed_dir,
                               'metadata.json')) as f:
            meta = json_mod.load(f)

        train_dataset = MmapValueDataset(
            preprocessed_dir, train=True)
        val_dataset = MmapValueDataset(
            preprocessed_dir, train=False)

        state.n_train = len(train_dataset)
        state.n_val = len(val_dataset)
        state.samples_loaded = (
            state.n_train + state.n_val)
        state.files_total = meta.get('n_files', 0)
        state.files_loaded = state.files_total

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=True,
            num_workers=0,
            pin_memory=backend.pin_memory,
            drop_last=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=0,
            pin_memory=backend.pin_memory)

        # Model
        model = MTGModel().to(device)
        state.model_params = model.count_parameters()['total']

        optimizer = optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs)
        scaler = (torch.amp.GradScaler('cuda')
                  if use_amp else None)

        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(
            args.save_dir, 'best_value_model.pt')

        state.phase = "training"
        state.start_time = time.time()

        for epoch in range(1, args.epochs + 1):
            state.epoch = epoch
            state.status = f"Training epoch {epoch}/{args.epochs}"
            t0 = time.time()

            # Train
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            n_batches = len(train_loader)

            for bi, batch in enumerate(train_loader):
                state.epoch_progress = bi / max(
                    n_batches, 1)

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
                with torch.amp.autocast(
                        'cuda', enabled=use_amp):
                    emb = model.encode_state(
                        g, mb, mbm, ob, obm, h, hm,
                        mg, mgm, og, ogm, s, sm)
                    pred = model.get_value(
                        emb).squeeze(-1)
                    loss = nn.functional.mse_loss(
                        pred, tgt)

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
                correct += (
                    (pred > 0) == (tgt > 0)
                ).sum().item()
                total += bs

            state.current_train_loss = (
                total_loss / max(1, total))
            state.current_train_acc = (
                correct / max(1, total))
            state.train_losses.append(
                state.current_train_loss)
            state.train_accs.append(
                state.current_train_acc)

            # Eval
            model.eval()
            total_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
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

                    with torch.amp.autocast(
                            'cuda', enabled=use_amp):
                        emb = model.encode_state(
                            g, mb, mbm, ob, obm, h, hm,
                            mg, mgm, og, ogm, s, sm)
                        pred = model.get_value(
                            emb).squeeze(-1)
                        loss = nn.functional.mse_loss(
                            pred, tgt)

                    bs = tgt.shape[0]
                    total_loss += loss.item() * bs
                    correct += (
                        (pred > 0) == (tgt > 0)).sum().item()
                    total += bs

            state.current_val_loss = (
                total_loss / max(1, total))
            state.current_val_acc = (
                correct / max(1, total))
            state.val_losses.append(state.current_val_loss)
            state.val_accs.append(state.current_val_acc)

            scheduler.step()

            # Save best
            if state.current_val_acc > state.best_val_acc:
                state.best_val_acc = state.current_val_acc
                state.best_epoch = epoch
                model.save(save_path)

            if epoch % 10 == 0:
                model.save(os.path.join(
                    args.save_dir,
                    f'value_model_epoch_{epoch}.pt'))

            state.epoch_time = time.time() - t0
            state.elapsed = time.time() - state.start_time
            remaining = args.epochs - epoch
            state.eta = remaining * state.epoch_time

            # Sync GPU and update state
            if backend.is_cuda:
                torch.cuda.synchronize()
                state.gpu_mem_used_mb = (
                    torch.cuda.memory_allocated() / 1024**2)
                state.gpu_mem_total_mb = (
                    torch.cuda.get_device_properties(0)
                    .total_memory / 1024**2)

            state.epoch_progress = 1.0
            state.chart_dirty = True
            # Brief yield to let UI thread catch up
            time.sleep(0.05)

        model.save(os.path.join(
            args.save_dir, 'value_model_final.pt'))
        state.status = "Training complete!"
        state.phase = "done"

    except Exception as e:
        state.status = f"ERROR: {e}"
        state.phase = "done"
        state.chart_dirty = True
        import traceback
        traceback.print_exc()
        # Write error to file for debugging
        with open('/tmp/rl_train_error.log', 'w') as f:
            traceback.print_exc(file=f)


# ── Tkinter Dashboard ────────────────────────────────

class TrainingDashboard:
    def __init__(self, root, state: TrainingState):
        self.root = root
        self.state = state
        self.root.title("MTG RL — Training Dashboard")
        self.root.geometry("900x700")
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
        style.configure('Dark.TFrame', background='#1e1e2e')
        style.configure('Card.TFrame', background='#313244')
        style.configure("green.Horizontal.TProgressbar",
            troughcolor='#313244', background='#a6e3a1')
        style.configure("blue.Horizontal.TProgressbar",
            troughcolor='#313244', background='#89b4fa')

        self._build_ui()
        self._update_loop()

    def _build_ui(self):
        main = ttk.Frame(self.root, style='Dark.TFrame')
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Title
        ttk.Label(main, text="MTG RL — Value Network Training",
                  style='Header.TLabel').pack(pady=(0, 10))

        # Status bar
        self.status_var = tk.StringVar(value="Initializing...")
        ttk.Label(main, textvariable=self.status_var,
                  style='Status.TLabel').pack()

        # Progress section
        prog_frame = ttk.Frame(main, style='Dark.TFrame')
        prog_frame.pack(fill=tk.X, pady=5)

        ttk.Label(prog_frame, text="Progress:",
                  style='Stat.TLabel').pack(anchor='w')
        self.epoch_progress = ttk.Progressbar(
            prog_frame, length=860, mode='determinate',
            style="blue.Horizontal.TProgressbar")
        self.epoch_progress.pack(fill=tk.X, pady=2)
        self.progress_label = tk.StringVar(value="")
        ttk.Label(prog_frame,
                  textvariable=self.progress_label,
                  style='Stat.TLabel').pack(anchor='w')

        # Stats grid
        stats_frame = ttk.Frame(main, style='Dark.TFrame')
        stats_frame.pack(fill=tk.X, pady=5)

        self.stat_vars = {}
        stat_items = [
            ('Epoch', '—'), ('Train Loss', '—'),
            ('Train Acc', '—'), ('Val Loss', '—'),
            ('Val Acc', '—'), ('Best Val Acc', '—'),
            ('Epoch Time', '—'), ('ETA', '—'),
            ('GPU Mem', '—'), ('Samples', '—'),
            ('Device', '—'), ('Parameters', '—'),
        ]

        for i, (label, default) in enumerate(stat_items):
            r, c = divmod(i, 4)
            lbl = ttk.Label(stats_frame, text=f"{label}:",
                            style='Stat.TLabel')
            lbl.grid(row=r, column=c*2, sticky='w',
                     padx=(10, 2), pady=2)
            var = tk.StringVar(value=default)
            val = ttk.Label(stats_frame, textvariable=var,
                            style='Value.TLabel')
            val.grid(row=r, column=c*2+1, sticky='w',
                     padx=(0, 15), pady=2)
            self.stat_vars[label] = var

        # Charts
        if HAS_MATPLOTLIB:
            chart_frame = ttk.Frame(main, style='Dark.TFrame')
            chart_frame.pack(fill=tk.BOTH, expand=True, pady=5)

            self.fig = Figure(figsize=(8.5, 3.5), dpi=100,
                              facecolor='#1e1e2e')

            # Loss chart
            self.ax_loss = self.fig.add_subplot(121)
            self.ax_loss.set_facecolor('#313244')
            self.ax_loss.set_title('Loss', color='#cdd6f4',
                                   fontsize=10)
            self.ax_loss.tick_params(colors='#6c7086',
                                    labelsize=8)
            for spine in self.ax_loss.spines.values():
                spine.set_color('#45475a')

            # Accuracy chart
            self.ax_acc = self.fig.add_subplot(122)
            self.ax_acc.set_facecolor('#313244')
            self.ax_acc.set_title('Accuracy', color='#cdd6f4',
                                  fontsize=10)
            self.ax_acc.tick_params(colors='#6c7086',
                                   labelsize=8)
            for spine in self.ax_acc.spines.values():
                spine.set_color('#45475a')

            self.fig.tight_layout(pad=2.0)
            self.canvas = FigureCanvasTkAgg(
                self.fig, master=chart_frame)
            self.canvas.get_tk_widget().pack(
                fill=tk.BOTH, expand=True)

    def _update_loop(self):
        s = self.state

        # Status
        self.status_var.set(s.status)

        # Progress bar
        if s.phase == "loading":
            pct = s.files_loaded / max(s.files_total, 1) * 100
            self.epoch_progress['value'] = pct
            self.progress_label.set(
                f"Loading: {s.files_loaded}/{s.files_total} "
                f"files ({s.samples_loaded} samples, "
                f"{s.wins}W/{s.losses}L)")
        elif s.phase == "training":
            total_pct = ((s.epoch - 1 + s.epoch_progress)
                         / max(s.total_epochs, 1) * 100)
            self.epoch_progress['value'] = total_pct
            self.progress_label.set(
                f"Epoch {s.epoch}/{s.total_epochs} "
                f"({s.epoch_progress:.0%} within epoch)")
        elif s.phase == "done":
            self.epoch_progress['value'] = 100

        # Stats
        self.stat_vars['Epoch'].set(
            f"{s.epoch}/{s.total_epochs}")
        self.stat_vars['Train Loss'].set(
            f"{s.current_train_loss:.6f}")
        self.stat_vars['Train Acc'].set(
            f"{s.current_train_acc:.1%}")
        self.stat_vars['Val Loss'].set(
            f"{s.current_val_loss:.6f}")
        self.stat_vars['Val Acc'].set(
            f"{s.current_val_acc:.1%}")
        self.stat_vars['Best Val Acc'].set(
            f"{s.best_val_acc:.1%} (ep {s.best_epoch})")
        self.stat_vars['Epoch Time'].set(
            f"{s.epoch_time:.1f}s")
        self.stat_vars['ETA'].set(
            f"{s.eta:.0f}s" if s.eta > 0 else "—")
        self.stat_vars['Samples'].set(
            f"{s.n_train}+{s.n_val}")
        self.stat_vars['Device'].set(s.device or s.gpu_name)
        self.stat_vars['Parameters'].set(
            f"{s.model_params:,}" if s.model_params else "—")

        if s.gpu_mem_total_mb > 0:
            self.stat_vars['GPU Mem'].set(
                f"{s.gpu_mem_used_mb:.0f}/"
                f"{s.gpu_mem_total_mb:.0f} MB")
        else:
            self.stat_vars['GPU Mem'].set("—")

        # Update charts only when new data arrives
        if HAS_MATPLOTLIB and s.chart_dirty and s.train_losses:
            s.chart_dirty = False
            self.ax_loss.clear()
            self.ax_loss.set_facecolor('#313244')
            self.ax_loss.set_title('Loss', color='#cdd6f4',
                                   fontsize=10)
            epochs = range(1, len(s.train_losses) + 1)
            self.ax_loss.plot(epochs, s.train_losses,
                              color='#89b4fa', linewidth=1.5,
                              label='Train')
            if s.val_losses:
                self.ax_loss.plot(epochs, s.val_losses,
                                  color='#f38ba8',
                                  linewidth=1.5,
                                  label='Val')
            self.ax_loss.legend(fontsize=8,
                                facecolor='#313244',
                                edgecolor='#45475a',
                                labelcolor='#cdd6f4')
            self.ax_loss.tick_params(colors='#6c7086',
                                    labelsize=8)
            for spine in self.ax_loss.spines.values():
                spine.set_color('#45475a')

            self.ax_acc.clear()
            self.ax_acc.set_facecolor('#313244')
            self.ax_acc.set_title('Accuracy', color='#cdd6f4',
                                  fontsize=10)
            self.ax_acc.plot(epochs, s.train_accs,
                             color='#89b4fa', linewidth=1.5,
                             label='Train')
            if s.val_accs:
                self.ax_acc.plot(epochs, s.val_accs,
                                 color='#f38ba8',
                                 linewidth=1.5, label='Val')
            self.ax_acc.set_ylim(0.4, 1.05)
            self.ax_acc.axhline(y=0.5, color='#585b70',
                                linestyle='--', linewidth=0.8)
            self.ax_acc.legend(fontsize=8,
                                facecolor='#313244',
                                edgecolor='#45475a',
                                labelcolor='#cdd6f4')
            self.ax_acc.tick_params(colors='#6c7086',
                                   labelsize=8)
            for spine in self.ax_acc.spines.values():
                spine.set_color('#45475a')

            self.fig.tight_layout(pad=2.0)
            self.canvas.draw()

        # Schedule next update
        self.root.after(500, self._update_loop)


# ── Main ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='MTG RL Training Dashboard')
    parser.add_argument('--data-dir',
        default='../../rl_data/trajectories')
    parser.add_argument('--save-dir',
        default='../../rl_data/checkpoints')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int,
        default=None)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', default=None)
    parser.add_argument('--max-files', type=int,
        default=None)
    args = parser.parse_args()

    # Shared state
    state = TrainingState()

    # Start trainer in background thread
    t = threading.Thread(target=trainer_thread,
                         args=(state, args), daemon=True)
    t.start()

    # Launch Tkinter UI (must be on main thread)
    root = tk.Tk()
    app = TrainingDashboard(root, state)
    root.mainloop()


if __name__ == '__main__':
    main()
