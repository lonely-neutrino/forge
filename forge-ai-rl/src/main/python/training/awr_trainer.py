#!/usr/bin/env python3
"""
Advantage-Weighted Regression (AWR) — Offline RL alternative to PPO.

Key difference from PPO:
- Collects games under ARGMAX (model plays at full strength, ~54% win rate)
- No stochastic sampling, no importance ratios, no clipping
- Updates policy by weighting supervised loss by advantage:
  loss = -exp(advantage / temperature) * log(π(a|s))
- More sample-efficient because training data reflects actual strong play

Usage:
    python training/awr_trainer.py \
        --checkpoint /path/to/model_with_decisions.pt \
        --device cuda --rounds 50 --games-per-round 100
"""

import argparse
import os
import sys
import time
import random
import socket
import threading
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

os.environ['PYTHONUNBUFFERED'] = '1'
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

from model.mtg_model import MTGModel
from model.backend import resolve_backend
from model.gpu_config import auto_detect_profile
from training.ppo_trainer import (
    load_ppo_data, run_games, start_model_server,
    find_free_port, ModelServerError, PROJECT_ROOT,
    GAE_GAMMA, GAE_LAMBDA)
from training.mmap_dataset import (
    parse_game_state, CARD_DIM, GLOBAL_DIM, ZONES_CONFIG)

import tkinter as tk
from tkinter import ttk

try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── AWR Loss Functions ──────────────────────────────

def compute_awr_priority_batch(model, head, samples, device,
                                use_amp, temperature=1.0,
                                clip_weight=20.0):
    """AWR loss for priority: advantage-weighted cross-entropy."""
    bs = len(samples)
    max_a = max(s['n_actions'] for s in samples)
    max_a = max(max_a, 1)
    cd = CARD_DIM

    af = torch.zeros(bs, max_a, 64, device=device)
    am = torch.zeros(bs, max_a, dtype=torch.bool, device=device)
    actions = torch.zeros(bs, dtype=torch.long, device=device)
    advantages = torch.zeros(bs, device=device)

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

    for i, s in enumerate(samples):
        na = s['n_actions']
        af[i, :na] = torch.from_numpy(s['action_features'])
        am[i, :na] = True
        actions[i] = s['selected_idx']
        advantages[i] = float(s['advantage'])

        g, zones, masks_d = parse_game_state(
            s['game_state_flat'], s['global_features'])
        gf[i] = torch.from_numpy(g)
        mb[i] = torch.from_numpy(zones['my_board'])
        mbm[i] = torch.from_numpy(masks_d['my_board_mask'])
        ob[i] = torch.from_numpy(zones['opp_board'])
        obm[i] = torch.from_numpy(masks_d['opp_board_mask'])
        h[i] = torch.from_numpy(zones['hand'])
        hm[i] = torch.from_numpy(masks_d['hand_mask'])
        mg[i] = torch.from_numpy(zones['my_gy'])
        mgm[i] = torch.from_numpy(masks_d['my_gy_mask'])
        og[i] = torch.from_numpy(zones['opp_gy'])
        ogm[i] = torch.from_numpy(masks_d['opp_gy_mask'])
        st[i] = torch.from_numpy(zones['stack'])
        stm[i] = torch.from_numpy(masks_d['stack_mask'])

    with torch.amp.autocast('cuda', enabled=use_amp):
        state = model.encode_state(
            gf, mb, mbm, ob, obm, h, hm,
            mg, mgm, og, ogm, st, stm)

        logits = head(state, af, am)
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(
            1, actions.unsqueeze(1)).squeeze(1)

        # AWR weights: exp(advantage / temperature), clamped
        weights = torch.exp(advantages / temperature)
        weights = weights.clamp(max=clip_weight)
        # Only upweight positive advantages
        weights = torch.where(advantages > 0, weights,
                              torch.ones_like(weights) * 0.1)

        loss = -(weights * selected_log_probs).mean()

        # Entropy for monitoring
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()

    metrics = {
        'policy_loss': loss.item(),
        'entropy': entropy.item(),
        'mean_advantage': advantages.mean().item(),
        'mean_weight': weights.mean().item(),
        'pos_adv_frac': (advantages > 0).float().mean().item(),
    }
    return loss, metrics, bs


def compute_awr_attack_batch(model, head, samples, device,
                              use_amp, temperature=1.0,
                              clip_weight=20.0):
    """AWR loss for attack: advantage-weighted BCE."""
    bs = len(samples)
    max_c = max(s['n_creatures'] for s in samples)
    max_c = max(max_c, 1)
    cd = CARD_DIM

    cf = torch.zeros(bs, max_c, cd, device=device)
    cm = torch.zeros(bs, max_c, dtype=torch.bool, device=device)
    targets = torch.zeros(bs, max_c, device=device)
    advantages = torch.zeros(bs, device=device)

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

    for i, s in enumerate(samples):
        nc = s['n_creatures']
        cf[i, :nc] = torch.from_numpy(s['creature_features'])
        cm[i, :nc] = True
        targets[i, :nc] = torch.from_numpy(s['action_mask'])
        advantages[i] = float(s['advantage'])

        g, zones, masks_d = parse_game_state(
            s['game_state_flat'], s['global_features'])
        gf[i] = torch.from_numpy(g)
        mb[i] = torch.from_numpy(zones['my_board'])
        mbm[i] = torch.from_numpy(masks_d['my_board_mask'])
        ob[i] = torch.from_numpy(zones['opp_board'])
        obm[i] = torch.from_numpy(masks_d['opp_board_mask'])
        h[i] = torch.from_numpy(zones['hand'])
        hm[i] = torch.from_numpy(masks_d['hand_mask'])
        mg[i] = torch.from_numpy(zones['my_gy'])
        mgm[i] = torch.from_numpy(masks_d['my_gy_mask'])
        og[i] = torch.from_numpy(zones['opp_gy'])
        ogm[i] = torch.from_numpy(masks_d['opp_gy_mask'])
        st[i] = torch.from_numpy(zones['stack'])
        stm[i] = torch.from_numpy(masks_d['stack_mask'])

    with torch.amp.autocast('cuda', enabled=use_amp):
        state = model.encode_state(
            gf, mb, mbm, ob, obm, h, hm,
            mg, mgm, og, ogm, st, stm)

        logits = head(state, cf, cm)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none')
        bce = (bce * cm.float()).sum(dim=1) / cm.float().sum(dim=1).clamp(min=1)

        weights = torch.exp(advantages / temperature).clamp(max=clip_weight)
        weights = torch.where(advantages > 0, weights,
                              torch.ones_like(weights) * 0.1)

        loss = (weights * bce).mean()

        probs = torch.sigmoid(logits)
        entropy = -(probs * torch.log(probs.clamp(min=1e-8))
                     + (1-probs) * torch.log((1-probs).clamp(min=1e-8)))
        entropy = (entropy * cm.float()).sum() / cm.float().sum().clamp(min=1)

    metrics = {
        'policy_loss': loss.item(),
        'entropy': entropy.item(),
        'mean_advantage': advantages.mean().item(),
        'mean_weight': weights.mean().item(),
        'pos_adv_frac': (advantages > 0).float().mean().item(),
    }
    return loss, metrics, bs


# ── AWR Training Loop ───────────────────────────────

@dataclass
class AWRState:
    status: str = "Starting..."
    phase: str = "init"
    round: int = 0
    total_rounds: int = 0
    games_this_round: int = 0
    games_total_this_round: int = 0
    current_win_rate: float = 0.0
    best_win_rate: float = 0.0
    best_round: int = 0
    current_policy_loss: float = 0.0
    current_entropy: float = 0.0
    device: str = ""
    gpu_name: str = ""

    win_rates: List[float] = field(default_factory=list)
    policy_losses: List[float] = field(default_factory=list)
    entropies: List[float] = field(default_factory=list)

    log_lines: List[str] = field(default_factory=list)
    log_dirty: bool = False


def log(state, msg):
    print(msg, flush=True)
    state.log_lines.append(msg)
    if len(state.log_lines) > 300:
        state.log_lines = state.log_lines[-300:]
    state.log_dirty = True


def awr_thread(state, args):
    try:
        profile = auto_detect_profile(args.device)
        backend = resolve_backend(args.device)
        device = backend.torch_device
        use_amp = profile.use_amp and backend.use_amp
        port = args.port or find_free_port()

        state.device = backend.name
        state.gpu_name = f'{backend.name} ({profile.name})'
        state.total_rounds = args.rounds

        log(state, f"=== AWR Offline RL Training ===")
        log(state, f"Device: {backend.name} ({profile.name})")
        log(state, f"Port: {port}")
        log(state, f"Temperature: {args.temperature}")
        log(state, f"Using ARGMAX for data collection")

        # Load model
        log(state, f"Loading: {args.checkpoint}")
        model = MTGModel.load(args.checkpoint, device=backend.name)
        log(state, "Model loaded.")

        # Unfreeze encoder with very low LR
        encoder_params = list(model.state_encoder.parameters())

        head_params = (
            list(model.priority_head.parameters()) +
            list(model.attack_head.parameters()) +
            list(model.block_head.parameters()) +
            list(model.target_head.parameters()))
        value_params = list(model.value_network.parameters())

        optimizer = optim.AdamW([
            {'params': encoder_params, 'lr': args.lr * 0.1},  # encoder: slow
            {'params': head_params, 'lr': args.lr},
            {'params': value_params, 'lr': args.lr * 3},
        ], weight_decay=1e-5)
        log(state, f"Encoder UNFROZEN at LR={args.lr * 0.1:.1e}")
        scaler = (torch.amp.GradScaler('cuda')
                  if use_amp else None)

        # Start model server in ARGMAX mode
        server = start_model_server(
            model, device, port, use_argmax=True)
        log(state, "Server ready (ARGMAX mode).")

        traj_dir = os.path.join(
            PROJECT_ROOT, 'rl_data/awr_trajectories')
        save_dir = args.save_dir
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(traj_dir, exist_ok=True)

        for rnd in range(1, args.rounds + 1):
            state.round = rnd
            t0 = time.time()

            # ── Collect games under argmax ──
            state.phase = "collecting"
            state.status = f"Round {rnd}: collecting (argmax)..."
            log(state, f"\n--- Round {rnd}/{args.rounds} ---")
            log(state, "  Collecting (argmax)...")

            def on_progress(done, total):
                state.games_this_round = done
                state.games_total_this_round = total

            try:
                _, stdout = run_games(
                    args.games_per_round, traj_dir,
                    mode='evaluate', port=port,
                    progress_callback=on_progress)
            except ModelServerError as e:
                log(state, f"  FATAL: {e}")
                state.phase = "done"
                break

            # Load data with GAE advantages
            attack_data, block_data, priority_data, \
                target_data, mulligan_data, \
                value_data = load_ppo_data(traj_dir)
            log(state,
                f"  Data: {len(priority_data)} priority, "
                f"{len(attack_data)} attack, "
                f"{len(target_data)} target")

            if not priority_data:
                log(state, "  No data — skipping")
                continue

            # ── AWR update ──
            state.phase = "training"
            state.status = f"Round {rnd}: AWR update..."
            model.train()

            total_pl, total_ent, n_updates = 0, 0, 0

            for epoch in range(args.awr_epochs):
                # Priority
                random.shuffle(priority_data)
                for bi in range(0, len(priority_data),
                                args.batch_size):
                    batch = priority_data[
                        bi:bi + args.batch_size]
                    if len(batch) < 2:
                        continue

                    loss, metrics, _ = \
                        compute_awr_priority_batch(
                            model, model.priority_head,
                            batch, device, use_amp,
                            temperature=args.temperature)

                    if torch.isnan(loss):
                        continue

                    optimizer.zero_grad()
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

                    total_pl += metrics['policy_loss']
                    total_ent += metrics['entropy']
                    n_updates += 1

                # Attack
                if attack_data:
                    random.shuffle(attack_data)
                    for bi in range(0, len(attack_data),
                                    args.batch_size):
                        batch = attack_data[
                            bi:bi + args.batch_size]
                        if len(batch) < 2:
                            continue

                        loss, metrics, _ = \
                            compute_awr_attack_batch(
                                model, model.attack_head,
                                batch, device, use_amp,
                                temperature=args.temperature)

                        if torch.isnan(loss):
                            continue

                        optimizer.zero_grad()
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

                        total_pl += metrics['policy_loss']
                        total_ent += metrics['entropy']
                        n_updates += 1

            # Log metrics
            avg_pl = total_pl / max(n_updates, 1)
            avg_ent = total_ent / max(n_updates, 1)
            state.current_policy_loss = avg_pl
            state.current_entropy = avg_ent
            state.policy_losses.append(avg_pl)
            state.entropies.append(avg_ent)

            # ── Eval (also argmax — this IS deployment performance) ──
            state.phase = "evaluating"
            state.status = f"Round {rnd}: evaluating..."
            model.eval()

            log(state, f"  AWR update: {n_updates} steps, "
                f"loss={avg_pl:.4f}, entropy={avg_ent:.3f}")

            try:
                eval_dir = traj_dir + '_eval'
                os.makedirs(eval_dir, exist_ok=True)
                _, stdout = run_games(
                    args.eval_games, eval_dir,
                    mode='evaluate', port=port)

                # Parse win rate from output
                win_rate = 0.0
                for line in stdout.split('\n'):
                    if 'RL Wins:' in line:
                        try:
                            pct = line.split('(')[1].split('%')[0]
                            win_rate = float(pct) / 100
                        except:
                            pass

                state.current_win_rate = win_rate
                state.win_rates.append(win_rate)
                log(state, f"  Eval: {win_rate*100:.0f}% "
                    f"(argmax, {args.eval_games} games)")

                if win_rate > state.best_win_rate:
                    state.best_win_rate = win_rate
                    state.best_round = rnd
                    model.save(os.path.join(
                        save_dir, 'best_awr_model.pt'))
                    log(state, f"  NEW BEST: {win_rate*100:.0f}%")

            except Exception as e:
                log(state, f"  Eval error: {e}")

            # Save latest
            model.save(os.path.join(
                save_dir, 'awr_model_latest.pt'))

            elapsed = time.time() - t0
            log(state, f"  Round time: {elapsed:.0f}s")

            # Save training state
            with open(os.path.join(save_dir,
                                    'awr_training_state.json'), 'w') as f:
                json.dump({
                    'completed_rounds': rnd,
                    'best_win_rate': state.best_win_rate,
                    'best_round': state.best_round,
                    'win_rates': state.win_rates,
                    'policy_losses': state.policy_losses,
                    'entropies': state.entropies,
                }, f)

        state.phase = "done"
        state.status = (f"AWR complete. Best: "
                        f"{state.best_win_rate*100:.0f}% "
                        f"(round {state.best_round})")
        log(state, f"\n=== AWR Complete ===")
        log(state, f"Best: {state.best_win_rate*100:.0f}% "
            f"at round {state.best_round}")

    except Exception as e:
        log(state, f"ERROR: {e}")
        state.status = f"ERROR: {e}"
        state.phase = "done"
        import traceback
        traceback.print_exc()


# ── Simple Dashboard ─────────────────────────────────

class AWRDashboard:
    def __init__(self, root, state):
        self.root = root
        self.state = state
        root.title("MTG RL — AWR Offline Training")
        root.geometry("900x700")
        root.configure(bg='#1e1e2e')

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('H.TLabel', font=('Helvetica', 16, 'bold'),
                         background='#1e1e2e', foreground='#cdd6f4')
        style.configure('S.TLabel', font=('Consolas', 11),
                         background='#1e1e2e', foreground='#a6adc8')
        style.configure('V.TLabel', font=('Consolas', 11, 'bold'),
                         background='#1e1e2e', foreground='#a6e3a1')
        style.configure('D.TFrame', background='#1e1e2e')

        m = ttk.Frame(root, style='D.TFrame')
        m.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        ttk.Label(m, text="MTG RL — AWR Offline Training",
                  style='H.TLabel').pack(pady=(0, 6))

        self.status_v = tk.StringVar(value="Starting...")
        ttk.Label(m, textvariable=self.status_v,
                  style='S.TLabel').pack()

        sf = ttk.Frame(m, style='D.TFrame')
        sf.pack(fill=tk.X, pady=4)
        self.svars = {}
        stats = [
            ('Round', '—'), ('Win Rate', '—'),
            ('Best', '—'), ('Policy Loss', '—'),
            ('Entropy', '—'), ('Phase', '—'),
        ]
        for i, (k, v) in enumerate(stats):
            r, c = divmod(i, 3)
            ttk.Label(sf, text=f"{k}:", style='S.TLabel').grid(
                row=r, column=c*2, sticky='w', padx=(8, 2))
            sv = tk.StringVar(value=v)
            ttk.Label(sf, textvariable=sv, style='V.TLabel').grid(
                row=r, column=c*2+1, sticky='w', padx=(0, 12))
            self.svars[k] = sv

        lf = ttk.Frame(m, style='D.TFrame')
        lf.pack(fill=tk.BOTH, expand=True, pady=4)
        self.log_text = tk.Text(lf, height=20,
                                 bg='#181825', fg='#a6adc8',
                                 font=('Consolas', 9),
                                 wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        self._tick()

    def _tick(self):
        s = self.state
        self.status_v.set(s.status)
        self.svars['Round'].set(
            f"{s.round}/{s.total_rounds}")
        self.svars['Win Rate'].set(
            f"{s.current_win_rate*100:.0f}%"
            if s.current_win_rate > 0 else "—")
        self.svars['Best'].set(
            f"{s.best_win_rate*100:.0f}% (R{s.best_round})"
            if s.best_win_rate > 0 else "—")
        self.svars['Policy Loss'].set(
            f"{s.current_policy_loss:.4f}")
        self.svars['Entropy'].set(
            f"{s.current_entropy:.3f}")
        self.svars['Phase'].set(s.phase)

        if s.log_dirty:
            s.log_dirty = False
            self.log_text.config(state=tk.NORMAL)
            self.log_text.delete('1.0', tk.END)
            self.log_text.insert('1.0',
                                  '\n'.join(s.log_lines[-40:]))
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)

        self.root.after(500, self._tick)


def main():
    parser = argparse.ArgumentParser(
        description='AWR Offline RL Training')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--save-dir',
                        default=os.path.join(PROJECT_ROOT,
                                              'rl_data/checkpoints'))
    parser.add_argument('--device', default=None)
    parser.add_argument('--rounds', type=int, default=50)
    parser.add_argument('--games-per-round', type=int,
                        default=100)
    parser.add_argument('--eval-games', type=int, default=50)
    parser.add_argument('--awr-epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float,
                        default=2.0,
                        help='AWR temperature — higher = '
                             'more uniform weighting')
    parser.add_argument('--port', type=int, default=0)
    args = parser.parse_args()

    state = AWRState()
    t = threading.Thread(target=awr_thread,
                         args=(state, args), daemon=True)
    t.start()

    root = tk.Tk()
    AWRDashboard(root, state)
    root.mainloop()


if __name__ == '__main__':
    main()
