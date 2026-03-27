#!/usr/bin/env python3
"""
PPO Training Dashboard — visual monitoring of self-play training.

Shows:
- Win rate vs heuristic over rounds (the key metric)
- Policy loss, value loss, entropy curves
- Per-round stats (games played, decisions, timing)
- Console log panel

Launch:
    python training/ppo_ui.py \
        --checkpoint /path/to/model_with_decisions.pt \
        --device cuda --rounds 20 --games-per-round 200
"""

import argparse
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import List

import tkinter as tk
from tkinter import ttk

os.environ['PYTHONUNBUFFERED'] = '1'
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg)
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


@dataclass
class PPOState:
    status: str = "Idle"
    phase: str = ""  # collecting, training, evaluating, done
    chart_dirty: bool = False

    round: int = 0
    total_rounds: int = 20
    games_this_round: int = 0
    games_total_this_round: int = 0
    attacks_this_round: int = 0
    blocks_this_round: int = 0

    # History
    win_rates: List[float] = field(default_factory=list)
    policy_losses: List[float] = field(default_factory=list)
    value_losses: List[float] = field(default_factory=list)
    entropies: List[float] = field(default_factory=list)

    current_win_rate: float = 0.0
    best_win_rate: float = 0.0
    best_round: int = 0
    current_policy_loss: float = 0.0
    current_value_loss: float = 0.0
    current_entropy: float = 0.0

    elapsed: float = 0.0
    round_time: float = 0.0
    eta: float = 0.0

    device: str = ""
    gpu_name: str = ""

    # Self-play mode
    training_mode: str = "vs_heuristic"  # vs_heuristic, selfplay, mixed
    elo_ratings: List[float] = field(default_factory=list)
    current_elo: float = 1000.0
    eval_interval: int = 5

    # Console log
    log_lines: List[str] = field(default_factory=list)
    log_dirty: bool = False


def update_elo(rating, opponent_rating, score, k=32):
    """Update Elo rating. score: 1=win, 0=loss, 0.5=draw."""
    expected = 1.0 / (1.0 + 10 ** ((opponent_rating - rating) / 400))
    return rating + k * (score - expected)


def log(state, msg):
    print(msg, flush=True)
    state.log_lines.append(msg)
    if len(state.log_lines) > 300:
        state.log_lines = state.log_lines[-300:]
    state.log_dirty = True


# ── PPO training thread (delegates to ppo_trainer) ───

def ppo_thread(state, args):
    try:
        from training.ppo_trainer import (
            load_ppo_data, compute_ppo_batch,
            compute_ppo_block_batch,
            compute_ppo_priority_batch,
            compute_ppo_target_batch,
            compute_ppo_mulligan_batch,
            run_games, start_model_server,
            find_free_port,
            ModelServerError, PROJECT_ROOT)
        from training.deck_config import load_rl_decks
        from training.mmap_dataset import (
            parse_game_state, CARD_DIM, GLOBAL_DIM,
            ZONES_CONFIG)
        from model.mtg_model import MTGModel
        from model.backend import resolve_backend
        from model.gpu_config import auto_detect_profile
        import torch
        import torch.optim as optim
        import torch.nn.functional as F
        import random

        profile = auto_detect_profile(args.device)
        backend = resolve_backend(args.device)
        device = backend.torch_device
        use_amp = profile.use_amp and backend.use_amp
        port = args.port or find_free_port()

        state.device = backend.name
        state.gpu_name = f'{backend.name} ({profile.name})'
        state.total_rounds = args.rounds
        decks = load_rl_decks(overrides=args.deck)

        log(state, f"Device: {backend.name} ({profile.name})")
        log(state, f"Port: {port}")
        log(state, f"Rounds: {args.rounds}, "
            f"Games/round: {args.games_per_round}")
        log(state, f"Decks: {', '.join(decks)}")

        # Load model
        log(state, f"Loading: {args.checkpoint}")
        if os.path.exists(args.checkpoint):
            model = MTGModel.load(
                args.checkpoint, device=backend.name)
        else:
            model = MTGModel().to(device)
        log(state, "Model loaded.")

        # Freeze encoder — protect imitation-trained representations
        for p in model.state_encoder.parameters():
            p.requires_grad = False

        # Separate param groups with different learning rates
        head_params = (
            list(model.priority_head.parameters()) +
            list(model.attack_head.parameters()) +
            list(model.block_head.parameters()) +
            list(model.target_head.parameters()) +
            list(model.card_select_head.parameters()) +
            list(model.mulligan_head.parameters()) +
            list(model.binary_head.parameters()))
        value_params = list(model.value_network.parameters())

        optimizer = optim.AdamW([
            {'params': head_params, 'lr': args.lr * 10},     # heads: 1e-4
            {'params': value_params, 'lr': args.lr * 30},     # value: 3e-4
        ], weight_decay=1e-5)
        scaler = (torch.amp.GradScaler('cuda')
                  if use_amp else None)

        # Resume training state if available
        import json as json_mod
        save_dir = args.save_dir
        state_path = os.path.join(
            save_dir, 'ppo_training_state.json')
        start_round = 0
        if os.path.exists(state_path):
            try:
                with open(state_path) as f:
                    saved = json_mod.load(f)
                start_round = saved.get(
                    'completed_rounds', 0)
                state.best_win_rate = saved.get(
                    'best_win_rate', 0.0)
                state.best_round = saved.get(
                    'best_round', 0)
                state.win_rates = saved.get(
                    'win_rates', [])
                state.policy_losses = saved.get(
                    'policy_losses', [])
                state.value_losses = saved.get(
                    'value_losses', [])
                state.entropies = saved.get(
                    'entropies', [])
                state.elo_ratings = saved.get(
                    'elo_ratings', [])
                state.current_elo = saved.get(
                    'current_elo', 1000.0)
                log(state,
                    f"Resumed from round {start_round}, "
                    f"best WR: "
                    f"{state.best_win_rate:.1%}")
            except Exception:
                pass

        # Start server(s)
        n_servers = getattr(args, 'servers', 1)
        servers = []
        ports = []
        for si in range(n_servers):
            p = find_free_port() if port == 0 else port + si
            log(state, f"Starting model server {si+1}/{n_servers} on :{p}")
            srv = start_model_server(model, device, p)
            servers.append(srv)
            ports.append(p)
        log(state, f"{n_servers} server(s) ready.")
        # For run_games: pass list if multi-server, single int if one
        port_arg = ports if len(ports) > 1 else ports[0]

        traj_dir = os.path.join(
            PROJECT_ROOT, 'rl_data/ppo_trajectories')
        eval_dir = traj_dir + '_eval'
        os.makedirs(save_dir, exist_ok=True)

        start_time = time.time()

        total_rounds = start_round + args.rounds
        for rnd in range(start_round + 1,
                         total_rounds + 1):
            state.round = rnd
            state.total_rounds = total_rounds
            t0 = time.time()

            # Collect
            collect_mode = getattr(args, 'collect_mode',
                                   'evaluate')
            state.training_mode = collect_mode
            mode_label = ('self-play' if collect_mode
                         == 'selfplay' else 'vs heuristic')
            state.phase = "collecting"
            state.status = (
                f"Round {rnd}: collecting "
                f"{args.games_per_round} games "
                f"({mode_label})...")
            log(state, f"\n--- Round {rnd}/{total_rounds} "
                f"({mode_label}) ---")
            log(state, "  Collecting games...")

            def on_progress(done, total):
                state.games_this_round = done
                state.games_total_this_round = total
                state.status = (
                    f"Round {rnd}: game {done}/{total}")

            state.games_this_round = 0
            state.games_total_this_round = \
                args.games_per_round

            def on_java_log(line):
                log(state, f"  [java] {line}")

            try:
                _, stdout = run_games(
                    args.games_per_round, traj_dir,
                    mode=collect_mode, port=port_arg,
                    progress_callback=on_progress,
                    threads=args.threads,
                    java_procs=args.java_procs,
                    log_callback=on_java_log,
                    decks=decks)
            except ModelServerError as e:
                log(state, f"  FATAL: {e}")
                log(state, "  Stopping PPO — model server "
                    "is down.")
                state.status = "ABORTED: model server down"
                state.phase = "done"
                break

            attack_data, block_data, priority_data, \
                target_data, mulligan_data, \
                value_data = load_ppo_data(traj_dir)
            state.attacks_this_round = len(attack_data)
            state.blocks_this_round = len(block_data)
            log(state,
                f"  Data: {len(attack_data)} attacks, "
                f"{len(block_data)} blocks, "
                f"{len(priority_data)} priority, "
                f"{len(target_data)} target, "
                f"{len(mulligan_data)} mulligan, "
                f"{len(value_data)} value samples")

            if not value_data:
                log(state, "  No data — skipping")
                continue

            # PPO update
            state.phase = "training"
            state.status = (
                f"Round {rnd}: PPO update...")
            model.train()

            total_pl, total_vl, total_ent = 0, 0, 0
            n_updates = 0

            for ppo_ep in range(args.ppo_epochs):
                # Train on attack data
                for data, head in [(attack_data, model.attack_head)]:
                    if not data:
                        continue
                    random.shuffle(data)
                    for bi in range(0, len(data),
                                    args.batch_size):
                        batch = data[
                            bi:bi + args.batch_size]
                        if len(batch) < 2:
                            continue

                        loss, metrics, _ = \
                            compute_ppo_batch(
                                model, head,
                                batch, device, use_amp)

                        if torch.isnan(loss):
                            continue

                        optimizer.zero_grad()
                        if scaler:
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), 0.5)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), 0.5)
                            optimizer.step()

                        total_pl += metrics['policy_loss']
                        total_vl += metrics['value_loss']
                        total_ent += metrics['entropy']
                        n_updates += 1

                # Block head (separate blocker/attacker tensors)
                if block_data:
                    random.shuffle(block_data)
                    for bi in range(0, len(block_data),
                                    args.batch_size):
                        batch = block_data[
                            bi:bi + args.batch_size]
                        if len(batch) < 2:
                            continue

                        loss, metrics, _ = \
                            compute_ppo_block_batch(
                                model,
                                batch, device, use_amp)

                        if torch.isnan(loss):
                            continue

                        optimizer.zero_grad()
                        if scaler:
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), 0.5)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), 0.5)
                            optimizer.step()

                        total_pl += metrics['policy_loss']
                        total_vl += metrics['value_loss']
                        total_ent += metrics['entropy']
                        n_updates += 1

                # Priority head (Categorical, not Bernoulli)
                if priority_data:
                    random.shuffle(priority_data)
                    for bi in range(0, len(priority_data),
                                    args.batch_size):
                        batch = priority_data[
                            bi:bi + args.batch_size]
                        if len(batch) < 2:
                            continue

                        loss, metrics, _ = \
                            compute_ppo_priority_batch(
                                model,
                                model.priority_head,
                                batch, device, use_amp)

                        if torch.isnan(loss):
                            continue

                        optimizer.zero_grad()
                        if scaler:
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), 0.5)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), 0.5)
                            optimizer.step()

                        total_pl += metrics['policy_loss']
                        total_vl += metrics['value_loss']
                        total_ent += metrics['entropy']
                        n_updates += 1

                # Target head updates
                if target_data:
                    random.shuffle(target_data)
                    for bi in range(0, len(target_data),
                                    args.batch_size):
                        batch = target_data[
                            bi:bi + args.batch_size]
                        if len(batch) < 2:
                            continue

                        loss, metrics, _ = \
                            compute_ppo_target_batch(
                                model,
                                batch, device, use_amp)

                        if torch.isnan(loss):
                            continue

                        optimizer.zero_grad()
                        if scaler:
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), 0.5)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), 0.5)
                            optimizer.step()

                        total_pl += metrics['policy_loss']
                        total_vl += metrics['value_loss']
                        total_ent += metrics['entropy']
                        n_updates += 1

                # Mulligan head: DISABLED for PPO.
                # Too few samples per round (~100) causes wild
                # policy swings. The imitation-learned policy
                # (98.5% accuracy) is already near-optimal.

                # Train value network on outcomes
                # with FULL game state (not zeros!)
                random.shuffle(value_data)
                for bi in range(0, len(value_data),
                                args.batch_size):
                    batch = value_data[
                        bi:bi + args.batch_size]
                    if len(batch) < 2:
                        continue
                    try:
                        bs = len(batch)
                        cd = CARD_DIM
                        gf = torch.zeros(bs, GLOBAL_DIM,
                            device=device)
                        tgt = torch.zeros(bs,
                            device=device)
                        mb = torch.zeros(bs, 40, cd,
                            device=device)
                        mbm = torch.zeros(bs, 40,
                            dtype=torch.bool,
                            device=device)
                        ob = torch.zeros(bs, 40, cd,
                            device=device)
                        obm = torch.zeros(bs, 40,
                            dtype=torch.bool,
                            device=device)
                        h = torch.zeros(bs, 15, cd,
                            device=device)
                        hm = torch.zeros(bs, 15,
                            dtype=torch.bool,
                            device=device)
                        mg = torch.zeros(bs, 20, cd,
                            device=device)
                        mgm = torch.zeros(bs, 20,
                            dtype=torch.bool,
                            device=device)
                        og = torch.zeros(bs, 20, cd,
                            device=device)
                        ogm = torch.zeros(bs, 20,
                            dtype=torch.bool,
                            device=device)
                        st = torch.zeros(bs, 10, cd,
                            device=device)
                        stm = torch.zeros(bs, 10,
                            dtype=torch.bool,
                            device=device)

                        for i, s in enumerate(batch):
                            g_np, zones, masks = \
                                parse_game_state(
                                    s['game_state_flat'],
                                    s['global_features'])
                            gf[i] = torch.from_numpy(g_np)
                            tgt[i] = s['outcome']
                            mb[i] = torch.from_numpy(
                                zones['my_board'])
                            mbm[i] = torch.from_numpy(
                                masks['my_board_mask'])
                            ob[i] = torch.from_numpy(
                                zones['opp_board'])
                            obm[i] = torch.from_numpy(
                                masks['opp_board_mask'])
                            h[i] = torch.from_numpy(
                                zones['hand'])
                            hm[i] = torch.from_numpy(
                                masks['hand_mask'])
                            mg[i] = torch.from_numpy(
                                zones['my_gy'])
                            mgm[i] = torch.from_numpy(
                                masks['my_gy_mask'])
                            og[i] = torch.from_numpy(
                                zones['opp_gy'])
                            ogm[i] = torch.from_numpy(
                                masks['opp_gy_mask'])
                            st[i] = torch.from_numpy(
                                zones['stack'])
                            stm[i] = torch.from_numpy(
                                masks['stack_mask'])

                        with torch.amp.autocast('cuda',
                                enabled=use_amp):
                            emb = model.encode_state(
                                gf, mb, mbm, ob, obm,
                                h, hm, mg, mgm,
                                og, ogm, st, stm)
                            val = model.get_value(
                                emb).squeeze(-1)
                            vl = F.mse_loss(val, tgt)

                        optimizer.zero_grad()
                        if scaler:
                            scaler.scale(vl).backward()
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), 0.5)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            vl.backward()
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), 0.5)
                            optimizer.step()
                        total_vl += vl.item()
                        n_updates += 1
                    except Exception:
                        pass

            avg_pl = total_pl / max(n_updates, 1)
            avg_vl = total_vl / max(n_updates, 1)
            avg_ent = total_ent / max(n_updates, 1)

            # Free trajectory data before launching eval/next round
            del attack_data, block_data, priority_data
            del target_data, mulligan_data, value_data
            import gc; gc.collect()
            if backend.is_cuda:
                torch.cuda.empty_cache()

            # Evaluate vs heuristic
            eval_interval = getattr(args, 'eval_interval', 1)
            run_eval = (collect_mode == 'evaluate'
                        or rnd % eval_interval == 0
                        or rnd == total_rounds)

            eval_wr = state.current_win_rate  # keep prev
            if run_eval:
                state.phase = "evaluating"
                state.status = (
                    f"Round {rnd}: evaluating vs "
                    f"heuristic...")
                model.eval()

                try:
                    eval_wr, _ = run_games(
                        args.eval_games, eval_dir,
                        mode='evaluate', port=port_arg,
                        log_callback=on_java_log,
                        threads=args.threads,
                        java_procs=args.java_procs,
                        decks=decks)
                    eval_wr = eval_wr or 0.0
                except ModelServerError as e:
                    log(state, f"  FATAL: {e}")
                    log(state, "  Stopping PPO — model "
                        "server is down during eval.")
                    state.status = ("ABORTED: model "
                        "server down")
                    state.phase = "done"
                    break

                # Update Elo from eval result
                heuristic_elo = 1200.0
                n_eval = args.eval_games
                wins = int(round(eval_wr * n_eval))
                elo = state.current_elo
                for _ in range(wins):
                    elo = update_elo(elo, heuristic_elo,
                                     1.0, k=16)
                for _ in range(n_eval - wins):
                    elo = update_elo(elo, heuristic_elo,
                                     0.0, k=16)
                state.current_elo = elo
                state.elo_ratings.append(elo)
            else:
                next_eval = (rnd + eval_interval
                    - rnd % eval_interval)
                log(state, f"  (eval skipped — next at "
                    f"round {next_eval})")

            # Update state
            state.current_win_rate = eval_wr
            state.current_policy_loss = avg_pl
            state.current_value_loss = avg_vl
            state.current_entropy = avg_ent
            if run_eval:
                state.win_rates.append(eval_wr)
            state.policy_losses.append(avg_pl)
            state.value_losses.append(avg_vl)
            state.entropies.append(avg_ent)

            if eval_wr > state.best_win_rate:
                state.best_win_rate = eval_wr
                state.best_round = rnd
                model.save(os.path.join(
                    save_dir, 'best_ppo_model.pt'))

            # Always save latest (for resume)
            model.save(os.path.join(
                save_dir, 'ppo_model_latest.pt'))

            # Save training state for resume
            training_state = {
                'completed_rounds': rnd,
                'best_win_rate': state.best_win_rate,
                'best_round': state.best_round,
                'win_rates': state.win_rates,
                'policy_losses': state.policy_losses,
                'value_losses': state.value_losses,
                'entropies': state.entropies,
                'elo_ratings': state.elo_ratings,
                'current_elo': state.current_elo,
                'collect_mode': collect_mode,
            }
            with open(os.path.join(
                    save_dir,
                    'ppo_training_state.json'), 'w') as f:
                json_mod.dump(training_state, f, indent=2)

            if rnd % 5 == 0:
                model.save(os.path.join(
                    save_dir,
                    f'ppo_model_round_{rnd}.pt'))

            state.round_time = time.time() - t0
            state.elapsed = time.time() - start_time
            state.eta = (
                (args.rounds - rnd) * state.round_time)
            state.chart_dirty = True

            log(state,
                f"  Policy: {avg_pl:.4f} | "
                f"Value: {avg_vl:.4f} | "
                f"Entropy: {avg_ent:.3f} | "
                f"Win rate: {eval_wr:.1%}"
                f"{'  ★ BEST' if eval_wr >= state.best_win_rate else ''}")

            time.sleep(0.05)

        model.save(os.path.join(
            save_dir, 'ppo_model_final.pt'))
        state.status = "Training complete!"
        state.phase = "done"
        state.chart_dirty = True
        log(state, f"\nBest win rate: "
            f"{state.best_win_rate:.1%} "
            f"(round {state.best_round})")

    except Exception as e:
        log(state, f"ERROR: {e}")
        state.status = f"ERROR: {e}"
        state.phase = "done"
        import traceback
        traceback.print_exc()


# ── Dashboard ────────────────────────────────────────

class PPODashboard:
    def __init__(self, root, state):
        self.root = root
        self.state = state
        root.title("MTG RL — PPO Self-Play Training")
        root.geometry("950x750")
        root.configure(bg='#1e1e2e')

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('H.TLabel',
            font=('Helvetica', 16, 'bold'),
            background='#1e1e2e', foreground='#cdd6f4')
        style.configure('S.TLabel',
            font=('Consolas', 11),
            background='#1e1e2e', foreground='#a6adc8')
        style.configure('V.TLabel',
            font=('Consolas', 11, 'bold'),
            background='#1e1e2e', foreground='#89b4fa')
        style.configure('St.TLabel',
            font=('Consolas', 10),
            background='#1e1e2e', foreground='#f9e2af')
        style.configure('D.TFrame',
            background='#1e1e2e')
        style.configure("b.Horizontal.TProgressbar",
            troughcolor='#313244', background='#89b4fa')

        self._build(root)
        self._tick()

    def _build(self, root):
        m = ttk.Frame(root, style='D.TFrame')
        m.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        ttk.Label(m, text="MTG RL — PPO Self-Play",
            style='H.TLabel').pack(pady=(0, 6))
        self.status_v = tk.StringVar(value="Starting...")
        ttk.Label(m, textvariable=self.status_v,
            style='St.TLabel').pack()

        pf = ttk.Frame(m, style='D.TFrame')
        pf.pack(fill=tk.X, pady=4)
        self.prog = ttk.Progressbar(pf, length=900,
            style="b.Horizontal.TProgressbar")
        self.prog.pack(fill=tk.X)
        self.prog_v = tk.StringVar()
        ttk.Label(pf, textvariable=self.prog_v,
            style='S.TLabel').pack(anchor='w')

        sf = ttk.Frame(m, style='D.TFrame')
        sf.pack(fill=tk.X, pady=4)
        self.svars = {}
        for i, (k, v) in enumerate([
            ('Round', '—'), ('Win Rate', '—'),
            ('Best WR', '—'), ('Elo', '—'),
            ('Policy Loss', '—'), ('Entropy', '—'),
            ('Round Time', '—'), ('ETA', '—'),
        ]):
            r, c = divmod(i, 4)
            ttk.Label(sf, text=f"{k}:",
                style='S.TLabel').grid(
                row=r, column=c*2, sticky='w',
                padx=(8, 2), pady=2)
            sv = tk.StringVar(value=v)
            ttk.Label(sf, textvariable=sv,
                style='V.TLabel').grid(
                row=r, column=c*2+1, sticky='w',
                padx=(0, 12), pady=2)
            self.svars[k] = sv

        if HAS_MPL:
            cf = ttk.Frame(m, style='D.TFrame')
            cf.pack(fill=tk.BOTH, expand=True, pady=4)
            self.fig = Figure(figsize=(9, 4), dpi=100,
                facecolor='#1e1e2e')
            self.ax_wr = self.fig.add_subplot(121)
            self.ax_loss = self.fig.add_subplot(122)
            for ax in [self.ax_wr, self.ax_loss]:
                ax.set_facecolor('#313244')
                ax.tick_params(colors='#6c7086',
                    labelsize=8)
                for sp in ax.spines.values():
                    sp.set_color('#45475a')
            self.fig.tight_layout(pad=2)
            self.canvas = FigureCanvasTkAgg(
                self.fig, master=cf)
            self.canvas.get_tk_widget().pack(
                fill=tk.BOTH, expand=True)

        # Console log
        lf = ttk.Frame(m, style='D.TFrame')
        lf.pack(fill=tk.BOTH, expand=True, pady=4)
        self.log_text = tk.Text(lf, height=8,
            bg='#181825', fg='#a6adc8',
            font=('Consolas', 9),
            insertbackground='#cdd6f4',
            selectbackground='#45475a',
            wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _tick(self):
        s = self.state
        self.status_v.set(s.status)

        pct = s.round / max(s.total_rounds, 1) * 100
        self.prog['value'] = pct
        if s.phase == 'collecting' and s.games_total_this_round > 0:
            self.prog_v.set(
                f"Round {s.round}/{s.total_rounds} "
                f"— game {s.games_this_round}/"
                f"{s.games_total_this_round}")
        elif s.phase == 'evaluating':
            self.prog_v.set(
                f"Round {s.round}/{s.total_rounds} "
                f"— evaluating...")
        else:
            self.prog_v.set(
                f"Round {s.round}/{s.total_rounds} "
                f"({s.phase})")

        self.svars['Round'].set(
            f"{s.round}/{s.total_rounds}")
        self.svars['Win Rate'].set(
            f"{s.current_win_rate:.1%}")
        self.svars['Best WR'].set(
            f"{s.best_win_rate:.1%} (r{s.best_round})")
        self.svars['Elo'].set(
            f"{s.current_elo:.0f}")
        self.svars['Policy Loss'].set(
            f"{s.current_policy_loss:.4f}")
        self.svars['Entropy'].set(
            f"{s.current_entropy:.3f}")
        self.svars['Round Time'].set(
            f"{s.round_time:.0f}s")
        self.svars['ETA'].set(
            f"{s.eta:.0f}s" if s.eta > 0 else "—")

        if HAS_MPL and s.chart_dirty and s.win_rates:
            s.chart_dirty = False

            self.ax_wr.clear()
            self.ax_wr.set_facecolor('#313244')
            self.ax_wr.set_title('Win Rate vs Heuristic',
                color='#cdd6f4', fontsize=10)
            rr = range(1, len(s.win_rates) + 1)
            self.ax_wr.plot(rr, s.win_rates,
                color='#a6e3a1', linewidth=2,
                marker='o', markersize=4)
            self.ax_wr.axhline(y=0.5, color='#f38ba8',
                linestyle='--', linewidth=1,
                label='50% baseline')
            self.ax_wr.set_ylim(0.0, 1.0)
            self.ax_wr.set_ylabel('Win Rate',
                color='#a6adc8', fontsize=9)
            self.ax_wr.set_xlabel('Round',
                color='#a6adc8', fontsize=9)
            self.ax_wr.legend(fontsize=8,
                facecolor='#313244',
                edgecolor='#45475a',
                labelcolor='#cdd6f4')
            self.ax_wr.tick_params(colors='#6c7086',
                labelsize=8)
            for sp in self.ax_wr.spines.values():
                sp.set_color('#45475a')

            self.ax_loss.clear()
            self.ax_loss.set_facecolor('#313244')
            self.ax_loss.set_title('Training Losses',
                color='#cdd6f4', fontsize=10)
            self.ax_loss.plot(rr, s.policy_losses,
                color='#89b4fa', linewidth=1.5,
                label='Policy')
            self.ax_loss.plot(rr, s.value_losses,
                color='#f38ba8', linewidth=1.5,
                label='Value')
            self.ax_loss.legend(fontsize=8,
                facecolor='#313244',
                edgecolor='#45475a',
                labelcolor='#cdd6f4')
            self.ax_loss.set_xlabel('Round',
                color='#a6adc8', fontsize=9)
            self.ax_loss.tick_params(colors='#6c7086',
                labelsize=8)
            for sp in self.ax_loss.spines.values():
                sp.set_color('#45475a')

            self.fig.tight_layout(pad=2)
            self.canvas.draw()

        if s.log_dirty:
            s.log_dirty = False
            self.log_text.config(state=tk.NORMAL)
            self.log_text.delete('1.0', tk.END)
            self.log_text.insert('1.0',
                '\n'.join(s.log_lines[-50:]))
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)

        self.root.after(500, self._tick)


def main():
    from training.ppo_trainer import PROJECT_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',
        default=os.path.join(PROJECT_ROOT,
            'rl_data/checkpoints/model_with_decisions.pt'))
    parser.add_argument('--save-dir',
        default=os.path.join(PROJECT_ROOT,
            'rl_data/checkpoints'))
    parser.add_argument('--device', default=None)
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--games-per-round', type=int,
        default=200)
    parser.add_argument('--ppo-epochs', type=int,
        default=4)
    parser.add_argument('--batch-size', type=int,
        default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--eval-games', type=int,
        default=50)
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--threads', type=int, default=16,
        help='Java game threads for data collection')
    parser.add_argument('--servers', type=int, default=1,
        help='Number of model servers for parallel inference')
    parser.add_argument('--java-procs', type=int, default=1,
        help='Number of Java processes to split games across')
    parser.add_argument('--collect-mode', default='evaluate',
        choices=['evaluate', 'selfplay'],
        help='Collection mode: evaluate (vs heuristic) '
             'or selfplay (RL vs RL)')
    parser.add_argument('--eval-interval', type=int,
        default=1,
        help='Eval vs heuristic every N rounds '
             '(selfplay mode, default: 1)')
    parser.add_argument('--deck', action='append',
        help='Override the configured RL deck list '
             '(may be repeated)')
    args = parser.parse_args()

    state = PPOState()
    t = threading.Thread(target=ppo_thread,
        args=(state, args), daemon=True)
    t.start()
    root = tk.Tk()
    PPODashboard(root, state)
    root.mainloop()


if __name__ == '__main__':
    main()
