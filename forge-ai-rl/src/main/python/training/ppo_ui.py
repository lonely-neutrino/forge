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

    # Gameplay metrics (from eval trajectories)
    attack_rate: float = 0.0         # % of candidates that attacked
    attack_all_in_pct: float = 0.0   # % of phases where all creatures attacked
    attack_hold_pct: float = 0.0     # % of phases where no creatures attacked
    spells_per_turn: float = 0.0     # avg spells cast per turn
    spells_per_game: float = 0.0     # avg spells cast per game
    avg_turns: float = 0.0           # avg game length in turns
    idle_turns_pct: float = 0.0      # % of turns with no spells when possible
    targeting_issues: int = 0        # fallback decisions in eval
    spell_type_counts: dict = field(default_factory=dict)

    # History for charts
    attack_rates: List[float] = field(default_factory=list)
    spells_per_turns: List[float] = field(default_factory=list)
    idle_turns_pcts: List[float] = field(default_factory=list)

    # Console log
    log_lines: List[str] = field(default_factory=list)
    log_dirty: bool = False


def update_elo(rating, opponent_rating, score, k=32):
    """Update Elo rating. score: 1=win, 0=loss, 0.5=draw."""
    expected = 1.0 / (1.0 + 10 ** ((opponent_rating - rating) / 400))
    return rating + k * (score - expected)


class LeagueManager:
    """Manages opponent pool and Elo-based matchmaking."""

    def __init__(self, checkpoint_dir, snapshot_interval=5,
                 max_snapshots=15, temperature=200.0):
        self.checkpoint_dir = checkpoint_dir
        self.snapshot_interval = snapshot_interval
        self.max_snapshots = max_snapshots
        self.temperature = temperature
        self.snapshots = []  # [{path, elo, round}]
        self.current_elo = 1000.0
        self.heuristic_elo = 1200.0
        self.heuristic_in_pool = False

    def save_snapshot(self, model, round_num):
        """Save current model as an opponent snapshot."""
        path = os.path.join(
            self.checkpoint_dir,
            f'league_snapshot_r{round_num}.pt')
        model.save(path)
        self.snapshots.append({
            'path': path,
            'elo': self.current_elo,
            'round': round_num,
        })
        # Prune if too many — remove lowest Elo
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots.sort(key=lambda s: s['elo'])
            removed = self.snapshots.pop(0)
            try:
                os.remove(removed['path'])
            except OSError:
                pass
        return path

    def should_add_heuristic(self, eval_win_rate):
        """Add heuristic to pool once model is strong enough."""
        if not self.heuristic_in_pool and eval_win_rate >= 0.35:
            self.heuristic_in_pool = True
            return True
        return False

    def select_opponents(self, n_games, max_opponents=3):
        """Select opponents weighted by Elo proximity.
        Returns: [(opponent_info, n_games_allocated)]
        opponent_info is either a snapshot dict or 'heuristic'.
        """
        import math

        candidates = list(self.snapshots)
        if self.heuristic_in_pool:
            candidates.append({
                'path': 'heuristic',
                'elo': self.heuristic_elo,
                'round': -1,
            })

        if not candidates:
            # No opponents yet — selfplay
            return [('selfplay', n_games)]

        # Weight by Elo proximity
        weights = []
        for c in candidates:
            diff = abs(c['elo'] - self.current_elo)
            w = math.exp(-diff / self.temperature)
            weights.append(w)

        total_w = sum(weights)
        if total_w == 0:
            weights = [1.0] * len(candidates)
            total_w = len(candidates)

        # Normalize and select top opponents
        scored = sorted(zip(candidates, weights),
                        key=lambda x: -x[1])
        selected = scored[:max_opponents]

        # Allocate games proportionally
        sel_total = sum(w for _, w in selected)
        result = []
        allocated = 0
        for i, (cand, w) in enumerate(selected):
            if i == len(selected) - 1:
                games = n_games - allocated
            else:
                games = max(10, int(n_games * w / sel_total))
            allocated += games
            result.append((cand, games))

        return result

    def update_elo_from_result(self, opponent_elo, wins,
                               losses, k=16):
        """Update current model Elo from game results."""
        for _ in range(wins):
            self.current_elo = update_elo(
                self.current_elo, opponent_elo, 1.0, k)
        for _ in range(losses):
            self.current_elo = update_elo(
                self.current_elo, opponent_elo, 0.0, k)

    def update_opponent_elo(self, opp_path, wins_against,
                            losses_against, k=16):
        """Update a snapshot's Elo based on results."""
        for snap in self.snapshots:
            if snap['path'] == opp_path:
                for _ in range(losses_against):
                    snap['elo'] = update_elo(
                        snap['elo'], self.current_elo,
                        1.0, k)
                for _ in range(wins_against):
                    snap['elo'] = update_elo(
                        snap['elo'], self.current_elo,
                        0.0, k)
                break

    def to_dict(self):
        return {
            'snapshots': self.snapshots,
            'current_elo': self.current_elo,
            'heuristic_elo': self.heuristic_elo,
            'heuristic_in_pool': self.heuristic_in_pool,
        }

    def from_dict(self, d):
        self.snapshots = d.get('snapshots', [])
        self.current_elo = d.get('current_elo', 1000.0)
        self.heuristic_elo = d.get('heuristic_elo', 1200.0)
        self.heuristic_in_pool = d.get(
            'heuristic_in_pool', False)


def log(state, msg):
    print(msg, flush=True)
    state.log_lines.append(msg)
    if len(state.log_lines) > 300:
        state.log_lines = state.log_lines[-300:]
    state.log_dirty = True


# ── Trajectory analysis (gameplay metrics) ───

TYPE_NAMES = ['Creature', 'Instant', 'Sorcery', 'Enchantment',
              'Artifact', 'Planeswalker', 'Land']


def analyze_eval_trajectories(traj_dir, state):
    """Analyze eval trajectory JSONL files and update state with gameplay metrics."""
    import json
    from collections import defaultdict
    from pathlib import Path

    files = sorted(Path(traj_dir).glob('traj_*.jsonl'))
    if not files:
        return

    attack_all = attack_partial = attack_none = 0
    attackers_total = candidates_total = 0
    total_fallback = total_decisions = 0
    game_spells = []
    game_turns = []
    type_counts = defaultdict(int)
    turns_with_options = 0   # turns where a spell could have been played
    turns_with_no_spell = 0  # turns where we had options but played nothing

    for fpath in files:
        with open(fpath) as fh:
            lines = fh.readlines()
        if not lines:
            continue

        g_spells_by_turn = defaultdict(int)
        g_options_by_turn = defaultdict(bool)
        max_turn = 0

        for line in lines[1:]:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            total_decisions += 1
            if rec.get('usedFallback', False):
                total_fallback += 1

            gf = rec.get('globalFeatures', [])
            game_turn = int(round(gf[4] * 30)) if len(gf) > 4 else 0
            max_turn = max(max_turn, game_turn)
            dt = rec.get('decisionType', '')

            if dt == 'PRIORITY_ACTION':
                cands = rec.get('candidateFeatures', [])
                sel = rec.get('selectedIndices', [])
                # Mark that this turn had spell options
                # (exclude pass-only situations: 1 candidate = pass only)
                if len(cands) > 1:
                    g_options_by_turn[game_turn] = True
                if sel and sel[0] < len(cands):
                    chosen = cands[sel[0]]
                    card_type = None
                    for ti in range(min(7, len(chosen))):
                        if chosen[ti] > 0.5:
                            card_type = TYPE_NAMES[ti]
                            break
                    if card_type and card_type != 'Land':
                        g_spells_by_turn[game_turn] += 1
                        type_counts[card_type] += 1

            elif dt == 'DECLARE_ATTACKERS':
                sel = rec.get('selectedIndices', [])
                cands = rec.get('candidateFeatures', [])
                n_sel, n_cand = len(sel), len(cands)
                attackers_total += n_sel
                candidates_total += n_cand
                if n_sel == 0:
                    attack_none += 1
                elif n_sel == n_cand:
                    attack_all += 1
                else:
                    attack_partial += 1

        game_spells.append(sum(g_spells_by_turn.values()))
        game_turns.append(max_turn)
        for t, had_options in g_options_by_turn.items():
            if had_options:
                turns_with_options += 1
                if g_spells_by_turn.get(t, 0) == 0:
                    turns_with_no_spell += 1

    n = len(files)
    if n == 0:
        return

    total_atk = attack_all + attack_partial + attack_none
    avg_spells = sum(game_spells) / n
    avg_turns_val = sum(game_turns) / n if game_turns else 0

    state.attack_rate = (attackers_total / candidates_total * 100
                         if candidates_total > 0 else 0)
    state.attack_all_in_pct = (attack_all / total_atk * 100
                                if total_atk > 0 else 0)
    state.attack_hold_pct = (attack_none / total_atk * 100
                              if total_atk > 0 else 0)
    state.spells_per_game = avg_spells
    state.avg_turns = avg_turns_val
    state.spells_per_turn = (avg_spells / avg_turns_val
                              if avg_turns_val > 0 else 0)
    state.idle_turns_pct = (turns_with_no_spell / turns_with_options * 100
                             if turns_with_options > 0 else 0)
    state.targeting_issues = total_fallback
    state.spell_type_counts = dict(type_counts)

    # Append to history for charting
    state.attack_rates.append(state.attack_rate)
    state.spells_per_turns.append(state.spells_per_turn)
    state.idle_turns_pcts.append(state.idle_turns_pct)


# ── PPO training thread (delegates to ppo_trainer) ───

def ppo_thread(state, args):
    try:
        from training.ppo_trainer import (
            load_ppo_data, compute_ppo_batch,
            compute_ppo_block_batch,
            compute_ppo_priority_batch,
            compute_ppo_target_batch,
            compute_ppo_mulligan_batch,
            run_games, run_league_games,
            start_model_server,
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
            model = MTGModel.from_size('xl').to(device)
        log(state, "Model loaded.")

        # Unfreeze encoder with very low LR for adaptation
        encoder_params = list(model.state_encoder.parameters())

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
            {'params': encoder_params, 'lr': args.lr},        # encoder: 1e-5 (very slow)
            {'params': head_params, 'lr': args.lr * 10},      # heads: 1e-4
            {'params': value_params, 'lr': args.lr * 30},     # value: 3e-4
        ], weight_decay=1e-5)
        log(state, f"Encoder UNFROZEN at LR={args.lr:.1e}")
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
                state.attack_rates = saved.get(
                    'attack_rates', [])
                state.spells_per_turns = saved.get(
                    'spells_per_turns', [])
                state.idle_turns_pcts = saved.get(
                    'idle_turns_pcts', [])
                # Restore reward shaping coefficient
                saved_coeff = saved.get(
                    'reward_shaping_coeff', None)
                if saved_coeff is not None:
                    args.reward_shaping_coeff = saved_coeff
                # Restore league state
                if 'league' in saved and use_league \
                        and league is not None:
                    league.from_dict(saved['league'])
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

        shaping_coeff = getattr(
            args, 'reward_shaping_coeff', 0.0)
        shaping_decay = getattr(
            args, 'reward_shaping_decay', 0.95)
        if shaping_coeff > 0:
            log(state, f"Reward shaping: coeff={shaping_coeff}, "
                f"decay={shaping_decay}/round")

        # League play setup
        use_league = getattr(args, 'league', False)
        league = None
        if use_league:
            league = LeagueManager(
                save_dir,
                snapshot_interval=getattr(
                    args, 'snapshot_interval', 5),
                max_snapshots=15)
            log(state, "League play ENABLED")

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
            state.phase = "collecting"

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

            # Clean trajectories before collection
            from pathlib import Path as _Path
            for _f in _Path(traj_dir).glob('traj_*.jsonl'):
                _f.unlink()

            if use_league and league is not None:
                # League collection
                opponents = league.select_opponents(
                    args.games_per_round,
                    max_opponents=getattr(
                        args, 'max_opponents', 3))
                opp_labels = []
                for opp_info, opp_games in opponents:
                    if opp_info == 'selfplay':
                        opp_labels.append(
                            f"selfplay({opp_games}g)")
                    elif opp_info['path'] == 'heuristic':
                        opp_labels.append(
                            f"heuristic(Elo {opp_info['elo']:.0f}, {opp_games}g)")
                    else:
                        opp_labels.append(
                            f"r{opp_info['round']}(Elo {opp_info['elo']:.0f}, {opp_games}g)")
                mode_label = "league: " + ", ".join(
                    opp_labels)
                state.status = (
                    f"Round {rnd}: collecting "
                    f"{args.games_per_round} games "
                    f"(league)...")
                log(state,
                    f"\n--- Round {rnd}/{total_rounds} "
                    f"(league) ---")
                log(state,
                    f"  Opponents: {', '.join(opp_labels)}")
                log(state, "  Collecting games...")

                collect_ok = True
                for opp_info, opp_games in opponents:
                    if opp_info == 'selfplay':
                        # Both players use current model
                        try:
                            _, _ = run_games(
                                opp_games, traj_dir,
                                mode='selfplay',
                                port=port_arg,
                                progress_callback=
                                    on_progress,
                                threads=args.threads,
                                java_procs=args.java_procs,
                                log_callback=on_java_log)
                        except ModelServerError as e:
                            log(state, f"  FATAL: {e}")
                            collect_ok = False
                            break
                    elif opp_info['path'] == 'heuristic':
                        # Current model vs heuristic
                        try:
                            wr, _ = run_games(
                                opp_games, traj_dir,
                                mode='evaluate',
                                port=port_arg,
                                progress_callback=
                                    on_progress,
                                threads=args.threads,
                                java_procs=args.java_procs,
                                log_callback=on_java_log)
                            wins = int((wr or 0) * opp_games)
                            league.update_elo_from_result(
                                league.heuristic_elo,
                                wins, opp_games - wins)
                        except ModelServerError as e:
                            log(state, f"  FATAL: {e}")
                            collect_ok = False
                            break
                    else:
                        # Current model vs snapshot
                        try:
                            opp_model = MTGModel.load(
                                opp_info['path'],
                                device=device)
                            opp_model.eval()
                            opp_ports = []
                            opp_servers = []
                            for _ in range(2):
                                op = find_free_port()
                                osrv = start_model_server(
                                    opp_model, device, op)
                                opp_ports.append(op)
                                opp_servers.append(osrv)

                            wr, _ = run_league_games(
                                opp_games, traj_dir,
                                current_ports=ports,
                                opponent_ports=opp_ports,
                                threads=args.threads,
                                progress_callback=
                                    on_progress,
                                log_callback=on_java_log)
                            wins = int(
                                (wr or 0) * opp_games)
                            losses = opp_games - wins
                            league.update_elo_from_result(
                                opp_info['elo'],
                                wins, losses)
                            league.update_opponent_elo(
                                opp_info['path'],
                                wins, losses)
                            log(state,
                                f"  vs r{opp_info['round']}"
                                f": {wr:.1%} "
                                f"({wins}/{opp_games})")
                        except ModelServerError as e:
                            log(state, f"  FATAL: {e}")
                            collect_ok = False
                            break
                        except Exception as e:
                            log(state,
                                f"  Opponent load error: "
                                f"{e}")
                            continue
                        finally:
                            # Clean up opponent servers
                            del opp_model
                            import gc; gc.collect()

                if not collect_ok:
                    log(state,
                        "  Stopping PPO — server down.")
                    state.status = (
                        "ABORTED: model server down")
                    state.phase = "done"
                    break

                # Save snapshot at interval
                if rnd % league.snapshot_interval == 0:
                    snap_path = league.save_snapshot(
                        model, rnd)
                    log(state,
                        f"  Saved league snapshot: "
                        f"{os.path.basename(snap_path)} "
                        f"(Elo {league.current_elo:.0f})")

                log(state,
                    f"  League Elo: "
                    f"{league.current_elo:.0f}")
            else:
                # Standard collection (no league)
                mode_label = ('self-play' if collect_mode
                             == 'selfplay'
                             else 'vs heuristic')
                state.status = (
                    f"Round {rnd}: collecting "
                    f"{args.games_per_round} games "
                    f"({mode_label})...")
                log(state,
                    f"\n--- Round {rnd}/{total_rounds} "
                    f"({mode_label}) ---")
                log(state, "  Collecting games...")

                try:
                    _, stdout = run_games(
                        args.games_per_round, traj_dir,
                        mode=collect_mode,
                        port=port_arg,
                        progress_callback=on_progress,
                        threads=args.threads,
                        java_procs=args.java_procs,
                        log_callback=on_java_log)
                except ModelServerError as e:
                    log(state, f"  FATAL: {e}")
                    log(state,
                        "  Stopping PPO — model server "
                        "is down.")
                    state.status = (
                        "ABORTED: model server down")
                    state.phase = "done"
                    break
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
                value_data = load_ppo_data(traj_dir,
                    shaping_coeff=shaping_coeff)
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

            # Decay reward shaping
            if shaping_coeff > 0:
                shaping_coeff *= shaping_decay
                log(state,
                    f"  Reward shaping coeff: "
                    f"{shaping_coeff:.4f}")

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

                # Analyze eval gameplay metrics
                try:
                    analyze_eval_trajectories(eval_dir, state)
                    types_str = ' '.join(
                        f"{k}={v}" for k, v in sorted(
                            state.spell_type_counts.items(),
                            key=lambda x: -x[1]))
                    log(state,
                        f"  Gameplay: atk_rate={state.attack_rate:.0f}% "
                        f"all-in={state.attack_all_in_pct:.0f}% "
                        f"hold={state.attack_hold_pct:.0f}% "
                        f"spells/turn={state.spells_per_turn:.2f} "
                        f"idle={state.idle_turns_pct:.0f}% "
                        f"fallbacks={state.targeting_issues}")
                    log(state, f"  Types: {types_str}")
                except Exception as e:
                    log(state, f"  (gameplay analysis failed: {e})")
            else:
                next_eval = (rnd + eval_interval
                    - rnd % eval_interval)
                log(state, f"  (eval skipped — next at "
                    f"round {next_eval})")

            # Update state
            state.current_win_rate = eval_wr
            # Check if heuristic should enter league
            if league and league.should_add_heuristic(
                    eval_wr):
                log(state,
                    f"  Heuristic added to league pool "
                    f"(eval WR {eval_wr:.1%} >= 35%)")
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
                'attack_rates': state.attack_rates,
                'spells_per_turns': state.spells_per_turns,
                'idle_turns_pcts': state.idle_turns_pcts,
                'reward_shaping_coeff': shaping_coeff,
                'league': league.to_dict()
                    if league else None,
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
        root.geometry("1050x800")
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
            ('Atk Rate', '—'), ('Spells/Turn', '—'),
            ('Idle Turns', '—'), ('Fallbacks', '—'),
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
            self.ax_wr = self.fig.add_subplot(131)
            self.ax_loss = self.fig.add_subplot(132)
            self.ax_gp = self.fig.add_subplot(133)
            for ax, title in [
                (self.ax_wr, 'Win Rate vs Heuristic'),
                (self.ax_loss, 'Training Losses'),
                (self.ax_gp, 'Gameplay Metrics'),
            ]:
                ax.set_facecolor('#313244')
                ax.set_title(title, color='#cdd6f4',
                    fontsize=10)
                ax.tick_params(colors='#6c7086',
                    labelsize=8)
                for sp in ax.spines.values():
                    sp.set_color('#45475a')
            self.ax_wr.set_ylabel('Win Rate',
                color='#a6adc8', fontsize=9)
            self.ax_loss.set_ylabel('Loss',
                color='#a6adc8', fontsize=9)
            self.ax_gp.set_ylabel('Percent',
                color='#a6adc8', fontsize=8)
            self.ax_gp2 = self.ax_gp.twinx()
            self.ax_gp2.set_ylabel('Spells/Turn',
                color='#a6e3a1', fontsize=8)
            self.ax_gp2.tick_params(colors='#6c7086',
                labelsize=7)
            self.ax_gp2.spines['right'].set_color('#45475a')
            for ax in [self.ax_wr, self.ax_loss, self.ax_gp]:
                ax.set_xlabel('Round',
                    color='#a6adc8', fontsize=9)
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
        self.svars['Atk Rate'].set(
            f"{s.attack_rate:.0f}% "
            f"(all={s.attack_all_in_pct:.0f}% "
            f"hold={s.attack_hold_pct:.0f}%)")
        self.svars['Spells/Turn'].set(
            f"{s.spells_per_turn:.2f} "
            f"({s.spells_per_game:.1f}/game)")
        self.svars['Idle Turns'].set(
            f"{s.idle_turns_pct:.0f}%")
        self.svars['Fallbacks'].set(
            f"{s.targeting_issues}")

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

            self.ax_gp.clear()
            self.ax_gp2.clear()
            self.ax_gp.set_facecolor('#313244')
            self.ax_gp.set_title('Gameplay Metrics',
                color='#cdd6f4', fontsize=10)
            if s.attack_rates:
                gr = range(1, len(s.attack_rates) + 1)
                self.ax_gp.plot(gr, s.attack_rates,
                    color='#f9e2af', linewidth=1.5,
                    label='Atk Rate %', marker='o',
                    markersize=3)
                self.ax_gp.plot(gr, s.idle_turns_pcts,
                    color='#f38ba8', linewidth=1.5,
                    label='Idle Turns %', marker='s',
                    markersize=3)
                self.ax_gp2.plot(gr, s.spells_per_turns,
                    color='#a6e3a1', linewidth=1.5,
                    label='Spells/Turn', marker='^',
                    markersize=3)
            self.ax_gp.set_ylabel('Percent',
                color='#a6adc8', fontsize=8)
            self.ax_gp.set_xlabel('Round',
                color='#a6adc8', fontsize=9)
            self.ax_gp2.set_ylabel('Spells/Turn',
                color='#a6e3a1', fontsize=8)
            self.ax_gp2.tick_params(colors='#6c7086',
                labelsize=7)
            self.ax_gp2.spines['right'].set_color('#45475a')
            self.ax_gp.legend(fontsize=7, loc='upper left',
                facecolor='#313244',
                edgecolor='#45475a',
                labelcolor='#cdd6f4')
            self.ax_gp.tick_params(colors='#6c7086',
                labelsize=8)
            for sp in self.ax_gp.spines.values():
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
    parser.add_argument('--reward-shaping-coeff',
        type=float, default=0.0,
        help='Initial coefficient for intermediate '
             'reward shaping (0=disabled)')
    parser.add_argument('--reward-shaping-decay',
        type=float, default=0.95,
        help='Per-round multiplicative decay for '
             'shaping coefficient')
    parser.add_argument('--league',
        action='store_true', default=False,
        help='Enable league-based opponent selection')
    parser.add_argument('--snapshot-interval',
        type=int, default=5,
        help='Save league snapshot every N rounds')
    parser.add_argument('--max-opponents',
        type=int, default=3,
        help='Max opponents per collection round')
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
