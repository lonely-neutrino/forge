#!/usr/bin/env python3
"""
Data Collection Dashboard — visual monitoring of trajectory collection.

Launches the Java game runner and displays live progress:
- Games completed / total
- Win/loss counts
- Trajectory files and decision counts
- Average turns per game
- Collection speed

Usage:
    python training/collect_ui.py --games 1000
"""

import argparse
import os
import sys
import threading
import time
import json
from dataclasses import dataclass, field
from pathlib import Path
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
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from training.ppo_trainer import PROJECT_ROOT, FORGE_JAR
from training.deck_config import load_rl_decks

import subprocess


@dataclass
class CollectState:
    status: str = "Starting..."
    phase: str = "init"  # init, collecting, analyzing, preprocessing, done

    games_done: int = 0
    games_total: int = 0

    p1_wins: int = 0
    p2_wins: int = 0
    draws: int = 0
    errors: int = 0

    traj_files: int = 0
    attack_decisions: int = 0
    block_decisions: int = 0
    priority_decisions: int = 0

    games_per_sec: float = 0.0
    eta_sec: float = 0.0
    elapsed_sec: float = 0.0
    avg_turns: float = 0.0

    # Preprocessing
    preprocess_phase: str = ""  # scan, write
    preprocess_files_done: int = 0
    preprocess_files_total: int = 0
    preprocess_records: int = 0
    preprocess_pct: float = 0.0
    target_decisions: int = 0
    card_select_decisions: int = 0
    mulligan_decisions: int = 0
    binary_decisions: int = 0
    n_games: int = 0
    preprocess_disk_gb: float = 0.0

    # For chart
    speed_history: List[float] = field(default_factory=list)
    files_history: List[int] = field(default_factory=list)
    chart_dirty: bool = False

    log_lines: List[str] = field(default_factory=list)
    log_dirty: bool = False


def log(state, msg):
    print(msg, flush=True)
    state.log_lines.append(msg)
    if len(state.log_lines) > 200:
        state.log_lines = state.log_lines[-200:]
    state.log_dirty = True


def _run_preprocessing(state, traj_dir):
    """Run preprocessing (scan + write) with progress updates."""
    from training.preprocess_trajectories import (
        scan_files, preprocess, GAME_STATE_DIM,
        GLOBAL_DIM, CARD_DIM, GAMMA,
        _build_game_id_map)

    output_dir = os.path.join(
        os.path.dirname(traj_dir), 'preprocessed')

    state.phase = "preprocessing"
    state.preprocess_phase = "scan"
    state.status = "Preprocessing: scanning files..."
    log(state, "")
    log(state, "=== Preprocessing ===")

    path = Path(traj_dir)
    files = sorted(path.glob('traj_*.jsonl'))
    state.preprocess_files_total = len(files)
    log(state, f"  Found {len(files)} trajectory files")

    if not files:
        log(state, "  ERROR: No trajectory files found")
        return

    # Pass 1: scan with progress
    def scan_progress(done, total):
        state.preprocess_files_done = done
        state.preprocess_pct = done / max(total, 1) * 100
        state.status = (f"Scanning: {done}/{total} files "
                        f"({state.preprocess_pct:.0f}%)")

    counts, max_cand = scan_files(files,
                                  progress_cb=scan_progress)
    state.preprocess_records = counts['total']
    state.attack_decisions = counts['attack']
    state.block_decisions = counts['block']
    state.priority_decisions = counts['priority']
    state.target_decisions = counts.get('target', 0)
    state.card_select_decisions = counts.get('card_select', 0)
    state.mulligan_decisions = counts.get('mulligan', 0)
    state.binary_decisions = counts.get('binary', 0)

    log(state, f"  Total records: {counts['total']}")
    log(state, f"  Priority: {counts['priority']}, "
        f"Attack: {counts['attack']}, "
        f"Block: {counts['block']}")
    log(state, f"  Target: {counts.get('target', 0)}, "
        f"Card Select: {counts.get('card_select', 0)}, "
        f"Mulligan: {counts.get('mulligan', 0)}, "
        f"Binary: {counts.get('binary', 0)}")
    log(state, f"  Max candidates: {max_cand}")

    # Pass 2: write mmap arrays with progress
    state.preprocess_phase = "write"
    os.makedirs(output_dir, exist_ok=True)

    t0_pp = time.time()

    def write_progress(done, total):
        state.preprocess_files_done = done
        state.preprocess_pct = done / max(total, 1) * 100
        state.status = (f"Writing arrays: {done}/{total} "
                        f"files ({state.preprocess_pct:.0f}%)")

    final = preprocess(files, output_dir, counts, max_cand,
                       progress_cb=write_progress)

    # Compute game count
    game_ids = _build_game_id_map(files)
    state.n_games = len(set(game_ids))

    # Save metadata
    metadata = {
        'source_dir': str(traj_dir),
        'n_files': len(files),
        'counts': counts,
        'final_counts': final,
        'max_candidates': max_cand,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'game_state_dim': GAME_STATE_DIM,
        'global_feature_dim': GLOBAL_DIM,
        'card_feature_dim': CARD_DIM,
        'shared_game_state': True,
        'discount_gamma': GAMMA,
        'value_targets': 'discounted_returns',
    }
    with open(os.path.join(output_dir,
                           'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Disk usage
    total_bytes = 0
    for root, dirs, fnames in os.walk(output_dir):
        for fn in fnames:
            total_bytes += os.path.getsize(
                os.path.join(root, fn))
    state.preprocess_disk_gb = total_bytes / 1024**3

    pp_time = time.time() - t0_pp
    state.preprocess_pct = 100.0
    state.preprocess_files_done = len(files)

    log(state, f"  Preprocessing complete in {pp_time:.0f}s")
    log(state, f"  Written: {final['total']} shared, "
        f"{final['priority']} priority, "
        f"{final['attack']} attack, "
        f"{final['block']} block")
    log(state, f"  Target: {final.get('target', 0)}, "
        f"Card Select: {final.get('card_select', 0)}, "
        f"Mulligan: {final.get('mulligan', 0)}, "
        f"Binary: {final.get('binary', 0)}")
    log(state, f"  Games: {state.n_games} unique")
    log(state, f"  Disk: {state.preprocess_disk_gb:.1f} GB")
    log(state, f"  Output: {output_dir}")


def collect_thread(state, args):
    """Run Java data collection and monitor progress."""
    try:
        output_dir = os.path.join(PROJECT_ROOT, 'rl_data/trajectories')
        preproc_dir = os.path.join(PROJECT_ROOT, 'rl_data/preprocessed')

        if args.clean:
            import shutil
            for d in [output_dir, preproc_dir]:
                if os.path.exists(d):
                    n = len(list(Path(d).glob('*')))
                    log(state, f"Cleaning {d} ({n} items)...")
                    shutil.rmtree(d)
            log(state, "Old data removed.")

        os.makedirs(output_dir, exist_ok=True)

        state.games_total = args.games
        state.phase = "collecting"
        state.status = f"Collecting {args.games} games..."

        log(state, f"Output: {output_dir}")
        log(state, f"Games: {args.games}, Threads: 16")

        decks = load_rl_decks(overrides=args.deck)
        log(state, f"Decks: {', '.join(decks)}")
        deck_args = []
        for d in decks:
            deck_args.extend(['-d', d])

        cmd = [
            'java', '-Xmx8192m',
            '--add-opens', 'java.base/java.lang=ALL-UNNAMED',
            '--add-opens', 'java.base/java.util=ALL-UNNAMED',
            '--add-opens', 'java.base/java.text=ALL-UNNAMED',
            '--add-opens', 'java.base/java.lang.reflect=ALL-UNNAMED',
            '--add-opens', 'java.desktop/javax.imageio.spi=ALL-UNNAMED',
            '-jar', FORGE_JAR,
            'rltrain', 'collect',
        ] + deck_args + [
            '-n', str(args.games),
            '-t', '16',
            '-o', output_dir,
        ]

        cwd = os.path.join(PROJECT_ROOT, 'forge-gui-desktop')
        t0 = time.time()

        proc = subprocess.Popen(
            cmd, cwd=cwd, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True)

        # Monitor thread: poll files for progress + track time
        def monitor():
            while state.phase == "collecting":
                try:
                    # Count trajectory files for continuous progress
                    files = list(Path(output_dir).glob('traj_*.jsonl'))
                    file_count = len(files)
                    if file_count > state.traj_files:
                        state.traj_files = file_count
                    # 2 files per game (one per player)
                    games_from_files = file_count // 2
                    if games_from_files > state.games_done:
                        state.games_done = games_from_files

                    elapsed = time.time() - t0
                    state.elapsed_sec = elapsed
                    if elapsed > 0 and state.games_done > 0:
                        state.games_per_sec = (
                            state.games_done / elapsed)
                    if state.games_per_sec > 0:
                        remaining = (state.games_total
                                     - state.games_done)
                        state.eta_sec = remaining / state.games_per_sec
                    if state.games_done > 0:
                        state.status = (
                            f"Game {state.games_done}/"
                            f"{state.games_total}"
                            f" ({state.games_per_sec:.1f}/s)")
                    state.speed_history.append(state.games_per_sec)
                    state.files_history.append(state.traj_files)
                    state.chart_dirty = True
                except Exception:
                    pass
                time.sleep(1)

        mon = threading.Thread(target=monitor, daemon=True)
        mon.start()

        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue

            # Parse progress table rows (pipe-delimited with │)
            # Format: Game │ Done/Total │ Games/s │ P1 │ P2 │ Draw│ Err │ Turns │ Files │ ETA
            if '│' in line and not line.startswith('─'):
                try:
                    cols = [c.strip() for c in line.split('│')]
                    if len(cols) >= 10:
                        # cols: ['Game', 'Done/Total', 'Games/s', 'P1', 'P2',
                        #        'Draw', 'Err', 'Turns', 'Files', 'ETA']
                        # Skip header row
                        if cols[1] and '/' in cols[1]:
                            parts = cols[1].split('/')
                            done = int(parts[0])
                            total = int(parts[1])
                            state.games_done = done
                            state.games_total = max(
                                state.games_total, total)
                        if cols[2]:
                            try:
                                state.games_per_sec = float(cols[2])
                            except ValueError:
                                pass
                        if cols[3]:
                            try:
                                state.p1_wins = int(cols[3])
                            except ValueError:
                                pass
                        if cols[4]:
                            try:
                                state.p2_wins = int(cols[4])
                            except ValueError:
                                pass
                        if cols[5]:
                            try:
                                state.draws = int(cols[5])
                            except ValueError:
                                pass
                        if cols[6]:
                            try:
                                state.errors = int(cols[6])
                            except ValueError:
                                pass
                        if cols[7]:
                            try:
                                state.avg_turns = float(cols[7])
                            except ValueError:
                                pass
                        if cols[8]:
                            try:
                                state.traj_files = int(cols[8])
                            except ValueError:
                                pass
                except Exception:
                    pass

            # Parse summary lines (after collection complete)
            # "Games: 1000 | P1: 500 | P2: 420 | Draw: 70 | Errors: 10"
            if 'P1:' in line and 'P2:' in line:
                try:
                    for part in line.split('|'):
                        part = part.strip()
                        if part.startswith('P1:'):
                            state.p1_wins = int(part[3:].strip())
                        elif part.startswith('P2:'):
                            state.p2_wins = int(part[3:].strip())
                        elif part.startswith('Draw:'):
                            state.draws = int(part[5:].strip())
                        elif part.startswith('Errors:'):
                            state.errors = int(part[7:].strip())
                except (ValueError, IndexError):
                    pass

            # "Avg turns: 8.5 | Files: 200"
            if 'Avg turns:' in line:
                try:
                    for part in line.split('|'):
                        part = part.strip()
                        if part.startswith('Avg turns:'):
                            state.avg_turns = float(
                                part[10:].strip())
                        elif part.startswith('Files:'):
                            state.traj_files = int(
                                part[6:].strip())
                except (ValueError, IndexError):
                    pass

            if 'Collection Complete' in line:
                state.phase = "analyzing"
                state.status = "Analyzing trajectories..."

            log(state, line)

        proc.wait()
        state.phase = "analyzing"

        # Analyze trajectory files
        state.status = "Analyzing trajectories..."
        attacks, blocks, priority = 0, 0, 0
        tapped_leak = 0
        total_cand, total_sel = 0, 0
        files = sorted(Path(output_dir).glob('traj_*.jsonl'))
        state.traj_files = len(files)
        n_files = len(files)

        for fi, f in enumerate(files):
            if fi % 20 == 0:
                pct = fi / max(n_files, 1) * 100
                state.status = (
                    f"Analyzing: {fi}/{n_files} files "
                    f"({pct:.0f}%)")
            for fline in open(f):
                try:
                    r = json.loads(fline)
                    dt = r.get('decisionType', '')
                    cand = r.get('candidateFeatures', [])
                    sel = r.get('selectedIndices', [])
                    if dt == 'DECLARE_ATTACKERS' and len(cand) > 0:
                        attacks += 1
                        total_cand += len(cand)
                        total_sel += len(sel)
                        for i, cf in enumerate(cand):
                            if len(cf) > 17 and cf[17] > 0.5 and i in sel:
                                tapped_leak += 1
                    elif dt == 'DECLARE_BLOCKERS' and len(cand) > 0:
                        blocks += 1
                    elif dt == 'PRIORITY_ACTION':
                        priority += 1
                except Exception:
                    pass

        state.attack_decisions = attacks
        state.block_decisions = blocks
        state.priority_decisions = priority
        state.elapsed_sec = time.time() - t0

        log(state, "")
        log(state, f"=== Collection Summary ===")
        log(state, f"Files: {state.traj_files}")
        log(state, f"Attacks: {attacks}, Blocks: {blocks}, Priority: {priority}")
        log(state, f"Avg candidates/attack: {total_cand/max(attacks,1):.1f}")
        log(state, f"Avg selected/attack: {total_sel/max(attacks,1):.1f}")
        log(state, f"Attack rate: {total_sel*100/max(total_cand,1):.1f}%")
        log(state, f"Tapped leak: {tapped_leak} (MUST be 0)")
        log(state, f"Collection time: {state.elapsed_sec:.0f}s")

        # === Preprocessing phase ===
        _run_preprocessing(state, output_dir)

        state.phase = "done"
        state.status = "Collection + preprocessing complete!"
        state.elapsed_sec = time.time() - t0
        log(state, f"Total time: {state.elapsed_sec:.0f}s")

    except Exception as e:
        log(state, f"ERROR: {e}")
        state.status = f"ERROR: {e}"
        state.phase = "done"
        import traceback
        traceback.print_exc()


class CollectDashboard:
    def __init__(self, root, state):
        self.root = root
        self.state = state
        root.title("MTG RL — Data Collection")
        root.geometry("900x750")
        root.configure(bg='#1e1e2e')

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('H.TLabel', font=('Helvetica', 16, 'bold'),
                         background='#1e1e2e', foreground='#cdd6f4')
        style.configure('S.TLabel', font=('Consolas', 11),
                         background='#1e1e2e', foreground='#a6adc8')
        style.configure('V.TLabel', font=('Consolas', 11, 'bold'),
                         background='#1e1e2e', foreground='#89b4fa')
        style.configure('D.TFrame', background='#1e1e2e')
        style.configure("b.Horizontal.TProgressbar",
                         troughcolor='#313244', background='#89b4fa')

        self._build(root)
        self._tick()

    def _build(self, root):
        m = ttk.Frame(root, style='D.TFrame')
        m.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        ttk.Label(m, text="MTG RL — Data Collection",
                  style='H.TLabel').pack(pady=(0, 6))

        self.status_v = tk.StringVar(value="Starting...")
        ttk.Label(m, textvariable=self.status_v,
                  style='S.TLabel').pack()

        # Progress bar
        pf = ttk.Frame(m, style='D.TFrame')
        pf.pack(fill=tk.X, pady=4)
        self.prog = ttk.Progressbar(pf, length=850,
                                     style="b.Horizontal.TProgressbar")
        self.prog.pack(fill=tk.X)
        self.prog_v = tk.StringVar()
        ttk.Label(pf, textvariable=self.prog_v,
                  style='S.TLabel').pack(anchor='w')

        # Stats grid
        sf = ttk.Frame(m, style='D.TFrame')
        sf.pack(fill=tk.X, pady=4)
        self.svars = {}
        stats = [
            ('Games', '—'), ('Speed', '—'),
            ('P1 Wins', '—'), ('P2 Wins', '—'),
            ('Draws', '—'), ('Errors', '—'),
            ('Avg Turns', '—'), ('ETA', '—'),
            ('Files', '—'), ('Attacks', '—'),
            ('Blocks', '—'), ('Priority', '—'),
            ('Target', '—'), ('Mulligan', '—'),
            ('Binary', '—'), ('Unique Games', '—'),
            ('Records', '—'), ('Disk', '—'),
            ('Elapsed', '—'),
        ]
        for i, (k, v) in enumerate(stats):
            r, c = divmod(i, 4)
            ttk.Label(sf, text=f"{k}:", style='S.TLabel').grid(
                row=r, column=c*2, sticky='w', padx=(8, 2), pady=2)
            sv = tk.StringVar(value=v)
            ttk.Label(sf, textvariable=sv, style='V.TLabel').grid(
                row=r, column=c*2+1, sticky='w', padx=(0, 12), pady=2)
            self.svars[k] = sv

        # Console log
        lf = ttk.Frame(m, style='D.TFrame')
        lf.pack(fill=tk.BOTH, expand=True, pady=4)
        self.log_text = tk.Text(lf, height=15,
                                 bg='#181825', fg='#a6adc8',
                                 font=('Consolas', 9),
                                 wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _tick(self):
        s = self.state
        self.status_v.set(s.status)

        if s.phase == "preprocessing":
            self.prog['value'] = s.preprocess_pct
            self.prog_v.set(
                f"Preprocessing... {s.preprocess_phase}")
        elif s.games_total > 0:
            pct = s.games_done / s.games_total * 100
            self.prog['value'] = pct
            self.prog_v.set(
                f"{s.games_done}/{s.games_total} games "
                f"({pct:.0f}%)")

        self.svars['Games'].set(
            f"{s.games_done}/{s.games_total}")
        self.svars['Speed'].set(
            f"{s.games_per_sec:.1f}/s")
        self.svars['P1 Wins'].set(str(s.p1_wins))
        self.svars['P2 Wins'].set(str(s.p2_wins))
        self.svars['Draws'].set(str(s.draws))
        self.svars['Errors'].set(str(s.errors))
        self.svars['Avg Turns'].set(
            f"{s.avg_turns:.1f}" if s.avg_turns > 0 else "—")
        self.svars['ETA'].set(
            f"{s.eta_sec:.0f}s" if s.eta_sec > 0 else "—")
        self.svars['Files'].set(str(s.traj_files))
        self.svars['Attacks'].set(str(s.attack_decisions))
        self.svars['Blocks'].set(str(s.block_decisions))
        self.svars['Priority'].set(str(s.priority_decisions))
        self.svars['Target'].set(
            str(s.target_decisions) if s.target_decisions else "—")
        self.svars['Mulligan'].set(
            str(s.mulligan_decisions) if s.mulligan_decisions else "—")
        self.svars['Binary'].set(
            str(s.binary_decisions) if s.binary_decisions else "—")
        self.svars['Unique Games'].set(
            str(s.n_games) if s.n_games else "—")
        self.svars['Records'].set(
            str(s.preprocess_records) if s.preprocess_records else "—")
        self.svars['Disk'].set(
            f"{s.preprocess_disk_gb:.1f} GB"
            if s.preprocess_disk_gb > 0 else "—")
        self.svars['Elapsed'].set(
            f"{s.elapsed_sec:.0f}s" if s.elapsed_sec > 0 else "—")

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
        description='MTG RL Data Collection Dashboard')
    parser.add_argument('--games', type=int, default=1000,
                        help='Number of games to collect')
    parser.add_argument('--clean', action='store_true',
                        help='Delete old trajectories and '
                             'preprocessed data before collecting')
    parser.add_argument('--deck', action='append',
                        help='Override the configured RL deck list '
                             '(may be repeated)')
    args = parser.parse_args()

    state = CollectState()
    t = threading.Thread(target=collect_thread,
                         args=(state, args), daemon=True)
    t.start()

    root = tk.Tk()
    CollectDashboard(root, state)
    root.mainloop()


if __name__ == '__main__':
    main()
