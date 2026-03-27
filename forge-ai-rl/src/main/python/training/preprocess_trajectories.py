#!/usr/bin/env python3
"""
Preprocess trajectory JSONL files into memory-mapped numpy arrays.

Structure:
    preprocessed/
    ├── metadata.json
    ├── shared/
    │   ├── game_state.npy      (N_total × 37216, float32)
    │   ├── global_features.npy (N_total × 96, float32)
    │   ├── outcome.npy         (N_total, float32)
    │   └── file_id.npy         (N_total, int32)
    ├── priority/
    │   ├── gs_index.npy        (N_pri, int64) → row in shared
    │   ├── candidates.npy      (N_pri × MAX × 64)
    │   ├── candidate_mask.npy  (N_pri × MAX, bool)
    │   ├── selected_idx.npy    (N_pri, int32)
    │   └── action_probs.npy    (N_pri × MAX, float32)
    ├── attack/
    │   ├── gs_index.npy        → row in shared
    │   ├── creatures.npy       (N_atk × MAX × 256)
    │   ├── creature_mask.npy
    │   ├── action_mask.npy
    │   └── action_probs.npy
    └── block/
        ├── gs_index.npy        → row in shared
        ├── pairs.npy           (N_blk × MAX × 512)
        ├── pair_mask.npy
        ├── action_mask.npy
        └── action_probs.npy

Game state is stored ONCE in shared/. Each decision type stores
only an index into shared/ plus its type-specific candidate data.
Disk usage: ~13GB instead of ~25GB.

Usage:
    python training/preprocess_trajectories.py \
        --data-dir /path/to/trajectories \
        --output-dir /path/to/preprocessed
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from pathlib import Path

# Canonical dimension constants
GLOBAL_DIM = 96
CARD_DIM = 256
GAME_STATE_DIM = 37216  # 96 + 145*256


def scan_files(files, progress_cb=None):
    """Pass 1: Count samples, find max candidate sizes.

    Args:
        progress_cb: optional callback(files_done, files_total)
    """
    counts = {'total': 0, 'priority': 0,
              'attack': 0, 'block': 0,
              'target': 0, 'card_select': 0,
              'mulligan': 0, 'binary': 0}
    max_cand = {'priority': 0, 'attack': 0, 'block': 0,
                'target': 0, 'card_select': 0,
                'mulligan': 0}

    print(f"  Pass 1: Scanning {len(files)} files...",
          flush=True)

    for fi, filepath in enumerate(files):
        if (fi + 1) % 200 == 0:
            print(f"    {fi+1}/{len(files)}", flush=True)
        if progress_cb and fi % 20 == 0:
            progress_cb(fi, len(files))
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            if len(lines) < 2:
                continue
            for line in lines[1:]:
                counts['total'] += 1
                rec = json.loads(line)
                dt = rec.get('decisionType', '')
                cand = rec.get('candidateFeatures', [])
                nc = len(cand)

                if dt == 'PRIORITY_ACTION':
                    counts['priority'] += 1
                    max_cand['priority'] = max(
                        max_cand['priority'], nc)
                elif dt == 'DECLARE_ATTACKERS':
                    counts['attack'] += 1
                    max_cand['attack'] = max(
                        max_cand['attack'], nc)
                elif dt == 'DECLARE_BLOCKERS':
                    if cand and len(cand[0]) > CARD_DIM + 10:
                        counts['block'] += 1
                        max_cand['block'] = max(
                            max_cand['block'], nc)
                elif dt == 'TARGET_SELECTION':
                    if nc > 1:  # skip trivial choices
                        counts['target'] += 1
                        max_cand['target'] = max(
                            max_cand['target'], nc)
                elif dt == 'CARD_SELECTION':
                    if nc >= 1:  # scry 1 has 1 candidate (top or bottom)
                        counts['card_select'] += 1
                        max_cand['card_select'] = max(
                            max_cand['card_select'], nc)
                elif dt == 'MULLIGAN':
                    counts['mulligan'] += 1
                    max_cand['mulligan'] = max(
                        max_cand['mulligan'], nc)
                elif dt == 'BINARY_CHOICE':
                    counts['binary'] += 1
        except Exception as e:
            print(f"    Warning: {filepath}: {e}",
                  flush=True)

    if progress_cb:
        progress_cb(len(files), len(files))
    print(f"  Counts: {counts}", flush=True)
    print(f"  Max candidates: {max_cand}", flush=True)
    return counts, max_cand


GAMMA = 0.99  # discount factor for value returns


def _extract_game_id(filepath):
    """Extract game ID from trajectory filename.
    Both files from the same game share a timestamp suffix.
    e.g. traj_P1_473_vs_P2_473_P1_473_W_1773863635351.jsonl
         traj_P1_473_vs_P2_473_P2_473_L_1773863635351.jsonl
    Both have game_id = 1773863635351
    """
    name = os.path.basename(str(filepath)).replace('.jsonl', '')
    # Timestamp is the last underscore-separated field
    parts = name.split('_')
    if len(parts) >= 2:
        return parts[-1]  # the timestamp
    return name


def _build_game_id_map(files):
    """Build mapping from file index to game_id (integer).
    Files from the same game get the same game_id."""
    ts_to_game = {}
    game_ids = []
    next_id = 0
    for filepath in files:
        ts = _extract_game_id(filepath)
        if ts not in ts_to_game:
            ts_to_game[ts] = next_id
            next_id += 1
        game_ids.append(ts_to_game[ts])
    return game_ids

def sanitize(arr):
    """Clip and replace NaN — matches training code."""
    np.clip(arr, -10, 10, out=arr)
    np.nan_to_num(arr, copy=False)
    return arr


def preprocess(files, output_dir, counts, max_cand,
               progress_cb=None):
    """Pass 2: Create and fill memory-mapped arrays.

    Args:
        progress_cb: optional callback(files_done, files_total)
    """
    nt = counts['total']
    np_ = counts['priority']
    na = counts['attack']
    nb = counts['block']
    n_tgt = counts['target']
    n_cs = counts['card_select']
    n_mul = counts['mulligan']
    n_bin = counts['binary']

    # Create directories
    for d in ['shared', 'priority', 'attack', 'block',
              'target', 'card_select', 'mulligan', 'binary']:
        os.makedirs(os.path.join(output_dir, d),
                    exist_ok=True)

    # Shared arrays (one row per decision record)
    print(f"  Creating shared arrays ({nt} records)...",
          flush=True)
    sh = os.path.join(output_dir, 'shared')
    gs = np.lib.format.open_memmap(
        os.path.join(sh, 'game_state.npy'),
        mode='w+', dtype=np.float32,
        shape=(nt, GAME_STATE_DIM))
    gf = np.lib.format.open_memmap(
        os.path.join(sh, 'global_features.npy'),
        mode='w+', dtype=np.float32,
        shape=(nt, GLOBAL_DIM))
    outcome = np.lib.format.open_memmap(
        os.path.join(sh, 'outcome.npy'),
        mode='w+', dtype=np.float32, shape=(nt,))
    file_id = np.lib.format.open_memmap(
        os.path.join(sh, 'file_id.npy'),
        mode='w+', dtype=np.int32, shape=(nt,))

    # Priority arrays
    print(f"  Creating priority arrays ({np_} records, "
          f"max_cand={max_cand['priority']})...",
          flush=True)
    pd = os.path.join(output_dir, 'priority')
    mp = max_cand['priority']
    p_gs_idx = np.lib.format.open_memmap(
        os.path.join(pd, 'gs_index.npy'),
        mode='w+', dtype=np.int64, shape=(np_,))
    p_cand = np.lib.format.open_memmap(
        os.path.join(pd, 'candidates.npy'),
        mode='w+', dtype=np.float32,
        shape=(np_, mp, 64))
    p_cmask = np.lib.format.open_memmap(
        os.path.join(pd, 'candidate_mask.npy'),
        mode='w+', dtype=np.bool_, shape=(np_, mp))
    p_sel = np.lib.format.open_memmap(
        os.path.join(pd, 'selected_idx.npy'),
        mode='w+', dtype=np.int32, shape=(np_,))
    p_aprobs = np.lib.format.open_memmap(
        os.path.join(pd, 'action_probs.npy'),
        mode='w+', dtype=np.float32, shape=(np_, mp))

    # Attack arrays
    print(f"  Creating attack arrays ({na} records, "
          f"max_cand={max_cand['attack']})...",
          flush=True)
    ad = os.path.join(output_dir, 'attack')
    ma = max_cand['attack']
    a_gs_idx = np.lib.format.open_memmap(
        os.path.join(ad, 'gs_index.npy'),
        mode='w+', dtype=np.int64, shape=(na,))
    a_crt = np.lib.format.open_memmap(
        os.path.join(ad, 'creatures.npy'),
        mode='w+', dtype=np.float32,
        shape=(na, ma, CARD_DIM))
    a_cmask = np.lib.format.open_memmap(
        os.path.join(ad, 'creature_mask.npy'),
        mode='w+', dtype=np.bool_, shape=(na, ma))
    a_amask = np.lib.format.open_memmap(
        os.path.join(ad, 'action_mask.npy'),
        mode='w+', dtype=np.float32, shape=(na, ma))
    a_aprobs = np.lib.format.open_memmap(
        os.path.join(ad, 'action_probs.npy'),
        mode='w+', dtype=np.float32, shape=(na, ma))

    # Block arrays
    print(f"  Creating block arrays ({nb} records, "
          f"max_cand={max_cand['block']})...",
          flush=True)
    bd = os.path.join(output_dir, 'block')
    mb = max_cand['block']
    b_gs_idx = np.lib.format.open_memmap(
        os.path.join(bd, 'gs_index.npy'),
        mode='w+', dtype=np.int64, shape=(nb,))
    b_pairs = np.lib.format.open_memmap(
        os.path.join(bd, 'pairs.npy'),
        mode='w+', dtype=np.float32,
        shape=(nb, mb, CARD_DIM * 2))
    b_pmask = np.lib.format.open_memmap(
        os.path.join(bd, 'pair_mask.npy'),
        mode='w+', dtype=np.bool_, shape=(nb, mb))
    b_amask = np.lib.format.open_memmap(
        os.path.join(bd, 'action_mask.npy'),
        mode='w+', dtype=np.float32, shape=(nb, mb))
    b_aprobs = np.lib.format.open_memmap(
        os.path.join(bd, 'action_probs.npy'),
        mode='w+', dtype=np.float32, shape=(nb, mb))

    # Target arrays (single-select from candidates, 256-dim features)
    if n_tgt > 0:
        mt = max_cand['target']
        print(f"  Creating target arrays ({n_tgt} records, "
              f"max_cand={mt})...", flush=True)
        td = os.path.join(output_dir, 'target')
        t_gs_idx = np.lib.format.open_memmap(
            os.path.join(td, 'gs_index.npy'),
            mode='w+', dtype=np.int64, shape=(n_tgt,))
        t_cand = np.lib.format.open_memmap(
            os.path.join(td, 'candidates.npy'),
            mode='w+', dtype=np.float32,
            shape=(n_tgt, mt, CARD_DIM))
        t_cmask = np.lib.format.open_memmap(
            os.path.join(td, 'candidate_mask.npy'),
            mode='w+', dtype=np.bool_, shape=(n_tgt, mt))
        t_sel = np.lib.format.open_memmap(
            os.path.join(td, 'selected_idx.npy'),
            mode='w+', dtype=np.int32, shape=(n_tgt,))
    else:
        mt = 0

    # Card select arrays (multi-select, 256-dim features)
    if n_cs > 0:
        mcs = max_cand['card_select']
        print(f"  Creating card_select arrays ({n_cs} records, "
              f"max_cand={mcs})...", flush=True)
        csd = os.path.join(output_dir, 'card_select')
        cs_gs_idx = np.lib.format.open_memmap(
            os.path.join(csd, 'gs_index.npy'),
            mode='w+', dtype=np.int64, shape=(n_cs,))
        cs_cand = np.lib.format.open_memmap(
            os.path.join(csd, 'candidates.npy'),
            mode='w+', dtype=np.float32,
            shape=(n_cs, mcs, CARD_DIM))
        cs_cmask = np.lib.format.open_memmap(
            os.path.join(csd, 'candidate_mask.npy'),
            mode='w+', dtype=np.bool_, shape=(n_cs, mcs))
        cs_amask = np.lib.format.open_memmap(
            os.path.join(csd, 'action_mask.npy'),
            mode='w+', dtype=np.float32, shape=(n_cs, mcs))
    else:
        mcs = 0

    # Mulligan arrays (hand features + keep/mull decision)
    if n_mul > 0:
        mmul = max_cand['mulligan']
        if mmul == 0:
            mmul = 7  # max hand size
        print(f"  Creating mulligan arrays ({n_mul} records, "
              f"max_hand={mmul})...", flush=True)
        muld = os.path.join(output_dir, 'mulligan')
        mul_gs_idx = np.lib.format.open_memmap(
            os.path.join(muld, 'gs_index.npy'),
            mode='w+', dtype=np.int64, shape=(n_mul,))
        mul_hand = np.lib.format.open_memmap(
            os.path.join(muld, 'hand_features.npy'),
            mode='w+', dtype=np.float32,
            shape=(n_mul, mmul, CARD_DIM))
        mul_hmask = np.lib.format.open_memmap(
            os.path.join(muld, 'hand_mask.npy'),
            mode='w+', dtype=np.bool_, shape=(n_mul, mmul))
        mul_keep = np.lib.format.open_memmap(
            os.path.join(muld, 'keep_decision.npy'),
            mode='w+', dtype=np.float32, shape=(n_mul,))
    else:
        mmul = 7

    # Binary arrays (just game state + decision)
    if n_bin > 0:
        print(f"  Creating binary arrays ({n_bin} records)...",
              flush=True)
        bind = os.path.join(output_dir, 'binary')
        bin_gs_idx = np.lib.format.open_memmap(
            os.path.join(bind, 'gs_index.npy'),
            mode='w+', dtype=np.int64, shape=(n_bin,))
        bin_decision = np.lib.format.open_memmap(
            os.path.join(bind, 'decision.npy'),
            mode='w+', dtype=np.float32, shape=(n_bin,))

    # Build game-level file IDs (both perspectives of same game get same ID)
    game_ids = _build_game_id_map(files)
    n_games = len(set(game_ids))
    print(f"  Game IDs: {n_games} unique games from {len(files)} files",
          flush=True)

    # Pass 2: Fill arrays
    print(f"\n  Pass 2: Writing {len(files)} files...",
          flush=True)
    si = 0  # shared index
    pi = 0  # priority index
    ai = 0  # attack index
    bi = 0  # block index
    ti = 0  # target index
    ci = 0  # card select index
    mi = 0  # mulligan index
    bni = 0  # binary index

    for fi, filepath in enumerate(files):
        if (fi + 1) % 200 == 0:
            print(f"    {fi+1}/{len(files)} "
                  f"(s={si} p={pi} a={ai} b={bi})",
                  flush=True)
        if progress_cb and fi % 20 == 0:
            progress_cb(fi, len(files))
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            if len(lines) < 2:
                continue
            header = json.loads(lines[0])
            won = header.get('won', False)

            # Parse all records first to compute
            # discounted returns backward
            records = []
            for line in lines[1:]:
                records.append(json.loads(line))

            # Compute discounted returns:
            # G_t = r_t + γ*G_{t+1}
            # where r_t = intermediateReward (shaping)
            # and terminal reward on last step
            n_recs = len(records)
            returns = np.zeros(n_recs, dtype=np.float32)
            G = 0.0
            for t in range(n_recs - 1, -1, -1):
                r = records[t].get(
                    'intermediateReward', 0.0)
                tr = records[t].get(
                    'terminalReward', 0.0)
                G = r + tr + GAMMA * G
                returns[t] = G

            for rec_idx, rec in enumerate(records):
                dt = rec.get('decisionType', '')
                cand = rec.get('candidateFeatures', [])
                sel = rec.get('selectedIndices', [])
                aprobs = rec.get(
                    'actionProbabilities', [])

                # Parse game state
                flat_raw = np.array(
                    rec.get('gameStateFlat', []),
                    dtype=np.float32)
                if len(flat_raw) == 0:
                    continue
                flat = np.zeros(GAME_STATE_DIM,
                                dtype=np.float32)
                fl = min(len(flat_raw), GAME_STATE_DIM)
                flat[:fl] = flat_raw[:fl]
                sanitize(flat)

                gf_raw = np.array(
                    rec.get('globalFeatures', []),
                    dtype=np.float32)
                g = np.zeros(GLOBAL_DIM, dtype=np.float32)
                gl = min(len(gf_raw), GLOBAL_DIM)
                if gl > 0:
                    g[:gl] = gf_raw[:gl]
                sanitize(g)

                # Write shared — discounted return, not
                # binary outcome
                gs[si] = flat
                gf[si] = g
                outcome[si] = returns[rec_idx]
                file_id[si] = game_ids[fi]  # game-level ID, not file-level
                shared_row = si
                si += 1

                nc = len(cand)

                if dt == 'PRIORITY_ACTION':
                    p_gs_idx[pi] = shared_row
                    for j in range(min(nc, mp)):
                        cf = cand[j]
                        cl = min(len(cf), 64)
                        p_cand[pi, j, :cl] = sanitize(
                            np.array(cf[:cl],
                                     dtype=np.float32))
                        p_cmask[pi, j] = True
                    si_val = sel[0] if sel else nc - 1
                    if si_val >= mp:
                        si_val = mp - 1
                    p_sel[pi] = si_val
                    for j in range(min(len(aprobs), mp)):
                        p_aprobs[pi, j] = aprobs[j]
                    pi += 1

                elif dt == 'DECLARE_ATTACKERS':
                    a_gs_idx[ai] = shared_row
                    for j in range(min(nc, ma)):
                        cf = cand[j]
                        cl = min(len(cf), CARD_DIM)
                        a_crt[ai, j, :cl] = sanitize(
                            np.array(cf[:cl],
                                     dtype=np.float32))
                        a_cmask[ai, j] = True
                    for s in sel:
                        if 0 <= s < ma:
                            a_amask[ai, s] = 1.0
                    for j in range(min(len(aprobs), ma)):
                        a_aprobs[ai, j] = aprobs[j]
                    ai += 1

                elif dt == 'DECLARE_BLOCKERS':
                    if not cand or len(cand[0]) <= CARD_DIM + 10:
                        continue
                    b_gs_idx[bi] = shared_row
                    for j in range(min(nc, mb)):
                        cf = cand[j]
                        cl = min(len(cf), CARD_DIM * 2)
                        b_pairs[bi, j, :cl] = sanitize(
                            np.array(cf[:cl],
                                     dtype=np.float32))
                        b_pmask[bi, j] = True
                    for s in sel:
                        if 0 <= s < mb:
                            b_amask[bi, s] = 1.0
                    for j in range(min(len(aprobs), mb)):
                        b_aprobs[bi, j] = aprobs[j]
                    bi += 1

                elif dt == 'TARGET_SELECTION' and n_tgt > 0:
                    if nc <= 1:
                        continue  # trivial choice
                    t_gs_idx[ti] = shared_row
                    for j in range(min(nc, mt)):
                        cf = cand[j]
                        cl = min(len(cf), CARD_DIM)
                        t_cand[ti, j, :cl] = sanitize(
                            np.array(cf[:cl],
                                     dtype=np.float32))
                        t_cmask[ti, j] = True
                    si_val = sel[0] if sel else 0
                    if si_val >= mt:
                        si_val = mt - 1
                    t_sel[ti] = si_val
                    ti += 1

                elif dt == 'CARD_SELECTION' and n_cs > 0:
                    if nc < 1:
                        continue
                    cs_gs_idx[ci] = shared_row
                    for j in range(min(nc, mcs)):
                        cf = cand[j]
                        cl = min(len(cf), CARD_DIM)
                        cs_cand[ci, j, :cl] = sanitize(
                            np.array(cf[:cl],
                                     dtype=np.float32))
                        cs_cmask[ci, j] = True
                    for s in sel:
                        if 0 <= s < mcs:
                            cs_amask[ci, s] = 1.0
                    ci += 1

                elif dt == 'MULLIGAN' and n_mul > 0:
                    mul_gs_idx[mi] = shared_row
                    for j in range(min(nc, mmul)):
                        cf = cand[j]
                        cl = min(len(cf), CARD_DIM)
                        mul_hand[mi, j, :cl] = sanitize(
                            np.array(cf[:cl],
                                     dtype=np.float32))
                        mul_hmask[mi, j] = True
                    # sel[0]=1 means keep, sel[0]=0 means mull
                    mul_keep[mi] = 1.0 if (sel and sel[0] == 1) else 0.0
                    mi += 1

                elif dt == 'BINARY_CHOICE' and n_bin > 0:
                    bin_gs_idx[bni] = shared_row
                    # sel[0]=1 means yes, sel[0]=0 means no
                    bin_decision[bni] = 1.0 if (sel and sel[0] == 1) else 0.0
                    bni += 1

        except Exception as e:
            print(f"    Warning: {filepath}: {e}",
                  flush=True)

    # Flush all arrays
    for arr in [gs, gf, outcome, file_id,
                p_gs_idx, p_cand, p_cmask, p_sel,
                p_aprobs,
                a_gs_idx, a_crt, a_cmask, a_amask,
                a_aprobs,
                b_gs_idx, b_pairs, b_pmask, b_amask,
                b_aprobs]:
        arr.flush()
    if n_tgt > 0:
        for arr in [t_gs_idx, t_cand, t_cmask, t_sel]:
            arr.flush()
    if n_cs > 0:
        for arr in [cs_gs_idx, cs_cand, cs_cmask, cs_amask]:
            arr.flush()
    if n_mul > 0:
        for arr in [mul_gs_idx, mul_hand, mul_hmask, mul_keep]:
            arr.flush()
    if n_bin > 0:
        for arr in [bin_gs_idx, bin_decision]:
            arr.flush()

    if progress_cb:
        progress_cb(len(files), len(files))

    return {'total': si, 'priority': pi,
            'attack': ai, 'block': bi,
            'target': ti, 'card_select': ci,
            'mulligan': mi, 'binary': bni}


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess trajectories to mmap')
    parser.add_argument('--data-dir',
        default='../../rl_data/trajectories')
    parser.add_argument('--output-dir',
        default='../../rl_data/preprocessed')
    args = parser.parse_args()

    t0 = time.time()
    print("=== Trajectory Preprocessing ===", flush=True)

    path = Path(args.data_dir)
    files = sorted(path.glob('traj_*.jsonl'))
    print(f"  Found {len(files)} trajectory files",
          flush=True)
    if not files:
        print("  ERROR: No files found", flush=True)
        sys.exit(1)

    # Pass 1
    counts, max_cand = scan_files(files)

    # Create output
    os.makedirs(args.output_dir, exist_ok=True)

    # Pass 2
    final = preprocess(files, args.output_dir,
                       counts, max_cand)

    # Metadata
    metadata = {
        'source_dir': str(args.data_dir),
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
    with open(os.path.join(args.output_dir,
                           'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n=== Preprocessing Complete ===", flush=True)
    print(f"  Time: {elapsed:.1f}s", flush=True)
    print(f"  Shared: {final['total']} records",
          flush=True)
    print(f"  Priority: {final['priority']}", flush=True)
    print(f"  Attack: {final['attack']}", flush=True)
    print(f"  Block: {final['block']}", flush=True)
    print(f"  Target: {final['target']}", flush=True)
    print(f"  Card Select: {final['card_select']}", flush=True)
    print(f"  Mulligan: {final['mulligan']}", flush=True)
    print(f"  Binary: {final['binary']}", flush=True)

    # Disk usage
    total_bytes = 0
    for root, dirs, fnames in os.walk(args.output_dir):
        for fn in fnames:
            total_bytes += os.path.getsize(
                os.path.join(root, fn))
    print(f"  Disk: {total_bytes/1024**3:.1f} GB",
          flush=True)


if __name__ == '__main__':
    main()
