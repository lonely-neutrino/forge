#!/usr/bin/env python3
"""
PPO Self-Play Training Loop

Alternates between:
1. Playing games (Java subprocess) with RL agent using model server
2. Loading trajectory data with action probabilities
3. Computing advantages via value network
4. PPO policy gradient updates on attack/block heads
5. Saving updated model and repeating

Usage:
    python training/ppo_trainer.py \
        --checkpoint /path/to/model_with_decisions.pt \
        --device cuda \
        --rounds 20 \
        --games-per-round 200
"""

import argparse
import json
import os
import sys
import time
import subprocess
import signal
import random
import socket
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
from serving.model_server import ModelServer
from training.mmap_dataset import parse_game_state, GAME_STATE_DIM, CARD_DIM, GLOBAL_DIM, ZONES_CONFIG
from training.deck_config import load_rl_decks
from training.runtime_paths import (
    PROJECT_ROOT as RUNTIME_PROJECT_ROOT,
    get_checkpoint_dir,
    get_device_override,
    get_ppo_traj_dir,
    resolve_forge_jar,
)

import threading


# ── Config ───────────────────────────────────────────

PROJECT_ROOT = str(RUNTIME_PROJECT_ROOT)
FORGE_JAR = resolve_forge_jar()


# ── Data loading for PPO ─────────────────────────────

GAE_GAMMA = 0.99  # must match value network training gamma
GAE_LAMBDA = 0.95  # lower = trust per-step value deltas more, less terminal outcome influence


def _compute_gae_returns(records, won, shaping_coeff=0.0):
    """Compute per-decision discounted returns using GAE.
    Uses intermediate rewards (scaled by shaping_coeff) + terminal reward."""
    n = len(records)
    if n == 0:
        return []

    # Extract rewards and value estimates
    rewards = np.zeros(n, dtype=np.float32)
    values = np.zeros(n, dtype=np.float32)
    for i, rec in enumerate(records):
        values[i] = rec.get('valueEstimate', 0.0)
        if shaping_coeff > 0:
            rewards[i] = shaping_coeff * rec.get('intermediateReward', 0.0)

    # Terminal reward on last step
    terminal = 1.0 if won else -1.0
    rewards[-1] += terminal

    # GAE: delta_t = r_t + gamma * V(t+1) - V(t)
    # A_t = sum_{l=0}^{T-t} (gamma*lambda)^l * delta_l
    advantages = np.zeros(n, dtype=np.float32)
    last_gae = 0.0
    for t in range(n - 1, -1, -1):
        if t == n - 1:
            next_value = 0.0  # terminal
        else:
            next_value = values[t + 1]
        delta = rewards[t] + GAE_GAMMA * next_value - values[t]
        last_gae = delta + GAE_GAMMA * GAE_LAMBDA * last_gae
        advantages[t] = last_gae

    # Returns = advantages + values (for value network training)
    returns = advantages + values
    return list(zip(advantages, returns))


def load_ppo_data(traj_dir, shaping_coeff=0.0):
    """Load trajectory data for PPO training.
    Computes per-decision GAE advantages from intermediate rewards."""
    path = Path(traj_dir)
    files = sorted(path.glob('traj_*.jsonl'))

    attack_samples = []
    block_samples = []
    priority_samples = []
    target_samples = []
    mulligan_samples = []
    value_samples = []

    for filepath in files:
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            if len(lines) < 2:
                continue
            header = json.loads(lines[0])
            won = header.get('won', False)

            # Parse all records first to compute GAE
            all_records = [json.loads(line) for line in lines[1:]]
            gae_data = _compute_gae_returns(all_records, won, shaping_coeff=shaping_coeff)

            for rec_idx, rec in enumerate(all_records):
                dt = rec.get('decisionType', '')
                cand = rec.get('candidateFeatures', [])
                sel = rec.get('selectedIndices', [])

                # Per-decision advantage and return from GAE
                if rec_idx < len(gae_data):
                    advantage, gae_return = gae_data[rec_idx]
                else:
                    advantage = 1.0 if won else -1.0
                    gae_return = advantage
                outcome = gae_return  # use GAE return for value training

                gf = np.array(
                    rec.get('globalFeatures', []),
                    dtype=np.float32)
                np.clip(gf, -10, 10, out=gf)
                gf = np.nan_to_num(gf)

                flat = np.array(
                    rec.get('gameStateFlat', []),
                    dtype=np.float32)
                np.clip(flat, -10, 10, out=flat)
                flat = np.nan_to_num(flat)

                # Always collect value training data
                if len(flat) > 0:
                    value_samples.append({
                        'global_features': gf,
                        'game_state_flat': flat,
                        'outcome': outcome,
                    })

                if len(cand) < 1:
                    continue

                # Old policy probabilities for PPO ratio
                old_probs = np.array(
                    rec.get('actionProbabilities', []),
                    dtype=np.float32)

                if dt == 'PRIORITY_ACTION':
                    # Priority: 64-dim action features,
                    # single-select
                    n = len(cand)
                    actions = np.zeros(
                        (n, 64), dtype=np.float32)
                    for j, cf in enumerate(cand):
                        cl = min(len(cf), 64)
                        actions[j, :cl] = np.array(
                            cf[:cl], dtype=np.float32)
                    np.clip(actions, -10, 10, out=actions)
                    actions = np.nan_to_num(actions)

                    selected_idx = sel[0] if sel else n - 1
                    if selected_idx >= n:
                        selected_idx = n - 1

                    # Old log prob for the selected action
                    old_lp = 0.0
                    if len(old_probs) > selected_idx:
                        p = max(old_probs[selected_idx],
                                1e-8)
                        old_lp = float(np.log(p))

                    priority_samples.append({
                        'global_features': gf,
                        'game_state_flat': flat,
                        'action_features': actions,
                        'selected_idx': selected_idx,
                        'n_actions': n,
                        'outcome': outcome,
                        'advantage': advantage,
                        'old_log_prob': old_lp,
                    })
                    continue

                # Combat: card features, multi-select
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

                action_mask = np.zeros(
                    n, dtype=np.float32)
                for idx in sel:
                    if 0 <= idx < n:
                        action_mask[idx] = 1.0

                # Old log prob: sum of per-creature
                # log probs for the joint action
                old_lp = 0.0
                if len(old_probs) >= n:
                    for j in range(n):
                        p = max(old_probs[j], 1e-8)
                        if action_mask[j] > 0.5:
                            old_lp += float(np.log(p))
                        else:
                            old_lp += float(
                                np.log(max(1-p, 1e-8)))

                sample = {
                    'global_features': gf,
                    'game_state_flat': flat,
                    'creature_features': creatures,
                    'action_mask': action_mask,
                    'n_creatures': n,
                    'outcome': outcome,
                    'advantage': advantage,
                    'old_log_prob': old_lp,
                }
                if dt == 'DECLARE_ATTACKERS':
                    attack_samples.append(sample)
                elif dt == 'DECLARE_BLOCKERS':
                    block_samples.append(sample)
                    continue

                if dt == 'TARGET_SELECTION':
                    if n <= 1:
                        continue  # trivial choice
                    targets = np.zeros(
                        (n, CARD_DIM), dtype=np.float32)
                    for j, cf in enumerate(cand):
                        cl = min(len(cf), CARD_DIM)
                        targets[j, :cl] = np.array(
                            cf[:cl], dtype=np.float32)
                    np.clip(targets, -10, 10, out=targets)
                    targets = np.nan_to_num(targets)

                    selected_idx = sel[0] if sel else 0
                    if selected_idx >= n:
                        selected_idx = n - 1

                    old_lp = 0.0
                    if len(old_probs) > selected_idx:
                        p = max(old_probs[selected_idx], 1e-8)
                        old_lp = float(np.log(p))

                    # Spell features for targeting context
                    sf = rec.get('spellFeatures')
                    spell_feats = np.zeros(64, dtype=np.float32)
                    if sf is not None:
                        sl = min(len(sf), 64)
                        spell_feats[:sl] = np.array(
                            sf[:sl], dtype=np.float32)

                    target_samples.append({
                        'global_features': gf,
                        'game_state_flat': flat,
                        'target_features': targets,
                        'selected_idx': selected_idx,
                        'n_targets': n,
                        'outcome': outcome,
                        'advantage': advantage,
                        'old_log_prob': old_lp,
                        'spell_features': spell_feats,
                    })

                elif dt == 'MULLIGAN':
                    # Mulligan: hand features + keep/mull
                    hand = np.zeros(
                        (n, CARD_DIM), dtype=np.float32)
                    for j, cf in enumerate(cand):
                        cl = min(len(cf), CARD_DIM)
                        hand[j, :cl] = np.array(
                            cf[:cl], dtype=np.float32)
                    np.clip(hand, -10, 10, out=hand)
                    hand = np.nan_to_num(hand)

                    kept = 1.0 if (sel and sel[0] == 1) else 0.0

                    old_lp = 0.0
                    if len(old_probs) > 0:
                        p = old_probs[0]
                        if kept > 0.5:
                            old_lp = float(np.log(max(p, 1e-8)))
                        else:
                            old_lp = float(np.log(max(1-p, 1e-8)))

                    mulligan_samples.append({
                        'global_features': gf,
                        'game_state_flat': flat,
                        'hand_features': hand,
                        'n_cards': n,
                        'kept': kept,
                        'outcome': outcome,
                        'advantage': advantage,
                        'old_log_prob': old_lp,
                    })
        except Exception:
            pass

    return (attack_samples, block_samples,
            priority_samples, target_samples,
            mulligan_samples, value_samples)


# ── PPO batch computation ────────────────────────────

def compute_ppo_batch(model, head, samples, device,
                      use_amp, clip_eps=0.1):
    """
    Compute PPO loss for a batch of attack/block decisions.

    For each creature, the old policy chose attack (1) or not (0).
    We compute the new policy's log prob for that same action,
    then apply the PPO clipped objective.
    """
    if not samples:
        return torch.tensor(0.0, device=device), {}, 0

    max_c = max(s['n_creatures'] for s in samples)
    max_c = max(max_c, 1)
    bs = len(samples)

    cd = CARD_DIM
    cf = torch.zeros(bs, max_c, cd, device=device)
    cm = torch.zeros(bs, max_c, dtype=torch.bool,
                      device=device)
    actions = torch.zeros(bs, max_c, device=device)
    outcomes = torch.zeros(bs, device=device)
    gae_advantages = torch.zeros(bs, device=device)
    old_log_probs = torch.zeros(bs, device=device)
    gf = torch.zeros(bs, GLOBAL_DIM, device=device)

    # Zone tensors for encoder
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

    for i, s in enumerate(samples):
        nc = s['n_creatures']
        cf[i, :nc] = torch.from_numpy(
            s['creature_features'])
        cm[i, :nc] = True
        actions[i, :nc] = torch.from_numpy(
            s['action_mask'])
        outcomes[i] = float(s['outcome'])
        gae_advantages[i] = float(s.get('advantage', s['outcome']))
        old_log_probs[i] = float(s.get('old_log_prob', 0.0))

        g, zones, masks_d = parse_game_state(
            s['game_state_flat'], s['global_features'])
        gf[i] = torch.from_numpy(g)
        mb[i] = torch.from_numpy(zones['my_board'])
        mbm[i] = torch.from_numpy(
            masks_d['my_board_mask'])
        ob[i] = torch.from_numpy(zones['opp_board'])
        obm[i] = torch.from_numpy(
            masks_d['opp_board_mask'])
        h[i] = torch.from_numpy(zones['hand'])
        hm[i] = torch.from_numpy(masks_d['hand_mask'])
        mg[i] = torch.from_numpy(zones['my_gy'])
        mgm[i] = torch.from_numpy(
            masks_d['my_gy_mask'])
        og[i] = torch.from_numpy(zones['opp_gy'])
        ogm[i] = torch.from_numpy(
            masks_d['opp_gy_mask'])
        st[i] = torch.from_numpy(zones['stack'])
        stm[i] = torch.from_numpy(
            masks_d['stack_mask'])

    with torch.amp.autocast('cuda', enabled=use_amp):
        # Encode state
        state = model.encode_state(
            gf, mb, mbm, ob, obm, h, hm,
            mg, mgm, og, ogm, st, stm)

        # Value estimate (critic) — detach state so value
        # gradients don't flow through shared encoder (PPG)
        value = model.get_value(
            state.detach()).squeeze(-1)

        # Policy logits (actor) — gradients DO flow to encoder
        logits = head(state, cf, cm)

        # Compute log probs for chosen actions
        safe_logits = logits.clone()
        safe_logits[~cm] = 0.0

        log_probs = (
            F.logsigmoid(safe_logits) * actions +
            F.logsigmoid(-safe_logits) * (1 - actions))
        log_probs = (log_probs * cm.float()).sum(dim=1)

        # Use pre-computed GAE advantages
        advantage = gae_advantages
        if advantage.numel() > 1:
            advantage = (advantage - advantage.mean()) / \
                (advantage.std() + 1e-8)

        # PPO clipped objective
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - clip_eps,
                            1.0 + clip_eps) * advantage
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss (encoder detached — only updates
        # value network MLP weights)
        value_loss = F.mse_loss(value, outcomes)

        # Entropy bonus
        probs = torch.sigmoid(safe_logits)
        entropy = -(
            probs * F.logsigmoid(safe_logits) +
            (1 - probs) * F.logsigmoid(-safe_logits))
        entropy = (entropy * cm.float()).sum(dim=1).mean()

        total_loss = (
            policy_loss +
            0.5 * value_loss -
            0.03 * entropy)  # attack: high exploration ok

    metrics = {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'entropy': entropy.item(),
        'mean_advantage': advantage.mean().item(),
        'mean_value': value.mean().item(),
        'win_rate': (outcomes > 0).float().mean().item(),
    }
    return total_loss, metrics, bs


def compute_ppo_block_batch(model, samples, device,
                            use_amp, clip_eps=0.1):
    """
    Compute PPO loss for block decisions using the proper BlockHead.

    Block candidates are concatenated (blocker, attacker) pairs.
    We reconstruct separate blocker/attacker tensors and compute
    per-blocker categorical PPO loss.
    """
    if not samples:
        return torch.tensor(0.0, device=device), {}, 0

    cd = CARD_DIM
    bs = len(samples)

    # Parse each sample to extract blocker/attacker structure
    parsed = []
    for s in samples:
        creatures = s['creature_features']  # (n_pairs, card_dim*2) or (n_pairs, card_dim)
        n_total = s['n_creatures']
        action_mask = s['action_mask']  # which pairs are active

        # Candidates are (blocker||attacker) pairs, last is "no block" zero vector
        real_pairs = n_total - 1 if n_total > 1 else n_total
        if real_pairs <= 0:
            continue

        # Infer n_attackers from pair structure
        first_blocker = creatures[0, :cd]
        n_attackers = 1
        for j in range(1, real_pairs):
            if np.allclose(first_blocker, creatures[j, :cd], atol=0.01):
                n_attackers += 1
            else:
                break
        n_blockers = real_pairs // max(n_attackers, 1)
        if n_blockers == 0 or n_attackers == 0:
            continue

        # Extract unique blocker and attacker features
        bf = np.zeros((n_blockers, cd), dtype=np.float32)
        af = np.zeros((n_attackers, cd), dtype=np.float32)
        for b in range(n_blockers):
            pidx = b * n_attackers
            if pidx < real_pairs:
                bf[b] = creatures[pidx, :cd]
        for a in range(n_attackers):
            if a < real_pairs and creatures.shape[1] >= cd * 2:
                af[a] = creatures[a, cd:cd*2]
            elif a < real_pairs:
                af[a] = creatures[a, :cd]  # fallback

        # Convert action_mask to per-blocker assignments
        # assignments[b] = attacker index, or n_attackers for "don't block"
        assignments = np.full(n_blockers, n_attackers, dtype=np.int64)
        for pidx in range(real_pairs):
            if action_mask[pidx] > 0.5:
                b = pidx // n_attackers
                a = pidx % n_attackers
                if b < n_blockers:
                    assignments[b] = a

        parsed.append({
            'blocker_features': bf,
            'attacker_features': af,
            'n_blockers': n_blockers,
            'n_attackers': n_attackers,
            'assignments': assignments,
            'outcome': s['outcome'],
            'advantage': s.get('advantage', s['outcome']),
            'old_log_prob': s.get('old_log_prob', 0.0),
            'global_features': s['global_features'],
            'game_state_flat': s['game_state_flat'],
        })

    if not parsed:
        return torch.tensor(0.0, device=device), {}, 0

    bs = len(parsed)
    max_b = max(p['n_blockers'] for p in parsed)
    max_a = max(p['n_attackers'] for p in parsed)

    bf_t = torch.zeros(bs, max_b, cd, device=device)
    bm_t = torch.zeros(bs, max_b, dtype=torch.bool, device=device)
    af_t = torch.zeros(bs, max_a, cd, device=device)
    am_t = torch.zeros(bs, max_a, dtype=torch.bool, device=device)
    assign_t = torch.full((bs, max_b), max_a, dtype=torch.long, device=device)
    outcomes = torch.zeros(bs, device=device)
    gae_advantages_b = torch.zeros(bs, device=device)
    old_log_probs = torch.zeros(bs, device=device)

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
    st_t = torch.zeros(bs, 10, cd, device=device)
    stm = torch.zeros(bs, 10, dtype=torch.bool, device=device)

    for i, p in enumerate(parsed):
        nb, na = p['n_blockers'], p['n_attackers']
        bf_t[i, :nb] = torch.from_numpy(p['blocker_features'])
        bm_t[i, :nb] = True
        af_t[i, :na] = torch.from_numpy(p['attacker_features'])
        am_t[i, :na] = True
        assign_t[i, :nb] = torch.from_numpy(p['assignments'])
        outcomes[i] = float(p['outcome'])
        gae_advantages_b[i] = float(p.get('advantage', p['outcome']))
        old_log_probs[i] = float(p['old_log_prob'])

        g, zones, masks_d = parse_game_state(
            p['game_state_flat'], p['global_features'])
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
        st_t[i] = torch.from_numpy(zones['stack'])
        stm[i] = torch.from_numpy(masks_d['stack_mask'])

    with torch.amp.autocast('cuda', enabled=use_amp):
        state = model.encode_state(
            gf, mb, mbm, ob, obm, h, hm,
            mg, mgm, og, ogm, st_t, stm)

        value = model.get_value(state.detach()).squeeze(-1)

        # BlockHead: (bs, max_b, max_a + 1)
        logits = model.block_head(state, bf_t, bm_t, af_t, am_t)

        # Per-blocker categorical log prob for the chosen assignment
        total_log_prob = torch.zeros(bs, device=device)
        total_entropy = torch.zeros(bs, device=device)
        for b in range(max_b):
            dist = torch.distributions.Categorical(logits=logits[:, b, :])
            lp = dist.log_prob(assign_t[:, b])
            total_log_prob += lp * bm_t[:, b].float()
            total_entropy += dist.entropy() * bm_t[:, b].float()

        advantage = gae_advantages_b
        if advantage.numel() > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        ratio = torch.exp(total_log_prob - old_log_probs)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(value, outcomes)
        entropy = total_entropy.mean()

        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy  # block: moderate

    metrics = {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'entropy': entropy.item(),
        'mean_advantage': advantage.mean().item(),
        'mean_value': value.mean().item(),
        'win_rate': (outcomes > 0).float().mean().item(),
    }
    return total_loss, metrics, bs


def compute_ppo_priority_batch(model, head, samples,
                               device, use_amp,
                               clip_eps=0.1):
    """
    Compute PPO loss for a batch of priority decisions.

    Uses Categorical distribution (single-select softmax)
    instead of Bernoulli (binary per-creature).
    """
    if not samples:
        return torch.tensor(0.0, device=device), {}, 0

    max_a = max(s['n_actions'] for s in samples)
    max_a = max(max_a, 1)
    bs = len(samples)

    cd = CARD_DIM
    af = torch.zeros(bs, max_a, 64, device=device)
    am = torch.zeros(bs, max_a, dtype=torch.bool,
                      device=device)
    actions = torch.zeros(bs, dtype=torch.long,
                          device=device)
    outcomes = torch.zeros(bs, device=device)
    gae_advantages = torch.zeros(bs, device=device)
    old_log_probs = torch.zeros(bs, device=device)
    gf = torch.zeros(bs, GLOBAL_DIM, device=device)

    # Zone tensors for encoder
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

    for i, s in enumerate(samples):
        na = s['n_actions']
        af[i, :na] = torch.from_numpy(
            s['action_features'])
        am[i, :na] = True
        actions[i] = s['selected_idx']
        outcomes[i] = float(s['outcome'])
        gae_advantages[i] = float(s.get('advantage', s['outcome']))
        old_log_probs[i] = float(s.get('old_log_prob', 0.0))

        g, zones, masks_d = parse_game_state(
            s['game_state_flat'], s['global_features'])
        gf[i] = torch.from_numpy(g)
        mb[i] = torch.from_numpy(zones['my_board'])
        mbm[i] = torch.from_numpy(
            masks_d['my_board_mask'])
        ob[i] = torch.from_numpy(zones['opp_board'])
        obm[i] = torch.from_numpy(
            masks_d['opp_board_mask'])
        h[i] = torch.from_numpy(zones['hand'])
        hm[i] = torch.from_numpy(masks_d['hand_mask'])
        mg[i] = torch.from_numpy(zones['my_gy'])
        mgm[i] = torch.from_numpy(
            masks_d['my_gy_mask'])
        og[i] = torch.from_numpy(zones['opp_gy'])
        ogm[i] = torch.from_numpy(
            masks_d['opp_gy_mask'])
        st[i] = torch.from_numpy(zones['stack'])
        stm[i] = torch.from_numpy(
            masks_d['stack_mask'])

    with torch.amp.autocast('cuda', enabled=use_amp):
        # Encode state
        state = model.encode_state(
            gf, mb, mbm, ob, obm, h, hm,
            mg, mgm, og, ogm, st, stm)

        # Value estimate (critic) — detach state so value
        # gradients don't flow through shared encoder (PPG)
        value = model.get_value(
            state.detach()).squeeze(-1)

        # Policy logits (actor) — gradients DO flow to encoder
        logits = head(state, af, am)

        # Categorical distribution over actions
        dist = torch.distributions.Categorical(
            logits=logits)
        log_probs = dist.log_prob(actions)

        # Use pre-computed GAE advantages
        advantage = gae_advantages
        if advantage.numel() > 1:
            advantage = (advantage - advantage.mean()) / \
                (advantage.std() + 1e-8)

        # PPO clipped objective
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - clip_eps,
                            1.0 + clip_eps) * advantage
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss — train to predict GAE returns
        value_loss = F.mse_loss(value, outcomes)

        # Entropy bonus
        entropy = dist.entropy().mean()

        total_loss = (
            policy_loss +
            0.5 * value_loss -
            0.01 * entropy)  # priority: mild exploration

    metrics = {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'entropy': entropy.item(),
        'mean_advantage': advantage.mean().item(),
        'mean_value': value.mean().item(),
        'win_rate': (outcomes > 0).float().mean().item(),
    }
    return total_loss, metrics, bs


def compute_ppo_target_batch(model, samples, device,
                              use_amp, clip_eps=0.1):
    """PPO loss for target selection (single-select softmax,
    same as priority but with 256-dim card features)."""
    bs = len(samples)
    max_t = max(s['n_targets'] for s in samples)
    max_t = max(max_t, 1)
    cd = CARD_DIM

    tf = torch.zeros(bs, max_t, cd, device=device)
    tm = torch.zeros(bs, max_t, dtype=torch.bool,
                      device=device)
    actions = torch.zeros(bs, dtype=torch.long,
                           device=device)
    outcomes = torch.zeros(bs, device=device)
    gae_advantages = torch.zeros(bs, device=device)
    old_log_probs = torch.zeros(bs, device=device)

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

    sf = torch.zeros(bs, 64, device=device)

    for i, s in enumerate(samples):
        nt = s['n_targets']
        tf[i, :nt] = torch.from_numpy(s['target_features'])
        tm[i, :nt] = True
        actions[i] = s['selected_idx']
        outcomes[i] = float(s['outcome'])
        gae_advantages[i] = float(s['advantage'])
        old_log_probs[i] = float(s['old_log_prob'])
        spell = s.get('spell_features')
        if spell is not None:
            sf[i] = torch.from_numpy(spell)

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
        value = model.get_value(state.detach()).squeeze(-1)
        logits = model.target_head(state, tf, tm,
                                    spell_features=sf)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)

        advantage = gae_advantages
        if advantage.numel() > 1:
            advantage = (advantage - advantage.mean()) / \
                (advantage.std() + 1e-8)

        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - clip_eps,
                            1.0 + clip_eps) * advantage
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(value, outcomes)
        entropy = dist.entropy().mean()
        total_loss = policy_loss + 0.5 * value_loss - 0.005 * entropy

    metrics = {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'entropy': entropy.item(),
    }
    return total_loss, metrics, bs


def compute_ppo_mulligan_batch(model, samples, device,
                                use_amp, clip_eps=0.1):
    """PPO loss for mulligan (binary keep/mull)."""
    bs = len(samples)
    max_h = max(s['n_cards'] for s in samples)
    max_h = max(max_h, 1)
    cd = CARD_DIM

    hf = torch.zeros(bs, max_h, cd, device=device)
    hmask = torch.zeros(bs, max_h, dtype=torch.bool,
                         device=device)
    kept = torch.zeros(bs, device=device)
    outcomes = torch.zeros(bs, device=device)
    gae_advantages = torch.zeros(bs, device=device)
    old_log_probs = torch.zeros(bs, device=device)

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

    for i, s in enumerate(samples):
        nc = s['n_cards']
        hf[i, :nc] = torch.from_numpy(s['hand_features'])
        hmask[i, :nc] = True
        kept[i] = s['kept']
        outcomes[i] = float(s['outcome'])
        gae_advantages[i] = float(s['advantage'])
        old_log_probs[i] = float(s['old_log_prob'])

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
        value = model.get_value(state.detach()).squeeze(-1)

        # Mulligan head returns keep logit
        keep_logit = model.mulligan_head(state, hf, hmask)
        keep_prob = torch.sigmoid(keep_logit)

        # Log prob of the taken action
        log_probs = torch.where(
            kept > 0.5,
            torch.log(keep_prob.clamp(min=1e-8)),
            torch.log((1 - keep_prob).clamp(min=1e-8)))

        advantage = gae_advantages
        if advantage.numel() > 1:
            advantage = (advantage - advantage.mean()) / \
                (advantage.std() + 1e-8)

        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - clip_eps,
                            1.0 + clip_eps) * advantage
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(value, outcomes)
        entropy = -(keep_prob * torch.log(keep_prob.clamp(min=1e-8))
                     + (1-keep_prob) * torch.log(
                         (1-keep_prob).clamp(min=1e-8))).mean()
        total_loss = policy_loss + 0.5 * value_loss - 0.005 * entropy

    metrics = {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'entropy': entropy.item(),
    }
    return total_loss, metrics, bs


# ── Game runner subprocess ───────────────────────────

class ModelServerError(Exception):
    """Raised when Java reports the model server is down."""
    pass


def run_games(n_games, traj_dir, mode='evaluate',
              port=50051, quiet=True,
              progress_callback=None,
              log_callback=None,
              threads=4,
              java_procs=1,
              decks=None):
    """Run games via Java subprocess(es).
    Raises ModelServerError if the server is detected as down.
    progress_callback(completed, total) called as games complete.
    log_callback(line) called for each stdout line from Java.
    java_procs: number of separate Java processes to split games across."""
    os.makedirs(traj_dir, exist_ok=True)
    # Clean old trajectories
    for f in Path(traj_dir).glob('traj_*.jsonl'):
        f.unlink()

    if java_procs > 1:
        return _run_games_multi(
            n_games, traj_dir, mode, port, quiet,
            progress_callback, log_callback, threads,
            java_procs, decks)

    return _run_games_single(
        n_games, traj_dir, mode, port, quiet,
        progress_callback, log_callback, threads, decks)

def run_league_games(n_games, traj_dir, current_ports,
                     opponent_ports, threads=16,
                     progress_callback=None,
                     log_callback=None,
                     clean_traj=False):
    """Run league play games: current model vs opponent model.
    current_ports: list of ports serving current model
    opponent_ports: list of ports serving opponent model
    Returns (win_rate, stdout_lines)."""
    os.makedirs(traj_dir, exist_ok=True)
    if clean_traj:
        for f in Path(traj_dir).glob('traj_*.jsonl'):
            f.unlink()

    if isinstance(current_ports, (list, tuple)):
        port_str = ','.join(str(p) for p in current_ports)
    else:
        port_str = str(current_ports)
    if isinstance(opponent_ports, (list, tuple)):
        port2_str = ','.join(str(p) for p in opponent_ports)
    else:
        port2_str = str(opponent_ports)

    cmd = _build_java_cmd(n_games, traj_dir, 'leagueplay',
                          port_str, threads, heap='12g',
                          port2_str=port2_str)
    cwd = os.path.join(PROJECT_ROOT, 'forge-gui-desktop')

    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True)

    stdout_lines = []
    try:
        for line in proc.stdout:
            stdout_lines.append(line)
            if log_callback:
                log_callback(line.rstrip())
            if progress_callback and 'Game ' in line:
                try:
                    parts = line.split('Game ')[1].split('/')
                    done = int(parts[0])
                    progress_callback(done, n_games)
                except (IndexError, ValueError):
                    pass
    except Exception:
        pass

    proc.wait()

    # Parse win rate from output
    win_rate = None
    for line in reversed(stdout_lines):
        if 'current win rate:' in line.lower():
            try:
                pct = line.split('win rate:')[1].split('%')[0]
                win_rate = float(pct.strip()) / 100.0
            except (IndexError, ValueError):
                pass
            break

    if win_rate is None:
        # Count from trajectory files
        wins = losses = 0
        for f in Path(traj_dir).glob('traj_*.jsonl'):
            try:
                with open(f) as fh:
                    header = json.loads(fh.readline())
                if header.get('won', False):
                    wins += 1
                else:
                    losses += 1
            except Exception:
                pass
        total = wins + losses
        win_rate = wins / total if total > 0 else 0.0

    return win_rate, stdout_lines

def _build_java_cmd(n_games, traj_dir, mode, port_str,
                    threads, heap='3g', port2_str=None, decks=None):
    """Build the Java command for a single process."""
    decks = decks or load_rl_decks()
    deck_args = []
    for d in decks:
        deck_args.extend(['-d', d])

    cmd = [
        'java', f'-Xmx{heap}',
        '-XX:+UseG1GC',
        '-XX:MaxGCPauseMillis=50',
        '-XX:ParallelGCThreads=2',
        '--add-opens', 'java.base/java.lang=ALL-UNNAMED',
        '--add-opens', 'java.base/java.util=ALL-UNNAMED',
        '--add-opens', 'java.base/java.text=ALL-UNNAMED',
        '--add-opens',
        'java.base/java.lang.reflect=ALL-UNNAMED',
        '--add-opens',
        'java.desktop/javax.imageio.spi=ALL-UNNAMED',
        '-jar', resolve_forge_jar(),
        'rltrain', mode,
    ] + deck_args + [
        '-n', str(n_games),
        '-t', str(threads),
        '-o', traj_dir,
        '-host', 'localhost',
        '-port', port_str,
    ]
    if port2_str:
        cmd.extend(['-port2', port2_str])
    return cmd


def _run_games_single(n_games, traj_dir, mode, port,
                      quiet, progress_callback,
                      log_callback, threads, decks):
    """Run games in a single Java process."""
    if isinstance(port, (list, tuple)):
        port_str = ','.join(str(p) for p in port)
    else:
        port_str = str(port)

    cmd = _build_java_cmd(n_games, traj_dir, mode,
                          port_str, threads, heap='12g',
                          decks=decks)
    cwd = os.path.join(PROJECT_ROOT, 'forge-gui-desktop')

    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True)

    stdout_lines = []
    try:
        for line in proc.stdout:
            stdout_lines.append(line)
            if log_callback:
                log_callback(line.rstrip())
            if progress_callback and 'Game ' in line:
                try:
                    parts = line.split('Game ')[1].split('/')
                    done = int(parts[0])
                    progress_callback(done, n_games)
                except (IndexError, ValueError):
                    pass

        proc.wait(timeout=600)
    except subprocess.TimeoutExpired:
        proc.kill()

    stdout = ''.join(stdout_lines)
    return _parse_game_results(stdout_lines, stdout,
                               n_games)


def _run_games_multi(n_games, traj_dir, mode, port,
                     quiet, progress_callback,
                     log_callback, threads, java_procs,
                     decks):
    """Run games across multiple Java processes."""
    if isinstance(port, (list, tuple)):
        port_str = ','.join(str(p) for p in port)
    else:
        port_str = str(port)

    # Split games and threads across processes
    games_per = n_games // java_procs
    threads_per = max(1, threads // java_procs)
    remainder = n_games - games_per * java_procs
    # Memory per process — 2GB is sufficient (actual RSS ~1.5GB)
    heap_str = '2g'

    cwd = os.path.join(PROJECT_ROOT, 'forge-gui-desktop')
    procs = []
    for i in range(java_procs):
        ng = games_per + (1 if i < remainder else 0)
        if ng == 0:
            continue
        cmd = _build_java_cmd(ng, traj_dir, mode,
                              port_str, threads_per,
                              heap=heap_str, decks=decks)
        proc = subprocess.Popen(
            cmd, cwd=cwd, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True)
        procs.append((proc, ng))

    if log_callback:
        log_callback(f"  Launched {len(procs)} Java "
                     f"processes ({games_per} games, "
                     f"{threads_per} threads, "
                     f"{heap_str} heap each)")

    # Monitor all processes via file counting
    completed = [0]
    all_stdout = [[] for _ in procs]

    def _read_proc(idx, proc):
        for line in proc.stdout:
            all_stdout[idx].append(line)
            if log_callback:
                log_callback(f"  [java:{idx}] "
                             f"{line.rstrip()}")
            if 'Game ' in line:
                try:
                    parts = (line.split('Game ')[1]
                             .split('/'))
                    done = int(parts[0])
                    # Approximate total progress
                    completed[0] += 10  # each report = 10
                    if progress_callback:
                        progress_callback(
                            min(completed[0], n_games),
                            n_games)
                except (IndexError, ValueError):
                    pass

    reader_threads = []
    for idx, (proc, _) in enumerate(procs):
        t = threading.Thread(target=_read_proc,
                             args=(idx, proc),
                             daemon=True)
        t.start()
        reader_threads.append(t)

    # Wait for all processes
    for proc, _ in procs:
        try:
            proc.wait(timeout=900)
        except subprocess.TimeoutExpired:
            proc.kill()

    for t in reader_threads:
        t.join(timeout=5)

    # Aggregate results
    all_lines = []
    for lines in all_stdout:
        all_lines.extend(lines)
    all_text = ''.join(all_lines)

    return _parse_game_results(all_lines, all_text,
                               n_games)


def _parse_game_results(stdout_lines, stdout, n_games):
    """Parse win rate and errors from game output."""
    if 'ABORT' in stdout:
        raise ModelServerError(
            "Model server is down — Java aborted the run. "
            "Check server logs.")

    # Aggregate win rates across all processes
    total_rl_wins = 0
    total_heuristic_wins = 0
    server_errors = 0
    for line in stdout_lines:
        if 'RL Wins:' in line:
            try:
                wins = int(line.split('RL Wins:')[1]
                           .strip().split()[0])
                total_rl_wins += wins
            except (IndexError, ValueError):
                pass
        # Not elif — RL Wins and Heuristic Wins are on same line
        if 'Heuristic Wins:' in line:
            try:
                hw = int(line.split('Heuristic Wins:')[1]
                         .strip().split()[0])
                total_heuristic_wins += hw
            except (IndexError, ValueError):
                pass
        if 'MODEL_SERVER_ERROR' in line:
            server_errors += 1

    total_games = total_rl_wins + total_heuristic_wins
    win_rate = (total_rl_wins / total_games
                if total_games > 0 else None)

    if server_errors > 0:
        print(f"  WARNING: {server_errors} model server "
              f"errors during run", flush=True)

    return win_rate, stdout


# ── Model server management ──────────────────────────

def start_model_server(model, device, port=50051,
                       use_argmax=False):
    """Start model server in background thread."""
    server = ModelServer(model, port=port, device=device,
                         use_argmax=use_argmax)
    thread = threading.Thread(
        target=server.start, daemon=True)
    thread.start()
    time.sleep(1)  # Wait for server to bind
    # Verify server is actually listening
    if not check_server_health(port):
        raise RuntimeError(
            f"Model server failed to start on port {port}")
    return server


def check_server_health(port, host='localhost'):
    """Verify the model server is reachable and responding."""
    import struct
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect((host, port))
        # Send a minimal ping request
        ping = json.dumps({
            'decisionType': 'PING',
            'globalFeatures': [],
            'candidateFeatures': [],
        }).encode('utf-8')
        sock.sendall(struct.pack('>I', len(ping)))
        sock.sendall(ping)
        # Read response
        length_bytes = sock.recv(4)
        if len(length_bytes) == 4:
            resp_len = struct.unpack('>I', length_bytes)[0]
            resp = sock.recv(resp_len)
            sock.close()
            return True
        sock.close()
        return False
    except Exception:
        return False


def find_free_port():
    """Find a free TCP port."""
    with socket.socket() as s:
        s.bind(('', 0))
        return s.getsockname()[1]


# ── Main PPO loop ────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='PPO Self-Play Training')
    parser.add_argument('--checkpoint',
        default=os.path.join(
            get_checkpoint_dir(),
            'model_with_decisions.pt'))
    parser.add_argument('--save-dir',
        default=get_checkpoint_dir())
    parser.add_argument('--traj-dir',
        default=get_ppo_traj_dir())
    parser.add_argument('--device', default=get_device_override())
    parser.add_argument('--rounds', type=int, default=20,
        help='Number of collect→train rounds')
    parser.add_argument('--games-per-round', type=int,
        default=200)
    parser.add_argument('--ppo-epochs', type=int,
        default=4,
        help='PPO update epochs per round')
    parser.add_argument('--batch-size', type=int,
        default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--eval-games', type=int,
        default=50,
        help='Evaluation games per round')
    parser.add_argument('--port', type=int, default=0,
        help='Model server port (0=auto)')
    parser.add_argument('--reward-shaping-coeff', type=float, default=0.0,
        help='Initial coefficient for intermediate reward shaping (0=disabled)')
    parser.add_argument('--reward-shaping-decay', type=float, default=0.95,
        help='Per-round multiplicative decay for shaping coefficient')
    parser.add_argument('--deck', action='append',
        help='Override the configured RL deck list '
             '(may be repeated)')
    args = parser.parse_args()

    profile = auto_detect_profile(args.device)
    backend = resolve_backend(args.device)
    device = backend.torch_device
    use_amp = profile.use_amp and backend.use_amp
    port = args.port or find_free_port()
    decks = load_rl_decks(overrides=args.deck)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.traj_dir, exist_ok=True)

    print('┌────────────────────────────────────────┐',
          flush=True)
    print('│     MTG RL — PPO Self-Play Training    │',
          flush=True)
    print('└────────────────────────────────────────┘',
          flush=True)
    print(f'  Device: {backend.name} ({profile.name})',
          flush=True)
    print(f'  Rounds: {args.rounds}', flush=True)
    print(f'  Games/round: {args.games_per_round}',
          flush=True)
    print(f'  PPO epochs: {args.ppo_epochs}', flush=True)
    print(f'  Eval games: {args.eval_games}', flush=True)
    print(f'  Server port: {port}', flush=True)
    print(f"  Decks: {', '.join(decks)}", flush=True)

    # Load model
    print(f'\n  Loading model: {args.checkpoint}',
          flush=True)
    if os.path.exists(args.checkpoint):
        model = MTGModel.load(
            args.checkpoint, device=backend.name)
    else:
        print('  No checkpoint, using random init',
              flush=True)
        model = MTGModel().to(device)

    # Unfreeze everything for PPO
    for p in model.parameters():
        p.requires_grad = True

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # Start model server
    print(f'  Starting model server on port {port}...',
          flush=True)
    server = start_model_server(model, device, port)

    best_win_rate = 0.0
    history = []

    print(flush=True)
    print('  Round │ Games │ Attacks │ Blocks │'
          ' Priority │ Policy Loss │ Value Loss │'
          ' Entropy │ Eval WR │ Status', flush=True)
    print('  ──────┼───────┼─────────┼────────┼'
          '──────────┼─────────────┼────────────┼'
          '─────────┼─────────┼───────', flush=True)

    shaping_coeff = args.reward_shaping_coeff

    for rnd in range(1, args.rounds + 1):
        t0 = time.time()

        # ── Step 1: Collect games with RL agent ──
        # Use 'evaluate' mode so RL plays vs heuristic
        # (captures RL's combat decisions)
        try:
            _, stdout = run_games(
                args.games_per_round, args.traj_dir,
                mode='evaluate', port=port,
                decks=decks)
        except ModelServerError as e:
            print(f'\n  FATAL: {e}', flush=True)
            print('  Stopping PPO — model server is down.',
                  flush=True)
            break

        # ── Step 2: Load trajectories ──
        attack_data, block_data, priority_data, \
            value_data = load_ppo_data(args.traj_dir,
                shaping_coeff=shaping_coeff)

        if not attack_data and not block_data \
                and not priority_data:
            print(f'  {rnd:>4d}   │ {args.games_per_round:>5d} │'
                  f'    0    │   0    │'
                  f'     0    │'
                  f' no data     │            │'
                  f'         │         │ SKIP',
                  flush=True)
            continue

        # ── Step 3: PPO updates ──
        model.train()
        total_pl, total_vl, total_ent = 0, 0, 0
        n_updates = 0

        for ppo_epoch in range(args.ppo_epochs):
            # Attack head updates
            random.shuffle(attack_data)
            for bi in range(0, len(attack_data),
                            args.batch_size):
                batch = attack_data[
                    bi:bi + args.batch_size]
                if len(batch) < 2:
                    continue
                loss, metrics, _ = compute_ppo_batch(
                    model, model.attack_head, batch,
                    device, use_amp)

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

            # Block head updates
            random.shuffle(block_data)
            for bi in range(0, len(block_data),
                            args.batch_size):
                batch = block_data[
                    bi:bi + args.batch_size]
                if len(batch) < 2:
                    continue
                loss, metrics, _ = compute_ppo_batch(
                    model, model.block_head, batch,
                    device, use_amp)

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

            # Priority head updates (Categorical, not Bernoulli)
            random.shuffle(priority_data)
            for bi in range(0, len(priority_data),
                            args.batch_size):
                batch = priority_data[
                    bi:bi + args.batch_size]
                if len(batch) < 2:
                    continue
                loss, metrics, _ = \
                    compute_ppo_priority_batch(
                        model, model.priority_head,
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

        avg_pl = total_pl / max(n_updates, 1)
        avg_vl = total_vl / max(n_updates, 1)
        avg_ent = total_ent / max(n_updates, 1)

        # ── Step 4: Evaluate vs heuristic ──
        model.eval()
        try:
            eval_wr, _ = run_games(
                args.eval_games, args.traj_dir + '_eval',
                mode='evaluate', port=port,
                decks=decks)
            eval_wr = eval_wr or 0.0
        except ModelServerError as e:
            print(f'\n  FATAL: {e}', flush=True)
            print('  Stopping PPO — model server down '
                  'during eval.', flush=True)
            break

        # Save
        status = ''
        if eval_wr > best_win_rate:
            best_win_rate = eval_wr
            model.save(os.path.join(
                args.save_dir, 'best_ppo_model.pt'))
            status = '★ best'
        if rnd % 5 == 0:
            model.save(os.path.join(
                args.save_dir,
                f'ppo_model_round_{rnd}.pt'))
            if not status:
                status = 'saved'

        elapsed = time.time() - t0
        history.append({
            'round': rnd,
            'eval_wr': eval_wr,
            'policy_loss': avg_pl,
            'value_loss': avg_vl,
        })

        print(
            f'  {rnd:>4d}   │ {args.games_per_round:>5d} │'
            f' {len(attack_data):>7d} │ {len(block_data):>6d} │'
            f' {len(priority_data):>8d} │'
            f' {avg_pl:>11.4f} │ {avg_vl:>10.4f} │'
            f' {avg_ent:>7.3f} │'
            f' {eval_wr:>6.1%} │ {status}'
            f'  ({elapsed:.0f}s)',
            flush=True)

        # Decay reward shaping
        if shaping_coeff > 0:
            shaping_coeff *= args.reward_shaping_decay
            print(f'  Reward shaping coeff: {shaping_coeff:.4f}',
                  flush=True)

    # Final summary
    print(flush=True)
    print('  ╔════════════════════════════════╗',
          flush=True)
    print(f'  ║ Best win rate: {best_win_rate:>6.1%}'
          f'          ║', flush=True)
    print(f'  ║ Total rounds: {args.rounds:>3d}'
          f'             ║', flush=True)
    print('  ╚════════════════════════════════╝',
          flush=True)

    # Save final
    model.save(os.path.join(
        args.save_dir, 'ppo_model_final.pt'))

    # Write history
    with open(os.path.join(
            args.save_dir, 'ppo_history.json'), 'w') as f:
        json.dump(history, f, indent=2)


if __name__ == '__main__':
    main()
