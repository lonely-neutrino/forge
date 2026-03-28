# Plan: Re-add Reward Shaping to PPO Training

## Context

PPO training currently uses only sparse terminal rewards (+1 win, -1 loss). With MTG games spanning 50+ turns and thousands of micro-decisions, the training signal is weak — especially early when the value network hasn't learned to propagate credit through long games. The Java trajectory recorder already computes and records per-decision intermediate rewards (life/card/board advantage deltas), but `_compute_gae_returns()` explicitly zeroes them out. Re-enabling these with a decaying coefficient will bootstrap learning while the value network is still weak, then fade out so the agent optimizes for pure win rate.

## Approach: Decaying Additive Shaping

Use the **pre-computed `intermediateReward`** already in JSONL trajectories (Java computes life delta * 0.01, card delta * 0.05, board delta * 0.02). These are mathematically equivalent to potential-based shaping (deltas telescope), so they approximately preserve the optimal policy. A per-round exponential decay ensures the shaping fades as the value network improves.

**Formula:** `rewards[t] = shaping_coeff * intermediateReward[t]` where `shaping_coeff = initial * (decay ^ round)`

**Defaults:** `--reward-shaping-coeff 1.0`, `--reward-shaping-decay 0.95` (halves every ~14 rounds, near-zero by round 60)

## Changes

### 1. `ppo_trainer.py` — `_compute_gae_returns()` (~line 69)
- Add `shaping_coeff` parameter (default 0.0 for backward compat)
- When `shaping_coeff > 0`, populate `rewards[i]` from `rec['intermediateReward']` scaled by the coefficient
- Terminal reward still added on final step as before

### 2. `ppo_trainer.py` — `load_ppo_data()` (~line 108)
- Add `shaping_coeff` parameter, pass through to `_compute_gae_returns()`

### 3. `ppo_trainer.py` — `main()` argparse
- Add `--reward-shaping-coeff` (float, default 0.0) and `--reward-shaping-decay` (float, default 0.95)
- In training loop: pass coeff to `load_ppo_data()`, decay after each round, save/restore in state

### 4. `ppo_ui.py` — `main()` argparse
- Add same two CLI args

### 5. `ppo_ui.py` — `ppo_thread()` training loop
- Initialize `shaping_coeff` from args or resumed state
- Pass to `load_ppo_data(traj_dir, shaping_coeff=shaping_coeff)`
- Decay after PPO update: `shaping_coeff *= decay`
- Log current coefficient each round
- Persist in `ppo_training_state.json`

### 6. `scripts/06_ppo_train.sh`
- Add `--reward-shaping-coeff 1.0 --reward-shaping-decay 0.95` to the launch command

## No Changes Needed
- **Java side** — `TrajectoryRecorder.java` already computes and records `intermediateReward`
- **RLConfig.java** — coefficients already defined (0.01, 0.05, 0.02)
- **GAE hyperparameters** — keep gamma=0.99, lambda=0.95
- **PPOState dataclass** — no new UI fields needed, just console logging

## Key Files
- `forge-ai-rl/src/main/python/training/ppo_trainer.py`
- `forge-ai-rl/src/main/python/training/ppo_ui.py`
- `scripts/06_ppo_train.sh`

## Verification
1. Run PPO with `--reward-shaping-coeff 1.0` for a few rounds
2. Check console log shows "Reward shaping coeff: X.XXXX" decaying each round
3. Compare early-round win rates vs a no-shaping baseline (should see faster initial improvement)
4. Verify that after ~40 rounds the coefficient is negligible and the agent still improves on pure win rate
