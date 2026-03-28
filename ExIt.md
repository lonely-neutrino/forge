# Expert Iteration (ExIt) for Forge MTG RL

## Context

PPO has plateaued at 30-39% win rate after multiple attempts with different hyperparameters. The core problem: stochastic sampling degrades play quality, and the sparse terminal reward signal can't provide per-decision credit assignment at our compute scale. MageZero demonstrated that MCTS-guided training (AlphaZero-style) can reach 66% win rate on MTG.

Expert Iteration (ExIt) sidesteps PPO's exploration problem entirely: instead of learning from random play, use **search** to find better moves than the current policy, then **train supervised** on the search-improved decisions. The model learns from search-quality play without needing to play randomly.

## How ExIt Works

```
Repeat:
  1. Play games using MCTS at each decision point
     - For each decision: try N options via game simulation
     - Pick the option that leads to the best outcome (by value network or game rollout)
     - Record the search-improved decision as training data
  2. Train the model supervised on MCTS-improved decisions
     - The model learns to predict what MCTS would choose
     - Policy network improves → MCTS gets better guidance → better training data
  3. Evaluate vs heuristic to track improvement
```

## Existing Infrastructure

**Already built (Forge engine):**
- `GameCopier.java` — full game state deep copy (all zones, players, combat)
- `GameSimulator.java` — plays a SpellAbility forward and scores the result
- `SimulationController.java` — recursive depth-limited lookahead (currently MAX_DEPTH=3)
- `GameStateEvaluator.java` — heuristic position scoring
- `GameStateEncoder.java` — works on any Game object, not just current

**Already built (RL infrastructure):**
- `RLController.java` — clean decision intercept points for all 5 types
- `requestDecision()` — single routing point that can be redirected to MCTS
- Model server with batched inference (~200 inferences/sec on GPU)
- Joint training pipeline with unfrozen encoder

## Implementation Plan

### Phase 1: Lightweight Search at Decision Points (Java)

Create `MCTSDecisionMaker.java` in `forge-ai-rl/src/main/java/forge/ai/rl/`:

**For each decision point (priority, attack, target):**
1. Copy the game state via `GameCopier`
2. For each candidate action (up to N candidates):
   - Apply the action to a copy
   - Roll out the rest of the game using heuristic AI (fast, ~10ms/game)
   - Record the outcome (win/loss)
3. Repeat M rollouts per candidate to reduce variance
4. Pick the candidate with highest win rate
5. Record as training data: (game_state, candidates, MCTS-chosen action)

**Parameters:**
- `N` = number of candidate actions to evaluate (all available, capped at ~10)
- `M` = rollouts per candidate (start with 5-10)
- Total simulations per decision: N × M = 50-100

**Time budget:** With heuristic rollouts at ~10ms/game, 100 simulations = ~1 second per decision. A 50-decision game takes ~50 seconds. 1,000 games = ~14 hours. Slower than pure heuristic collection but feasible overnight.

### Phase 2: Hybrid Approach — Search Only Where It Matters

Not all decisions need search. Optimize by:

1. **Priority decisions:** Full search (most impactful — which spell to play)
2. **Attack decisions:** Full search (combat outcomes are directly simulatable)
3. **Target decisions:** Full search (targeting errors are costly, search eliminates them)
4. **Block decisions:** Full search (defensive combat is complex)
5. **Mulligan/Binary:** Skip search (model already >98% accurate)

**Further optimization:** Only search when multiple candidates exist and the decision is non-trivial. If there's only 1 creature to attack with, skip search.

### Phase 3: Training Pipeline

**New script: `scripts/09_exit_collect.sh`**
1. Run games with MCTS at decision points
2. Record trajectory JSONL with search-improved choices (same format as current)
3. Output to `rl_data/exit_trajectories/`

**Training uses existing pipeline:**
1. Preprocess MCTS trajectories (same `02b_preprocess_data.sh`)
2. Joint train on MCTS data (same `04_train_decisions.sh --joint`)
3. The model learns to predict what MCTS chose, not what the heuristic chose

**Iterate:**
1. Train on MCTS data → model improves
2. Use improved model's value network to guide future MCTS
3. MCTS makes better decisions → better training data → model improves further

### Phase 4: Use Value Network Instead of Full Rollouts

Initially: full game rollout with heuristic AI to evaluate each candidate.
Later optimization: replace rollouts with the trained value network.

For each candidate action:
1. Copy game state
2. Apply the action
3. Encode the resulting state
4. Query value network for V(s')
5. Pick the candidate with highest V(s')

This is much faster (~5ms per evaluation instead of ~10ms per rollout) and enables deeper search or more candidates.

## Files to Create/Modify

| File | Change |
|------|--------|
| **NEW:** `forge-ai-rl/.../MCTSDecisionMaker.java` | Core MCTS logic: copy state, try candidates, rollout, pick best |
| `RLController.java` | Add MCTS mode that routes decisions to MCTSDecisionMaker |
| `SimulateRLTraining.java` | Add `rltrain mcts-collect` command for MCTS data collection |
| `RLConfig.java` / `RLModelMode` | Add MCTS mode enum |
| **NEW:** `scripts/09_exit_collect.sh` | MCTS data collection script |
| `ppo_ui.py` or new `exit_trainer.py` | ExIt training loop: collect MCTS data → train → repeat |

## Performance Estimates

| Approach | Sims/Decision | Time/Decision | Time/Game | Time/1000 Games |
|----------|--------------|---------------|-----------|-----------------|
| Heuristic rollout (M=10, N=5) | 50 | 0.5s | 25s | 7 hours |
| Heuristic rollout (M=5, N=10) | 50 | 0.5s | 25s | 7 hours |
| Value network eval (N=10) | 10 | 0.05s | 2.5s | 40 min |
| Current PPO collection | 1 | 0.01s | 0.5s | 8 min |

Starting with heuristic rollouts (7 hours for 1,000 games) is practical for overnight runs. Value network evaluation is a later optimization.

## Verification

1. **MCTS data quality:** Check that MCTS-chosen actions differ from policy's argmax in >10% of decisions (otherwise search isn't adding value)
2. **Training improvement:** After training on MCTS data, imitation accuracy on MCTS targets should be >90%
3. **Win rate improvement:** Eval vs heuristic should show improvement over the 30% baseline
4. **Targeting correctness:** MCTS should produce ~0% enchantment/pump mis-targeting (search directly evaluates outcomes)

## Why This Should Work

1. **Search eliminates targeting errors** — MCTS tries "burn own creature" and "burn opponent creature," sees that burning opponent wins more often, and picks correctly. No need for the model to learn polarity from features.

2. **Per-decision credit assignment is automatic** — each decision is evaluated by its direct simulation outcome, not by GAE propagating a terminal reward backward through 50 noisy steps.

3. **No exploration-exploitation tradeoff** — search explores by trying alternatives, not by degrading the policy. The model always plays its best during training.

4. **Compound improvement** — better model → better value network → better MCTS guidance → better training data → better model. This is the AlphaZero virtuous cycle.

## Comparison: PPO vs AWR vs ExIt

| | PPO | AWR | ExIt |
|---|-----|-----|------|
| Exploration | Stochastic sampling (degrades play) | None (argmax only) | Tree search (no degradation) |
| Credit assignment | GAE (noisy over 50 decisions) | GAE (same) | Direct simulation (per-decision) |
| Data quality | Low (random play) | Medium (best current play) | High (search-improved play) |
| Compute per round | Low (800 games, ~20 min) | Low (same) | High (1000 games, ~7 hours) |
| Can discover new strategies? | Yes (random exploration) | No (refines existing) | Yes (search tries alternatives) |
| Risk of degradation | High (round 1 dips) | Low | None |

## MageZero Reference

MageZero (Will Wroble, github.com/WillWroble/MageZero) achieved 66% win rate using AlphaZero-style MCTS on XMage engine. Key differences from our approach:
- 300 MCTS simulations per decision (we'd start with 50-100)
- Sparse feature hashing (2M-dim) vs our dense 37K features
- TD-blended value targets (lambda 0.9-0.95)
- No Dirichlet noise — MTG's randomness provides exploration
- ~250 games/hour throughput
