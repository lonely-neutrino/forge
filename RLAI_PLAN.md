# Reinforcement Learning AI for Forge MTG — Architecture Plan

## Codebase Summary

The Forge codebase is well-structured for this project:

- **`PlayerController`** is the clean abstraction layer — 72+ abstract methods define every decision point (mulligan, cast spell, choose targets, declare attackers/blockers, pay costs, etc.)
- **`PlayerControllerAi`** implements all of these with heuristics via `AiController`
- **`GameSimulator`** already copies full game state for lookahead
- **32,300+ cards** implemented via a text-based scripting DSL
- **204 effect types**, **152 AI ability handlers**, **47 cost types**
- The game is fully deterministic given decisions — perfect for RL

The key insight: **we don't need to touch the game engine at all**. We create a new `PlayerControllerRL` that implements the same interface, and the game engine treats it like any other player.

---

## The Core Challenge

MTG is uniquely hard for RL because:

1. **Massive action space** — at any priority window, dozens of spells/abilities may be legal, each with different targets, modes, costs
2. **Variable-length games** — 5 to 50+ turns with thousands of micro-decisions
3. **Hidden information** — opponent's hand, library order
4. **Enormous card variety** — 32,300 cards with unique rules text
5. **Combinatorial explosion** — card interactions create emergent complexity
6. **Sparse rewards** — you only win or lose, many decisions earlier are what mattered

A single monolithic neural network won't work. We need a **hierarchical, modular architecture**.

---

## Proposed Architecture: Hierarchical RL with Specialized Decision Models

### Layer 1: Game State Encoder (shared foundation)

A single neural network that converts the raw game state into a dense vector representation. All decision models consume this.

**Inputs** (encoded as feature vectors):
- **Board state**: each permanent's power/toughness/keywords/types/tapped status/counters/attachments
- **Hand**: cards in hand with mana costs, types, key abilities
- **Life totals**: both players
- **Mana available**: lands untapped, mana pool contents
- **Graveyard/exile**: key cards and counts
- **Stack**: spells/abilities currently resolving
- **Phase/turn**: current phase, turn number, who's active
- **Combat state**: declared attackers/blockers if in combat
- **Game metadata**: cards in library (count), cards in opponent's hand (count)

**Architecture**: Transformer-based encoder with attention over sets of cards (since board state is a variable-size set of objects). Similar to how DeepMind handles StarCraft units.

**Output**: Fixed-size game state embedding (e.g., 512-1024 dimensions)

### Layer 2: Strategic Value Network

A network that evaluates "how good is this game state for me?" — analogous to the existing `GameStateEvaluator` but learned.

- Takes game state embedding → outputs win probability estimate
- Trained from game outcomes (Monte Carlo returns)
- Used by all decision models as a baseline/critic
- Also used for MCTS-style lookahead when time permits

### Layer 3: Specialized Decision Heads

Rather than one model for all decisions, we use **specialized heads** for each major decision category. Each head shares the game state encoder but has its own policy network.

#### Decision Head 1: **Priority Action Selection** (most critical)
- **When**: Every time the player gets priority
- **Decides**: Play a spell/ability from the available options, or pass priority
- **Architecture**:
  - Encode each available action (spell/ability) as a feature vector
  - Cross-attention between game state and available actions
  - Output: probability distribution over actions + "pass" option
- **Key features per action**: mana cost, card type, effect type (from ApiType), target requirements, whether it's a creature/removal/draw/etc.

#### Decision Head 2: **Target Selection**
- **When**: A spell/ability requires choosing targets
- **Decides**: Which legal target(s) to select
- **Architecture**:
  - Encode each legal target (card or player) as a feature vector
  - Pointer network that selects from the candidate set
  - Handles multi-target by iterative selection

#### Decision Head 3: **Combat — Attackers**
- **When**: Declare attackers step
- **Decides**: Which creatures attack, and which opponent/planeswalker they attack
- **Architecture**:
  - For each creature: binary attack/don't-attack decision
  - Encode creatures as feature vectors with combat-relevant stats
  - Joint decision (not independent per creature — attacking patterns matter)
  - Use combinatorial action representation

#### Decision Head 4: **Combat — Blockers**
- **When**: Declare blockers step
- **Decides**: Which creatures block which attackers
- **Architecture**:
  - Assignment problem: each blocker maps to zero or one attacker
  - Attention-based matching network
  - Must respect blocking restrictions/requirements

#### Decision Head 5: **Card Selection** (general purpose)
- **When**: Choose cards for effects (discard, sacrifice, scry, etc.)
- **Decides**: Which card(s) to select from a set
- **Architecture**:
  - Pointer network over candidate cards
  - Context includes the reason for selection (effect type)
  - Handles variable min/max selection counts

#### Decision Head 6: **Mulligan & Game Start**
- **When**: Pre-game mulligan decisions
- **Decides**: Keep or mulligan, which cards to bottom (London mulligan)
- **Architecture**:
  - Hand evaluation network
  - Trained on correlation between opening hands and win rates

#### Decision Head 7: **Binary/Numeric Choices**
- **When**: confirmAction, chooseNumber, chooseBinary, etc.
- **Decides**: Yes/no or numeric value
- **Architecture**: Simple MLP head on game state embedding + context

---

## Module Structure (new `forge-ai-rl` module)

```
forge-ai-rl/
├── src/main/java/forge/ai/rl/
│   ├── PlayerControllerRL.java          # Implements PlayerController
│   ├── RLController.java                # Orchestrates decision heads
│   ├── GameStateEncoder.java            # Converts game state → feature vectors
│   ├── ActionEncoder.java               # Encodes available actions
│   ├── CardEncoder.java                 # Encodes individual cards
│   │
│   ├── decisions/                        # Decision head interfaces
│   │   ├── PriorityDecision.java
│   │   ├── TargetDecision.java
│   │   ├── AttackDecision.java
│   │   ├── BlockDecision.java
│   │   ├── CardSelectDecision.java
│   │   ├── MulliganDecision.java
│   │   └── BinaryDecision.java
│   │
│   ├── model/                            # Neural network integration
│   │   ├── ModelServer.java             # gRPC/socket client to Python model server
│   │   ├── InferenceRequest.java
│   │   ├── InferenceResponse.java
│   │   └── ModelConfig.java
│   │
│   ├── training/                         # Training infrastructure
│   │   ├── GameRunner.java              # Runs AI vs AI games at scale
│   │   ├── ExperienceBuffer.java        # Stores game trajectories
│   │   ├── TrajectoryRecorder.java      # Records state-action-reward tuples
│   │   ├── RewardShaper.java            # Intermediate reward signals
│   │   └── SelfPlayManager.java         # Manages self-play population
│   │
│   └── features/                         # Feature extraction
│       ├── BoardFeatures.java
│       ├── CardFeatures.java
│       ├── CombatFeatures.java
│       ├── ManaFeatures.java
│       └── FeatureNormalizer.java
│
├── src/main/python/                      # Python ML side
│   ├── model/
│   │   ├── game_state_encoder.py        # Transformer encoder
│   │   ├── value_network.py             # Win probability estimator
│   │   ├── priority_head.py             # Action selection policy
│   │   ├── target_head.py               # Target selection policy
│   │   ├── combat_attack_head.py        # Attacker declaration policy
│   │   ├── combat_block_head.py         # Blocker declaration policy
│   │   ├── card_select_head.py          # Card selection policy
│   │   ├── mulligan_head.py             # Mulligan policy
│   │   └── binary_head.py              # Yes/no decisions
│   │
│   ├── training/
│   │   ├── trainer.py                   # PPO/IMPALA training loop
│   │   ├── self_play.py                 # Self-play orchestration
│   │   ├── curriculum.py                # Curriculum learning scheduler
│   │   ├── elo_tracker.py               # Track model strength over time
│   │   └── replay_buffer.py             # Prioritized experience replay
│   │
│   ├── serving/
│   │   ├── model_server.py              # gRPC server for inference
│   │   └── batch_inference.py           # Batch requests for throughput
│   │
│   └── evaluation/
│       ├── benchmark.py                 # Evaluate vs heuristic AI
│       ├── card_understanding.py        # Test card-specific knowledge
│       └── visualize.py                 # Training curves, attention maps
```

---

## Training Strategy

### Phase 1: Bootstrap — Learn from the Heuristic AI (Imitation Learning)

Before any RL, we **imitate the existing heuristic AI** to get a reasonable starting policy.

1. Run thousands of heuristic AI vs AI games
2. Record every decision point: (game_state, available_actions, chosen_action)
3. Train each decision head via supervised learning to predict the heuristic AI's choices
4. This gives us a "warm start" — a policy that's roughly as good as the heuristic AI

**Why**: Starting RL from scratch with random play would take astronomically long. The heuristic AI is already decent — we start there and improve.

### Phase 2: Curriculum Learning — Start Simple

Don't train on all 32,300 cards at once. Use a curriculum:

1. **Stage A**: Vanilla creatures only (no abilities). Learn combat math, life total management, mana curve
2. **Stage B**: Add common keywords (flying, trample, first strike, deathtouch, lifelink)
3. **Stage C**: Add removal spells, combat tricks
4. **Stage D**: Add card draw, counterspells, stack interaction
5. **Stage E**: Add planeswalkers, enchantments, complex abilities
6. **Stage F**: Full card pool (or competitive format subsets like Standard/Modern)

At each stage, cards are restricted to a curated pool. The AI learns fundamentals before complexity.

### Phase 3: Self-Play RL (PPO + Population-Based Training)

Once bootstrapped, the RL agent plays against itself and a pool of opponents:

**Algorithm**: Proximal Policy Optimization (PPO) with:
- **Population-based training**: Maintain a population of ~10-20 agents with different hyperparameters
- **League training** (AlphaStar-style):
  - "Main agents" that train against all opponents
  - "Exploiter agents" that specifically target weaknesses in main agents
  - "League exploiters" that target the full history of agents
- **Historical snapshots**: Keep checkpoints of past agents as training opponents to prevent forgetting

**Reward Shaping** (critical for sparse-reward games):
- +1.0 for winning, -1.0 for losing (terminal reward)
- Small intermediate rewards to guide learning:
  - +0.01 per point of life advantage gained
  - +0.05 per card advantage gained
  - +0.02 per creature advantage on board
  - -0.1 for illegal action attempts (shouldn't happen but safety net)
  - Rewards decay over training — eventually rely only on win/loss

**Discount factor**: γ = 0.999 (very long horizon — early decisions matter for late game outcomes)

### Phase 4: Targeted Training Against Heuristic AI

Periodically pit the RL agent against the heuristic AI to:
- Measure improvement (Elo tracking)
- Identify specific weaknesses (does it lose to aggro? control? combo?)
- Generate hard examples for focused training

---

## Java-Python Bridge

The game engine runs in Java; the neural networks run in Python. We need efficient communication.

### Option A: gRPC Service (recommended for training)
- Python model server exposes gRPC endpoints for each decision type
- Java client sends encoded game state, receives action
- Supports batching for throughput during training
- ~1-5ms latency per decision (acceptable for AI vs AI)

### Option B: ONNX Runtime (for deployment)
- Export trained PyTorch models to ONNX format
- Load ONNX models directly in Java via ONNX Runtime
- Zero inter-process communication overhead
- Use this for the final deployed model (no Python dependency)

### Recommended approach: gRPC during training, ONNX for deployment.

---

## Card Representation

Each card is represented as:

1. **Structural features** (fixed, interpretable):
   - Mana cost (CMC + color breakdown)
   - Card types (creature, instant, sorcery, enchantment, artifact, planeswalker, land)
   - Subtypes (encoded categorically)
   - Power/toughness (for creatures)
   - Keywords (binary vector for ~100 common keywords)
   - Zone it's currently in
   - Tapped/untapped, summoning sick, counters

2. **Ability features** (extracted from card script):
   - ApiType of each ability (DealDamage, Draw, Counter, ChangeZone, etc.)
   - Target types
   - Numeric parameters (damage amount, cards drawn, etc.)
   - Cost to activate

3. **Learned embedding** (trainable):
   - Each unique card gets a trainable embedding vector (like word embeddings)
   - Initialized randomly, learned during training
   - Captures subtle card interactions that aren't in structural features

4. **Card text embedding** (optional, advanced):
   - Encode Oracle text with a small language model
   - Helps generalize to unseen cards

The final card representation is the **concatenation of all four**, projected to a fixed size.

---

## Handling MTG Complexity

### Stack Interactions
- When the RL agent has priority with spells on the stack, the stack contents are encoded as part of game state
- The priority decision head sees what's on the stack and can choose to respond or pass
- Train specifically on counter-spell scenarios

### Hidden Information
- The RL agent only sees what a legal player would see (no peeking at opponent's hand/library)
- Use the game's existing visibility rules
- The value network must learn to estimate under uncertainty

### Variable Action Spaces
- Each decision presents a different set of legal actions
- Pointer networks / attention over action sets handle this naturally
- "Pass" is always an option during priority

### Multi-step Decisions
- Some effects require sequences of choices (e.g., cast a spell → choose targets → choose modes → pay costs)
- Model as a sequence of decision head invocations, each conditioned on previous choices

---

## Implementation Phases

### Phase 1: Infrastructure (weeks 1-4) ✅ COMPLETE
1. ✅ Create `forge-ai-rl` Maven module
2. ✅ Implement `PlayerControllerRL` — extends PlayerControllerAi, overrides combat + priority decisions
3. ✅ Build `GameStateEncoder` — 37,216-float game state (96 global + 145×256 card features)
4. ✅ Build `CardFeatures` (256-dim) + `ActionEncoder` (64-dim) — card and spell representations
5. ✅ Build `TrajectoryRecorder` — JSONL trajectory files with full state + action features
6. ✅ Build `SimulateRLTraining` — headless parallel game runner (16 threads, 1.3 games/sec)
7. ✅ JSON-over-TCP bridge with batched inference server
8. ✅ Python project with PyTorch, MTGModel (11M params), training dashboards

### Phase 2: Imitation Learning (weeks 5-8) ✅ COMPLETE
1. ✅ Run 1,000 heuristic AI vs AI games, recording all 7 decision types
   - Latest data (v3, 2026-03-23): 127,156 records from 987 unique games
   - 104,514 priority, 9,562 attack, 2,768 block, 4,995 target, 2,635 mulligan, 2,032 binary, 650 card_select
2. ✅ Train game state encoder + value network
3. ✅ Train ALL 7 decision heads with 256-dim card features
   - Previous run (v2 data): Priority 95.7%, Attack 82.8%, Target 74.3%, Card Select 76.5%, Block 64.2%, Mulligan 99.0%, Binary 80.9%
   - Retraining on v3 data in progress (bugfixes applied: leakage, feature encoding, aura targeting, multi-target)
4. ✅ Pure RL gameplay — all 7 heads make decisions via ONNX, no heuristic involvement
5. ✅ Baseline: ~25% win rate vs heuristic (imitation model, no PPO)
6. ✅ ONNX deployment — 9 model files loaded in Java, verified identical to PyTorch output

### Phase 3: RL Training — Simple Cards (weeks 9-14) 🔄 IN PROGRESS
1. 4 aggro decks (Red Aggro, Green Stompy, White Weenie, Blue Tempo)
2. ✅ PPO training loop with GAE per-decision advantages
3. ✅ Reward shaping implemented (life/card/board advantage signals with decay)
4. ✅ Frozen encoder during PPO (separate LRs: heads 3e-5, value 1e-4)
5. ✅ PPO vs heuristic: 43 rounds (17,200 games), plateaued at 30-39% win rate
   - Best: 39% at round 21, no sustained upward trend
   - Per-deck: White 46%, Green 32%, Red 21-40%, Blue (old) 5-15%
   - Weight analysis: heads changed 10-20% L2 but win rate didn't improve
   - Root cause: frozen encoder limits what heads can learn (see Phase 3b below)
6. ✅ Encoder reconstruction experiment confirmed the problem:
   - Value-only encoder retained only 24% R² of basic game state features
   - Opponent life R² = -0.02 (worse than mean), creature counts R² negative
   - Combat math features largely discarded (power/toughness R² negative)
   - Encoder was optimized for value prediction, threw away decision-critical info
7. ✅ Blue Tempo deck replaced with Mono Blue Tempest Djinn (MTGGoldfish #1361608)
   - Old deck: Delver/Counterspell — heuristic couldn't play it (~5-15% win rate)
   - New deck: Mist-Cloaked Herald, Tempest Djinn, Curious Obsession, Dive Down
   - New meta: Green 61%, White 56%, Blue 44%, Red 38% (much healthier)
8. ⬜ Elo tracking and benchmarking

### Phase 3b: Joint Training + PPO Improvements (2026-03-27) 🔄 IN PROGRESS

Key insight: the encoder must be trained jointly with ALL heads (not value-only then freeze).
Dense supervised signal from imitation data is far more effective for encoder training than
sparse PPO signal. Joint training gives the encoder balanced gradients from all 7 heads +
value network simultaneously.

1. ✅ Implemented `--joint` mode in `train_decisions_ui.py`:
   - Round-robin training: all heads + value network + unfrozen encoder
   - Encoder LR at 1/10th of head LR (1e-4 vs 1e-3)
   - Per-head optimizers with gradient accumulation across heads
   - Encoder stepped once per round-robin cycle with combined gradients
2. ✅ PPO hyperparameter improvements:
   - GAE gamma 0.95 → 0.99 (better early-game credit assignment)
   - GAE lambda 0.90 → 0.95 (longer effective horizon: ~20 steps vs ~7)
   - Entropy coefficient 0.005 → 0.03 (prevents policy collapse)
   - Games per round 400 → 800 (more data for rare heads)
3. 🔄 Joint training in progress (3 epochs completed before interruption):
   - After 3 epochs: encoder R² improved from 0.24 → 0.47 (basic globals)
   - Card combat R²: 0.31 → 0.54 (survives_block 0.59→0.80, evasive 0.26→0.54)
   - Decision head accuracies: priority 86.8%, attack 86.4%, block 59.0%
   - Training continuing to 10 epochs
4. ⬜ PPO with improved encoder + hyperparameters
5. ⬜ Self-play if PPO vs heuristic plateaus again

### Phase 4: RL Training — Full Complexity (weeks 15-22)
1. Train on full card pools (Standard, Modern, or curated sets)
2. Implement league training with exploiter agents
3. Implement population-based training for hyperparameter search
4. Run large-scale self-play (millions of games)
5. Regular benchmarking vs heuristic AI at different difficulty levels

### Phase 5: Deployment & Polish (weeks 23-26)
1. Export best models to ONNX for Java-native inference
2. Integrate into Forge as a selectable AI opponent ("RL AI" option)
3. Add difficulty scaling (use earlier checkpoints = easier, latest = hardest)
4. Performance optimization for acceptable game speed
5. Add AI personality profiles that bias the RL agent's style

---

## Key Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Game simulation too slow for millions of training games | Profile and optimize headless game execution; strip UI/logging; parallelize across cores |
| Action space too large for effective learning | Hierarchical decisions + curriculum learning + imitation pre-training |
| Card interactions create long-tail edge cases | Focus on competitive format card pools (Standard ~2000 cards); expand gradually |
| gRPC latency slows training | Batch inference; async game execution; eventually ONNX in-process |
| RL agent learns degenerate strategies | League training with exploiters; periodic evaluation on diverse matchups |
| New card sets invalidate training | Card embedding approach generalizes; fine-tune on new sets; structural features transfer |

## Success Metrics

1. **Phase 2 target**: RL agent (imitation) wins 45-55% vs heuristic AI (parity)
2. **Phase 3 target**: RL agent wins 60%+ vs heuristic AI on simple card pools
3. **Phase 4 target**: RL agent wins 65%+ vs heuristic AI on full card pools
4. **Stretch goal**: RL agent discovers non-obvious strategies that surprise experienced players


---

## Bugs Fixed (v3 data collection cycle, 2026-03-23)

All critical bugs have been fixed. Data regenerated with corrected feature encoding.

- **~~ppo_ui.py used attack_head for block data~~** — FIXED. Block decisions trained through attack head. Fixed with `(data, head)` tuple pairing.
- **~~RewardShaper.initialized never set to true~~** — FIXED. Intermediate rewards always returned 0.
- **~~PPO used flat terminal +1/-1 for all decisions~~** — FIXED. Now computes GAE advantages from `intermediateReward` per decision with gamma=0.95, lambda=0.90.
- **~~GAE gamma mismatched value function~~** — FIXED 2026-03-24. GAE_GAMMA changed from 0.999 to 0.95 to match the value function training gamma (0.95 in preprocess_trajectories.py). The mismatch caused systematic bias: delta = r_t + 0.999*V(t+1) - V(t), but V was trained to satisfy V(t) ≈ r_t + 0.95*V(t+1), so deltas were always proportional to V(t+1) regardless of decision quality. Every decision in winning games got positive advantage, every decision in losing games got negative advantage. Value loss dropped from ~0.20 to 0.032 after the fix, confirming the value function was already accurate but miscalibrated.
- **~~GAE lambda too high for credit assignment~~** — FIXED 2026-03-24. GAE_LAMBDA changed from 0.98 to 0.90. Lower lambda puts more weight on per-step value deltas vs terminal game outcome. With λ=0.98 in a 25-decision game, terminal reward dominated credit assignment. At λ=0.90, effective discount is (γλ)^k = 0.855^k, so per-step value improvements drive advantages — good decisions in losing games get positive advantage (value improved) rather than being penalized by the eventual loss. Note: intermediate rewards (life delta × 0.01, board delta × 0.02, card advantage × 0.05) are very small compared to value-based signal; the value function is the primary credit assignment mechanism through GAE.
- **~~Encoder corrupted by PPO gradients~~** — FIXED. Now encoder is frozen during PPO, heads get lr=3e-5, value network gets lr=1e-4.
- **~~Heuristic vetoed RL spell choices~~** — FIXED. RL uses `decideTargets` for targeting directly, no heuristic strategic veto.
- **~~Duplicate ApiType.ChangeZone in feature encoding~~** — FIXED 2026-03-23. Index 29 replaced with `RearrangeTopOfLibrary` in both `ActionEncoder.java` and `CardFeatures.java`.
- **~~Aura targeting flags wrong in ActionEncoder~~** — FIXED 2026-03-23. Added `source.isAura()` check to use enchant keyword instead of `getValidTgts()`.
- **~~Multi-target spells (Searing Blaze) hardcoded to 1 target~~** — FIXED 2026-03-23. Now uses `getMinTargets()`/`getMaxTargets()` from `TargetRestrictions`.
- **~~Train/val data leakage~~** — FIXED 2026-03-23. All training scripts now split by game_id (filename timestamp) instead of random shuffle. Both P1/P2 perspectives of the same game stay in the same split.
- **~~is_sorcery_speed feature missing~~** — FIXED 2026-03-23. Added at global feature index 54.

### Moderate — Fixed

- **~~Block candidate maxSelections constraint~~** — FIXED 2026-03-23. `maxSelections` now capped to `min(possibleBlockers.size(), candidates.size())`. Inference-only fix, no data regeneration needed (ONNX block head uses per-blocker argmax and doesn't consume this value).

### Diagnostics

- **All decision logging**: `PlayerControllerRL` logs every RL decision:
  - `RL_PRIORITY_PLAY: {card} ({api}) -> target: {target}` — spell played with target
  - `RL_PRIORITY: PASS ({n} options available)` — priority pass
  - `RL_SPELL_REJECTED: {card} ({api}) reason={reason}` — spell couldn't be played
  - `RL_MULLIGAN: KEEP/MULLIGAN ({n} cards)` — mulligan decision
  - `RL_BINARY: YES/NO ({context})` — binary decision
  - `RL_MODEL_ATTACK: probs=[...] selected=[...] value={v}` — attack decision
- **Per-game diagnostics**: `RL_DIAG:` summary at game end with model_asked, play, pass, rejected, bypass counts

## Future Enhancements

- **Record opponent's plays**: Currently only the RL player's decisions are recorded. Recording opponent plays would help the model learn reactive strategies.

## Feature Encoding Gaps

62 of 256 CardFeature slots and 8 of 64 ActionEncoder slots are reserved/unused. All gaps below can be filled without changing the feature dimensions or model architecture — just richer extraction in Java, requiring data regeneration.

### CardFeatures (256-dim) — Current Gaps

#### Gap 1: Trigger effects are opaque (Priority: High)
Indices 178-181 encode `has_etb`, `has_death`, `has_combat`, `has_upkeep` as binary flags but not what the triggers DO. An ETB that draws a card and one that deals 2 damage both show `has_etb=1`.

**4-deck impact:** Augur of Bolas (ETB: dig for instant/sorcery), Thalia's Lieutenant (ETB: +1/+1 all humans), Experiment One (evolve on ETB of others), Snapcaster Mage (ETB: flashback), Eidolon of the Great Revel (cast trigger: 2 damage).

**Fix:** Use reserved slots 190-199. For each trigger category, extract `getOverridingAbility().getApi()` and effect magnitude:
- [190-194]: ETB trigger ApiType (one-hot top 5: DealDamage, Draw, Pump, PumpAll, ChangeZone)
- [195]: ETB effect magnitude (normalized damage/draw/pump amount)
- [196-197]: Death trigger ApiType (top 2) + magnitude
- [198]: Combat trigger type (attacks/blocks/damage)
- [199]: Other trigger magnitude

**Full card pool considerations:** 30,000+ cards have triggers. The top-5 ETB ApiType covers ~80% of ETB effects. For full coverage, would need either: (A) expand to top-10 ETB types using more reserved slots, or (B) encode trigger text via a small language model embedding — but that's a v3 architectural change. The magnitude extraction (`NumDmg`, `NumAtt`, `NumDef`, `NumCards`) covers most parameterized effects; conditional/complex triggers (e.g., "when ~ deals combat damage to a player, draw cards equal to its power") would need special handling or would just show as `has_combat=1, combat_magnitude=0` — an acceptable approximation for now.

#### Gap 2: Pump spell magnitude missing (Priority: High for green deck)
The model sees `ApiType.Pump` but not the magnitude. Giant Growth (+3/+3), Aspect of Hydra (+N/+N), Rancor (+2/+0 trample) all look identical. The effect magnitude fields (103-106) only extract damage, draw, life, tokens — no pump amount.

**Fix:** Add at indices 200-201:
- [200]: `est_pump_power` — extract from `sa.getParam("NumAtt")`
- [201]: `est_pump_toughness` — extract from `sa.getParam("NumDef")`

**Full card pool:** Variable pump amounts (e.g., "gets +X/+X where X is your devotion") would return the default 0.3f fallback. For Aspect of Hydra specifically, computing actual devotion at encode time would give the real value — but this requires game state access during card encoding, which the current static `CardFeatures.encode(Card)` signature doesn't support. A future refactor to `encode(Card, Player)` would enable board-relative card features.

#### Gap 3: Protection/evasion details not encoded (Priority: Medium)
Brave the Elements gives protection from a chosen color. Vines of Vastwood gives hexproof + pump. The model sees `ApiType.Protection` or `ApiType.Pump` but not which color, or that hexproof prevents targeting. The value of these combat tricks depends entirely on the opponent's board composition.

**Fix:** Add at indices 202-205:
- [202]: protection_from_white, [203]: protection_from_blue, etc. — or more generally, `protection_matches_opponent_colors` as a single bit computed at encode time by checking opponent's board colors.

**Full card pool:** Protection from arbitrary qualities ("protection from converted mana cost 3 or greater") is common. A single `has_relevant_protection` bit computed against the current board is more useful than trying to enumerate all protection types.

#### Gap 4: Recursion / graveyard value not encoded (Priority: Low for current meta)
Snapcaster Mage's value depends on the graveyard contents. Undying creatures (Strangleroot Geist) come back — but this IS encoded via `Keyword.UNDYING`.

**Full card pool:** Flashback, unearth, escape, embalm, aftermath — all create graveyard-to-battlefield or graveyard-to-stack value. Would need: `graveyard_castable_count` as a global feature, and per-card `has_flashback`, `has_escape` flags in the extended keyword set.

#### Gap 5: Conditional growth not projected (Priority: Low)
Experiment One (evolve) and Monastery Swiftspear (prowess) have conditional growth. The model sees the keywords but can't project future growth. Prowess value depends on noncreature spell count in hand; evolve depends on upcoming creature sizes.

**Full card pool:** Prowess, evolve, modular, and similar "grows over time" mechanics are common in aggro/tempo. A `projected_growth` feature would need game state access (hand contents, deck composition). Deferred to the `encode(Card, Player)` refactor.

### ActionEncoder (64-dim) — Current Gaps

#### Gap 6: No board-relative utility features (Priority: High)
The 64-dim action features encode what a spell IS but not what it DOES in this specific board state. Reserved slots 56-63 (8 available).

**Fix:** Compute at encode time from the game state:
- [56]: `would_kill_creature` — does this damage spell kill any opposing creature? Check `NumDmg >= min(opp_creature_toughness - damage_marked)`
- [57]: `kills_biggest_threat` — would it kill the highest-power opposing creature?
- [58]: `is_lethal` — does burn-to-face win the game? Check `NumDmg >= opp_life`
- [59]: `n_creatures_affected` — for PumpAll/DestroyAll, how many creatures would this hit?
- [60]: `opponent_can_respond` — does opponent have mana open for instant-speed response?
- [61]: `leaves_mana_for_followup` — after casting, how much mana remains? Normalized.
- [62]: `creature_would_trade` — if playing a creature, would it trade favorably in combat?
- [63]: `saves_creature` — for protection/hexproof spells, is a creature currently targeted?

**Requires refactoring `ActionEncoder.encode(SpellAbility)` to `ActionEncoder.encode(SpellAbility, Game, Player)` since board-relative features need the game state.** This is a breaking change that requires updating all call sites in PlayerControllerRL, RLController, and HumanGameRecorder. Data regeneration required.

**Full card pool:** The 8 utility features above are universal — they apply to any damage spell, any creature, any pump spell. No card-specific logic needed. The `opponent_can_respond` feature becomes more important with counterspells and interaction-heavy formats.

### GameStateEncoder (96-dim global) — Current Gaps

#### Gap 7: No race/clock calculation (Priority: High for aggro)
In aggro mirrors, the key question is "how many turns until I die vs how many until I kill them." The model has raw data (life totals, creature stats) but no pre-computed clock.

**Fix:** Add at global indices 55-56 (currently reserved):
- [55]: `turns_to_lethal_me` — estimate based on opponent's total attacking power vs my life
- [56]: `turns_to_lethal_opp` — estimate based on my total attacking power vs opponent's life

Simple computation: `ceil(life / max(total_attack_power - total_block_power, 1))`. Doesn't account for evasion or removal but gives a directional signal.

**Full card pool:** The race clock becomes less meaningful in control/combo matchups where the game isn't decided by creature damage. Could add a `game_archetype` signal (aggro mirror, aggro-vs-control, etc.) but that requires the belief-state modeling discussed elsewhere.

#### Gap 8: No opponent threat quality assessment (Priority: Medium)
Global features encode creature counts and total power but not threat quality. A 5/5 trampler is very different from five 1/1 tokens.

**Fix:** Add at global indices 57-58:
- [57]: `opp_max_creature_power` — normalized power of opponent's biggest creature
- [58]: `opp_evasion_power` — total power of opponent's creatures with flying/trample/menace/unblockable

**Full card pool:** Add `opp_has_planeswalker`, `opp_has_equipment`, `opp_noncreature_threat_count` for non-combat threats.

### Aura/Equipment Association (Priority: Low for current meta)

Cards encode attachments count [27] and auras show as separate board cards, but there is no explicit "card X is attached to card Y" pointer. The per-zone self-attention in `CardSetEncoder` can learn these associations implicitly (both are in `my_board`). Options: (A) encode host card stats inline — no model change; (B) positional index — fragile; (C) attachment attention mask — new layer. See Encoder Architecture section.

**Full card pool:** Equipment-heavy strategies (Voltron, Bogles) rely heavily on aura/equipment stacking. For these archetypes, Option C (attachment attention) becomes necessary. For our 4-deck meta, only Rancor matters, and the model can learn "Rancor + creature = bigger creature" from the P/T boost visible in `getNetPower()`.

### Implementation Priority for Next Data Regeneration

All changes below fit in reserved slots — no model architecture changes required.

| Change | Slots Used | Impact | Effort |
|--------|-----------|--------|--------|
| Pump magnitude (CardFeatures 200-201) | 2 | High (green deck) | Trivial |
| Race clock (Global 55-56) | 2 | High (aggro mirrors) | Easy |
| Trigger effects (CardFeatures 190-199) | 10 | High (ETB creatures) | Medium |
| Board-relative action utility (Action 56-63) | 8 | High (burn targeting) | Medium — requires signature change |
| Threat quality (Global 57-58) | 2 | Medium | Easy |
| Protection details (CardFeatures 202-205) | 4 | Medium (white deck) | Easy |
| Total | 28 slots | | |

This leaves 34 CardFeatures slots and 32 global slots still reserved for full card pool expansion (planeswalker abilities, saga chapters, adventure/split cards, companion, etc.).

### Priority Head Class Imbalance — IMPLEMENTED 2026-03-23

- **Class-weighted cross-entropy for priority head**: The heuristic passes priority ~85% of the time even with playable spells available. Standard cross-entropy lets the model achieve high accuracy by learning a pass-heavy distribution, which is exposed during PPO's stochastic sampling (78% creature miss rate during main phases). Under argmax (ONNX deployment) the model performs well (29% win rate, heuristic parity), but the underlying distribution is too peaked at pass for PPO to explore effectively.

  **Fix applied**: Inverse-frequency per-sample weighting in `make_priority_batch()`. Each sample gets weight `n_pass/n_total` if it's a play decision, or `n_play/n_total` if it's a pass. This equalizes gradient contribution without discarding data. The pass action is identified per-sample as `selected_idx == n_actions - 1` (pass is always the last candidate). Only the priority head needs this — other heads have naturally balanced distributions.

  **Evidence**: Heuristic vs heuristic baseline shows only a mild P2 advantage (47%/53%), but the RL imitation model amplifies this to 34%/74% — indicating the model learned an exaggerated pass preference that disproportionately hurts going-first play. Supported by Parekh et al. (2025, "Towards Balanced Behavior Cloning from Imbalanced Datasets") which formally proves equally-weighted BC on imbalanced data produces policies that emulate the dominant behavior.

### Encoder Architecture

- **Joint multi-task encoder training (IMPLEMENTED 2026-03-27)**: The encoder is now trained jointly with ALL 7 decision heads + value network from the start via `--joint` mode. Each task contributes gradients via round-robin batching with per-head optimizers. Encoder uses 1/10th LR (1e-4 vs 1e-3 for heads) to prevent any single head from dominating. This replaced the previous two-phase approach (value-only encoder → frozen encoder + sequential heads) which produced an encoder optimized only for value prediction that discarded decision-critical information (life totals R²=0.23, creature counts R²<0, combat math largely lost). After 3 epochs of joint training, encoder reconstruction R² improved from 0.24 → 0.47 for basic globals and 0.31 → 0.54 for combat features.

- **Per-head encoders (investigated, not recommended at current scale)**: Giving each head its own 3.5M-param encoder would allow full specialization (attack encoder emphasizes combat stats, priority encoder focuses on mana/timing). However: (1) rare decision types (block 2.7K, binary 2K samples) cannot train a 3.5M encoder — they'd memorize instantly; (2) inference cost multiplies by 6× since each decision type needs a full encoder forward pass instead of reusing one shared embedding; (3) the value network's critic assessment becomes inconsistent with each head's world model, breaking PPO advantage estimation; (4) model size triples (~32M params) and ONNX deployment goes from 9 to 14 files. Per-head encoders make sense at much larger data scale (millions of samples per type) or if decision types were truly unrelated tasks. At current scale, the shared encoder with joint training is the right approach.

- **Encoder capacity analysis (2026-03-27)**: The 512-dim bottleneck compresses 37,216 input floats at 72:1 raw ratio, but effective compression is ~3:1 (most card slots are empty/zero-padded, ~1,300-1,500 meaningful floats). PCA shows 90% of embedding variance in 49 dims, 99% in 92 dims — all 512 dims are active (std 0.44-1.07) but information is concentrated. The encoder size appears adequate for the current 4-deck meta; the bottleneck was training methodology (value-only) not capacity.

### Evaluation and Metrics (Priority: High)

- **Stop using head accuracy as the primary metric.** Head accuracy is a debugging tool, not a quality metric. Priority at 93.9% masks a 78% creature miss rate; mulligan at 98.5% mostly reflects the baseline keep rate. The class-weighted cross-entropy fix (already implemented) partially addresses this — by equalizing gradient contribution between play and pass, accuracy will drop (the model can no longer coast on the 85% pass baseline) but the reported number becomes more meaningful as a measure of actual decision quality. Track additionally: (1) win rate under argmax deployment, (2) win rate by play/draw, (3) creature play rate during own main phases, (4) per-head contribution via ablation (disable one head → heuristic fallback → measure win rate drop). These metrics reveal whether the model is actually learning MTG strategy vs just fitting the label distribution.

- **Held-out evaluation.** Current eval uses the same 4 decks for training and testing. Add held-out decks (different builds of the same archetypes, or a 5th archetype) to test generalization. Also add exploitability tests: a targeted opponent bot that exploits known weaknesses (e.g., never blocking because the model rarely attacks).

- **Argmax eval during PPO.** PPO eval uses stochastic sampling, which conflates "is the policy better?" with "how much does exploration hurt?". The true baseline under both argmax and sampling is ~29-31% (the previously reported 54% argmax result was a heuristic fallback bug). Periodic argmax eval would still be useful for checkpoint selection and learning rate scheduling, but the gap between sampling and argmax performance is smaller than previously believed.

### Hidden Information and Belief State (Priority: High impact, High effort — v2)

- **Deck archetype posterior.** The model sees the opponent's board and graveyard but has no explicit representation of which deck they're playing. After observing 2-3 opponent cards, the deck identity is usually determinable (Mountain + Goblin Guide = Red Aggro). Encode a deck-archetype embedding conditioned on revealed cards. This is tractable for our 4-deck meta and would help the model anticipate likely threats (e.g., hold removal against Green Stompy because Leatherback Baloth is coming).

- **Inferred hand range.** Knowing "opponent has 1G open and a card in hand" should trigger different play from "opponent is tapped out." The current encoding captures mana availability and hand size but not the probability distribution over what the opponent holds. A latent belief state (updated recurrently across decisions within a game) could model this. Complex to implement — requires either a recurrent component or a memory mechanism across decision points within a game. This is the single highest-impact enhancement for playing beyond heuristic level, but also the hardest.

### PPO Sample Efficiency — Scale Reality Check (2026-03-23)

PPO in complex game domains requires vastly more data than we're currently generating:

| System | Games for RL improvement | Decisions | Compute |
|--------|------------------------|-----------|---------|
| Atari (PPO) | 10-100M timesteps | Millions | Hours on single GPU |
| AlphaStar (imitation) | Millions of human replays | Billions | 3 days on TPUs |
| AlphaStar (RL) | Millions of self-play games | Billions | 200 years real-time equiv, 16 TPUs/agent × 12 agents |
| **Our PPO** | **5,000 games (50×100)** | **~250K decisions** | **~2 hours on RTX 3080** |

We are 3-4 orders of magnitude below typical PPO data requirements for complex games. The oscillation and flat win rate we observe (20-34% across 20 rounds, with catastrophic mulligan collapses) may be normal variance at this scale rather than a fundamentally broken algorithm.

**Key insight:** AlphaStar's imitation-only model (before any RL) already beat 84% of human players — because it trained on millions of games. Our imitation model at 29% win rate from 1,000 games is actually a strong result given the data scale. PPO improving beyond this may require 100-1,000× more games than we've tried.

**Practical options:**
1. Run PPO much longer (500+ rounds × 400 games = 200K+ games, ~50 hours). Accept that improvement will be gradual.
2. Switch to offline RL (AWR) which is more sample-efficient for our setup (see below).
3. Focus on improving the imitation model instead — better features, human data, class-weighted loss — since that's where our wins have come from so far.
4. Accept 29% win rate as a strong baseline and invest in feature encoding improvements (documented above) rather than RL training compute.

### Self-Play + Reward Shaping (Priority: Medium — after joint training + improved PPO)

**Reward Shaping — investigated and intentionally disabled.**
Intermediate rewards (life/card/board deltas) are recorded in Java but zeroed during GAE computation. The value network's per-step delta `V(s_{t+1}) - V(s_t)` already provides implicit reward shaping — and more accurately than hand-tuned weights, since V sees the full 37K-float game state. Adding explicit shaped rewards risks double-counting (the value function learns the shaping signal, then GAE subtracts V but also adds the raw shaping reward).

**Self-Play — attempted, useful but not a silver bullet.**
Tried 3+ rounds of self-play PPO from the 39% vs-heuristic model. Infrastructure works (`runSelfPlayMode`, `--collect-mode selfplay`, Elo tracking). Self-play guarantees 50% win rate for balanced signal and 2x data per game. However, at 30-39% strength the model is too weak for productive self-play — risk of degenerate strategies. Better to improve the model via joint training + improved PPO first, then revisit self-play once it's consistently >45% vs heuristic.

**Current approach:** Joint encoder+head training (dense supervised signal) → PPO vs heuristic with improved hyperparameters (gamma 0.99, entropy 0.03, 800 games/round) → self-play if PPO plateaus again.

### Alternative RL Approaches (Priority: High — PPO may not be viable at our scale)

- **Advantage-weighted regression (AWR) / offline RL.** The most promising alternative to PPO for our compute budget. Instead of PPO's stochastic sampling (which degrades play quality and requires importance sampling ratios), AWR:
  1. Collects games under **argmax** — the model plays at full strength (29% win rate, not 28%)
  2. Computes GAE advantages for each decision in the trajectory
  3. Updates the policy by **weighting the supervised loss** by the advantage — actions with positive advantage get upweighted, negative get downweighted
  4. No importance sampling ratios needed, no clipping, no stochastic sampling

  **Why this may work better for us:** The fundamental PPO problem is that stochastic sampling produces games where the model plays badly (passes on creatures 78% of the time), wins rarely (~28%), and therefore generates mostly negative advantage signals. The model can't learn what good play looks like because it almost never plays well during data collection. AWR avoids this entirely — the model plays its best (~29% under argmax), and learns from both wins and losses at full strength rather than degraded sampling.

  **Implementation:** Replace the GRPC model server's `multinomial` sampling with `argmax`, record action probabilities for all candidates (not just the chosen one), and replace the PPO clipped objective with advantage-weighted cross-entropy: `loss = -advantage * log(π(a|s))` for positive-advantage actions only (or with exponential weighting).

  **Reference:** Peng et al. (2019), "Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning."

- **Preference learning from game pairs.** Instead of per-action credit assignment, compare pairs of games from similar starting positions and learn "this sequence of decisions led to a win, that one to a loss." Avoids the need for per-step advantage estimation. Could use the value network to identify similar game states across different games and construct preference pairs.

### Autoregressive Action Sequences (Priority: Low now, Critical for scaling)

- **Turn-plan modeling.** The current priority head makes one-step picks (play a spell or pass). In reality, MTG turns are short action programs: cast spell A → maintain priority → respond to trigger → choose mode → choose targets → pass. The current factorization works for simple aggro play but risks myopic decisions in stack-heavy or combo lines. A constrained autoregressive decoder over legal action tokens, or a "turn-plan" latent that conditions multiple sub-decisions, would handle complex sequencing. Not needed for the current 4-deck aggro meta but essential for scaling to control/combo archetypes.

### Selective Decision-Time Search (Priority: Medium)

- **Shallow search at tactical decision points.** Rather than pure policy+value everywhere, use the value network to do a shallow 2-3 move lookahead at combat decisions and lethal checks. The Forge engine's `GameSimulator` already supports state copying for lookahead. Even a value-guided beam over a handful of legal actions could catch tactical blunders (missed lethal, bad blocks) without the full cost of MCTS. Most impactful for combat decisions where the branching factor is manageable (5-10 creatures × attack/hold).

### Matchup-Aware Curriculum (Priority: Medium — next phase)

- **Curriculum over archetypes, not just card complexity.** The current curriculum stages (vanilla creatures → keywords → removal → etc.) focus on card mechanics. MTG strategy is heavily archetype-structured: aggro mirrors reward tempo, aggro-vs-control rewards threat sequencing, midrange battles reward card advantage. Add matchup families to the curriculum so the model learns "plans" (deploy threats before answers, sequence removal by priority) not just "card mechanics." The current 4-deck aggro meta implicitly covers aggro mirrors but misses the aggro-vs-control dynamic that teaches the model when to hold back resources.