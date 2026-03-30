package forge.ai.rl;

import forge.ai.rl.decisions.DecisionContext;
import forge.ai.rl.decisions.DecisionResult;
import forge.ai.rl.decisions.DecisionType;
import forge.ai.rl.features.ActionEncoder;
import forge.ai.rl.features.GameStateEncoder;
import forge.ai.rl.features.GameStateFeatures;
import forge.ai.rl.model.ModelServerClient;
import forge.ai.rl.model.ONNXModelClient;
import forge.ai.rl.training.TrajectoryRecorder;
import forge.game.Game;
import forge.game.GameEntity;
import forge.game.card.Card;
import forge.game.card.CardCollectionView;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.tinylog.Logger;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Central orchestrator for the RL AI system.
 * Routes decisions to the appropriate model head, manages the model server connection,
 * and records trajectories for training.
 */
public class RLController {
    // Static registry of latest value estimates per player name, for GUI display
    private static final ConcurrentHashMap<String, Float> latestValueEstimates = new ConcurrentHashMap<>();

    /** Get the latest value estimate for a player (by name). Returns null if no estimate available. */
    public static Float getLatestValueEstimate(String playerName) {
        return latestValueEstimates.get(playerName);
    }

    /** Clear all stored value estimates (call on game end). */
    public static void clearValueEstimates() {
        latestValueEstimates.clear();
    }

    private final RLConfig config;
    private final GameStateEncoder stateEncoder;
    private final ModelServerClient modelClient;
    private final ONNXModelClient onnxClient;
    private final TrajectoryRecorder trajectoryRecorder;
    private final ReplayTraceRecorder replayTraceRecorder;

    private Player player;
    private Game game;

    public RLController(RLConfig config) {
        this.config = config;
        this.stateEncoder = new GameStateEncoder(config);
        this.modelClient = new ModelServerClient(config);

        // Initialize ONNX client if in ONNX mode
        if (config.getMode() == RLModelMode.ONNX) {
            this.onnxClient = new ONNXModelClient(config);
            this.onnxClient.loadModels();
        } else {
            this.onnxClient = null;
        }

        this.trajectoryRecorder = config.isRecordTrajectories()
                ? new TrajectoryRecorder(config.getTrajectoryOutputDir(),
                config.isZeroIntermediateReward())
                : null;
        this.replayTraceRecorder = config.isReplayMode()
                ? new ReplayTraceRecorder(config.getReplayConfig().getTraceOutputDir(),
                config.getReplayConfig())
                : null;
    }

    public void setPlayer(Player player) {
        this.player = player;
        this.game = player.getGame();
    }

    public Player getPlayer() { return player; }
    public Game getGame() { return game; }
    public RLConfig getConfig() { return config; }

    /**
     * Check if the model server is reachable.
     * When false, callers should use heuristic fallback for all decisions.
     */
    public boolean isModelServerAvailable() {
        if (config.getMode() == RLModelMode.ONNX) {
            return onnxClient != null && onnxClient.isLoaded();
        }
        if (config.getMode() == RLModelMode.GRPC) {
            if (modelClient.isConnected()) return true;
            // Try to connect — retry a few times
            for (int attempt = 0; attempt < 3; attempt++) {
                if (modelClient.connect()) return true;
                try { Thread.sleep(500 * (attempt + 1)); } catch (InterruptedException e) { break; }
            }
            // GRPC mode MUST have a server — throw, don't silently fall back
            throw new ModelServerException(
                    "Cannot connect to model server at " + config.getGrpcHost() + ":" + config.getGrpcPort()
                    + " — refusing to fall back to heuristic");
        }
        return false;
    }

    /**
     * Start a new game — initialize trajectory recording.
     */
    public void onGameStart(String gameId) {
        onGameStart(gameId, null, null, null, null, null);
    }

    public void onGameStart(String gameId,
                            String matchId,
                            String playerName,
                            String opponentName,
                            String playerDeck,
                            String opponentDeck) {
        onGameStart(gameId, matchId, playerName, opponentName, playerDeck, opponentDeck, -1);
    }

    public void onGameStart(String gameId,
                            String matchId,
                            String playerName,
                            String opponentName,
                            String playerDeck,
                            String opponentDeck,
                            int seatIndex) {
        if (trajectoryRecorder != null) {
            trajectoryRecorder.startGame(
                    gameId, matchId, playerName, opponentName,
                    playerDeck, opponentDeck);
        }
        if (config.getMode() == RLModelMode.GRPC) {
            modelClient.connect();
            if (config.isDeterministicPolicyRequired()) {
                modelClient.ensureDeterministicReady();
            }
        }
        if (replayTraceRecorder != null) {
            replayTraceRecorder.start(
                    playerName,
                    seatIndex,
                    opponentName,
                    playerDeck,
                    opponentDeck,
                    config.getMode().name(),
                    getBackendLabel(),
                    getModelIdentifier(),
                    config.isDeterministicPolicyRequired());
        }
    }

    /**
     * End the game — finalize trajectory and compute terminal reward.
     */
    public void onGameEnd(boolean won) {
        if (trajectoryRecorder != null) {
            trajectoryRecorder.endGame(won);
        }
        if (replayTraceRecorder != null) {
            replayTraceRecorder.finish(game, player, won);
            replayTraceRecorder.close();
        }
    }

    /**
     * Make a priority action decision: choose which spell/ability to play, or pass.
     *
     * @param availableActions list of playable SpellAbilities
     * @return index into availableActions, or -1 to pass priority
     */
    public int decidePriorityAction(List<SpellAbility> availableActions) {
        if (availableActions.isEmpty()) return -1;

        GameStateFeatures gameState = stateEncoder.encode(game, player);

        // Encode each available action + pass
        List<float[]> candidates = new ArrayList<>();
        for (SpellAbility sa : availableActions) {
            candidates.add(ActionEncoder.encode(sa));
        }
        candidates.add(ActionEncoder.encodePassAction()); // last index = pass

        DecisionContext context = new DecisionContext(
                DecisionType.PRIORITY_ACTION, gameState, candidates,
                describeSpellAbilities(availableActions, true),
                1, 1, "priority_action", null);

        DecisionResult result = requestDecision(context);
        if (result == null) return -1; // fallback: pass

        int selectedIdx = result.getSelectedIndex();
        recordDecision(context, result);

        // If selected index is the pass action (last candidate)
        if (selectedIdx >= availableActions.size()) return -1;
        return selectedIdx;
    }

    /**
     * Choose targets for a spell/ability from a list of legal targets.
     *
     * @param targets list of legal target entities
     * @param min minimum targets to select
     * @param max maximum targets to select
     * @param spellFeatures 64-dim feature vector of the source spell (null if unknown)
     * @return list of indices into targets
     */
    public List<Integer> decideTargets(List<GameEntity> targets, int min, int max,
                                        float[] spellFeatures) {
        if (targets.isEmpty()) return List.of();

        GameStateFeatures gameState = stateEncoder.encode(game, player);
        List<float[]> candidates = new ArrayList<>();
        for (GameEntity entity : targets) {
            if (entity instanceof Card) {
                candidates.add(forge.ai.rl.features.CardFeatures.encode((Card) entity, player));
            } else {
                // Player targets — pad to card_feature_dim (256)
                float[] playerFeats = ActionEncoder.encodeTarget(entity);
                float[] padded = new float[256];
                System.arraycopy(playerFeats, 0, padded, 0, Math.min(playerFeats.length, 256));
                candidates.add(padded);
            }
        }

        DecisionContext context = new DecisionContext(
                DecisionType.TARGET_SELECTION, gameState, candidates,
                describeEntities(targets),
                min, max, "target_selection", spellFeatures);

        DecisionResult result = requestDecision(context);
        if (result == null) return List.of(0); // fallback: first target
        recordDecision(context, result);

        Logger.info("RL_TARGET: n={} selected={} value={}", targets.size(),
                result.getSelectedIndices(), String.format("%.3f", result.getValueEstimate()));

        return result.getSelectedIndices();
    }

    /** Overload without spell features for backward compatibility. */
    public List<Integer> decideTargets(List<GameEntity> targets, int min, int max) {
        return decideTargets(targets, min, max, null);
    }

    /**
     * Choose which creatures to attack with.
     *
     * @param possibleAttackers creatures that can legally attack
     * @return list of indices of creatures that should attack
     */
    public List<Integer> decideAttackers(List<Card> possibleAttackers) {
        if (possibleAttackers.isEmpty()) return List.of();

        GameStateFeatures gameState = stateEncoder.encode(game, player);
        List<float[]> candidates = new ArrayList<>();
        for (Card c : possibleAttackers) {
            candidates.add(forge.ai.rl.features.CardFeatures.encode(c, player));
        }
        DecisionContext context = new DecisionContext(
                DecisionType.DECLARE_ATTACKERS, gameState, candidates,
                describeCards(possibleAttackers),
                0, possibleAttackers.size(), "declare_attackers", null);

        DecisionResult result = requestDecision(context);
        if (result == null) {
            throw new ModelServerException(
                    "Model server unavailable for attack decision");
        }
        float[] probs = result.getActionProbabilities();
        StringBuilder sb = new StringBuilder("RL_ATTACK: n=");
        sb.append(possibleAttackers.size()).append(" selected=").append(result.getSelectedIndices());
        sb.append(" probs=[");
        for (int i = 0; i < probs.length && i < possibleAttackers.size(); i++) {
            if (i > 0) sb.append(",");
            sb.append(String.format("%.2f", probs[i]));
        }
        sb.append("] value=").append(String.format("%.3f", result.getValueEstimate()));
        Logger.info(sb.toString());
        recordDecision(context, result);
        return result.getSelectedIndices();
    }

    /**
     * Choose blocking assignments.
     *
     * @param possibleBlockers creatures that can legally block
     * @param attackers the attacking creatures
     * @return list of pairs (blocker_index, attacker_index), or empty for no blocks
     */
    public List<int[]> decideBlockers(List<Card> possibleBlockers, List<Card> attackers) {
        if (possibleBlockers.isEmpty() || attackers.isEmpty()) return List.of();

        GameStateFeatures gameState = stateEncoder.encode(game, player);

        // Pre-encode all blockers and attackers (combat math injected automatically by encode)
        List<float[]> blockerFeats = new ArrayList<>();
        for (Card c : possibleBlockers) {
            blockerFeats.add(forge.ai.rl.features.CardFeatures.encode(c, player));
        }

        List<float[]> attackerFeats = new ArrayList<>();
        for (Card c : attackers) {
            attackerFeats.add(forge.ai.rl.features.CardFeatures.encode(c, player));
        }

        // Each candidate is a (blocker, attacker) pair
        List<float[]> candidates = new ArrayList<>();
        List<int[]> pairIndices = new ArrayList<>();

        for (int b = 0; b < possibleBlockers.size(); b++) {
            for (int a = 0; a < attackers.size(); a++) {
                float[] bf = blockerFeats.get(b);
                float[] af = attackerFeats.get(a);
                // Concatenate blocker + attacker features
                float[] combined = new float[bf.length + af.length];
                System.arraycopy(bf, 0, combined, 0, bf.length);
                System.arraycopy(af, 0, combined, bf.length, af.length);
                candidates.add(combined);
                pairIndices.add(new int[]{b, a});
            }
        }
        // Add "no block" option
        candidates.add(new float[candidates.get(0).length]); // zero vector = no block

        DecisionContext context = new DecisionContext(
                DecisionType.DECLARE_BLOCKERS, gameState, candidates,
                describeBlockPairs(possibleBlockers, attackers),
                0, Math.min(possibleBlockers.size(), candidates.size()), "declare_blockers", null);

        DecisionResult result = requestDecision(context);
        if (result == null) {
            throw new ModelServerException(
                    "Model server unavailable for block decision");
        }
        recordDecision(context, result);

        // Convert selected indices to blocker-attacker pairs
        List<int[]> assignments = new ArrayList<>();
        for (int idx : result.getSelectedIndices()) {
            if (idx < pairIndices.size()) {
                assignments.add(pairIndices.get(idx));
            }
        }

        StringBuilder sb = new StringBuilder("RL_BLOCK: blockers=");
        sb.append(possibleBlockers.size()).append(" attackers=").append(attackers.size());
        sb.append(" assignments=").append(assignments.size());
        sb.append(" value=").append(String.format("%.3f", result.getValueEstimate()));
        Logger.info(sb.toString());

        return assignments;
    }

    /**
     * Choose cards from a set (for discard, sacrifice, scry, etc.)
     *
     * @param candidates the cards to choose from
     * @param min minimum to select
     * @param max maximum to select
     * @return indices of selected cards
     */
    public List<Integer> decideCardSelection(CardCollectionView candidates, int min, int max) {
        if (candidates.isEmpty()) return List.of();

        GameStateFeatures gameState = stateEncoder.encode(game, player);
        List<float[]> candidateFeatures = new ArrayList<>();
        List<Card> cardList = new ArrayList<>();
        for (Card c : candidates) {
            candidateFeatures.add(forge.ai.rl.features.CardFeatures.encode(c, player));
            cardList.add(c);
        }
        DecisionContext context = new DecisionContext(
                DecisionType.CARD_SELECTION, gameState, candidateFeatures,
                describeCards(cardList),
                min, max, "card_selection", null);

        DecisionResult result = requestDecision(context);
        if (result == null) {
            // Fallback: select first min cards
            List<Integer> fallback = new ArrayList<>();
            for (int i = 0; i < min && i < candidates.size(); i++) fallback.add(i);
            return fallback;
        }
        recordDecision(context, result);
        return result.getSelectedIndices();
    }

    /**
     * Make a binary (yes/no) decision.
     */
    public boolean decideBinary(String contextInfo) {
        GameStateFeatures gameState = stateEncoder.encode(game, player);
        DecisionContext context = new DecisionContext(
                DecisionType.BINARY_CHOICE, gameState, List.of(),
                List.of("NO", "YES"),
                1, 1, contextInfo, null);

        DecisionResult result = requestDecision(context);
        if (result == null) return false; // fallback: no
        recordDecision(context, result);
        return result.getSelectedIndex() == 1;
    }

    /**
     * Make a mulligan keep/reject decision.
     *
     * @param handCards the current hand
     * @param cardsToReturn number of cards to return (0 = keep)
     * @return true to keep, false to mulligan
     */
    public boolean decideMulligan(CardCollectionView handCards, int cardsToReturn) {
        GameStateFeatures gameState = stateEncoder.encode(game, player);

        List<float[]> candidates = new ArrayList<>();
        for (Card c : handCards) {
            candidates.add(forge.ai.rl.features.CardFeatures.encode(c, player));
        }

        DecisionContext context = new DecisionContext(
                DecisionType.MULLIGAN, gameState, candidates,
                List.of("MULLIGAN", "KEEP"),
                0, 0, "mulligan_keep_" + cardsToReturn, null);

        DecisionResult result = requestDecision(context);
        if (result == null) return cardsToReturn <= 1; // fallback: keep if 0-1 cards to return
        recordDecision(context, result);
        return result.getSelectedIndex() == 1; // 1 = keep
    }

    /**
     * Choose a number within a range.
     */
    public int decideNumber(int min, int max, String contextInfo) {
        GameStateFeatures gameState = stateEncoder.encode(game, player);

        List<float[]> candidates = new ArrayList<>();
        for (int i = min; i <= max; i++) {
            float[] feat = new float[4];
            feat[0] = (float)(i - min) / Math.max(1, max - min); // normalized position
            feat[1] = (float) i; // raw value
            feat[2] = (float) min;
            feat[3] = (float) max;
            candidates.add(feat);
        }

        DecisionContext context = new DecisionContext(
                DecisionType.CATEGORICAL_CHOICE, gameState, candidates,
                describeNumbers(min, max),
                1, 1, contextInfo, null);

        DecisionResult result = requestDecision(context);
        if (result == null) return min; // fallback: minimum
        recordDecision(context, result);
        return min + result.getSelectedIndex();
    }

    // --- Pre-decision capture for heuristic recording ---

    // Cached pre-decision state for recording heuristic choices
    private GameStateFeatures cachedPreDecisionState;
    private List<float[]> cachedCandidateFeatures;

    /**
     * Capture the game state and candidate features BEFORE the heuristic
     * makes a combat decision. Called from PlayerControllerRL before
     * delegating to super.declareAttackers/declareBlockers.
     */
    public void capturePreDecisionState(List<Card> candidates) {
        cachedPreDecisionState = stateEncoder.encode(game, player);
        cachedCandidateFeatures = new ArrayList<>();
        for (Card c : candidates) {
            cachedCandidateFeatures.add(forge.ai.rl.features.CardFeatures.encode(c, player));
        }
    }

    /**
     * Record the heuristic's attack decision paired with the pre-decision
     * state captured earlier. This ensures training data has the same
     * state representation as live inference.
     */
    public void recordHeuristicAttack(List<Card> possibleAttackers, List<Integer> selectedIndices) {
        if ((trajectoryRecorder == null && replayTraceRecorder == null)
                || cachedPreDecisionState == null) return;

        DecisionContext context = new DecisionContext(
                DecisionType.DECLARE_ATTACKERS, cachedPreDecisionState,
                cachedCandidateFeatures,
                describeCards(possibleAttackers),
                0, possibleAttackers.size(),
                "attack_" + selectedIndices.size() + "_of_" + possibleAttackers.size(),
                null);

        DecisionResult result = new DecisionResult(
                selectedIndices, new float[0], 0f, true);

        recordDecision(context, result);
        cachedPreDecisionState = null;
        cachedCandidateFeatures = null;
    }

    /**
     * Record the heuristic's block decision paired with the pre-decision state.
     */
    public void recordHeuristicBlock(List<Card> possibleBlockers, List<Integer> selectedIndices) {
        if ((trajectoryRecorder == null && replayTraceRecorder == null)
                || cachedPreDecisionState == null) return;

        DecisionContext context = new DecisionContext(
                DecisionType.DECLARE_BLOCKERS, cachedPreDecisionState,
                cachedCandidateFeatures,
                describeCards(possibleBlockers),
                0, Math.min(possibleBlockers.size(),
                    cachedCandidateFeatures.size()),
                "block_" + selectedIndices.size(),
                null);

        DecisionResult result = new DecisionResult(
                selectedIndices, new float[0], 0f, true);

        recordDecision(context, result);
        cachedPreDecisionState = null;
        cachedCandidateFeatures = null;
    }

    /**
     * Record the heuristic's block decision with full (blocker, attacker) pair assignment.
     * Matches the format used at inference time in decideBlockers().
     *
     * Each candidate is a concatenated (blocker_features, attacker_features) = 256 dims.
     * selectedIndices marks which (blocker, attacker) pairs are active assignments.
     */
    public void recordHeuristicBlockAssignment(
            List<Card> possibleBlockers, List<Card> attackers,
            forge.game.combat.Combat combat) {
        if ((trajectoryRecorder == null && replayTraceRecorder == null)
                || cachedPreDecisionState == null) return;

        // Pre-encode and enrich all blockers and attackers with combat math
        List<float[]> blockerFeats = new ArrayList<>();
        for (Card c : possibleBlockers) {
            blockerFeats.add(forge.ai.rl.features.CardFeatures.encode(c, player));
        }

        List<float[]> attackerFeats = new ArrayList<>();
        for (Card c : attackers) {
            attackerFeats.add(forge.ai.rl.features.CardFeatures.encode(c, player));
        }

        // Build (blocker, attacker) pair candidates
        List<float[]> candidates = new ArrayList<>();
        List<int[]> pairIndices = new ArrayList<>();

        for (int b = 0; b < possibleBlockers.size(); b++) {
            for (int a = 0; a < attackers.size(); a++) {
                float[] bf = blockerFeats.get(b);
                float[] af = attackerFeats.get(a);
                float[] combined = new float[bf.length + af.length];
                System.arraycopy(bf, 0, combined, 0, bf.length);
                System.arraycopy(af, 0, combined, bf.length, af.length);
                candidates.add(combined);
                pairIndices.add(new int[]{b, a});
            }
        }
        // Add "no block" option (zero vector)
        if (!candidates.isEmpty()) {
            candidates.add(new float[candidates.get(0).length]);
        }

        // Find which pairs are active in the heuristic's assignment
        List<Integer> selectedIndices = new ArrayList<>();
        for (int pairIdx = 0; pairIdx < pairIndices.size(); pairIdx++) {
            int b = pairIndices.get(pairIdx)[0];
            int a = pairIndices.get(pairIdx)[1];
            Card blocker = possibleBlockers.get(b);
            Card attacker = attackers.get(a);
            if (combat.getBlockers(attacker).contains(blocker)) {
                selectedIndices.add(pairIdx);
            }
        }

        DecisionContext context = new DecisionContext(
                DecisionType.DECLARE_BLOCKERS, cachedPreDecisionState,
                candidates.isEmpty() ? cachedCandidateFeatures : candidates,
                describeBlockPairs(possibleBlockers, attackers),
                0, Math.min(possibleBlockers.size(),
                    candidates.isEmpty() ? cachedCandidateFeatures.size()
                        : candidates.size()),
                "block_assign_" + selectedIndices.size()
                    + "_of_" + possibleBlockers.size()
                    + "x" + attackers.size(),
                null);

        DecisionResult result = new DecisionResult(
                selectedIndices, new float[0], 0f, true);

        recordDecision(context, result);
        cachedPreDecisionState = null;
        cachedCandidateFeatures = null;
    }

    /**
     * Record the heuristic's priority decision (which spell to play, or pass).
     *
     * @param availableActions all playable spells/abilities (excluding lands/mana)
     * @param chosenSa the spell the heuristic chose, or null for pass
     */
    public void recordHeuristicPriority(List<SpellAbility> availableActions, SpellAbility chosenSa) {
        if (trajectoryRecorder == null && replayTraceRecorder == null) return;
        // Only record when there's at least 1 spell option (+ pass = 2 candidates)
        if (availableActions.isEmpty()) return;

        GameStateFeatures gameState = stateEncoder.encode(game, player);

        // Encode each available action + pass (same format as inference)
        List<float[]> candidates = new ArrayList<>();
        for (SpellAbility sa : availableActions) {
            candidates.add(ActionEncoder.encode(sa));
        }
        candidates.add(ActionEncoder.encodePassAction()); // pass is always last

        // Find which action the heuristic chose
        int selectedIdx = availableActions.size(); // default = pass (last index)
        if (chosenSa != null) {
            // Try identity match first
            for (int i = 0; i < availableActions.size(); i++) {
                if (availableActions.get(i) == chosenSa) {
                    selectedIdx = i;
                    break;
                }
            }
            // Fallback: match by card name + API type
            if (selectedIdx == availableActions.size() && chosenSa.getHostCard() != null) {
                String chosenName = chosenSa.getHostCard().getName();
                Object chosenApi = chosenSa.getApi();
                for (int i = 0; i < availableActions.size(); i++) {
                    SpellAbility sa = availableActions.get(i);
                    if (sa.getHostCard() != null
                            && sa.getHostCard().getName().equals(chosenName)
                            && sa.getApi() == chosenApi) {
                        selectedIdx = i;
                        break;
                    }
                }
            }
        }

        DecisionContext context = new DecisionContext(
                DecisionType.PRIORITY_ACTION, gameState, candidates,
                describeSpellAbilities(availableActions, true),
                1, 1,
                "priority_" + availableActions.size() + "_options",
                null);

        DecisionResult result = new DecisionResult(
                List.of(selectedIdx), new float[0], 0f, true);

        recordDecision(context, result);
    }

    /**
     * Record a priority decision made by MCTS with rollout win rates as soft targets.
     */
    public void recordMCTSPriority(List<SpellAbility> availableActions, SpellAbility chosenSa,
                                    float[] winRates, float[] visitProportions,
                                    float valueEstimate) {
        if (trajectoryRecorder == null) return;
        if (availableActions.isEmpty()) return;

        GameStateFeatures gameState = stateEncoder.encode(game, player);

        List<float[]> candidates = new ArrayList<>();
        for (SpellAbility sa : availableActions) {
            candidates.add(ActionEncoder.encode(sa));
        }
        candidates.add(ActionEncoder.encodePassAction());

        int selectedIdx = availableActions.size(); // default = pass
        if (chosenSa != null) {
            for (int i = 0; i < availableActions.size(); i++) {
                if (availableActions.get(i) == chosenSa) {
                    selectedIdx = i;
                    break;
                }
            }
            if (selectedIdx == availableActions.size() && chosenSa.getHostCard() != null) {
                String chosenName = chosenSa.getHostCard().getName();
                Object chosenApi = chosenSa.getApi();
                for (int i = 0; i < availableActions.size(); i++) {
                    SpellAbility sa = availableActions.get(i);
                    if (sa.getHostCard() != null
                            && sa.getHostCard().getName().equals(chosenName)
                            && sa.getApi() == chosenApi) {
                        selectedIdx = i;
                        break;
                    }
                }
            }
        }

        DecisionContext context = DecisionContext.singleSelect(
                DecisionType.PRIORITY_ACTION, gameState, candidates,
                "mcts_priority_" + availableActions.size() + "_options");

        // Store MCTS win rates and visit proportions
        DecisionResult result = new DecisionResult(
                List.of(selectedIdx), winRates, visitProportions,
                valueEstimate, false);

        recordDecision(context, result);
    }

    /**
     * Record an attack decision made by MCTS with per-creature win rates.
     */
    public void recordMCTSAttack(List<Card> possibleAttackers, List<Integer> selectedIndices,
                                  float[] creatureWinRates, float[] creatureVisitProps,
                                  float valueEstimate) {
        if (trajectoryRecorder == null) return;

        GameStateFeatures gameState = cachedPreDecisionState;
        if (gameState == null) {
            gameState = stateEncoder.encode(game, player);
        }

        List<float[]> candidateFeats = cachedCandidateFeatures;
        if (candidateFeats == null) {
            candidateFeats = new ArrayList<>();
            for (Card c : possibleAttackers) {
                candidateFeats.add(forge.ai.rl.features.CardFeatures.encode(c, player));
            }
        }

        DecisionContext context = DecisionContext.multiSelect(
                DecisionType.DECLARE_ATTACKERS, gameState, candidateFeats,
                0, possibleAttackers.size(), "mcts_attack");

        DecisionResult result = new DecisionResult(
                selectedIndices, creatureWinRates, creatureVisitProps,
                valueEstimate, false);

        recordDecision(context, result);
        cachedPreDecisionState = null;
        cachedCandidateFeatures = null;
    }

    /**
     * Record a target decision made by MCTS with per-target win rates.
     */
    public void recordMCTSTarget(SpellAbility spell, List<GameEntity> targets,
                                  int selectedTargetIdx,
                                  float[] targetWinRates, float[] targetVisitProps,
                                  float valueEstimate) {
        if (trajectoryRecorder == null) return;
        try {
            GameStateFeatures gameState = stateEncoder.encode(game, player);
            List<float[]> candidateFeats = new ArrayList<>();
            for (GameEntity entity : targets) {
                if (entity instanceof forge.game.card.Card) {
                    candidateFeats.add(forge.ai.rl.features.CardFeatures.encode(
                            (forge.game.card.Card) entity, player));
                } else {
                    float[] playerFeats = ActionEncoder.encodeTarget(entity);
                    float[] padded = new float[256];
                    System.arraycopy(playerFeats, 0, padded, 0,
                            Math.min(playerFeats.length, 256));
                    candidateFeats.add(padded);
                }
            }

            String spellName = spell.getHostCard() != null
                    ? spell.getHostCard().getName() : "?";
            DecisionContext context = new DecisionContext(
                    DecisionType.TARGET_SELECTION, gameState, candidateFeats,
                    1, 1, "mcts_target_" + spellName,
                    ActionEncoder.encode(spell));

            DecisionResult result = new DecisionResult(
                    List.of(selectedTargetIdx), targetWinRates,
                    targetVisitProps, valueEstimate, false);

            recordDecision(context, result);
        } catch (Exception e) {
            // Never crash
        }
    }

    // --- Internal helpers ---

    private DecisionResult requestDecision(DecisionContext context) {
        DecisionResult result;
        switch (config.getMode()) {
            case GRPC:
                result = modelClient.requestDecision(context);
                break;
            case ONNX:
                result = onnxClient != null ? onnxClient.requestDecision(context) : null;
                break;
            case HEURISTIC_FALLBACK:
            case RECORD_HEURISTIC:
            default:
                return null; // caller should use heuristic fallback
        }
        // Store latest value estimate for GUI display
        if (result != null && player != null) {
            latestValueEstimates.put(player.getName(), result.getValueEstimate());
        }
        return result;
    }

    private void recordDecision(DecisionContext context, DecisionResult result) {
        if ((trajectoryRecorder == null && replayTraceRecorder == null) || result == null) return;

        Player opp = player.getWeakestOpponent();
        if (trajectoryRecorder != null) {
            trajectoryRecorder.recordDecision(
                    context, result,
                    player.getLife(),
                    opp != null ? opp.getLife() : 0,
                    player.getCardsIn(ZoneType.Hand).size(),
                    opp != null ? opp.getCardsIn(ZoneType.Hand).size() : 0,
                    countCreatures(player),
                    opp != null ? countCreatures(opp) : 0
            );
        }
        if (replayTraceRecorder != null) {
            replayTraceRecorder.record(game, player, context, result);
        }
    }

    private int countCreatures(Player p) {
        int count = 0;
        for (Card c : p.getCardsIn(ZoneType.Battlefield)) {
            if (c.isCreature()) count++;
        }
        return count;
    }

    /**
     * Record a decision directly from PlayerControllerRL overrides.
     * Used for target selection, card selection, mulligan, and binary choices.
     */
    public void recordDecisionDirect(DecisionType type,
                                      int numCandidates,
                                      List<Integer> selected,
                                      List<float[]> candidateFeats,
                                      String info) {
        recordDecisionDirect(type, numCandidates, selected, candidateFeats, List.of(), info, null);
    }

    public void recordDecisionDirect(DecisionType type,
                                      int numCandidates,
                                      List<Integer> selected,
                                      List<float[]> candidateFeats,
                                      List<String> candidateLabels,
                                      String info) {
        recordDecisionDirect(type, numCandidates, selected, candidateFeats,
                candidateLabels, info, null);
    }

    public void recordDecisionDirect(DecisionType type,
                                      int numCandidates,
                                      List<Integer> selected,
                                      List<float[]> candidateFeats,
                                      List<String> candidateLabels,
                                      String info,
                                      float[] spellFeatures) {
        recordDecisionDirect(type, numCandidates, selected, candidateFeats,
                candidateLabels, candidateLabels, info, spellFeatures);
    }

    public void recordDecisionDirect(DecisionType type,
                                      int numCandidates,
                                      List<Integer> selected,
                                      List<float[]> candidateFeats,
                                      List<String> candidateLabels,
                                      List<String> traceCandidateLabels,
                                      String info,
                                      float[] spellFeatures) {
        if (trajectoryRecorder == null && replayTraceRecorder == null) return;
        try {
            GameStateFeatures gs = stateEncoder.encode(game, player);
            DecisionContext ctx = new DecisionContext(
                    type, gs,
                    candidateFeats != null ? candidateFeats : List.of(),
                    candidateLabels,
                    traceCandidateLabels,
                    selected.size(), numCandidates, info, spellFeatures);
            boolean isFallback = config.getMode() != RLModelMode.GRPC
                    && config.getMode() != RLModelMode.ONNX;
            DecisionResult res = new DecisionResult(
                    selected, new float[0], 0f, isFallback);
            recordDecision(ctx, res);
        } catch (Exception e) {
            // Never crash the game due to recording errors
        }
    }

    public void recordDecisionDirect(DecisionType type,
                                      int numCandidates,
                                      List<Integer> selected,
                                      List<float[]> candidateFeats,
                                      String info,
                                      float[] spellFeatures) {
        recordDecisionDirect(type, numCandidates, selected, candidateFeats,
                List.of(), info, spellFeatures);
    }

    private String getBackendLabel() {
        switch (config.getMode()) {
            case ONNX:
                return "onnx";
            case GRPC:
                return "grpc";
            case RECORD_HEURISTIC:
            case HEURISTIC_FALLBACK:
            default:
                return "heuristic";
        }
    }

    private String getModelIdentifier() {
        switch (config.getMode()) {
            case ONNX:
                return config.getOnnxModelDir();
            case GRPC:
                ModelServerClient.ServerMetadata metadata = modelClient.getCachedMetadata();
                return metadata != null ? metadata.modelId : config.getGrpcHost() + ":" + config.getGrpcPort();
            case RECORD_HEURISTIC:
            case HEURISTIC_FALLBACK:
            default:
                return "heuristic";
        }
    }

    private static List<String> describeCards(Collection<Card> cards) {
        List<String> labels = new ArrayList<>();
        for (Card card : cards) {
            labels.add(card.getName());
        }
        return labels;
    }

    private static List<String> describeEntities(Collection<? extends GameEntity> entities) {
        List<String> labels = new ArrayList<>();
        for (GameEntity entity : entities) {
            labels.add(entity.toString());
        }
        return labels;
    }

    private static List<String> describeSpellAbilities(List<SpellAbility> spells, boolean includePass) {
        List<String> labels = new ArrayList<>();
        for (SpellAbility sa : spells) {
            String name = sa.getHostCard() != null ? sa.getHostCard().getName() : "Spell";
            String api = sa.getApi() != null ? sa.getApi().name() : "None";
            labels.add(name + " [" + api + "]");
        }
        if (includePass) {
            labels.add("PASS");
        }
        return labels;
    }

    private static List<String> describeBlockPairs(List<Card> blockers, List<Card> attackers) {
        List<String> labels = new ArrayList<>();
        for (Card blocker : blockers) {
            for (Card attacker : attackers) {
                labels.add(blocker.getName() + " -> " + attacker.getName());
            }
        }
        if (!labels.isEmpty()) {
            labels.add("NO_BLOCK");
        }
        return labels;
    }

    private static List<String> describeNumbers(int min, int max) {
        List<String> labels = new ArrayList<>();
        for (int i = min; i <= max; i++) {
            labels.add(String.valueOf(i));
        }
        return labels;
    }
}
