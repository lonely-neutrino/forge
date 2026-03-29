package forge.ai.rl.training;

import com.google.common.eventbus.Subscribe;
import forge.ai.rl.RLConfig;
import forge.ai.rl.features.ActionEncoder;
import forge.ai.rl.features.CardFeatures;
import forge.ai.rl.features.GameStateEncoder;
import forge.ai.rl.features.GameStateFeatures;
import forge.ai.rl.decisions.DecisionContext;
import forge.ai.rl.decisions.DecisionResult;
import forge.ai.rl.decisions.DecisionType;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CardCollectionView;
import forge.game.combat.Combat;
import forge.game.player.IGameRecorder;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import org.tinylog.Logger;

import java.util.*;

/**
 * Records human player decisions during GUI games.
 *
 * Implements IGameRecorder so it can be attached to PlayerControllerHuman.
 * Uses the same feature encoding and JSONL format as the RL training
 * pipeline, so human game data can be directly used for imitation learning.
 *
 * Created via reflection from HostedMatch when UI_RECORD_HUMAN_GAMES is enabled.
 */
public class HumanGameRecorder implements IGameRecorder {
    private final Player player;
    private final GameStateEncoder stateEncoder;
    private final TrajectoryRecorder trajectoryRecorder;
    private final Game game;

    // Pre-decision state captured before human acts
    private GameStateFeatures cachedState;
    private List<float[]> cachedCandidates;

    public HumanGameRecorder(Player player, String outputDir) {
        this.player = player;
        this.game = player.getGame();
        this.stateEncoder = new GameStateEncoder(new RLConfig());
        this.trajectoryRecorder = new TrajectoryRecorder(outputDir, false);
    }

    public void onGameStart(String gameId) {
        trajectoryRecorder.startGame(gameId);
        Logger.info("HumanGameRecorder: Recording started for {}", gameId);
    }

    public void onGameEnd(boolean won) {
        trajectoryRecorder.endGame(won);
    }

    // ── Priority recording ──────────────────────────────

    /**
     * Call BEFORE the human makes a priority decision.
     * Captures the game state. Candidates may be null for human players
     * (no AiController to enumerate legal spells).
     */
    public void capturePrePriority(List<SpellAbility> candidates) {
        try {
            cachedState = stateEncoder.encode(game, player);
            if (candidates != null && !candidates.isEmpty()) {
                cachedCandidates = new ArrayList<>();
                for (SpellAbility sa : candidates) {
                    cachedCandidates.add(ActionEncoder.encode(sa));
                }
            } else {
                cachedCandidates = null;
            }
        } catch (Exception e) {
            Logger.warn("HumanGameRecorder: Failed to capture priority state: {}", e.getMessage());
            cachedState = null;
        }
    }

    /**
     * Call AFTER the human makes a priority decision.
     *
     * @param candidates the candidate list (may be null for human players)
     * @param chosen the SpellAbility the human chose, or null for pass
     */
    public void recordPriorityDecision(List<SpellAbility> candidates, SpellAbility chosen) {
        if (cachedState == null) return;

        try {
            // If we have candidates, record full decision with candidate features
            // If not, record just the chosen spell (or pass) with its features
            List<float[]> features = new ArrayList<>();
            int selectedIdx;

            if (chosen == null || chosen.isLandAbility()) {
                // Pass — record with just a pass candidate
                features.add(new float[64]); // pass = zero vector
                selectedIdx = 0;
            } else {
                // Play — record the chosen spell + pass
                features.add(ActionEncoder.encode(chosen));
                features.add(new float[64]); // pass option
                selectedIdx = 0; // chose to play (index 0)
            }

            if (cachedCandidates != null) {
                // Full candidate list available — use it
                features = cachedCandidates;
                features.add(new float[64]); // add pass
                if (chosen == null || chosen.isLandAbility()) {
                    selectedIdx = features.size() - 1; // pass
                } else {
                    // Find chosen in candidates
                    selectedIdx = features.size() - 1; // default to pass
                    // Can't match SpellAbility to cached features by reference
                    // since candidates list may be null — use the action features
                    float[] chosenFeats = ActionEncoder.encode(chosen);
                    for (int i = 0; i < features.size() - 1; i++) {
                        if (java.util.Arrays.equals(features.get(i), chosenFeats)) {
                            selectedIdx = i;
                            break;
                        }
                    }
                }
            }

            String contextInfo = chosen != null && !chosen.isLandAbility()
                    ? "play_" + chosen.getHostCard().getName()
                    : "pass";

            DecisionContext context = DecisionContext.multiSelect(
                    DecisionType.PRIORITY_ACTION, cachedState,
                    features, 0, 1, contextInfo);

            DecisionResult result = new DecisionResult(
                    List.of(selectedIdx), new float[0], 0f, false);

            recordWithStats(context, result);
        } catch (Exception e) {
            Logger.warn("HumanGameRecorder: Failed to record priority: {}", e.getMessage());
        } finally {
            cachedState = null;
            cachedCandidates = null;
        }
    }

    // ── Attack recording ────────────────────────────────

    /**
     * Call BEFORE declare attackers step.
     */
    public void capturePreAttack(List<Card> possibleAttackers) {
        if (possibleAttackers == null || possibleAttackers.isEmpty()) return;
        try {
            cachedState = stateEncoder.encode(game, player);
            cachedCandidates = new ArrayList<>();
            for (Card c : possibleAttackers) {
                cachedCandidates.add(CardFeatures.encode(c, player));
            }
        } catch (Exception e) {
            Logger.warn("HumanGameRecorder: Failed to capture attack state: {}", e.getMessage());
            cachedState = null;
        }
    }

    /**
     * Call AFTER declare attackers with the combat result.
     */
    public void recordAttackDecision(List<Card> possibleAttackers, Combat combat) {
        if (cachedState == null || cachedCandidates == null) return;
        if (possibleAttackers == null || possibleAttackers.isEmpty()) return;

        try {
            List<Integer> selected = new ArrayList<>();
            for (int i = 0; i < possibleAttackers.size(); i++) {
                if (combat.isAttacking(possibleAttackers.get(i))) {
                    selected.add(i);
                }
            }

            DecisionContext context = DecisionContext.multiSelect(
                    DecisionType.DECLARE_ATTACKERS, cachedState,
                    cachedCandidates,
                    0, possibleAttackers.size(),
                    "attack_" + selected.size() + "_of_" + possibleAttackers.size());

            DecisionResult result = new DecisionResult(
                    selected, new float[0], 0f, false);

            recordWithStats(context, result);
        } catch (Exception e) {
            Logger.warn("HumanGameRecorder: Failed to record attack: {}", e.getMessage());
        } finally {
            cachedState = null;
            cachedCandidates = null;
        }
    }

    // ── Block recording ─────────────────────────────────

    /**
     * Call AFTER declare blockers with the combat result.
     */
    public void recordBlockDecision(List<Card> possibleBlockers,
                                     List<Card> attackers, Combat combat) {
        if (possibleBlockers == null || possibleBlockers.isEmpty()) return;
        if (attackers == null || attackers.isEmpty()) return;

        try {
            GameStateFeatures state = stateEncoder.encode(game, player);

            // Pre-encode blockers and attackers (combat math injected automatically by encode)
            List<float[]> blockerFeats = new ArrayList<>();
            for (Card c : possibleBlockers) {
                blockerFeats.add(CardFeatures.encode(c, player));
            }

            List<float[]> attackerFeats = new ArrayList<>();
            for (Card c : attackers) {
                attackerFeats.add(CardFeatures.encode(c, player));
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

            // Find selected pairs
            List<Integer> selected = new ArrayList<>();
            for (int idx = 0; idx < pairIndices.size(); idx++) {
                int b = pairIndices.get(idx)[0];
                int a = pairIndices.get(idx)[1];
                Card blocker = possibleBlockers.get(b);
                Card attacker = attackers.get(a);
                if (combat.getBlockers(attacker).contains(blocker)) {
                    selected.add(idx);
                }
            }

            // Add no-block option
            candidates.add(new float[candidates.isEmpty() ? 512 : candidates.get(0).length]);

            DecisionContext context = DecisionContext.multiSelect(
                    DecisionType.DECLARE_BLOCKERS, state, candidates,
                    0, Math.min(possibleBlockers.size(), candidates.size()),
                    "block_assign_" + selected.size()
                            + "_of_" + possibleBlockers.size()
                            + "x" + attackers.size());

            DecisionResult result = new DecisionResult(
                    selected, new float[0], 0f, false);

            recordWithStats(context, result);
        } catch (Exception e) {
            Logger.warn("HumanGameRecorder: Failed to record block: {}", e.getMessage());
        }
    }

    // ── Mulligan recording ──────────────────────────────

    /**
     * Record a mulligan decision.
     */
    public void recordMulligan(CardCollectionView hand, boolean kept) {
        try {
            GameStateFeatures state = stateEncoder.encode(game, player);
            List<float[]> candidates = new ArrayList<>();
            for (Card c : hand) {
                candidates.add(CardFeatures.encode(c, player));
            }

            DecisionContext context = DecisionContext.multiSelect(
                    DecisionType.MULLIGAN, state, candidates,
                    0, 1,
                    "mulligan_" + (kept ? "keep" : "mull"));

            DecisionResult result = new DecisionResult(
                    List.of(kept ? 1 : 0), new float[0], 0f, false);

            recordWithStats(context, result);
        } catch (Exception e) {
            Logger.warn("HumanGameRecorder: Failed to record mulligan: {}", e.getMessage());
        }
    }

    // ── Target recording ────────────────────────────────

    /**
     * Record a target selection decision.
     */
    public void recordTargetDecision(List<?> candidates,
                                      int selectedIdx) {
        try {
            GameStateFeatures state = stateEncoder.encode(game, player);
            List<float[]> candidateFeatures = new ArrayList<>();
            for (Object obj : candidates) {
                if (obj instanceof Card) {
                    candidateFeatures.add(CardFeatures.encode((Card) obj, player));
                } else {
                    candidateFeatures.add(new float[256]); // player or other entity
                }
            }
            DecisionContext context = DecisionContext.multiSelect(
                    DecisionType.TARGET_SELECTION, state, candidateFeatures,
                    1, 1,
                    "target_" + candidates.size() + "_options");

            DecisionResult result = new DecisionResult(
                    List.of(selectedIdx), new float[0], 0f, false);

            recordWithStats(context, result);
        } catch (Exception e) {
            Logger.warn("HumanGameRecorder: Failed to record target: {}", e.getMessage());
        }
    }

    // ── Binary recording ────────────────────────────────

    /**
     * Record a binary yes/no decision.
     */
    public void recordBinaryDecision(boolean yes, String context) {
        try {
            GameStateFeatures state = stateEncoder.encode(game, player);
            List<float[]> candidates = List.of(); // no candidates for binary

            DecisionContext ctx = DecisionContext.multiSelect(
                    DecisionType.BINARY_CHOICE, state, candidates,
                    0, 1,
                    "binary_" + context);

            DecisionResult result = new DecisionResult(
                    List.of(yes ? 1 : 0), new float[0], 0f, false);

            recordWithStats(ctx, result);
        } catch (Exception e) {
            Logger.warn("HumanGameRecorder: Failed to record binary: {}", e.getMessage());
        }
    }

    // ── Shared recording helper ─────────────────────────

    private void recordWithStats(DecisionContext context, DecisionResult result) {
        Player opp = player.getWeakestOpponent();
        int myLife = player.getLife();
        int oppLife = opp != null ? opp.getLife() : 20;
        int myHand = player.getCardsIn(forge.game.zone.ZoneType.Hand).size();
        int oppHand = opp != null ? opp.getCardsIn(forge.game.zone.ZoneType.Hand).size() : 0;
        int myCreatures = player.getCreaturesInPlay().size();
        int oppCreatures = opp != null ? opp.getCreaturesInPlay().size() : 0;

        trajectoryRecorder.recordDecision(context, result,
                myLife, oppLife, myHand, oppHand, myCreatures, oppCreatures);
    }

    /**
     * Get the number of decisions recorded so far.
     */
    public int getDecisionCount() {
        return trajectoryRecorder.getCurrentGameDecisionCount();
    }
}
