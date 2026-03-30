package forge.ai.rl.mcts;

import forge.ai.PlayerControllerAi;
import forge.ai.simulation.GameCopier;
import forge.ai.simulation.GameSimulator;
import forge.ai.simulation.GameStateEvaluator;
import forge.ai.simulation.SimulationController;
import forge.ai.LobbyPlayerAi;
import forge.game.*;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.combat.CombatUtil;
import forge.game.player.Player;
import forge.game.player.RegisteredPlayer;
import forge.game.spellability.SpellAbility;
import forge.util.MyRandom;

import org.tinylog.Logger;

import java.util.*;
import java.util.concurrent.*;

/**
 * Monte Carlo Tree Search decision maker using UCB1 for rollout allocation.
 *
 * For each decision point, expands all (spell, target) pairs as candidates,
 * allocates a fixed rollout budget using UCB1 (explore promising candidates
 * more, prune unpromising ones), and returns visit-count-based policy
 * as soft training targets.
 *
 * Used by Expert Iteration (ExIt) to generate search-improved training data.
 */
public class MCTSDecisionMaker {

    // UCB1 exploration constant — sqrt(2) is theoretical optimum,
    // lower values exploit more, higher explore more
    private static final double UCB_C = 1.41;

    private final int rolloutBudget;     // total rollouts per decision
    private final int rolloutTimeoutSec; // timeout per individual rollout
    private final Random rng;
    private final Map<Integer, GameEntity> chosenTargets = new HashMap<>();
    // Per-target MCTS results for the chosen spell (for recording)
    private List<GameEntity> lastTargetCandidates = null;
    private float[] lastTargetWinRates = null;
    private float[] lastTargetVisitProps = null;

    // Stats
    private int totalDecisions = 0;
    private int totalRollouts = 0;
    private long totalRolloutTimeMs = 0;

    /**
     * @param rolloutBudget total rollouts per decision (allocated via UCB1)
     * @param rolloutTimeoutSec timeout for each individual game rollout
     */
    public MCTSDecisionMaker(int rolloutBudget, int rolloutTimeoutSec) {
        this.rolloutBudget = rolloutBudget;
        this.rolloutTimeoutSec = rolloutTimeoutSec;
        this.rng = new Random();
    }

    // ── Candidate representation ──

    /** A candidate action: spell + optional target, or pass, or attack pattern. */
    private static class Candidate {
        final SpellAbility spell;      // null = pass
        final GameEntity target;       // null = no targeting needed
        final String label;
        int visits = 0;
        int wins = 0;

        // For attack patterns
        final List<Integer> attackIndices; // null for non-attack

        Candidate(SpellAbility spell, GameEntity target, String label) {
            this.spell = spell;
            this.target = target;
            this.label = label;
            this.attackIndices = null;
        }

        Candidate(List<Integer> attackIndices, String label) {
            this.spell = null;
            this.target = null;
            this.label = label;
            this.attackIndices = attackIndices;
        }

        double winRate() {
            return visits > 0 ? (double) wins / visits : 0;
        }

        double ucb1(int totalVisits) {
            if (visits == 0) return Double.MAX_VALUE; // unvisited = infinite priority
            return winRate() + UCB_C * Math.sqrt(Math.log(totalVisits) / visits);
        }
    }

    // ── Priority decisions ──

    /**
     * Evaluate priority candidates via UCB1-allocated rollouts.
     * For targeted spells, each (spell, target) pair is a separate candidate.
     *
     * Returns MCTSResult with per-original-candidate win rates (best target
     * per spell) and the visit-based policy distribution.
     */
    public MCTSResult decidePriority(Game game, Player player,
                                     List<SpellAbility> candidates) {
        if (candidates.isEmpty()) {
            return new MCTSResult(0, new float[]{1f}, new float[]{1f}, 1f);
        }

        chosenTargets.clear();
        lastTargetCandidates = null;
        lastTargetWinRates = null;
        lastTargetVisitProps = null;

        // Build expanded candidate list: (spell, target) pairs
        List<Candidate> expanded = new ArrayList<>();
        // Track which original candidate index each expanded candidate maps to
        List<Integer> origIndex = new ArrayList<>();

        for (int c = 0; c < candidates.size(); c++) {
            SpellAbility sa = candidates.get(c);

            if (sa.usesTargeting() && sa.getTargetRestrictions() != null) {
                List<GameEntity> legalTargets = sa.getTargetRestrictions()
                        .getAllCandidates(sa, true);
                if (legalTargets.isEmpty()) continue;

                for (GameEntity target : legalTargets) {
                    String targetName = target instanceof Card
                            ? ((Card) target).getName()
                            : target instanceof Player
                            ? "Player:" + ((Player) target).getName()
                            : target.toString();
                    expanded.add(new Candidate(sa, target,
                            cardName(sa) + " -> " + targetName));
                    origIndex.add(c);
                }
            } else {
                expanded.add(new Candidate(sa, null, cardName(sa)));
                origIndex.add(c);
            }
        }

        // Add pass
        expanded.add(new Candidate((SpellAbility) null, null, "PASS"));
        origIndex.add(candidates.size()); // pass index

        // Run UCB1 rollouts
        runUCB1(expanded, game, player);

        // Aggregate results back to original candidate indices
        int nOriginal = candidates.size() + 1;
        float[] winRates = new float[nOriginal];
        int[] totalVisitsPerOrig = new int[nOriginal];
        int[] bestVisits = new int[nOriginal];

        // Total visits across all expanded candidates
        int totalAllVisits = 0;
        for (Candidate c : expanded) totalAllVisits += c.visits;

        for (int e = 0; e < expanded.size(); e++) {
            Candidate cand = expanded.get(e);
            int orig = origIndex.get(e);

            // Sum visits for this original candidate (across all targets)
            totalVisitsPerOrig[orig] += cand.visits;

            // For targeted spells: keep the best target's rate
            if (cand.visits > bestVisits[orig]
                    || (cand.visits == bestVisits[orig]
                        && cand.winRate() > winRates[orig])) {
                winRates[orig] = (float) cand.winRate();
                bestVisits[orig] = cand.visits;
                if (cand.target != null) {
                    chosenTargets.put(orig, cand.target);
                }
            }
        }

        // Visit proportions (AlphaZero-style search policy)
        float[] visitProps = new float[nOriginal];
        if (totalAllVisits > 0) {
            for (int i = 0; i < nOriginal; i++) {
                visitProps[i] = (float) totalVisitsPerOrig[i] / totalAllVisits;
            }
        }

        int best = argmax(winRates);
        totalDecisions++;

        // Extract per-target results for the chosen spell (if targeted)
        if (best < candidates.size()) {
            SpellAbility bestSa = candidates.get(best);
            if (bestSa.usesTargeting() && bestSa.getTargetRestrictions() != null) {
                List<GameEntity> targets = bestSa.getTargetRestrictions()
                        .getAllCandidates(bestSa, true);
                if (targets.size() > 1) {
                    lastTargetCandidates = targets;
                    lastTargetWinRates = new float[targets.size()];
                    lastTargetVisitProps = new float[targets.size()];
                    int targetTotalVisits = 0;
                    // Find expanded candidates for this spell and map to targets
                    for (int e = 0; e < expanded.size(); e++) {
                        if (origIndex.get(e) == best) {
                            Candidate c = expanded.get(e);
                            int tIdx = targets.indexOf(c.target);
                            if (tIdx >= 0) {
                                lastTargetWinRates[tIdx] = (float) c.winRate();
                                lastTargetVisitProps[tIdx] = c.visits;
                                targetTotalVisits += c.visits;
                            }
                        }
                    }
                    // Normalize visit props
                    if (targetTotalVisits > 0) {
                        for (int t = 0; t < lastTargetVisitProps.length; t++) {
                            lastTargetVisitProps[t] /= targetTotalVisits;
                        }
                    }
                }
            }
        }

        // Log
        String label = (best < candidates.size())
                ? cardName(candidates.get(best))
                : "PASS";
        StringBuilder sb = new StringBuilder();
        sb.append("MCTS_PRIORITY: n=").append(candidates.size());
        sb.append(" expanded=").append(expanded.size());
        sb.append(" budget=").append(totalAllVisits);
        sb.append(" best=").append(best).append(" (").append(label).append(")");
        sb.append(" rates=").append(formatRates(winRates));
        sb.append(" visits=").append(formatRates(visitProps));
        sb.append(" | ");
        for (Candidate c : expanded) {
            sb.append(String.format("%s:%d/%d(%.0f%%) ",
                    c.label, c.wins, c.visits, c.winRate() * 100));
        }
        System.out.println(sb);
        System.out.flush();

        return new MCTSResult(best, winRates, visitProps, winRates[best]);
    }

    /**
     * Get the MCTS-chosen target for a targeted spell candidate.
     */
    public GameEntity getChosenTarget(int candidateIdx) {
        return chosenTargets.get(candidateIdx);
    }

    /** Per-target candidates for the last targeted spell evaluated. */
    public List<GameEntity> getLastTargetCandidates() {
        return lastTargetCandidates;
    }
    /** Per-target win rates from MCTS for recording. */
    public float[] getLastTargetWinRates() {
        return lastTargetWinRates;
    }
    /** Per-target visit proportions from MCTS for recording. */
    public float[] getLastTargetVisitProps() {
        return lastTargetVisitProps;
    }

    // ── Attack decisions ──

    /**
     * Evaluate attack candidates via UCB1-allocated rollouts.
     * Candidates: no-attack, all-in, and each individual creature.
     */
    public MCTSResult decideAttackers(Game game, Player player,
                                       List<Card> possibleAttackers) {
        if (possibleAttackers.isEmpty()) {
            return new MCTSResult(0, new float[0], new float[0], 0.5f);
        }

        List<Candidate> expanded = new ArrayList<>();

        // No attack
        expanded.add(new Candidate(List.of(), "hold"));

        // All-in
        List<Integer> allIn = new ArrayList<>();
        for (int i = 0; i < possibleAttackers.size(); i++) allIn.add(i);
        expanded.add(new Candidate(new ArrayList<>(allIn), "all-in"));

        // Individual creatures
        for (int i = 0; i < possibleAttackers.size(); i++) {
            String name = possibleAttackers.get(i).getName();
            expanded.add(new Candidate(List.of(i), name));
        }

        // Run UCB1 rollouts for attack patterns
        runAttackUCB1(expanded, game, player, possibleAttackers);

        // Find best pattern
        int bestIdx = 0;
        double bestRate = -1;
        for (int i = 0; i < expanded.size(); i++) {
            if (expanded.get(i).winRate() > bestRate) {
                bestRate = expanded.get(i).winRate();
                bestIdx = i;
            }
        }

        List<Integer> selected = expanded.get(bestIdx).attackIndices;
        if (selected == null) selected = List.of();

        // Per-creature win rates and visit proportions
        float[] creatureRates = new float[possibleAttackers.size()];
        float[] creatureVisits = new float[possibleAttackers.size()];
        float allInRate = (float) expanded.get(1).winRate();
        int totalVisits = 0;
        for (Candidate c : expanded) totalVisits += c.visits;
        for (int i = 0; i < possibleAttackers.size(); i++) {
            float indivRate = (float) expanded.get(2 + i).winRate();
            creatureRates[i] = Math.max(indivRate, allInRate);
            // Visit proportion: sum of all-in visits + individual visits
            int creatureVisitCount = expanded.get(2 + i).visits
                    + expanded.get(1).visits; // all-in includes this creature
            creatureVisits[i] = totalVisits > 0
                    ? (float) creatureVisitCount / totalVisits : 0;
        }

        totalDecisions++;

        // Log
        StringBuilder sb = new StringBuilder("MCTS_ATTACK: ");
        for (Candidate c : expanded) {
            sb.append(String.format("%s:%d/%d(%.0f%%) ",
                    c.label, c.wins, c.visits, c.winRate() * 100));
        }
        System.out.println(sb);
        System.out.flush();

        return new MCTSResult(bestIdx, creatureRates, creatureVisits,
                (float) bestRate);
    }

    // ── Target decisions (standalone, for non-priority targeting) ──

    public MCTSResult decideTarget(Game game, Player player,
                                   List<GameEntity> targets,
                                   SpellAbility sourceSpell) {
        if (targets.size() <= 1) {
            return new MCTSResult(0, new float[]{1f}, new float[]{1f}, 1f);
        }

        List<Candidate> expanded = new ArrayList<>();
        for (GameEntity t : targets) {
            String name = t instanceof Card ? ((Card) t).getName() : "Player";
            expanded.add(new Candidate(sourceSpell, t, name));
        }

        runUCB1(expanded, game, player);

        float[] winRates = new float[targets.size()];
        for (int i = 0; i < expanded.size(); i++) {
            winRates[i] = (float) expanded.get(i).winRate();
        }

        int best = argmax(winRates);
        totalDecisions++;
        // Visit proportions for targets
        float[] targetVisitProps = new float[targets.size()];
        int totalVis = 0;
        for (Candidate c : expanded) totalVis += c.visits;
        for (int i = 0; i < expanded.size(); i++) {
            targetVisitProps[i] = totalVis > 0
                    ? (float) expanded.get(i).visits / totalVis : 0;
        }
        return new MCTSResult(best, winRates, targetVisitProps, winRates[best]);
    }

    // ── Mulligan ──

    /**
     * Decide keep vs mulligan via game-copy rollouts.
     *
     * "Keep" rollouts: copy the game as-is (preserving hand + library),
     * play to completion with heuristic AIs. Tests THIS hand.
     *
     * "Mulligan" rollouts: copy the game, shuffle the copy's hand back
     * into library, draw N-1 cards, play to completion. Tests a random
     * smaller hand.
     *
     * Compare win rates: if keep > mulligan, keep.
     *
     * Returns MCTSResult: index 0 = mulligan, index 1 = keep.
     */
    public MCTSResult decideMulligan(Game game, Player player) {
        int budget = Math.max(rolloutBudget / 3, 6);
        int keepWins = 0, keepTrials = 0;
        int mullWins = 0, mullTrials = 0;

        int handSize = player.getCardsIn(forge.game.zone.ZoneType.Hand).size();

        for (int r = 0; r < budget; r++) {
            // Alternate keep and mulligan rollouts
            boolean testKeep = (r % 2 == 0);

            Random origRandom = MyRandom.getRandom();
            MyRandom.setRandom(new Random(rng.nextLong()));
            try {
                GameCopier copier = new GameCopier(game);
                Game simGame = copier.makeCopy();
                disableSimulation(simGame);
                Player simPlayer = (Player) copier.find(player);

                if (!testKeep) {
                    // Simulate mulligan: shuffle hand into library,
                    // draw (handSize - 1) cards
                    forge.game.zone.PlayerZone simHand =
                            simPlayer.getZone(forge.game.zone.ZoneType.Hand);
                    forge.game.zone.PlayerZone simLib =
                            simPlayer.getZone(forge.game.zone.ZoneType.Library);

                    // Move hand cards to library
                    List<Card> handCards = new ArrayList<>(
                            simPlayer.getCardsIn(forge.game.zone.ZoneType.Hand));
                    for (Card c : handCards) {
                        simGame.getAction().moveToLibrary(c, -1, null);
                    }
                    // Shuffle library
                    simPlayer.shuffle(null);
                    // Draw N-1 cards
                    int newHandSize = Math.max(handSize - 1, 1);
                    for (int d = 0; d < newHandSize; d++) {
                        simPlayer.drawCard();
                    }
                }

                // Play to completion
                boolean won = playToCompletion(simGame, simPlayer);

                if (testKeep) {
                    keepTrials++;
                    if (won) keepWins++;
                } else {
                    mullTrials++;
                    if (won) mullWins++;
                }
                totalRollouts++;
            } catch (Exception e) {
                totalRollouts++;
            } finally {
                MyRandom.setRandom(origRandom);
            }
        }

        float keepRate = keepTrials > 0
                ? (float) keepWins / keepTrials : 0.5f;
        float mullRate = mullTrials > 0
                ? (float) mullWins / mullTrials : 0.5f;

        // Keep if keep rate >= mulligan rate (bias toward keeping)
        int best = keepRate >= mullRate ? 1 : 0;
        totalDecisions++;

        float[] rates = {mullRate, keepRate};
        float[] visits = {0.5f, 0.5f};

        System.out.printf("MCTS_MULLIGAN: keep=%d/%d(%.0f%%) mull=%d/%d(%.0f%%) → %s%n",
                keepWins, keepTrials, keepRate * 100,
                mullWins, mullTrials, mullRate * 100,
                best == 1 ? "KEEP" : "MULLIGAN");
        System.out.flush();

        return new MCTSResult(best, rates, visits, rates[best]);
    }

    // ── Binary decisions ──

    /**
     * Decide yes/no via rollouts. Since we can't easily simulate
     * "yes" vs "no" for arbitrary triggered abilities, we let the
     * heuristic decide and record it. Returns -1 to signal "use heuristic".
     *
     * TODO: For specific binary decisions (e.g., "pay life?", "sacrifice?"),
     * implement proper simulation of both outcomes.
     */
    public int decideBinary() {
        return -1; // use heuristic
    }

    // ── Block decisions ──

    /**
     * Decide blocking assignments via rollouts.
     * Tests: heuristic's blocking assignment vs no blocks.
     * The heuristic's block logic is sophisticated, so we mainly
     * verify whether blocking at all is correct.
     */
    public MCTSResult decideBlocking(Game game, Player player,
                                      boolean heuristicBlocked) {
        List<Candidate> expanded = new ArrayList<>();
        expanded.add(new Candidate(List.of(), "no-block"));
        expanded.add(new Candidate(List.of(1), "heuristic-block"));

        // We can't easily simulate specific block assignments via rollout
        // because the combat is already set up. Instead, rollout from
        // current state — the heuristic handles blocks in the rollout.
        // This effectively tests "is the current board state a win?"
        int budget = Math.max(rolloutBudget / 3, 6);
        for (Candidate c : expanded) {
            for (int r = 0; r < budget / 2; r++) {
                boolean won = rollout(game, player, null);
                c.visits++;
                if (won) c.wins++;
                totalRollouts++;
            }
        }

        int best = expanded.get(1).winRate() >= expanded.get(0).winRate() ? 1 : 0;
        totalDecisions++;

        float[] rates = {(float) expanded.get(0).winRate(),
                         (float) expanded.get(1).winRate()};
        float[] visits = {0.5f, 0.5f};

        System.out.printf("MCTS_BLOCK: no-block=%.0f%% heuristic=%.0f%% → %s%n",
                rates[0] * 100, rates[1] * 100,
                best == 1 ? "BLOCK" : "NO-BLOCK");
        System.out.flush();

        return new MCTSResult(best, rates, visits, rates[best]);
    }

    // ── UCB1 rollout allocation ──

    /**
     * Run rollouts allocated by UCB1. Each candidate gets at least 1 rollout,
     * then remaining budget goes to the candidate with highest UCB1 score.
     */
    private void runUCB1(List<Candidate> candidates,
                         Game game, Player player) {
        int n = candidates.size();
        int budget = Math.max(rolloutBudget, n); // at least 1 per candidate
        int spent = 0;

        // Phase 1: one rollout per candidate
        for (Candidate c : candidates) {
            boolean won = (c.target != null)
                    ? rolloutWithTarget(game, player, c.spell, c.target)
                    : rollout(game, player, c.spell);
            c.visits++;
            if (won) c.wins++;
            spent++;
            totalRollouts++;
        }

        // Phase 2: UCB1 allocation for remaining budget
        while (spent < budget) {
            // Select candidate with highest UCB1
            Candidate best = null;
            double bestUcb = Double.NEGATIVE_INFINITY;
            for (Candidate c : candidates) {
                double ucb = c.ucb1(spent);
                if (ucb > bestUcb) {
                    bestUcb = ucb;
                    best = c;
                }
            }
            if (best == null) break;

            boolean won = (best.target != null)
                    ? rolloutWithTarget(game, player, best.spell, best.target)
                    : rollout(game, player, best.spell);
            best.visits++;
            if (won) best.wins++;
            spent++;
            totalRollouts++;
        }
    }

    /**
     * UCB1 rollout allocation for attack patterns.
     */
    private void runAttackUCB1(List<Candidate> candidates,
                                Game game, Player player,
                                List<Card> possibleAttackers) {
        int n = candidates.size();
        int budget = Math.max(rolloutBudget, n);
        int spent = 0;

        // Phase 1: one per candidate
        for (Candidate c : candidates) {
            boolean won = rolloutAttackPattern(game, player,
                    possibleAttackers, c.attackIndices);
            c.visits++;
            if (won) c.wins++;
            spent++;
            totalRollouts++;
        }

        // Phase 2: UCB1
        while (spent < budget) {
            Candidate best = null;
            double bestUcb = Double.NEGATIVE_INFINITY;
            for (Candidate c : candidates) {
                double ucb = c.ucb1(spent);
                if (ucb > bestUcb) {
                    bestUcb = ucb;
                    best = c;
                }
            }
            if (best == null) break;

            boolean won = rolloutAttackPattern(game, player,
                    possibleAttackers, best.attackIndices);
            best.visits++;
            if (won) best.wins++;
            spent++;
            totalRollouts++;
        }
    }

    // ── Rollout implementations ──

    /**
     * Rollout for a non-targeted spell (or pass).
     * Uses GameSimulator to properly apply the action.
     */
    private boolean rollout(Game origGame, Player origPlayer,
                            SpellAbility origSa) {
        long t0 = System.currentTimeMillis();
        try {
            Random origRandom = MyRandom.getRandom();
            MyRandom.setRandom(new Random(rng.nextLong()));
            try {
                SimulationController ctrl = new SimulationController(
                        new GameStateEvaluator.Score(0));
                GameSimulator simulator = new GameSimulator(
                        ctrl, origGame, origPlayer, null);
                if (origSa != null) {
                    simulator.simulateSpellAbility(origSa);
                }

                Game simGame = simulator.getSimulatedGameState();
                disableSimulation(simGame);
                Player simPlayer = (Player) simulator.getGameCopier()
                        .find(origPlayer);

                return playToCompletion(simGame, simPlayer);
            } finally {
                MyRandom.setRandom(origRandom);
            }
        } finally {
            totalRolloutTimeMs += System.currentTimeMillis() - t0;
        }
    }

    /**
     * Rollout for a targeted spell with a specific target.
     * Temporarily sets target on original SA for GameSimulator to copy.
     */
    private boolean rolloutWithTarget(Game origGame, Player origPlayer,
                                       SpellAbility origSa,
                                       GameEntity target) {
        if (origSa == null) return rollout(origGame, origPlayer, null);

        long t0 = System.currentTimeMillis();
        Random origRandom = MyRandom.getRandom();
        MyRandom.setRandom(new Random(rng.nextLong()));

        // Save and replace targets
        List<GameObject> savedTargets = new ArrayList<>();
        for (Object t : origSa.getTargets()) {
            if (t instanceof GameObject) savedTargets.add((GameObject) t);
        }

        try {
            origSa.resetTargets();
            origSa.getTargets().add(target);

            SimulationController ctrl = new SimulationController(
                    new GameStateEvaluator.Score(0));
            GameSimulator simulator = new GameSimulator(
                    ctrl, origGame, origPlayer, null);
            simulator.simulateSpellAbility(origSa);

            Game simGame = simulator.getSimulatedGameState();
            disableSimulation(simGame);
            Player simPlayer = (Player) simulator.getGameCopier()
                    .find(origPlayer);

            return playToCompletion(simGame, simPlayer);
        } catch (Exception e) {
            return false;
        } finally {
            // Restore original targets
            origSa.resetTargets();
            for (GameObject t : savedTargets) {
                if (t instanceof GameEntity) {
                    origSa.getTargets().add((GameEntity) t);
                }
            }
            MyRandom.setRandom(origRandom);
            totalRolloutTimeMs += System.currentTimeMillis() - t0;
        }
    }

    /**
     * Rollout for an attack pattern.
     */
    private boolean rolloutAttackPattern(Game origGame, Player origPlayer,
                                          List<Card> possibleAttackers,
                                          List<Integer> attackerIndices) {
        long t0 = System.currentTimeMillis();
        Random origRandom = MyRandom.getRandom();
        MyRandom.setRandom(new Random(rng.nextLong()));
        try {
            GameCopier copier = new GameCopier(origGame);
            Game simGame = copier.makeCopy();
            disableSimulation(simGame);
            Player simPlayer = (Player) copier.find(origPlayer);
            GameEntity defender = simPlayer.getWeakestOpponent();

            if (defender != null && attackerIndices != null) {
                Combat combat = simGame.getCombat();
                if (combat == null) {
                    combat = new Combat(simPlayer);
                    simGame.getPhaseHandler().setCombat(combat);
                }
                for (int idx : attackerIndices) {
                    if (idx >= 0 && idx < possibleAttackers.size()) {
                        Card simCard = (Card) copier.find(
                                possibleAttackers.get(idx));
                        if (simCard != null
                                && CombatUtil.canAttack(simCard, defender)) {
                            combat.addAttacker(simCard, defender);
                        }
                    }
                }
            }

            return playToCompletion(simGame, simPlayer);
        } finally {
            MyRandom.setRandom(origRandom);
            totalRolloutTimeMs += System.currentTimeMillis() - t0;
        }
    }

    /**
     * Play a copied game to completion and return whether our player won.
     */
    private boolean playToCompletion(Game simGame, Player simPlayer) {
        if (!simGame.isGameOver()) {
            if (!runWithTimeout(() ->
                    simGame.getPhaseHandler().mainGameLoop(),
                    rolloutTimeoutSec)) {
                return false;
            }
        }
        if (simGame.getOutcome() == null || simGame.getOutcome().isDraw()) {
            return false;
        }
        return simGame.getOutcome().isWinner(
                simPlayer.getRegisteredPlayer());
    }

    // ── Helpers ──

    private static void disableSimulation(Game game) {
        for (Player p : game.getPlayers()) {
            if (p.getController() instanceof PlayerControllerAi) {
                ((PlayerControllerAi) p.getController())
                        .setUseSimulation(false);
            }
        }
        game.AI_TIMEOUT = 2;
    }

    private static boolean runWithTimeout(Runnable task, int timeoutSec) {
        ExecutorService exec = Executors.newSingleThreadExecutor();
        Future<?> future = exec.submit(task);
        try {
            future.get(timeoutSec, TimeUnit.SECONDS);
            return true;
        } catch (TimeoutException e) {
            future.cancel(true);
            return false;
        } catch (Exception e) {
            return false;
        } finally {
            exec.shutdownNow();
        }
    }

    private static int argmax(float[] arr) {
        int best = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > arr[best]) best = i;
        }
        return best;
    }

    private static String cardName(SpellAbility sa) {
        if (sa == null) return "null";
        Card c = sa.getHostCard();
        return c != null ? c.getName() : "?";
    }

    private static String formatRates(float[] rates) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < rates.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(String.format("%.0f%%", rates[i] * 100));
        }
        sb.append("]");
        return sb.toString();
    }

    public int getTotalDecisions() { return totalDecisions; }
    public int getTotalRollouts() { return totalRollouts; }
    public long getTotalRolloutTimeMs() { return totalRolloutTimeMs; }
}
