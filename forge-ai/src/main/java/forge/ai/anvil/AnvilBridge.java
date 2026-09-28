package forge.ai.anvil;

import java.util.List;

/**
 * Decision-answering seam for PlayerControllerAnvil (Anvil bridge-protocol-v0
 * answer shapes, Java side). Implementations: LocalRandomBridge (in-process
 * random-legal, the transport-free control arm) and GrpcBridge (forge-gui-desktop,
 * talks to the Python decision server). Answers must be deterministic given the
 * game seed for replay to hold.
 *
 * Option labels are built by the controller for BOTH arms (equal cost outside
 * the measured transport delta); the local arm ignores them, the gRPC arm
 * serializes them so M0 payloads carry realistic option lists.
 */
public interface AnvilBridge {
    /**
     * True once the transport has lost request/response correspondence
     * (decision deadline expired or a mismatched decision_seq arrived —
     * bridge-protocol-v0's "mismatched sequence is a protocol bug" clause).
     * A poisoned bridge refuses further decisions; the worker must stop
     * starting games and exit so the harness recycles it. Local bridges
     * cannot desync.
     */
    default boolean poisoned() {
        return false;
    }

    /** True only when the transport/server can honor exact candidate forcing. */
    default boolean supportsForcedCandidate() {
        return false;
    }

    /** SELECT_ONE: pick over options (index into the list). */
    int selectOne(String tag, List<String> optionLabels);

    /** SELECT_K: pick k distinct indices from [0, n); returned ascending. */
    int[] selectK(String tag, int n, int k);

    /** BOOL. */
    boolean bool(String tag);

    /** INT_IN_RANGE: inclusive bounds. */
    int intInRange(String tag, int min, int max);

    /**
     * M1 one-shot cast: composite answer to a priority window
     * (bridge-protocol-v0 CastPlan; ServerHello.one_shot_cast). Null means
     * this bridge doesn't answer the composite shape — the caller falls back
     * to the M0 selectOne path (local-random and echo arms stay unchanged).
     * observation is Obs.lastDecForBridge() (null when obs logging is off).
     */
    default CastPlanAnswer priorityCastPlan(String tag, List<String> optionLabels,
            String observation) {
        return null;
    }

    /**
     * Re-ask variant (D6 run-2, re-ask-on-veto): attempt 0 is the window's
     * first ask; attempt k>0 re-asks after k vetoes with the vetoed
     * candidates removed from optionLabels. Transports that track wire seqs
     * mark re-asks via DecisionRequest.retry_of; shapes that don't answer
     * the composite keep returning null (M0 path never re-asks).
     */
    default CastPlanAnswer priorityCastPlan(String tag, List<String> optionLabels,
            String observation, int attempt) {
        return priorityCastPlan(tag, optionLabels, observation);
    }

    /**
     * Forced-branch variant (M7 D2): forbidDecline asks the server to mask
     * the null/pass answer — the response must be a cast. Transports that
     * don't understand the constraint fall back to the unconstrained ask
     * (the caller adjudicates a pass response as branch_skip, loudly).
     */
    default CastPlanAnswer priorityCastPlan(String tag, List<String> optionLabels,
            String observation, int attempt, boolean forbidDecline) {
        return priorityCastPlan(tag, optionLabels, observation, attempt);
    }

    /** Candidate-drill intervention: forcedOption is one-based in the
     *  wire option list (0 is PASS).  Implementations that do not advertise
     *  the additive protocol capability return null; the rollout driver
     *  records that as an explicit unsupported skip. */
    default CastPlanAnswer priorityCastPlan(String tag, List<String> optionLabels,
            String observation, int attempt, boolean forbidDecline, int forcedOption) {
        return priorityCastPlan(tag, optionLabels, observation, attempt, forbidDecline);
    }

    /**
     * M2 D5 combat declarations (bridge-protocol-v0 AttackMap/BlockMap,
     * entity-ref form). Null means this bridge doesn't answer the shape —
     * the caller keeps the heuristic path (local-random/echo arms, servers
     * that never bridge the combat tags). observation is
     * Obs.lastDecForBridge(game).
     */
    default CombatMapAnswer attackMap(String tag, String observation) {
        return null;
    }

    default CombatMapAnswer blockMap(String tag, String observation) {
        return null;
    }

    /** Lifecycle notifications (no-ops for the local arm). */
    default void gameStart(String gameId, long seed) {
    }

    /**
     * Explicit-header variant (M2 D1): under fork nesting the zero-arg
     * Obs.lastHeaderForBridge is ambiguous when the mainline is re-announced
     * after a fork replay, so drivers that juggle sessions pass the header
     * they mean (Obs.lastHeaderForBridge(game)). Null header falls back to
     * the two-arg path's behavior.
     */
    default void gameStart(String gameId, long seed, String header) {
        gameStart(gameId, seed);
    }

    default void gameEnd(String gameId, String winner, int turns, long wallMs) {
    }

    default void close() {
    }
}
