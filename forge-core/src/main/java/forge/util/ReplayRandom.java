package forge.util;

import java.util.Random;
import java.util.Objects;
import java.util.function.Supplier;

/**
 * Tracks deterministic replay metadata for the current process.
 *
 * Replay sessions are intentionally coarse-grained for now: deterministic
 * replay is only supported for single-threaded headless runs.
 */
public final class ReplayRandom {
    private static volatile ReplaySession currentSession;

    private ReplayRandom() {
    }

    public static void startSession(long seed, String replayId,
                                    String runLabel,
                                    boolean deterministic,
                                    boolean auditEnabled) {
        currentSession = new ReplaySession(seed,
                replayId != null ? replayId : "replay",
                runLabel != null ? runLabel : "run",
                deterministic,
                auditEnabled);
    }

    public static void endSession() {
        currentSession = null;
    }

    public static boolean isActive() {
        return currentSession != null;
    }

    public static boolean isDeterministic() {
        ReplaySession session = currentSession;
        return session != null && session.deterministic;
    }

    public static long getSeed() {
        ReplaySession session = Objects.requireNonNull(currentSession,
                "No replay session is active");
        return session.seed;
    }

    public static String getReplayId() {
        ReplaySession session = currentSession;
        return session == null ? null : session.replayId;
    }

    public static String getRunLabel() {
        ReplaySession session = currentSession;
        return session == null ? null : session.runLabel;
    }

    public static void audit(String source) {
        ReplaySession session = currentSession;
        if (session == null || !session.auditEnabled) {
            return;
        }
        System.err.println("[ReplayRandom] Noncompliant random source used during replay: "
                + source + " [replayId=" + session.replayId
                + ", run=" + session.runLabel
                + ", seed=" + session.seed + "]");
    }

    public static <T> T runWithDecisionRandom(String playerName,
                                              String decisionType,
                                              String detail,
                                              Supplier<T> action) {
        ReplaySession session = currentSession;
        if (session == null || !session.deterministic) {
            return action.get();
        }

        Random original = MyRandom.getRandom();
        try {
            MyRandom.setRandom(new Random(deriveDecisionSeed(
                    session.seed, playerName, decisionType, detail)));
            return action.get();
        } finally {
            MyRandom.setRandom(original);
        }
    }

    public static void runWithDecisionRandom(String playerName,
                                             String decisionType,
                                             String detail,
                                             Runnable action) {
        runWithDecisionRandom(playerName, decisionType, detail, () -> {
            action.run();
            return null;
        });
    }

    private static long deriveDecisionSeed(long seed,
                                           String playerName,
                                           String decisionType,
                                           String detail) {
        long h = seed ^ 0x9E3779B97F4A7C15L;
        h = mix(h, playerName);
        h = mix(h, decisionType);
        h = mix(h, detail);
        return h;
    }

    private static long mix(long acc, String value) {
        String s = value != null ? value : "";
        long h = acc;
        for (int i = 0; i < s.length(); i++) {
            h ^= s.charAt(i);
            h *= 0x100000001B3L;
            h ^= (h >>> 32);
        }
        return h;
    }

    private static final class ReplaySession {
        private final long seed;
        private final String replayId;
        private final String runLabel;
        private final boolean deterministic;
        private final boolean auditEnabled;

        private ReplaySession(long seed, String replayId,
                              String runLabel,
                              boolean deterministic,
                              boolean auditEnabled) {
            this.seed = seed;
            this.replayId = replayId;
            this.runLabel = runLabel;
            this.deterministic = deterministic;
            this.auditEnabled = auditEnabled;
        }
    }
}
