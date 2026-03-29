package forge.ai.rl;

/**
 * Replay contract for deterministic paired runs.
 */
public class ReplayConfig {
    private final long seed;
    private final String replayId;
    private final String runLabel;
    private final boolean deterministicPolicy;
    private final boolean rngAuditEnabled;
    private final String traceOutputDir;

    public ReplayConfig(long seed, String replayId, String runLabel,
                        boolean deterministicPolicy,
                        boolean rngAuditEnabled,
                        String traceOutputDir) {
        this.seed = seed;
        this.replayId = replayId;
        this.runLabel = runLabel;
        this.deterministicPolicy = deterministicPolicy;
        this.rngAuditEnabled = rngAuditEnabled;
        this.traceOutputDir = traceOutputDir;
    }

    public long getSeed() {
        return seed;
    }

    public String getReplayId() {
        return replayId;
    }

    public String getRunLabel() {
        return runLabel;
    }

    public boolean isDeterministicPolicy() {
        return deterministicPolicy;
    }

    public boolean isRngAuditEnabled() {
        return rngAuditEnabled;
    }

    public String getTraceOutputDir() {
        return traceOutputDir;
    }
}
