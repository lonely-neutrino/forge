package forge.ai.rl.decisions;

import forge.ai.rl.features.GameStateFeatures;

import java.util.List;

/**
 * Encapsulates the context for a single decision the RL agent must make.
 * Sent to the model server for inference, or used locally with ONNX.
 */
public class DecisionContext {
    private final DecisionType type;
    private final GameStateFeatures gameState;
    private final List<float[]> candidateFeatures; // feature vector per candidate option
    private final int minSelections;
    private final int maxSelections;
    private final String contextInfo; // human-readable description for logging
    private final float[] spellFeatures; // 64-dim source spell features for targeting

    public DecisionContext(DecisionType type, GameStateFeatures gameState,
                           List<float[]> candidateFeatures, int minSelections,
                           int maxSelections, String contextInfo) {
        this(type, gameState, candidateFeatures, minSelections, maxSelections, contextInfo, null);
    }

    public DecisionContext(DecisionType type, GameStateFeatures gameState,
                           List<float[]> candidateFeatures, int minSelections,
                           int maxSelections, String contextInfo, float[] spellFeatures) {
        this.type = type;
        this.gameState = gameState;
        this.candidateFeatures = candidateFeatures;
        this.minSelections = minSelections;
        this.maxSelections = maxSelections;
        this.contextInfo = contextInfo;
        this.spellFeatures = spellFeatures;
    }

    public DecisionType getType() { return type; }
    public GameStateFeatures getGameState() { return gameState; }
    public List<float[]> getCandidateFeatures() { return candidateFeatures; }
    public int getMinSelections() { return minSelections; }
    public int getMaxSelections() { return maxSelections; }
    public String getContextInfo() { return contextInfo; }
    public float[] getSpellFeatures() { return spellFeatures; }

    /**
     * Convenience constructor for binary decisions (yes/no).
     */
    public static DecisionContext binary(GameStateFeatures gameState, String contextInfo) {
        return new DecisionContext(DecisionType.BINARY_CHOICE, gameState, List.of(), 1, 1, contextInfo);
    }

    /**
     * Convenience constructor for single-select from candidates.
     */
    public static DecisionContext singleSelect(DecisionType type, GameStateFeatures gameState,
                                                List<float[]> candidates, String contextInfo) {
        return new DecisionContext(type, gameState, candidates, 1, 1, contextInfo);
    }

    /**
     * Convenience constructor for multi-select from candidates.
     */
    public static DecisionContext multiSelect(DecisionType type, GameStateFeatures gameState,
                                               List<float[]> candidates, int min, int max,
                                               String contextInfo) {
        return new DecisionContext(type, gameState, candidates, min, max, contextInfo);
    }
}
