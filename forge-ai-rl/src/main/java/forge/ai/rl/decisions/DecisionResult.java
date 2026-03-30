package forge.ai.rl.decisions;

import java.util.List;

/**
 * Result from the RL model for a single decision.
 * Contains the selected action indices, probabilities, and value estimate.
 */
public class DecisionResult {
    private final List<Integer> selectedIndices;
    private final float[] actionProbabilities;  // win rates / Q-values
    private final float[] visitProportions;     // UCB1 visit fractions (search policy)
    private final float valueEstimate;
    private final boolean usedFallback;

    public DecisionResult(List<Integer> selectedIndices, float[] actionProbabilities,
                          float valueEstimate, boolean usedFallback) {
        this(selectedIndices, actionProbabilities, null, valueEstimate, usedFallback);
    }

    public DecisionResult(List<Integer> selectedIndices, float[] actionProbabilities,
                          float[] visitProportions,
                          float valueEstimate, boolean usedFallback) {
        this.selectedIndices = selectedIndices;
        this.actionProbabilities = actionProbabilities;
        this.visitProportions = visitProportions;
        this.valueEstimate = valueEstimate;
        this.usedFallback = usedFallback;
    }

    public List<Integer> getSelectedIndices() { return selectedIndices; }
    public int getSelectedIndex() { return selectedIndices.isEmpty() ? -1 : selectedIndices.get(0); }
    public float[] getActionProbabilities() { return actionProbabilities; }
    public float[] getVisitProportions() { return visitProportions; }
    public float getValueEstimate() { return valueEstimate; }
    public boolean isUsedFallback() { return usedFallback; }

    /**
     * Create a result indicating the heuristic fallback was used.
     */
    public static DecisionResult fallback(int selectedIndex) {
        return new DecisionResult(List.of(selectedIndex), new float[0], 0f, true);
    }

    /**
     * Create a result for a binary decision (index 0 = false, index 1 = true).
     */
    public static DecisionResult binary(boolean value, float[] probs, float valueEstimate) {
        return new DecisionResult(List.of(value ? 1 : 0), probs, valueEstimate, false);
    }
}
