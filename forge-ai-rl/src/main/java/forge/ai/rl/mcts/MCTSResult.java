package forge.ai.rl.mcts;

/**
 * Result from MCTS rollout evaluation.
 * Stores two representations of the search policy:
 * - winRates: per-candidate win rates (Q-values)
 * - visitProportions: per-candidate visit fractions (search policy, AlphaZero-style)
 * Both are recorded in trajectories for the training pipeline to choose from.
 */
public class MCTSResult {
    private final int selectedIndex;
    private final float[] winRates;          // per-candidate Q-values from rollouts
    private final float[] visitProportions;  // per-candidate visit counts / total visits
    private final float valueEstimate;       // win rate of selected candidate

    public MCTSResult(int selectedIndex, float[] winRates,
                      float[] visitProportions, float valueEstimate) {
        this.selectedIndex = selectedIndex;
        this.winRates = winRates;
        this.visitProportions = visitProportions;
        this.valueEstimate = valueEstimate;
    }

    public int getSelectedIndex() { return selectedIndex; }
    public float[] getWinRates() { return winRates; }
    public float[] getVisitProportions() { return visitProportions; }
    public float getValueEstimate() { return valueEstimate; }
}
