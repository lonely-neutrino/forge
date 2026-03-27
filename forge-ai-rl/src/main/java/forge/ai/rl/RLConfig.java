package forge.ai.rl;

/**
 * Configuration for the RL AI system.
 */
public class RLConfig {
    // Model server connection
    private String grpcHost = "127.0.0.1";
    private int grpcPort = 50051;
    private int grpcTimeoutMs = 30000;

    // Feature dimensions
    private int gameStateEmbeddingDim = 512;
    private int cardEmbeddingDim = 256;
    private int maxBoardCreatures = 40;
    private int maxHandCards = 15;
    private int maxGraveyardCards = 20;
    private int maxStackEntries = 10;
    private int maxAvailableActions = 50;

    // Training
    private RLModelMode mode = RLModelMode.HEURISTIC_FALLBACK;
    private boolean recordTrajectories = false;
    private String trajectoryOutputDir = "rl_data/trajectories";
    private String onnxModelDir = "rl_data/models";
    private String onnxExecutionProvider = "auto";

    // Reward shaping
    private double winReward = 1.0;
    private double loseReward = -1.0;
    private double lifeAdvantageReward = 0.01;
    private double cardAdvantageReward = 0.05;
    private double boardAdvantageReward = 0.02;
    private double rewardShapingDecay = 0.9999; // multiply shaping rewards by this each training step

    // Discount factor
    private double gamma = 0.999;

    public String getGrpcHost() { return grpcHost; }
    public void setGrpcHost(String grpcHost) { this.grpcHost = grpcHost; }
    public int getGrpcPort() { return grpcPort; }
    public void setGrpcPort(int grpcPort) { this.grpcPort = grpcPort; }
    public int getGrpcTimeoutMs() { return grpcTimeoutMs; }
    public void setGrpcTimeoutMs(int grpcTimeoutMs) { this.grpcTimeoutMs = grpcTimeoutMs; }

    public int getGameStateEmbeddingDim() { return gameStateEmbeddingDim; }
    public void setGameStateEmbeddingDim(int dim) { this.gameStateEmbeddingDim = dim; }
    public int getCardEmbeddingDim() { return cardEmbeddingDim; }
    public void setCardEmbeddingDim(int dim) { this.cardEmbeddingDim = dim; }
    public int getMaxBoardCreatures() { return maxBoardCreatures; }
    public void setMaxBoardCreatures(int max) { this.maxBoardCreatures = max; }
    public int getMaxHandCards() { return maxHandCards; }
    public void setMaxHandCards(int max) { this.maxHandCards = max; }
    public int getMaxGraveyardCards() { return maxGraveyardCards; }
    public void setMaxGraveyardCards(int max) { this.maxGraveyardCards = max; }
    public int getMaxStackEntries() { return maxStackEntries; }
    public void setMaxStackEntries(int max) { this.maxStackEntries = max; }
    public int getMaxAvailableActions() { return maxAvailableActions; }
    public void setMaxAvailableActions(int max) { this.maxAvailableActions = max; }

    public RLModelMode getMode() { return mode; }
    public void setMode(RLModelMode mode) { this.mode = mode; }
    public boolean isRecordTrajectories() { return recordTrajectories; }
    public void setRecordTrajectories(boolean record) { this.recordTrajectories = record; }
    public String getTrajectoryOutputDir() { return trajectoryOutputDir; }
    public void setTrajectoryOutputDir(String dir) { this.trajectoryOutputDir = dir; }
    public String getOnnxModelDir() { return onnxModelDir; }
    public void setOnnxModelDir(String dir) { this.onnxModelDir = dir; }
    public String getOnnxExecutionProvider() { return onnxExecutionProvider; }
    public void setOnnxExecutionProvider(String provider) { this.onnxExecutionProvider = provider; }

    public double getWinReward() { return winReward; }
    public double getLoseReward() { return loseReward; }
    public double getLifeAdvantageReward() { return lifeAdvantageReward; }
    public double getCardAdvantageReward() { return cardAdvantageReward; }
    public double getBoardAdvantageReward() { return boardAdvantageReward; }
    public double getRewardShapingDecay() { return rewardShapingDecay; }
    public double getGamma() { return gamma; }
}
