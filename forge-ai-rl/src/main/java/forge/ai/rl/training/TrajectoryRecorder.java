package forge.ai.rl.training;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import forge.ai.rl.decisions.DecisionContext;
import forge.ai.rl.decisions.DecisionResult;
import org.tinylog.Logger;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 * Records game trajectories (sequences of state-action-reward tuples) for training.
 *
 * Each game produces one trajectory file containing all decisions made by the RL player.
 * These files are consumed by the Python training pipeline.
 *
 * File format: one JSON object per line (JSONL), with a header line and one line per decision.
 */
public class TrajectoryRecorder {
    private static final String METADATA_FILENAME = "game_metadata.jsonl";
    private final String outputDir;
    private final boolean zeroIntermediateReward;
    private final Gson gson;
    private final List<DecisionRecord> currentGame;
    private String gameId;
    private String matchId;
    private String playerName;
    private String opponentName;
    private String playerDeck;
    private String opponentDeck;
    private long gameStartTime;

    // Running state for reward shaping
    private int prevLifeAdvantage = 0;
    private int prevCardAdvantage = 0;
    private int prevBoardAdvantage = 0;

    public TrajectoryRecorder(String outputDir) {
        this(outputDir, false);
    }

    public TrajectoryRecorder(String outputDir,
                              boolean zeroIntermediateReward) {
        this.outputDir = outputDir;
        this.zeroIntermediateReward = zeroIntermediateReward;
        this.gson = new GsonBuilder().create();
        this.currentGame = new ArrayList<>();

        // Ensure output directory exists
        try {
            Files.createDirectories(Paths.get(outputDir));
        } catch (IOException e) {
            Logger.error("Failed to create trajectory output directory: {}", e.getMessage());
        }
    }

    /**
     * Start recording a new game.
     */
    public void startGame(String gameId) {
        startGame(gameId, null, null, null, null, null);
    }

    /**
     * Start recording a new game with optional deck metadata.
     */
    public void startGame(String gameId,
                          String matchId,
                          String playerName,
                          String opponentName,
                          String playerDeck,
                          String opponentDeck) {
        this.gameId = gameId;
        this.matchId = matchId;
        this.playerName = playerName;
        this.opponentName = opponentName;
        this.playerDeck = playerDeck;
        this.opponentDeck = opponentDeck;
        this.gameStartTime = System.currentTimeMillis();
        this.currentGame.clear();
        this.prevLifeAdvantage = 0;
        this.prevCardAdvantage = 0;
        this.prevBoardAdvantage = 0;
    }

    /**
     * Record a decision made during the game.
     */
    public void recordDecision(DecisionContext context, DecisionResult result,
                                int myLife, int oppLife,
                                int myHandSize, int oppHandSize,
                                int myCreatureCount, int oppCreatureCount) {
        DecisionRecord record = new DecisionRecord();
        record.turnIndex = currentGame.size();
        record.decisionType = context.getType().name();
        record.contextInfo = context.getContextInfo();
        record.globalFeatures = context.getGameState().getGlobalFeatures();
        record.candidateCount = context.getCandidateFeatures().size();
        record.selectedIndices = result.getSelectedIndices();
        record.actionProbabilities = result.getActionProbabilities();
        record.valueEstimate = result.getValueEstimate();
        record.usedFallback = result.isUsedFallback();
        record.source = result.isUsedFallback() ? "heuristic" : "ppo";
        record.modelSelectedIndices = result.getSelectedIndices();
        record.counterfactualHeuristicAvailable = false;

        // Compute intermediate reward from state changes
        int lifeAdv = myLife - oppLife;
        int cardAdv = myHandSize - oppHandSize;
        int boardAdv = myCreatureCount - oppCreatureCount;

        if (zeroIntermediateReward) {
            record.intermediateReward = 0.0;
        } else {
            record.intermediateReward = 0;
            record.intermediateReward += (lifeAdv - prevLifeAdvantage) * 0.01;
            record.intermediateReward += (cardAdv - prevCardAdvantage) * 0.05;
            record.intermediateReward += (boardAdv - prevBoardAdvantage) * 0.02;
        }

        prevLifeAdvantage = lifeAdv;
        prevCardAdvantage = cardAdv;
        prevBoardAdvantage = boardAdv;

        // Store flattened game state for training
        record.gameStateFlat = context.getGameState().flatten();

        // Store candidate features
        record.candidateFeatures = context.getCandidateFeatures().toArray(new float[0][]);

        // Store spell features for target decisions
        record.spellFeatures = context.getSpellFeatures();

        currentGame.add(record);
    }

    /**
     * Attach analysis-only heuristic counterfactual data to the last recorded
     * decision. This is used by tests and downstream analysis tooling; it does
     * not change the selected action for training.
     */
    public void annotateLastDecisionWithHeuristicCounterfactual(
            List<Integer> heuristicSelectedIndices) {
        if (currentGame.isEmpty()) {
            return;
        }
        DecisionRecord record = currentGame.get(currentGame.size() - 1);
        record.counterfactualHeuristicAvailable = true;
        record.counterfactualLabelSource = "shadow_heuristic";
        record.counterfactualHeuristicSelectedIndices = heuristicSelectedIndices;
    }

    /**
     * End the game and write the trajectory to disk.
     * @param won true if the RL player won
     */
    public void endGame(boolean won) {
        if (currentGame.isEmpty()) return;

        double terminalReward = won ? 1.0 : -1.0;

        // Set terminal reward on last decision
        if (!currentGame.isEmpty()) {
            currentGame.get(currentGame.size() - 1).terminalReward = terminalReward;
        }

        // Write trajectory file
        String filename = String.format("traj_%s_%s_%s.jsonl",
                gameId != null ? gameId : UUID.randomUUID().toString().substring(0, 8),
                won ? "W" : "L",
                gameStartTime);

        Path filePath = Paths.get(outputDir, filename);
        try (BufferedWriter writer = Files.newBufferedWriter(filePath)) {
            // Header
            TrajectoryHeader header = new TrajectoryHeader();
            header.gameId = gameId;
            header.won = won;
            header.totalDecisions = currentGame.size();
            header.durationMs = System.currentTimeMillis() - gameStartTime;
            writer.write(gson.toJson(header));
            writer.newLine();

            // Decision records
            for (DecisionRecord record : currentGame) {
                writer.write(gson.toJson(record));
                writer.newLine();
            }

            Logger.info("Wrote trajectory: {} ({} decisions, {})",
                    filePath.getFileName(), currentGame.size(), won ? "WIN" : "LOSS");

            appendMetadata(filePath.getFileName().toString(), won);

        } catch (IOException e) {
            Logger.error("Failed to write trajectory file: {}", e.getMessage());
        }

        currentGame.clear();
    }

    private void appendMetadata(String trajectoryFile, boolean won) {
        TrajectoryMetadata metadata = new TrajectoryMetadata();
        metadata.gameId = gameId;
        metadata.matchId = matchId;
        metadata.playerName = playerName;
        metadata.opponentName = opponentName;
        metadata.playerDeck = playerDeck;
        metadata.opponentDeck = opponentDeck;
        metadata.won = won;
        metadata.trajectoryFile = trajectoryFile;
        metadata.timestamp = gameStartTime;

        Path metadataPath = Paths.get(outputDir, METADATA_FILENAME);
        synchronized (TrajectoryRecorder.class) {
            try (BufferedWriter writer = Files.newBufferedWriter(
                    metadataPath,
                    java.nio.file.StandardOpenOption.CREATE,
                    java.nio.file.StandardOpenOption.APPEND)) {
                writer.write(gson.toJson(metadata));
                writer.newLine();
            } catch (IOException e) {
                Logger.error("Failed to append trajectory metadata: {}", e.getMessage());
            }
        }
    }

    /**
     * Get the number of decisions recorded in the current game.
     */
    public int getCurrentGameDecisionCount() {
        return currentGame.size();
    }

    // Internal data structures for serialization

    private static class TrajectoryHeader {
        String gameId;
        boolean won;
        int totalDecisions;
        long durationMs;
    }

    private static class TrajectoryMetadata {
        String gameId;
        String matchId;
        String playerName;
        String opponentName;
        String playerDeck;
        String opponentDeck;
        boolean won;
        String trajectoryFile;
        long timestamp;
    }

    private static class DecisionRecord {
        int turnIndex;
        String decisionType;
        String contextInfo;
        String source;
        float[] globalFeatures;
        float[] gameStateFlat;
        float[][] candidateFeatures;
        int candidateCount;
        List<Integer> selectedIndices;
        List<Integer> modelSelectedIndices;
        float[] actionProbabilities;
        float valueEstimate;
        boolean usedFallback;
        boolean counterfactualHeuristicAvailable;
        String counterfactualLabelSource;
        List<Integer> counterfactualHeuristicSelectedIndices;
        double intermediateReward;
        double terminalReward;
        float[] spellFeatures; // 64-dim source spell for target decisions
    }
}
