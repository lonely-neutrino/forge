package forge.ai.rl;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import forge.ai.rl.decisions.DecisionContext;
import forge.ai.rl.decisions.DecisionResult;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CardCollectionView;
import forge.game.player.Player;
import forge.game.phase.PhaseType;
import forge.game.spellability.SpellAbilityStackInstance;
import forge.game.zone.ZoneType;
import forge.util.BuildInfo;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Formatter;
import java.util.List;

/**
 * Replay-oriented trace writer.
 *
 * Each file contains a header line followed by step records. The payload is
 * intentionally self-contained so the Python viewer can work without the live
 * game engine.
 */
public class ReplayTraceRecorder {
    private final Gson gson = new GsonBuilder().create();
    private final Path outputDir;
    private final ReplayConfig replayConfig;
    private BufferedWriter writer;
    private int stepCounter;

    public ReplayTraceRecorder(String outputDir, ReplayConfig replayConfig) {
        this.outputDir = Paths.get(outputDir);
        this.replayConfig = replayConfig;
    }

    public void start(String actor,
                      int seatIndex,
                      String opponentName,
                      String playerDeck,
                      String opponentDeck,
                      String policyMode,
                      String backend,
                      String modelId,
                      boolean deterministicPolicy) {
        try {
            Files.createDirectories(outputDir);
            String filename = String.format("replay_%s_%s_seat%d_%s.jsonl",
                    safe(replayConfig.getReplayId()),
                    safe(replayConfig.getRunLabel()),
                    seatIndex,
                    safe(actor));
            writer = Files.newBufferedWriter(outputDir.resolve(filename),
                    StandardCharsets.UTF_8);

            Header header = new Header();
            header.recordType = "header";
            header.replayId = replayConfig.getReplayId();
            header.runLabel = replayConfig.getRunLabel();
            header.seed = replayConfig.getSeed();
            header.actor = actor;
            header.seatIndex = seatIndex;
            header.opponent = opponentName;
            header.playerDeck = playerDeck;
            header.opponentDeck = opponentDeck;
            header.policyMode = policyMode;
            header.backend = backend;
            header.modelId = modelId;
            header.deterministicPolicy = deterministicPolicy;
            header.codeVersion = BuildInfo.getVersionString();
            writer.write(gson.toJson(header));
            writer.newLine();
            writer.flush();
            stepCounter = 0;
        } catch (IOException e) {
            closeQuietly();
        }
    }

    public void record(Game game, Player actor,
                       DecisionContext context,
                       DecisionResult result) {
        if (writer == null || game == null || actor == null) {
            return;
        }
        try {
            Step step = new Step();
            step.recordType = "step";
            step.stepIndex = stepCounter++;
            step.turn = game.getPhaseHandler().getTurn();
            PhaseType phase = game.getPhaseHandler().getPhase();
            step.phase = phase != null ? phase.name() : "UNKNOWN";
            step.actor = actor.getName();
            step.decisionType = context.getType().name();
            step.contextInfo = context.getContextInfo();
            step.candidateLabels = new ArrayList<>(context.getCandidateLabels());
            step.traceCandidateLabels = new ArrayList<>(context.getTraceCandidateLabels());
            step.selectedIndices = new ArrayList<>(result.getSelectedIndices());
            step.selectedLabels = new ArrayList<>();
            for (int idx : step.selectedIndices) {
                if (idx >= 0 && idx < step.candidateLabels.size()) {
                    step.selectedLabels.add(step.candidateLabels.get(idx));
                } else {
                    step.selectedLabels.add(String.valueOf(idx));
                }
            }
            step.globalFeatures = context.getGameState().getGlobalFeatures();
            step.gameStateFlat = context.getGameState().flatten();
            step.candidateFeatures = context.getCandidateFeatures().toArray(new float[0][]);
            step.actionProbabilities = result.getActionProbabilities();
            step.valueEstimate = result.getValueEstimate();
            step.usedFallback = result.isUsedFallback();
            step.snapshot = buildSnapshot(game, actor);
            step.stateHash = sha256(step.globalFeatures, step.gameStateFlat);
            // Keep the replay decision hash focused on the semantic choice so
            // heuristic and RL traces line up when they genuinely made the
            // same decision from the same state. Context strings can differ in
            // formatting/order even when the underlying choice is identical.
            step.decisionHash = sha256(step.stateHash,
                    step.decisionType,
                    step.candidateLabels.toString(),
                    step.selectedIndices.toString(),
                    step.selectedLabels.toString());
            writer.write(gson.toJson(step));
            writer.newLine();
            writer.flush();
        } catch (IOException e) {
            closeQuietly();
        }
    }

    public void close() {
        closeQuietly();
    }

    public void finish(Game game, Player actor, boolean won) {
        if (writer == null || game == null || actor == null) {
            return;
        }
        try {
            ResultRecord record = new ResultRecord();
            record.recordType = "result";
            record.turn = game.getPhaseHandler().getTurn();
            PhaseType phase = game.getPhaseHandler().getPhase();
            record.phase = phase != null ? phase.name() : "UNKNOWN";
            record.actor = actor.getName();
            record.won = won;
            record.snapshot = buildSnapshot(game, actor);
            writer.write(gson.toJson(record));
            writer.newLine();
            writer.flush();
        } catch (IOException e) {
            closeQuietly();
        }
    }

    private static Snapshot buildSnapshot(Game game, Player actor) {
        Snapshot snapshot = new Snapshot();
        Player opponent = actor.getWeakestOpponent();
        snapshot.activePlayer = game.getPhaseHandler().getPlayerTurn() != null
                ? game.getPhaseHandler().getPlayerTurn().getName()
                : null;
        snapshot.myLife = actor.getLife();
        snapshot.oppLife = opponent != null ? opponent.getLife() : 0;
        snapshot.myPoison = actor.getPoisonCounters();
        snapshot.oppPoison = opponent != null ? opponent.getPoisonCounters() : 0;
        snapshot.myHandCount = actor.getCardsIn(ZoneType.Hand).size();
        snapshot.oppHandCount = opponent != null ? opponent.getCardsIn(ZoneType.Hand).size() : 0;
        snapshot.myLibraryCount = actor.getCardsIn(ZoneType.Library).size();
        snapshot.oppLibraryCount = opponent != null ? opponent.getCardsIn(ZoneType.Library).size() : 0;
        snapshot.myBoard = describeCards(actor.getCardsIn(ZoneType.Battlefield));
        snapshot.oppBoard = opponent != null
                ? describeCards(opponent.getCardsIn(ZoneType.Battlefield))
                : new ArrayList<>();
        snapshot.hand = describeCards(actor.getCardsIn(ZoneType.Hand));
        snapshot.myGraveyard = describeCards(actor.getCardsIn(ZoneType.Graveyard));
        snapshot.oppGraveyard = opponent != null
                ? describeCards(opponent.getCardsIn(ZoneType.Graveyard))
                : new ArrayList<>();
        snapshot.stack = describeStack(game);
        return snapshot;
    }

    private static List<SnapshotCard> describeCards(CardCollectionView cards) {
        List<SnapshotCard> result = new ArrayList<>();
        for (Card card : cards) {
            SnapshotCard entry = new SnapshotCard();
            entry.name = card.getName();
            entry.type = card.getType() != null ? card.getType().toString() : "";
            if (card.isCreature()) {
                entry.power = card.getNetPower();
                entry.toughness = card.getNetToughness();
            }
            entry.tapped = card.isTapped();
            entry.summoningSick = card.hasSickness();
            entry.counters = card.getCounters() != null ? card.getCounters().toString() : null;
            result.add(entry);
        }
        return result;
    }

    private static List<String> describeStack(Game game) {
        List<String> result = new ArrayList<>();
        for (SpellAbilityStackInstance item : game.getStack()) {
            Card source = item.getSourceCard();
            result.add(source != null ? source.getName() : item.toString());
        }
        return result;
    }

    private void closeQuietly() {
        if (writer == null) {
            return;
        }
        try {
            writer.close();
        } catch (IOException ignored) {
        } finally {
            writer = null;
        }
    }

    private static String safe(String value) {
        if (value == null || value.isBlank()) {
            return "unknown";
        }
        return value.replaceAll("[^a-zA-Z0-9._-]", "_");
    }

    private static String sha256(float[] a, float[] b) {
        StringBuilder sb = new StringBuilder();
        if (a != null) {
            for (float v : a) {
                sb.append(v).append('|');
            }
        }
        sb.append('#');
        if (b != null) {
            for (float v : b) {
                sb.append(v).append('|');
            }
        }
        return sha256(sb.toString());
    }

    private static String sha256(String... values) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            for (String value : values) {
                if (value != null) {
                    digest.update(value.getBytes(StandardCharsets.UTF_8));
                }
                digest.update((byte) 0);
            }
            byte[] hash = digest.digest();
            Formatter formatter = new Formatter();
            for (byte b : hash) {
                formatter.format("%02x", b);
            }
            String result = formatter.toString();
            formatter.close();
            return result;
        } catch (Exception e) {
            return "sha256_error";
        }
    }

    private static final class Header {
        String recordType;
        String replayId;
        String runLabel;
        long seed;
        String actor;
        int seatIndex;
        String opponent;
        String playerDeck;
        String opponentDeck;
        String policyMode;
        String backend;
        String modelId;
        boolean deterministicPolicy;
        String codeVersion;
    }

    private static final class Step {
        String recordType;
        int stepIndex;
        int turn;
        String phase;
        String actor;
        String decisionType;
        String contextInfo;
        List<String> candidateLabels;
        List<String> traceCandidateLabels;
        List<Integer> selectedIndices;
        List<String> selectedLabels;
        String stateHash;
        String decisionHash;
        float[] globalFeatures;
        float[] gameStateFlat;
        float[][] candidateFeatures;
        float[] actionProbabilities;
        float valueEstimate;
        boolean usedFallback;
        Snapshot snapshot;
    }

    private static final class ResultRecord {
        String recordType;
        int turn;
        String phase;
        String actor;
        boolean won;
        Snapshot snapshot;
    }

    private static final class Snapshot {
        String activePlayer;
        int myLife;
        int oppLife;
        int myPoison;
        int oppPoison;
        int myHandCount;
        int oppHandCount;
        int myLibraryCount;
        int oppLibraryCount;
        List<SnapshotCard> myBoard;
        List<SnapshotCard> oppBoard;
        List<SnapshotCard> hand;
        List<SnapshotCard> myGraveyard;
        List<SnapshotCard> oppGraveyard;
        List<String> stack;
    }

    private static final class SnapshotCard {
        String name;
        String type;
        Integer power;
        Integer toughness;
        boolean tapped;
        boolean summoningSick;
        String counters;
    }
}
