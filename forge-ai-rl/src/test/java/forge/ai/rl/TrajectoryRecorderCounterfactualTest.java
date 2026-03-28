package forge.ai.rl;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import forge.ai.rl.decisions.DecisionContext;
import forge.ai.rl.decisions.DecisionResult;
import forge.ai.rl.decisions.DecisionType;
import forge.ai.rl.features.GameStateFeatures;
import forge.ai.rl.training.TrajectoryRecorder;
import org.testng.annotations.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.testng.AssertJUnit.*;

public class TrajectoryRecorderCounterfactualTest {

    @Test
    public void testCounterfactualMetadataIsSerializedAsAnalysisOnlyFields() throws Exception {
        Path tempDir = Files.createTempDirectory("traj-counterfactual");
        try {
            TrajectoryRecorder recorder = new TrajectoryRecorder(tempDir.toString());
            recorder.startGame("game-1");

            DecisionContext context = DecisionContext.singleSelect(
                    DecisionType.PRIORITY_ACTION,
                    GameStateFeatures.empty(),
                    List.of(new float[64], new float[64]),
                    "priority_action");
            DecisionResult result = new DecisionResult(
                    List.of(1), new float[]{0.25f, 0.75f}, 0.2f, false);

            recorder.recordDecision(context, result, 20, 20, 3, 3, 1, 1);
            recorder.annotateLastDecisionWithHeuristicCounterfactual(List.of(0));
            recorder.endGame(true);

            Path trajFile = Files.list(tempDir)
                    .filter(p -> p.getFileName().toString().startsWith("traj_"))
                    .findFirst()
                    .orElseThrow();
            List<String> lines = Files.readAllLines(trajFile);
            assertEquals(2, lines.size());

            JsonObject record = JsonParser.parseString(lines.get(1)).getAsJsonObject();
            assertEquals("ppo", record.get("source").getAsString());
            assertTrue(record.get("counterfactualHeuristicAvailable").getAsBoolean());
            assertEquals("shadow_heuristic",
                    record.get("counterfactualLabelSource").getAsString());
            assertEquals(1, record.getAsJsonArray("modelSelectedIndices").size());
            assertEquals(1, record.getAsJsonArray("selectedIndices").get(0).getAsInt());
            assertEquals(0,
                    record.getAsJsonArray("counterfactualHeuristicSelectedIndices")
                            .get(0).getAsInt());
        } finally {
            Files.walk(tempDir)
                    .sorted((a, b) -> b.compareTo(a))
                    .forEach(path -> {
                        try {
                            Files.deleteIfExists(path);
                        } catch (Exception ignored) {
                        }
                    });
        }
    }
}
