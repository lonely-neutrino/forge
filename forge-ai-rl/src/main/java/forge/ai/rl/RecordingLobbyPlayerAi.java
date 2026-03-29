package forge.ai.rl;

import forge.ai.LobbyPlayerAi;
import forge.ai.rl.training.TrajectoryRecorder;

/**
 * Marker subclass of LobbyPlayerAi for trajectory recording.
 * Holds config and recorder but does NOT modify AI behavior.
 * Recorder and config are lazily initialized to avoid any
 * side effects during construction.
 */
public class RecordingLobbyPlayerAi extends LobbyPlayerAi {

    private RLConfig config;
    private TrajectoryRecorder recorder;

    public RecordingLobbyPlayerAi(String name) {
        super(name, null);
    }

    public void setRLConfig(RLConfig config) {
        this.config = config;
    }

    public RLConfig getConfig() {
        return config;
    }

    public TrajectoryRecorder getRecorder() {
        if (recorder == null && config != null) {
            recorder = new TrajectoryRecorder(
                    config.getTrajectoryOutputDir(),
                    config.isZeroIntermediateReward());
        }
        return recorder;
    }
}
