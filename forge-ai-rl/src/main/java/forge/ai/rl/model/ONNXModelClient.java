package forge.ai.rl.model;

import ai.onnxruntime.*;
import forge.ai.rl.RLConfig;
import forge.ai.rl.decisions.DecisionContext;
import forge.ai.rl.decisions.DecisionResult;
import forge.ai.rl.decisions.DecisionType;
import forge.ai.rl.features.GameStateFeatures;
import org.tinylog.Logger;

import java.io.File;
import java.lang.reflect.Method;
import java.nio.FloatBuffer;
import java.util.*;

/**
 * Local ONNX Runtime inference client — replaces the TCP ModelServerClient.
 * Loads 9 separate ONNX files (state encoder + 7 decision heads + value).
 * No Python dependency needed.
 */
public class ONNXModelClient {

    private static final int GLOBAL_DIM = 96;
    private static final int CARD_DIM = 256;
    private static final int ACTION_DIM = 64;
    private static final int STATE_DIM = 512;
    private static final int MAX_BOARD = 40;
    private static final int MAX_HAND = 15;
    private static final int MAX_GY = 20;
    private static final int MAX_STACK = 10;

    private final RLConfig config;
    private OrtEnvironment env;
    private OrtSession stateEncoder;
    private OrtSession valueHead;
    private OrtSession priorityHead;
    private OrtSession targetHead;
    private OrtSession attackHead;
    private OrtSession blockHead;
    private OrtSession cardSelectHead;
    private OrtSession mulliganHead;
    private OrtSession binaryHead;
    private boolean loaded = false;

    public ONNXModelClient(RLConfig config) {
        this.config = config;
    }

    public synchronized boolean loadModels() {
        if (loaded) return true;
        String dir = config.getOnnxModelDir();

        // Check multiple locations
        String[] candidates = {
            dir,
            System.getProperty("user.home") + "/.forge/res/rl/models",
            "rl_data/models",
            "forge-ai-rl/models",
        };

        String modelDir = null;
        for (String candidate : candidates) {
            if (new File(candidate, "state_encoder.onnx").exists()) {
                modelDir = candidate;
                break;
            }
        }

        if (modelDir == null) {
            Logger.warn("No ONNX model files found in: {}", Arrays.toString(candidates));
            return false;
        }

        try {
            env = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
            opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            configureExecutionProvider(opts);

            Logger.info("Loading ONNX models from: {}", modelDir);
            stateEncoder = env.createSession(modelDir + "/state_encoder.onnx", opts);
            valueHead = env.createSession(modelDir + "/value_head.onnx", opts);
            priorityHead = env.createSession(modelDir + "/priority_head.onnx", opts);
            targetHead = env.createSession(modelDir + "/target_head.onnx", opts);
            attackHead = env.createSession(modelDir + "/attack_head.onnx", opts);
            blockHead = env.createSession(modelDir + "/block_head.onnx", opts);
            cardSelectHead = env.createSession(modelDir + "/card_select_head.onnx", opts);
            mulliganHead = env.createSession(modelDir + "/mulligan_head.onnx", opts);
            binaryHead = env.createSession(modelDir + "/binary_head.onnx", opts);

            loaded = true;
            Logger.info("ONNX models loaded successfully (9 sessions)");
            return true;
        } catch (OrtException e) {
            Logger.error("Failed to load ONNX models: {}", e.getMessage());
            return false;
        }
    }

    public boolean isLoaded() {
        return loaded;
    }

    private void configureExecutionProvider(OrtSession.SessionOptions opts) {
        String provider = config.getOnnxExecutionProvider();
        if (provider == null || provider.isBlank()) {
            provider = "auto";
        }

        String normalized = provider.toLowerCase(Locale.ROOT);
        boolean isWindows = System.getProperty("os.name", "")
                .toLowerCase(Locale.ROOT).contains("win");

        if ("cpu".equals(normalized)) {
            Logger.info("Using ONNX Runtime CPU execution provider");
            return;
        }

        if (!"directml".equals(normalized) && !"auto".equals(normalized)) {
            Logger.warn("Unknown ONNX execution provider '{}', falling back to auto", provider);
            normalized = "auto";
        }

        if (!isWindows) {
            if ("directml".equals(normalized)) {
                Logger.warn("DirectML was requested but is only available on Windows; using CPU");
            }
            return;
        }

        try {
            Method addDirectML = opts.getClass().getMethod("addDirectML", int.class);
            addDirectML.invoke(opts, 0);
            Logger.info("Enabled ONNX Runtime DirectML execution provider");
        } catch (NoSuchMethodException e) {
            if ("directml".equals(normalized)) {
                Logger.warn("This ONNX Runtime build does not expose DirectML; using CPU");
            }
        } catch (Exception e) {
            if ("directml".equals(normalized)) {
                Logger.warn("Failed to enable DirectML execution provider: {}", e.getMessage());
            } else {
                Logger.info("DirectML unavailable, falling back to CPU: {}", e.getMessage());
            }
        }
    }

    public synchronized DecisionResult requestDecision(DecisionContext context) {
        if (!loaded) return null;

        try {
            // 1. Encode game state → 512-dim embedding
            float[] stateEmbedding = encodeGameState(context.getGameState());

            // 2. Get value estimate
            float value = computeValue(stateEmbedding);

            // 3. Route to appropriate head
            DecisionType type = context.getType();
            switch (type) {
                case PRIORITY_ACTION:
                    return handlePriority(stateEmbedding, context, value);
                case TARGET_SELECTION:
                    return handleTarget(stateEmbedding, context, value);
                case DECLARE_ATTACKERS:
                    return handleAttack(stateEmbedding, context, value);
                case DECLARE_BLOCKERS:
                    return handleBlock(stateEmbedding, context, value);
                case CARD_SELECTION:
                    return handleCardSelect(stateEmbedding, context, value);
                case MULLIGAN:
                    return handleMulligan(stateEmbedding, context, value);
                case BINARY_CHOICE:
                    return handleBinary(stateEmbedding, value);
                default:
                    return null;
            }
        } catch (OrtException e) {
            Logger.warn("ONNX inference error: {}", e.getMessage());
            return null;
        }
    }

    // ── Game State Encoding ──

    private float[] encodeGameState(GameStateFeatures gs) throws OrtException {
        float[] globalFeats = gs.getGlobalFeatures();
        float[] flat = gs.flatten();

        // Parse flat state into zones (same logic as Python parse_game_state)
        float[][] globalInput = new float[1][GLOBAL_DIM];
        System.arraycopy(globalFeats, 0, globalInput[0], 0,
                Math.min(globalFeats.length, GLOBAL_DIM));
        clip(globalInput[0]);

        int offset = GLOBAL_DIM;
        int[][] zoneSizes = {
            {MAX_BOARD, CARD_DIM},  // my_board
            {MAX_BOARD, CARD_DIM},  // opp_board
            {MAX_HAND, CARD_DIM},   // hand
            {MAX_GY, CARD_DIM},     // my_gy
            {MAX_GY, CARD_DIM},     // opp_gy
            {MAX_STACK, CARD_DIM},  // stack
        };

        float[][][][] zoneData = new float[6][1][][];
        float[][] zoneMasks = new float[6][];

        for (int z = 0; z < 6; z++) {
            int count = zoneSizes[z][0];
            int dim = zoneSizes[z][1];
            zoneData[z][0] = new float[count][dim];
            zoneMasks[z] = new float[count];

            for (int j = 0; j < count; j++) {
                int start = offset + j * dim;
                if (start + dim <= flat.length) {
                    boolean anyNonZero = false;
                    for (int k = 0; k < dim; k++) {
                        float v = flat[start + k];
                        v = Math.max(-10, Math.min(10, v));
                        if (Float.isNaN(v)) v = 0;
                        zoneData[z][0][j][k] = v;
                        if (v != 0) anyNonZero = true;
                    }
                    zoneMasks[z][j] = anyNonZero ? 1.0f : 0.0f;
                }
            }
            offset += count * dim;
        }

        // Create ONNX tensors
        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("global_features", OnnxTensor.createTensor(env, globalInput));
        String[] zoneNames = {"my_board", "opp_board", "hand", "my_gy", "opp_gy", "stack"};
        for (int z = 0; z < 6; z++) {
            inputs.put(zoneNames[z], OnnxTensor.createTensor(env, zoneData[z]));
            inputs.put(zoneNames[z] + "_mask",
                    OnnxTensor.createTensor(env, new float[][]{zoneMasks[z]}));
        }

        // Run state encoder
        try (OrtSession.Result result = stateEncoder.run(inputs)) {
            float[][] embedding = (float[][]) result.get(0).getValue();
            return embedding[0];
        } finally {
            for (OnnxTensor t : inputs.values()) t.close();
        }
    }

    // ── Value ──

    private float computeValue(float[] stateEmbedding) throws OrtException {
        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("state_embedding",
                OnnxTensor.createTensor(env, new float[][]{stateEmbedding}));
        try (OrtSession.Result result = valueHead.run(inputs)) {
            float[][] value = (float[][]) result.get(0).getValue();
            return value[0][0];
        } finally {
            for (OnnxTensor t : inputs.values()) t.close();
        }
    }

    // ── Priority ──

    // Fixed sizes matching ONNX export dummy inputs — must match export_onnx.py
    private static final int PRI_MAX = 50;
    private static final int TGT_MAX = 40;
    private static final int ATK_MAX = 40;
    private static final int BLK_MAX = 40;
    private static final int BLK_ATK_MAX = 40;
    private static final int CS_MAX = 40;
    private static final int MUL_MAX = 7;

    private DecisionResult handlePriority(float[] state, DecisionContext ctx, float value)
            throws OrtException {
        List<float[]> candidates = ctx.getCandidateFeatures();
        int n = candidates.size();
        int padN = Math.max(n, PRI_MAX);  // pad to at least export size

        float[][][] actionFeats = new float[1][padN][ACTION_DIM];
        float[][] mask = new float[1][padN];
        for (int i = 0; i < n; i++) {
            float[] cf = candidates.get(i);
            System.arraycopy(cf, 0, actionFeats[0][i], 0,
                    Math.min(cf.length, ACTION_DIM));
            mask[0][i] = 1.0f;
        }
        // Remaining slots have mask=0, features=0 (padding)

        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("state_embedding", OnnxTensor.createTensor(env, new float[][]{state}));
        inputs.put("action_features", OnnxTensor.createTensor(env, actionFeats));
        inputs.put("action_mask", OnnxTensor.createTensor(env, mask));

        try (OrtSession.Result result = priorityHead.run(inputs)) {
            float[][] logits = (float[][]) result.get(0).getValue();
            float[] probs = softmax(logits[0], n);
            int action = argmax(probs, n);
            // Debug: log probabilities for first few decisions
            StringBuilder sb = new StringBuilder();
            sb.append("ONNX_PRIORITY: n=").append(n).append(" pick=").append(action);
            sb.append(" probs=[");
            for (int i = 0; i < n; i++) {
                if (i > 0) sb.append(",");
                sb.append(String.format("%.2f", probs[i]));
            }
            sb.append("] value=").append(String.format("%.3f", value));
            Logger.info(sb.toString());
            return new DecisionResult(List.of(action), probs, value, false);
        } finally {
            for (OnnxTensor t : inputs.values()) t.close();
        }
    }

    // ── Target ──

    private DecisionResult handleTarget(float[] state, DecisionContext ctx, float value)
            throws OrtException {
        List<float[]> candidates = ctx.getCandidateFeatures();
        int n = candidates.size();
        int padN = Math.max(n, TGT_MAX);

        float[][][] feats = new float[1][padN][CARD_DIM];
        float[][] mask = new float[1][padN];
        for (int i = 0; i < n; i++) {
            float[] cf = candidates.get(i);
            System.arraycopy(cf, 0, feats[0][i], 0,
                    Math.min(cf.length, CARD_DIM));
            mask[0][i] = 1.0f;
        }

        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("state_embedding", OnnxTensor.createTensor(env, new float[][]{state}));
        inputs.put("target_features", OnnxTensor.createTensor(env, feats));
        inputs.put("target_mask", OnnxTensor.createTensor(env, mask));

        try (OrtSession.Result result = targetHead.run(inputs)) {
            float[][] logits = (float[][]) result.get(0).getValue();
            float[] probs = softmax(logits[0], n);
            int maxSelect = ctx.getMaxSelections();
            List<Integer> selected;
            if (maxSelect <= 1) {
                selected = List.of(argmax(probs, n));
            } else {
                selected = topK(probs, n, Math.min(maxSelect, n));
            }
            return new DecisionResult(selected, probs, value, false);
        } finally {
            for (OnnxTensor t : inputs.values()) t.close();
        }
    }

    // ── Attack ──

    private DecisionResult handleAttack(float[] state, DecisionContext ctx, float value)
            throws OrtException {
        List<float[]> candidates = ctx.getCandidateFeatures();
        int n = candidates.size();
        int padN = Math.max(n, ATK_MAX);

        float[][][] feats = new float[1][padN][CARD_DIM];
        float[][] mask = new float[1][padN];
        for (int i = 0; i < n; i++) {
            float[] cf = candidates.get(i);
            System.arraycopy(cf, 0, feats[0][i], 0,
                    Math.min(cf.length, CARD_DIM));
            mask[0][i] = 1.0f;
        }

        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("state_embedding", OnnxTensor.createTensor(env, new float[][]{state}));
        inputs.put("creature_features", OnnxTensor.createTensor(env, feats));
        inputs.put("creature_mask", OnnxTensor.createTensor(env, mask));

        try (OrtSession.Result result = attackHead.run(inputs)) {
            float[][] logits = (float[][]) result.get(0).getValue();
            float[] probs = new float[n];
            List<Integer> selected = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                // Clamp logits to [-5, 5] then sigmoid
                float clamped = Math.max(-5, Math.min(5, logits[0][i]));
                probs[i] = sigmoid(clamped);
                if (probs[i] > 0.5f) {
                    selected.add(i);
                }
            }
            return new DecisionResult(selected, probs, value, false);
        } finally {
            for (OnnxTensor t : inputs.values()) t.close();
        }
    }

    // ── Block ──

    private DecisionResult handleBlock(float[] state, DecisionContext ctx, float value)
            throws OrtException {
        List<float[]> candidates = ctx.getCandidateFeatures();
        if (candidates.isEmpty()) {
            return new DecisionResult(List.of(), new float[0], value, false);
        }

        // Reconstruct blockers and attackers from concatenated pairs
        // Same logic as Python model_server._handle_block
        int nPairs = candidates.size() - 1; // last is "no block" zero vector
        if (nPairs <= 0) {
            return new DecisionResult(List.of(), new float[0], value, false);
        }

        float[] firstBlocker = Arrays.copyOfRange(candidates.get(0), 0, CARD_DIM);
        int nAttackers = 1;
        for (int j = 1; j < nPairs; j++) {
            float[] other = Arrays.copyOfRange(candidates.get(j), 0, CARD_DIM);
            if (arraysClose(firstBlocker, other, 0.01f)) {
                nAttackers++;
            } else {
                break;
            }
        }
        int nBlockers = nPairs / Math.max(nAttackers, 1);
        if (nBlockers == 0 || nAttackers == 0) {
            return new DecisionResult(List.of(), new float[0], value, false);
        }

        int padB = Math.max(nBlockers, BLK_MAX);
        int padA = Math.max(nAttackers, BLK_ATK_MAX);
        float[][][] bf = new float[1][padB][CARD_DIM];
        float[][] bm = new float[1][padB];
        float[][][] af = new float[1][padA][CARD_DIM];
        float[][] am = new float[1][padA];

        for (int b = 0; b < nBlockers; b++) {
            int pairIdx = b * nAttackers;
            if (pairIdx < nPairs) {
                float[] pair = candidates.get(pairIdx);
                System.arraycopy(pair, 0, bf[0][b], 0,
                        Math.min(pair.length, CARD_DIM));
                bm[0][b] = 1.0f;
            }
        }
        for (int a = 0; a < nAttackers; a++) {
            if (a < nPairs) {
                float[] pair = candidates.get(a);
                int srcLen = Math.min(pair.length - CARD_DIM, CARD_DIM);
                if (srcLen > 0) {
                    System.arraycopy(pair, CARD_DIM, af[0][a], 0, srcLen);
                }
                am[0][a] = 1.0f;
            }
        }

        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("state_embedding", OnnxTensor.createTensor(env, new float[][]{state}));
        inputs.put("blocker_features", OnnxTensor.createTensor(env, bf));
        inputs.put("blocker_mask", OnnxTensor.createTensor(env, bm));
        inputs.put("attacker_features", OnnxTensor.createTensor(env, af));
        inputs.put("attacker_mask", OnnxTensor.createTensor(env, am));

        try (OrtSession.Result result = blockHead.run(inputs)) {
            float[][][] logits = (float[][][]) result.get(0).getValue();
            List<Integer> selectedPairs = new ArrayList<>();
            float[] allProbs = new float[nPairs + 1];

            for (int b = 0; b < nBlockers; b++) {
                float[] probs = softmax(logits[0][b], nAttackers + 1);
                int action = argmax(probs, nAttackers + 1);
                if (action < nAttackers) {
                    int pairIdx = b * nAttackers + action;
                    if (pairIdx < nPairs) {
                        selectedPairs.add(pairIdx);
                    }
                }
                for (int a = 0; a < nAttackers; a++) {
                    int pidx = b * nAttackers + a;
                    if (pidx < nPairs) {
                        allProbs[pidx] = probs[a];
                    }
                }
                allProbs[nPairs] += probs[nAttackers] / nBlockers;
            }

            return new DecisionResult(selectedPairs, allProbs, value, false);
        } finally {
            for (OnnxTensor t : inputs.values()) t.close();
        }
    }

    // ── Card Select ──

    private DecisionResult handleCardSelect(float[] state, DecisionContext ctx, float value)
            throws OrtException {
        List<float[]> candidates = ctx.getCandidateFeatures();
        int n = candidates.size();
        int padN = Math.max(n, CS_MAX);

        float[][][] feats = new float[1][padN][CARD_DIM];
        float[][] mask = new float[1][padN];
        for (int i = 0; i < n; i++) {
            float[] cf = candidates.get(i);
            System.arraycopy(cf, 0, feats[0][i], 0,
                    Math.min(cf.length, CARD_DIM));
            mask[0][i] = 1.0f;
        }

        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("state_embedding", OnnxTensor.createTensor(env, new float[][]{state}));
        inputs.put("card_features", OnnxTensor.createTensor(env, feats));
        inputs.put("card_mask", OnnxTensor.createTensor(env, mask));

        try (OrtSession.Result result = cardSelectHead.run(inputs)) {
            float[][] logits = (float[][]) result.get(0).getValue();
            float[] probs = softmax(logits[0], n);
            int numSelect = Math.max(ctx.getMinSelections(),
                    Math.min(ctx.getMaxSelections(), n));
            List<Integer> selected = topK(probs, n, numSelect);
            return new DecisionResult(selected, probs, value, false);
        } finally {
            for (OnnxTensor t : inputs.values()) t.close();
        }
    }

    // ── Mulligan ──

    private DecisionResult handleMulligan(float[] state, DecisionContext ctx, float value)
            throws OrtException {
        List<float[]> candidates = ctx.getCandidateFeatures();
        int n = candidates.size();
        int padN = Math.max(n, MUL_MAX);

        float[][][] feats = new float[1][padN][CARD_DIM];
        float[][] mask = new float[1][padN];
        for (int i = 0; i < n; i++) {
            float[] cf = candidates.get(i);
            System.arraycopy(cf, 0, feats[0][i], 0,
                    Math.min(cf.length, CARD_DIM));
            mask[0][i] = 1.0f;
        }

        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("state_embedding", OnnxTensor.createTensor(env, new float[][]{state}));
        inputs.put("hand_features", OnnxTensor.createTensor(env, feats));
        inputs.put("hand_mask", OnnxTensor.createTensor(env, mask));

        try (OrtSession.Result result = mulliganHead.run(inputs)) {
            // keep_logit is a scalar
            Object raw = result.get(0).getValue();
            float keepLogit;
            if (raw instanceof float[][]) {
                keepLogit = ((float[][]) raw)[0][0];
            } else if (raw instanceof float[]) {
                keepLogit = ((float[]) raw)[0];
            } else {
                keepLogit = 0f;
            }
            boolean keep = sigmoid(keepLogit) > 0.5f;
            float keepProb = sigmoid(keepLogit);
            return new DecisionResult(
                    List.of(keep ? 1 : 0),
                    new float[]{1 - keepProb, keepProb},
                    value, false);
        } finally {
            for (OnnxTensor t : inputs.values()) t.close();
        }
    }

    // ── Binary ──

    private DecisionResult handleBinary(float[] state, float value) throws OrtException {
        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("state_embedding",
                OnnxTensor.createTensor(env, new float[][]{state}));

        try (OrtSession.Result result = binaryHead.run(inputs)) {
            Object raw = result.get(0).getValue();
            float logit;
            if (raw instanceof float[][]) {
                logit = ((float[][]) raw)[0][0];
            } else if (raw instanceof float[]) {
                logit = ((float[]) raw)[0];
            } else {
                logit = 0f;
            }
            boolean yes = sigmoid(logit) > 0.5f;
            float prob = sigmoid(logit);
            return new DecisionResult(
                    List.of(yes ? 1 : 0),
                    new float[]{1 - prob, prob},
                    value, false);
        } finally {
            for (OnnxTensor t : inputs.values()) t.close();
        }
    }

    // ── Utility Functions ──

    private static float sigmoid(float x) {
        return 1.0f / (1.0f + (float) Math.exp(-x));
    }

    private static float[] softmax(float[] logits, int n) {
        float[] probs = new float[n];
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < n; i++) max = Math.max(max, logits[i]);
        float sum = 0;
        for (int i = 0; i < n; i++) {
            probs[i] = (float) Math.exp(logits[i] - max);
            sum += probs[i];
        }
        for (int i = 0; i < n; i++) probs[i] /= sum;
        return probs;
    }

    private static int argmax(float[] probs, int n) {
        int best = 0;
        for (int i = 1; i < n; i++) {
            if (probs[i] > probs[best]) best = i;
        }
        return best;
    }

    private static List<Integer> topK(float[] probs, int n, int k) {
        Integer[] indices = new Integer[n];
        for (int i = 0; i < n; i++) indices[i] = i;
        Arrays.sort(indices, (a, b) -> Float.compare(probs[b], probs[a]));
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < Math.min(k, n); i++) {
            result.add(indices[i]);
        }
        return result;
    }

    private static void clip(float[] arr) {
        for (int i = 0; i < arr.length; i++) {
            arr[i] = Math.max(-10, Math.min(10, arr[i]));
            if (Float.isNaN(arr[i])) arr[i] = 0;
        }
    }

    private static boolean arraysClose(float[] a, float[] b, float tol) {
        int n = Math.min(a.length, b.length);
        for (int i = 0; i < n; i++) {
            if (Math.abs(a[i] - b[i]) > tol) return false;
        }
        return true;
    }

    public void close() {
        try {
            if (stateEncoder != null) stateEncoder.close();
            if (valueHead != null) valueHead.close();
            if (priorityHead != null) priorityHead.close();
            if (targetHead != null) targetHead.close();
            if (attackHead != null) attackHead.close();
            if (blockHead != null) blockHead.close();
            if (cardSelectHead != null) cardSelectHead.close();
            if (mulliganHead != null) mulliganHead.close();
            if (binaryHead != null) binaryHead.close();
        } catch (Exception e) {
            Logger.warn("Error closing ONNX sessions: {}", e.getMessage());
        }
    }
}
