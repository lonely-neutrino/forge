package forge.anvil;

import com.google.protobuf.ByteString;

import forge.ai.anvil.AnvilBridge;
import forge.ai.anvil.CastPlanAnswer;
import forge.ai.anvil.CombatMapAnswer;
import forge.ai.anvil.Obs;
import forge.anvil.bridge.v0.AnswerShape;
import forge.anvil.bridge.v0.CastPlan;
import forge.anvil.bridge.v0.EntityRef;
import forge.anvil.bridge.v0.Constraints;
import forge.anvil.bridge.v0.DecisionBridgeGrpc;
import forge.anvil.bridge.v0.DecisionRequest;
import forge.anvil.bridge.v0.DecisionResponse;
import forge.anvil.bridge.v0.GameEnd;
import forge.anvil.bridge.v0.GameStart;
import forge.anvil.bridge.v0.IndexList;
import forge.anvil.bridge.v0.Option;
import forge.anvil.bridge.v0.ServerMsg;
import forge.anvil.bridge.v0.WorkerHello;
import forge.anvil.bridge.v0.WorkerMsg;
import forge.util.MyRandom;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.stub.StreamObserver;

import java.util.List;
import java.util.Set;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

/**
 * gRPC implementation of AnvilBridge (bridge-protocol-v0): one long-lived
 * bidirectional stream, one outstanding request at a time (the game thread
 * blocks). At M0 every answer is pre-drawn from the game's seeded RNG and sent
 * as echo_answer — the server echoes it back, so gRPC-arm games are
 * bit-identical to local-arm games and the throughput delta isolates
 * serialization + transport. On deadline or stream failure the pre-drawn
 * answer is used locally (counted; the run stays alive and deterministic).
 */
public final class GrpcBridge implements AnvilBridge {
    // Operational knob only — never part of a correctness argument (issue #9
    // point 6). Instance-read so tests can shrink it before construction.
    private final int deadlineMs = Integer.getInteger("anvil.bridge.deadline.ms", 5000);

    /**
     * The stream lost request/response correspondence: a decision deadline
     * expired with the request still in flight, or a response arrived whose
     * decision_seq is not the outstanding request's. Either way the shared
     * queue now carries response debt and the NEXT waiter would consume a
     * stale answer — the off-by-one desync of issue #9 (anvil), where a
     * late response was silently applied to the following decision (and
     * across game boundaries) as if it answered it. Per bridge-protocol-v0
     * the game is discarded and the worker drains: this exception fails the
     * current game loudly (crash_or_hang:BridgePoisonedException), and
     * AnvilRun stops starting games on a poisoned bridge so the harness
     * recycles the worker with a fresh stream.
     */
    public static final class BridgePoisonedException extends RuntimeException {
        BridgePoisonedException(String msg) {
            super(msg);
        }
    }

    private final ManagedChannel channel;
    private final StreamObserver<WorkerMsg> out;
    private final LinkedBlockingQueue<ServerMsg> in = new LinkedBlockingQueue<>();
    private final Set<String> serverTags;
    private final boolean oneShotCast;
    private final boolean forcedPriorityOption;
    private long seq;
    private long lastPrioritySeq;
    private String gameId = "";
    private int transportFailures;
    private int serverFallbacks;
    private String poisonReason;

    public GrpcBridge(String host, int port, String workerId, String forkCommit) {
        channel = ManagedChannelBuilder.forAddress(host, port).usePlaintext().build();
        out = DecisionBridgeGrpc.newStub(channel).session(new StreamObserver<ServerMsg>() {
            @Override
            public void onNext(ServerMsg msg) {
                in.add(msg);
            }

            @Override
            public void onError(Throwable t) {
                System.err.println("[GrpcBridge] stream error: " + t);
            }

            @Override
            public void onCompleted() {
            }
        });
        out.onNext(WorkerMsg.newBuilder().setHello(WorkerHello.newBuilder()
                .setProtocolVersion(0).setWorkerId(workerId).setForkCommit(forkCommit)).build());
        ServerMsg hello = await();
        if (hello == null || !hello.hasHello()) {
            throw new IllegalStateException("Anvil decision server handshake failed");
        }
        serverTags = Set.copyOf(hello.getHello().getBridgedTagsList());
        oneShotCast = hello.getHello().getOneShotCast();
        forcedPriorityOption = hello.getHello().getForcedPriorityOption();
    }

    /** Server-driven coverage: the tag set this session answers over the wire. */
    public Set<String> serverBridgedTags() {
        return serverTags;
    }

    /** Additive candidate-drill capability; false means the worker must
     * report an explicit unsupported arm rather than silently natural-cast. */
    public boolean forcedPriorityOption() {
        return forcedPriorityOption;
    }

    @Override
    public boolean supportsForcedCandidate() {
        return oneShotCast && forcedPriorityOption;
    }

    public int transportFailures() {
        return transportFailures;
    }

    private ServerMsg await() {
        try {
            return in.poll(deadlineMs, TimeUnit.MILLISECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return null;
        }
    }

    @Override
    public boolean poisoned() {
        return poisonReason != null;
    }

    private BridgePoisonedException poison(String reason) {
        poisonReason = reason;
        System.err.println("[GrpcBridge] POISONED: " + reason
                + " — game discarded, worker will drain (protocol-v0 mismatch clause)");
        return new BridgePoisonedException(reason);
    }

    private DecisionResponse roundTrip(String tag, AnswerShape shape, List<String> optionLabels,
            Constraints constraints, DecisionResponse echo) {
        if (poisonReason != null) {
            // No decision may read the queue once correspondence is lost —
            // the next message may be debt from an abandoned request.
            throw new BridgePoisonedException(poisonReason);
        }
        DecisionRequest.Builder req = DecisionRequest.newBuilder()
                .setGameId(gameId).setDecisionSeq(++seq).setDecisionTag(tag)
                .setShape(shape).setDeadlineMs(deadlineMs).setEchoAnswer(echo);
        if (constraints != null) {
            req.setConstraints(constraints);
        }
        if (optionLabels != null) {
            for (int i = 0; i < optionLabels.size(); i++) {
                String label = optionLabels.get(i);
                req.addOptions(Option.newBuilder().setId(i).setLabel(label == null ? "" : label));
            }
        }
        // M1: every bridged decision carries its observation (the model server
        // answers one-field tags from the same record the obs log writes).
        // Cheap when logging is on (string already built); null when off.
        String obs = Obs.lastDecForBridge();
        if (obs != null) {
            req.setObservation(ByteString.copyFromUtf8(obs));
        }
        out.onNext(WorkerMsg.newBuilder().setRequest(req).build());
        ServerMsg msg = await();
        if (msg == null) {
            // Deadline expired with request `seq` still alive server-side.
            // Answering locally and continuing (the pre-#9 behavior) leaves
            // one unit of response debt on the stream: every later decision
            // silently consumes its predecessor's answer. Fail the game.
            transportFailures++;
            throw poison("deadline on " + tag + " seq=" + seq
                    + " (transport failure " + transportFailures + ")");
        }
        if (!msg.hasResponse()) {
            throw poison("non-response message while awaiting " + tag + " seq=" + seq);
        }
        if (msg.getResponse().getDecisionSeq() != seq) {
            throw poison("decision_seq mismatch on " + tag + ": expected " + seq
                    + ", got " + msg.getResponse().getDecisionSeq());
        }
        // A fallback response means "answer locally, tagged" — reading the
        // oneof's default value instead looped mulligans forever (D8 smoke 1:
        // default flag=false read as never-keep, 14.8K mulligans, obs cap).
        if (msg.getResponse().getFallback()) {
            serverFallbacks++;
            return echo;
        }
        return msg.getResponse();
    }

    @Override
    public int selectOne(String tag, List<String> optionLabels) {
        int n = optionLabels.size();
        // M9 D3 (§3c): the payment-class window's local echo is AUTO (option
        // 0), never random — a server that declines the tag (fallback:true,
        // e.g. a model checkpoint without the payment head) must reproduce
        // today's behavior exactly, not pick random payment classes. Other
        // tags keep the M0 random-legal echo semantics.
        int local = n <= 1 || forge.ai.anvil.PlayerControllerAnvil.TAG_PAY_CLASS.equals(tag)
                ? 0 : MyRandom.getRandom().nextInt(n);
        DecisionResponse resp = roundTrip(tag, AnswerShape.SELECT_ONE, optionLabels, null,
                DecisionResponse.newBuilder().setIndex(local).build());
        return resp.getIndex();
    }

    @Override
    public int[] selectK(String tag, int n, int k) {
        int[] all = new int[n];
        for (int i = 0; i < n; i++) {
            all[i] = i;
        }
        for (int i = 0; i < k && i < n; i++) {
            int j = i + MyRandom.getRandom().nextInt(n - i);
            int tmp = all[i];
            all[i] = all[j];
            all[j] = tmp;
        }
        IndexList.Builder local = IndexList.newBuilder();
        java.util.Arrays.stream(all, 0, Math.min(k, n)).sorted().forEach(local::addIndices);
        DecisionResponse resp = roundTrip(tag, AnswerShape.SELECT_K, null,
                Constraints.newBuilder().setK(Math.min(k, n)).setMax(n).build(),
                DecisionResponse.newBuilder().setIndices(local).build());
        return resp.getIndices().getIndicesList().stream().mapToInt(Integer::intValue).toArray();
    }

    @Override
    public boolean bool(String tag) {
        boolean local = MyRandom.getRandom().nextBoolean();
        DecisionResponse resp = roundTrip(tag, AnswerShape.BOOL, null, null,
                DecisionResponse.newBuilder().setFlag(local).build());
        return resp.getFlag();
    }

    @Override
    public int intInRange(String tag, int min, int max) {
        int local = max <= min ? min : min + MyRandom.getRandom().nextInt(max - min + 1);
        DecisionResponse resp = roundTrip(tag, AnswerShape.INT_IN_RANGE, null,
                Constraints.newBuilder().setMin(min).setMax(max).build(),
                DecisionResponse.newBuilder().setValue(local).build());
        return (int) resp.getValue();
    }

    @Override
    public void gameStart(String id, long seed) {
        // M1: observation game header (AnvilRun calls Obs.startGame first, so
        // it exists whenever obs logging is on; absent -> M0-shape session).
        gameStart(id, seed, Obs.lastHeaderForBridge());
    }

    /** Explicit-header variant (M2 D1): fork drivers announce derived game
     *  ids and re-announce the mainline after a fork replay — the header
     *  they pass (per-game, session-routed) is authoritative. */
    @Override
    public void gameStart(String id, long seed, String header) {
        gameId = id;
        GameStart.Builder gs = GameStart.newBuilder()
                .setGameId(id).setSeed(seed).setFormatTag("mtg.commander");
        if (header != null) {
            gs.setHeader(ByteString.copyFromUtf8(header));
        }
        out.onNext(WorkerMsg.newBuilder().setGameStart(gs).build());
    }

    /**
     * M1 one-shot cast. Active only when the server declared one_shot_cast in
     * its hello; otherwise null routes the caller to the M0 selectOne path.
     * Transport failure or server fallback answers PASS (counted): an eval
     * arm must never silently substitute random or heuristic play for the
     * model on a bridged tag.
     */
    @Override
    public CastPlanAnswer priorityCastPlan(String tag, List<String> optionLabels,
            String observation) {
        return priorityCastPlan(tag, optionLabels, observation, 0);
    }

    @Override
    public CastPlanAnswer priorityCastPlan(String tag, List<String> optionLabels,
            String observation, int attempt) {
        return priorityCastPlan(tag, optionLabels, observation, attempt, false);
    }

    @Override
    public CastPlanAnswer priorityCastPlan(String tag, List<String> optionLabels,
            String observation, int attempt, boolean forbidDecline) {
        return priorityCastPlan(tag, optionLabels, observation, attempt, forbidDecline, 0);
    }

    @Override
    public CastPlanAnswer priorityCastPlan(String tag, List<String> optionLabels,
            String observation, int attempt, boolean forbidDecline, int forcedOption) {
        if (!oneShotCast) {
            return null;
        }
        if (forcedOption > 0 && !forcedPriorityOption) {
            // The caller's candidate labels distinguish this null from a
            // natural pass as UNSUPPORTED; no fallback arm is attempted.
            return null;
        }
        if (poisonReason != null) {
            throw new BridgePoisonedException(poisonReason);
        }
        long prevPrioritySeq = lastPrioritySeq;
        DecisionRequest.Builder req = DecisionRequest.newBuilder()
                .setGameId(gameId).setDecisionSeq(++seq).setDecisionTag(tag)
                .setShape(AnswerShape.CONSTRUCT).setDeadlineMs(deadlineMs);
        if (forbidDecline) {
            // M7 forced-branch act ask: server masks the pass logit.
            req.setForbidDecline(true);
        }
        if (forcedOption > 0) {
            req.setForceOption(true).setForcedOption(forcedOption);
        }
        lastPrioritySeq = seq;
        if (attempt > 0 && prevPrioritySeq > 0) {
            // Re-ask after veto: mark the superseded request so server stats
            // can count re-asks (the mu record keys off the obs seq, not this).
            req.setRetryOf(prevPrioritySeq);
        }
        if (optionLabels != null) {
            for (int i = 0; i < optionLabels.size(); i++) {
                String label = optionLabels.get(i);
                req.addOptions(Option.newBuilder().setId(i).setLabel(label == null ? "" : label));
            }
        }
        if (observation != null) {
            req.setObservation(ByteString.copyFromUtf8(observation));
        }
        out.onNext(WorkerMsg.newBuilder().setRequest(req).build());
        ServerMsg msg = await();
        if (msg == null) {
            transportFailures++;
            throw poison("deadline on " + tag + " seq=" + seq
                    + " (transport failure " + transportFailures + ")");
        }
        if (!msg.hasResponse()) {
            throw poison("non-response message while awaiting " + tag + " seq=" + seq);
        }
        if (msg.getResponse().getDecisionSeq() != seq) {
            throw poison("decision_seq mismatch on " + tag + ": expected " + seq
                    + ", got " + msg.getResponse().getDecisionSeq());
        }
        DecisionResponse resp = msg.getResponse();
        if (resp.getFallback() || !resp.hasConstruct() || !resp.getConstruct().hasCastPlan()) {
            transportFailures++;
            return pass();
        }
        CastPlan cp = resp.getConstruct().getCastPlan();
        java.util.List<CastPlanAnswer.Ref> refs =
                new java.util.ArrayList<>(cp.getTargetRefsCount());
        for (EntityRef r : cp.getTargetRefsList()) {
            if (r.getRefCase() == EntityRef.RefCase.PLAYER) {
                refs.add(new CastPlanAnswer.Ref(true, r.getPlayer(), -1, false));
            } else {
                refs.add(new CastPlanAnswer.Ref(false, -1, (int) r.getEntity(), r.getNs() == 1));
            }
        }
        return new CastPlanAnswer((int) cp.getSpellOption(), cp.getHostLevel(), refs,
                cp.getHasX(), (int) cp.getXValue());
    }

    private static CastPlanAnswer pass() {
        return new CastPlanAnswer(0, false, java.util.Collections.emptyList(), false, 0);
    }

    /**
     * M2 D5 combat declarations. Transport failure or server fallback answers
     * the EMPTY map (counted) — the realizer's validate tiers repair upward
     * only where the rules require it; never heuristic substitution on a
     * bridged tag (same principle as the one-shot PASS above).
     */
    @Override
    public CombatMapAnswer attackMap(String tag, String observation) {
        forge.anvil.bridge.v0.Construct c = roundTripConstruct(tag, observation);
        if (c == null || !c.hasAttackMap()) {
            if (c != null) {
                transportFailures++; // server answered the wrong construct kind
            }
            return CombatMapAnswer.empty();
        }
        java.util.List<CombatMapAnswer.Assignment> out =
                new java.util.ArrayList<>(c.getAttackMap().getAssignmentsCount());
        for (forge.anvil.bridge.v0.AttackMap.Assignment a : c.getAttackMap().getAssignmentsList()) {
            out.add(new CombatMapAnswer.Assignment(toRef(a.getAttacker()), toRef(a.getDefender())));
        }
        return new CombatMapAnswer(out);
    }

    @Override
    public CombatMapAnswer blockMap(String tag, String observation) {
        forge.anvil.bridge.v0.Construct c = roundTripConstruct(tag, observation);
        if (c == null || !c.hasBlockMap()) {
            if (c != null) {
                transportFailures++; // server answered the wrong construct kind
            }
            return CombatMapAnswer.empty();
        }
        java.util.List<CombatMapAnswer.Assignment> out =
                new java.util.ArrayList<>(c.getBlockMap().getAssignmentsCount());
        for (forge.anvil.bridge.v0.BlockMap.Assignment a : c.getBlockMap().getAssignmentsList()) {
            out.add(new CombatMapAnswer.Assignment(toRef(a.getBlocker()), toRef(a.getAttacker())));
        }
        return new CombatMapAnswer(out);
    }

    /** One CONSTRUCT round-trip; null = server fallback/shape miss (counted). */
    private forge.anvil.bridge.v0.Construct roundTripConstruct(String tag, String observation) {
        if (poisonReason != null) {
            throw new BridgePoisonedException(poisonReason);
        }
        DecisionRequest.Builder req = DecisionRequest.newBuilder()
                .setGameId(gameId).setDecisionSeq(++seq).setDecisionTag(tag)
                .setShape(AnswerShape.CONSTRUCT).setDeadlineMs(deadlineMs);
        if (observation != null) {
            req.setObservation(ByteString.copyFromUtf8(observation));
        }
        out.onNext(WorkerMsg.newBuilder().setRequest(req).build());
        ServerMsg msg = await();
        if (msg == null) {
            transportFailures++;
            throw poison("deadline on " + tag + " seq=" + seq
                    + " (transport failure " + transportFailures + ")");
        }
        if (!msg.hasResponse()) {
            throw poison("non-response message while awaiting " + tag + " seq=" + seq);
        }
        if (msg.getResponse().getDecisionSeq() != seq) {
            throw poison("decision_seq mismatch on " + tag + ": expected " + seq
                    + ", got " + msg.getResponse().getDecisionSeq());
        }
        DecisionResponse resp = msg.getResponse();
        if (resp.getFallback() || !resp.hasConstruct()) {
            transportFailures++;
            return null;
        }
        return resp.getConstruct();
    }

    private static CastPlanAnswer.Ref toRef(EntityRef r) {
        if (r.getRefCase() == EntityRef.RefCase.PLAYER) {
            return new CastPlanAnswer.Ref(true, r.getPlayer(), -1, false);
        }
        return new CastPlanAnswer.Ref(false, -1, (int) r.getEntity(), r.getNs() == 1);
    }

    @Override
    public void gameEnd(String id, String winner, int turns, long wallMs) {
        out.onNext(WorkerMsg.newBuilder().setGameEnd(GameEnd.newBuilder()
                .setGameId(id).setWinner(winner == null ? "" : winner)
                .setTurns(Math.max(turns, 0)).setWallMs(wallMs)).build());
    }

    @Override
    public void close() {
        out.onCompleted();
        channel.shutdown();
        try {
            channel.awaitTermination(2, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}
