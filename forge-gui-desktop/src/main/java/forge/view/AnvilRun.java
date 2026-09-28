package forge.view;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

import com.google.common.eventbus.Subscribe;

import forge.ai.AiProfileUtil;
import forge.ai.anvil.AnvilBridge;
import forge.ai.simulation.GameCopier;
import forge.ai.anvil.AnvilLobbyPlayer;
import forge.ai.anvil.Census;
import forge.ai.anvil.ChoiceDirective;
import forge.ai.anvil.LocalRandomBridge;
import forge.ai.anvil.Obs;
import forge.ai.anvil.PlayerControllerAnvil;
import forge.ai.anvil.ScheduleDirective;
import forge.deck.Deck;
import forge.game.Game;
import forge.game.GameEndReason;
import forge.game.GameRules;
import forge.game.GameType;
import forge.game.Match;
import forge.game.card.Card;
import forge.game.event.GameEventPlayerPriority;
import forge.game.event.GameEventTurnBegan;
import forge.game.phase.PhaseHandler;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.player.RegisteredPlayer;
import forge.game.zone.ZoneType;
import forge.model.FModel;
import forge.util.MyRandom;

/**
 * Anvil worker (M0 batch-harness chunk contract): plays a chunk of globally
 * indexed games, appends one JSONL record per completed game (the progress
 * record resume scans), checks a stop-file between games (graceful stop:
 * finish current game, flush, exit 0), and exits when the chunk is done
 * (recycling = chunk boundary). Seeds: SplitMix64(seed_base ^ index), same
 * function as the Python orchestrator (anvil/bridge/harness/seeds.py).
 *
 * Syntax: forge anvil (-d <deck1> <deck2> | -pairs <file> [-gpp <gamesPerPair>])
 *   [-f <format>] [-b local-random|grpc:host:port]
 *   [-tags <csv>] [-census <out.jsonl>] [-obs <out.zst>]
 *   chunk mode:  -range <start> <count> -seedbase <long> [-results <games.jsonl>] [-stopfile <path>]
 *   legacy mode: [-n <games>] [-s <baseSeed>]
 *
 * -pairs: one deck pair per line, tab-separated (deck names contain spaces);
 * game index i plays pair (i / gpp) % nPairs. AI personalities are drawn
 * per seat from the game seed (sorted profile list), so corpus provenance is
 * a pure function of (seedbase, index) — logged in results JSONL and the
 * observation game record.
 */
public final class AnvilRun {
    private static final int DRAW_CLOCK_S = 300;
    private static final int GAME_HARD_CAP_S = 360;

    private static final Set<String> DEFAULT_TAGS = new HashSet<>(Arrays.asList(
            PlayerControllerAnvil.TAG_PRIORITY, PlayerControllerAnvil.TAG_MULLIGAN,
            PlayerControllerAnvil.TAG_TUCK, PlayerControllerAnvil.TAG_TRIGGER,
            PlayerControllerAnvil.TAG_BINARY, PlayerControllerAnvil.TAG_NUMBER));

    private AnvilRun() {
    }

    /** Same constants as Python-side seeds.py; the pair must stay in lockstep. */
    static long splitmix64(long z) {
        z += 0x9E3779B97F4A7C15L;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    public static void run(String[] args) {
        FModel.initialize(null, null);

        Map<String, List<String>> params = parseParams(args);
        boolean fixedPair = params != null && params.containsKey("d") && params.get("d").size() == 2;
        boolean pairFile = params != null && params.containsKey("pairs");
        if (params == null || fixedPair == pairFile) {
            System.out.println("Syntax: forge anvil (-d <deck1> <deck2> | -pairs <file> [-gpp <n>]) [-f <format>] "
                    + "[-b local-random|grpc:host:port] [-tags <csv>] [-bridgeseats <csv>] [-reask] "
                    + "[-census <out.jsonl>] [-obs <out.zst>] [-paytelemetry] "
                    + "[-range <start> <count> -seedbase <long> [-results <jsonl>] [-stopfile <path>]] "
                    + "[-rollout <k> -points <m> -labels <jsonl> [-noreshuffle]] "
                    + "[-drillfile <txt> [-drillstop]] [-forkobs] [-forcebranch] [-forceseq <n>] "
                    + "[-forcecandidate <tsv>] "
                    + "[-seqarms nat|all] [-forceschedule <tsv>] [-forcechoice <tsv>] "
                    + "[-n <games>] [-s <baseSeed>]");
            return;
        }

        GameType type = params.containsKey("f")
                ? GameType.valueOf(params.get("f").get(0)) : GameType.Commander;
        String bridgeMode = params.containsKey("b") ? params.get("b").get(0) : "local-random";
        Set<String> tags = params.containsKey("tags")
                ? new HashSet<>(Arrays.asList(params.get("tags").get(0).split(","))) : DEFAULT_TAGS;
        // null = all seats bridged (self-play); "-bridgeseats 0" = seat 0 vs heuristic
        Set<Integer> bridgeSeats = null;
        if (params.containsKey("bridgeseats")) {
            bridgeSeats = new HashSet<>();
            for (String s : params.get("bridgeseats").get(0).split(",")) {
                bridgeSeats.add(Integer.parseInt(s.trim()));
            }
        }
        // D6 run-2: re-ask-on-veto (d6-vtrace-loop §6b). Per-JVM, all seats.
        boolean reask = params.containsKey("reask");
        PlayerControllerAnvil.setReaskOnVeto(reask);
        // M9 D3 §3c: payment-surface census telemetry-only mode (enumeration +
        // flag telemetry on every in-scope payManaCost, no bridging —
        // m9-payment-surface-spec.md §8). Trajectory-perturbing like -obs;
        // runs pin the flag.
        forge.ai.anvil.PaymentTelemetry.enabled = params.containsKey("paytelemetry");

        int rangeStart = 0;
        int nGames;
        Long seedBase = params.containsKey("seedbase")
                ? Long.parseLong(params.get("seedbase").get(0)) : null;
        long legacyBaseSeed = params.containsKey("s")
                ? Long.parseLong(params.get("s").get(0)) : 20260704L;
        if (params.containsKey("range")) {
            rangeStart = Integer.parseInt(params.get("range").get(0));
            nGames = Integer.parseInt(params.get("range").get(1));
        } else {
            nGames = params.containsKey("n") ? Integer.parseInt(params.get("n").get(0)) : 10;
        }
        File stopFile = params.containsKey("stopfile") ? new File(params.get("stopfile").get(0)) : null;

        // Rollout-label mode (M2 D4): at -points sampled quiescent MAIN1
        // priority windows per game, fork the live game and complete -rollout
        // copies to game end under the bridge (forks inherit Anvil
        // controllers; wire-only obs sessions keep them out of the store).
        // One labels-JSONL record per fork point; an Obs "mark" record keys
        // the fork point to the next mainline priority window. Unless
        // -noreshuffle, each rollout silently re-randomizes both libraries
        // (determinization: the label approximates E over unseen order, not
        // the outcome of the one concrete order nobody has seen).
        int rolloutK = params.containsKey("rollout")
                ? Integer.parseInt(params.get("rollout").get(0)) : 0;
        int rolloutPoints = params.containsKey("points")
                ? Integer.parseInt(params.get("points").get(0)) : 4;
        boolean rolloutReshuffle = !params.containsKey("noreshuffle");

        // Drill mode (M4 D2): -drillfile gives explicit per-game fork turns
        // ("<index> <t1>[,<t2>...]" per line, '#' comments) in place of
        // -points sampling; indices absent from the file are skipped without
        // creating a game (a "drill_skip" results row keeps harness resume
        // accounting exact). -drillstop ends the mainline right after its
        // last fork point — the completions are the product, the rest of the
        // replay is waste.
        Map<Integer, int[]> drillTargets = null;
        if (params.containsKey("drillfile")) {
            if (rolloutK <= 0) {
                System.err.println("FATAL: -drillfile requires -rollout <k>");
                System.exit(2);
            }
            drillTargets = new HashMap<>();
            try {
                for (String line : java.nio.file.Files.readAllLines(
                        java.nio.file.Paths.get(params.get("drillfile").get(0)))) {
                    line = line.trim();
                    if (line.isEmpty() || line.startsWith("#")) {
                        continue;
                    }
                    String[] parts = line.split("\\s+");
                    String[] ts = parts[1].split(",");
                    int[] turns = new int[ts.length];
                    for (int j = 0; j < ts.length; j++) {
                        turns[j] = Integer.parseInt(ts[j]);
                    }
                    drillTargets.put(Integer.parseInt(parts[0]), turns);
                }
            } catch (java.io.IOException | RuntimeException e) {
                System.err.println("FATAL: cannot read drillfile: " + e);
                System.exit(2);
            }
        }
        boolean drillStop = params.containsKey("drillstop");

        // Fork-session store (M4 D3): -forkobs streams every completion's
        // records to <obs>-forks.zst as a store frame of its own (synthetic
        // game id, fork provenance header, per-dec wire hist) and announces
        // the completion's OWN rollout seed to the bridge (per-completion
        // sampled noise decorrelates). Off = byte-identical to before.
        boolean forkObs = params.containsKey("forkobs");
        if (forkObs && rolloutK <= 0) {
            System.err.println("FATAL: -forkobs requires -rollout <k>");
            System.exit(2);
        }
        if (forkObs && rolloutK > 99) {
            // synthetic id = FORK_G_BASE + (gameIdx*100 + fp)*100 + r
            System.err.println("FATAL: -forkobs supports -rollout k <= 99");
            System.exit(2);
        }
        if (forkObs && !params.containsKey("obs")) {
            System.err.println("FATAL: -forkobs requires -obs (replay fidelity rule)");
            System.exit(2);
        }
        if (params.containsKey("forkns")) {
            long ns = Long.parseLong(params.get("forkns").get(0));
            if (ns < 0 || ns >= (Long.MAX_VALUE - FORK_G_BASE) / FORK_NS_STRIDE) {
                System.err.println("FATAL: -forkns out of range: " + ns);
                System.exit(2);
            }
            forkGBase = FORK_G_BASE + ns * FORK_NS_STRIDE;
        }

        // M7 forced-branch paired rollouts (m7-plan D2): per fork point, two
        // branches (act/hold) x k completions with PAIRED rollout seeds — the
        // drilled seat's first post-fork decision is forced (act = bridge ask
        // with pass masked; hold = one forced pass). Labels-only product
        // (pin 3): no fork stores, so -forkobs is excluded by design, and the
        // wire sessions announce per-completion seeds (instrument serving).
        boolean forceBranch = params.containsKey("forcebranch");
        if (forceBranch && (drillTargets == null || rolloutK <= 0)) {
            System.err.println("FATAL: -forcebranch requires -drillfile + -rollout <k>");
            System.exit(2);
        }
        if (forceBranch && forkObs) {
            System.err.println("FATAL: -forcebranch excludes -forkobs "
                    + "(m7-plan D2 pin 3: forced completions never build stores)");
            System.exit(2);
        }
        // M7 D2 sequence probe (routing pin 2026-08-11): -forceseq <n> = three
        // arms per fork point (natural / hold-n-turns / act-n-turns) x K
        // paired completions; persistent directive, labels-only.
        int forceSeq = params.containsKey("forceseq")
                ? Integer.parseInt(params.get("forceseq").get(0)) : 0;
        if (forceSeq > 0 && (drillTargets == null || rolloutK <= 0)) {
            System.err.println("FATAL: -forceseq requires -drillfile + -rollout <k>");
            System.exit(2);
        }
        if (forceSeq > 0 && (forkObs || forceBranch)) {
            System.err.println("FATAL: -forceseq excludes -forkobs and -forcebranch");
            System.exit(2);
        }
        // M8 D1 (m8-plan): -seqarms nat = the natural arm alone, OBSERVE
        // directive (records first-spell/first-land timing, never forces).
        // Labels-only like the rest of the seq machinery.
        String seqArms = params.containsKey("seqarms") ? params.get("seqarms").get(0) : "all";
        if (params.containsKey("seqarms")
                && (forceSeq <= 0 || !("nat".equals(seqArms) || "all".equals(seqArms)))) {
            System.err.println("FATAL: -seqarms requires -forceseq and takes nat|all");
            System.exit(2);
        }
        boolean seqNatOnly = forceSeq > 0 && "nat".equals(seqArms);

        // M10 sched mode (m10-ceiling-spec "Engine build owed"):
        // -forceschedule <file> = per-(game, turn) schedule arms; NATURAL +
        // each directed arm x K paired completions per fork point,
        // horizon-stopped (h > 0) or run to natural end (h = 0), one labels
        // row per completion (directive trace + certify-style snapshot).
        // TSV contract with the Python planner (labels must not contain
        // tabs; the planner asserts):
        //   gameIdx \t turn \t horizon \t seat \t armId \t joint|auto \t label...
        // armId >= 1 (0 = the implicit NATURAL arm); no labels = hold-all.
        Map<Integer, Map<Integer, SchedPoint>> schedJobs = null;
        int schedMaxArms = 0;
        if (params.containsKey("forceschedule")) {
            if (rolloutK <= 0 || !params.containsKey("labels")) {
                System.err.println("FATAL: -forceschedule requires -rollout <k> + -labels");
                System.exit(2);
            }
            if (forkObs || forceBranch || forceSeq > 0 || drillTargets != null) {
                System.err.println("FATAL: -forceschedule excludes -forkobs/-forcebranch/-forceseq/-drillfile");
                System.exit(2);
            }
            schedJobs = readSchedFile(params.get("forceschedule").get(0));
            // derive drill targeting from the jobs; the completions are the
            // product, so the mainline always stops after its last point.
            drillTargets = new HashMap<>();
            for (Map.Entry<Integer, Map<Integer, SchedPoint>> e : schedJobs.entrySet()) {
                int[] ts = new int[e.getValue().size()];
                int i = 0;
                for (int t : e.getValue().keySet()) {
                    ts[i++] = t;
                }
                java.util.Arrays.sort(ts);
                drillTargets.put(e.getKey(), ts);
                for (SchedPoint p : e.getValue().values()) {
                    schedMaxArms = Math.max(schedMaxArms, p.arms.size());
                }
            }
            drillStop = true;
        }

        // M11 choice mode (m11-routing-probes-spec.md): -forcechoice <file> =
        // per-(game, turn) forced-choice arms; NATURAL + each arm x K paired
        // completions per fork point, horizon-stopped (h > 0) or run to
        // natural end (h = 0), one labels row per completion (directive
        // trace + certify-style snapshot). TSV contract with the planner:
        //   gameIdx \t turn \t seat \t horizon \t armId \t tutor|prevent \t action
        // armId >= 1 (0 = the implicit NATURAL arm); action = 0-based
        // candidate index (tutor) or pay|decline (prevent).
        Map<Integer, Map<Integer, ChoicePoint>> choiceJobs = null;
        int choiceMaxArms = 0;
        if (params.containsKey("forcechoice")) {
            if (rolloutK <= 0 || !params.containsKey("labels")) {
                System.err.println("FATAL: -forcechoice requires -rollout <k> + -labels");
                System.exit(2);
            }
            if (forkObs || forceBranch || forceSeq > 0 || drillTargets != null
                    || schedJobs != null) {
                System.err.println("FATAL: -forcechoice excludes "
                        + "-forkobs/-forcebranch/-forceseq/-drillfile/-forceschedule");
                System.exit(2);
            }
            choiceJobs = readChoiceFile(params.get("forcechoice").get(0));
            drillTargets = new HashMap<>();
            for (Map.Entry<Integer, Map<Integer, ChoicePoint>> e : choiceJobs.entrySet()) {
                int[] ts = new int[e.getValue().size()];
                int i = 0;
                for (int t : e.getValue().keySet()) {
                    ts[i++] = t;
                }
                java.util.Arrays.sort(ts);
                drillTargets.put(e.getKey(), ts);
                for (ChoicePoint p : e.getValue().values()) {
                    choiceMaxArms = Math.max(choiceMaxArms, p.arms.size());
                }
            }
            drillStop = true;
        }

        // Generic counterfactual candidate mode: one natural arm plus one
        // exact (host entity, normalized SA) forced arm per candidate point.
        // The target file is outcome-independent and is joined by the stable
        // source window id, not only by turn.
        Map<Integer, Map<Integer, CandidatePoint>> candidateJobs = null;
        boolean forceCandidate = params.containsKey("forcecandidate");
        if (forceCandidate) {
            if (rolloutK <= 0 || !params.containsKey("labels")) {
                System.err.println("FATAL: -forcecandidate requires -rollout <k> + -labels");
                System.exit(2);
            }
            if (forkObs || forceBranch || forceSeq > 0 || schedJobs != null
                    || choiceJobs != null) {
                System.err.println("FATAL: -forcecandidate excludes other rollout arm modes");
                System.exit(2);
            }
            candidateJobs = readCandidateFile(params.get("forcecandidate").get(0));
            if (drillTargets == null) {
                drillTargets = new HashMap<>();
                for (Map.Entry<Integer, Map<Integer, CandidatePoint>> e : candidateJobs.entrySet()) {
                    int[] ts = e.getValue().values().stream().mapToInt(p -> p.turn).distinct().toArray();
                    Arrays.sort(ts);
                    drillTargets.put(e.getKey(), ts);
                }
            }
            drillStop = true;
        }

        final AnvilBridge bridge;
        if ("local-random".equals(bridgeMode)) {
            bridge = new LocalRandomBridge();
        } else if ("local-oneshot".equals(bridgeMode)) {
            // M10 smoke rig: random-legal through the one-shot path (the
            // realizer AI-fits targets), so forced-mask directive genres run
            // without a decision server. Never a corpus/measurement arm.
            bridge = new forge.ai.anvil.LocalOneShotBridge();
        } else if (bridgeMode.startsWith("grpc:")) {
            String[] hp = bridgeMode.substring(5).split(":");
            forge.anvil.GrpcBridge grpc = new forge.anvil.GrpcBridge(
                    hp[0], Integer.parseInt(hp[1]), "anvil-worker-r" + rangeStart, "");
            if (!grpc.serverBridgedTags().isEmpty()) {
                tags = grpc.serverBridgedTags(); // server-driven coverage
            }
            bridge = grpc;
        } else {
            System.out.println("Unknown bridge mode: " + bridgeMode);
            return;
        }

        GameRules rules = new GameRules(type);
        rules.setAppliedVariants(java.util.EnumSet.of(type));

        // Deck schedule: fixed pair (-d) or index-mapped pairs file (-pairs).
        List<String[]> pairNames = new ArrayList<>();
        int gamesPerPair = params.containsKey("gpp")
                ? Integer.parseInt(params.get("gpp").get(0)) : 5;
        if (fixedPair) {
            pairNames.add(new String[] { params.get("d").get(0), params.get("d").get(1) });
            gamesPerPair = Integer.MAX_VALUE;
        } else {
            try {
                for (String line : Files.readAllLines(Paths.get(params.get("pairs").get(0)),
                        StandardCharsets.UTF_8)) {
                    if (line.isEmpty() || line.startsWith("#")) {
                        continue;
                    }
                    String[] pq = line.split("\t");
                    if (pq.length != 2) {
                        System.err.println("FATAL: bad pairs line (need 2 tab-separated decks): " + line);
                        System.exit(2);
                    }
                    pairNames.add(pq);
                }
            } catch (java.io.IOException e) {
                System.err.println("FATAL: cannot read pairs file: " + e);
                System.exit(2);
            }
            if (pairNames.isEmpty()) {
                System.err.println("FATAL: pairs file has no pairs");
                System.exit(2);
            }
        }
        Map<String, Deck> deckCache = new HashMap<>();

        // AI personalities (ADR-0004: corpus is personality-randomized). Sorted
        // so the seed->profile map is stable across filesystems.
        List<String> profiles = new ArrayList<>(AiProfileUtil.getAvailableProfiles());
        Collections.sort(profiles);
        if (profiles.isEmpty()) {
            // Profiles are corpus provenance — an empty list would silently run
            // every seat on enum defaults while the log claims randomization.
            System.err.println("FATAL: no AI profiles found (AI_PROFILE_DIR missing?)");
            System.exit(2);
        }

        System.out.printf("Anvil worker: games [%d,%d), %s, seedbase=%s, bridge=%s, tags=%s, "
                        + "reask=%s, pairs=%d gpp=%s, profiles=%s%n",
                rangeStart, rangeStart + nGames, type,
                seedBase != null ? seedBase : ("legacy:" + legacyBaseSeed), bridgeMode, tags,
                reask, pairNames.size(), fixedPair ? "-" : String.valueOf(gamesPerPair), profiles);

        Map<String, Integer> tally = new TreeMap<>();
        // Direct ScheduledThreadPoolExecutor (not the Executors wrapper) so
        // remove-on-cancel can be set: cancelled rollout clocks otherwise
        // sit in the queue until their deadline, each lambda pinning its
        // whole Game copy — invisible at seq-probe scale (3 arms x K), an
        // OOM at sched scale (up to 33 arms x 8 rolls per point; the M10
        // serve-smoke JVM died here). Pure memory semantics, no game path.
        java.util.concurrent.ScheduledThreadPoolExecutor watchdogPool =
                new java.util.concurrent.ScheduledThreadPoolExecutor(1, r -> {
                    Thread t = new Thread(r, "anvil-watchdog");
                    t.setDaemon(true);
                    return t;
                });
        watchdogPool.setRemoveOnCancelPolicy(true);
        ScheduledExecutorService watchdogs = watchdogPool;
        long t0 = System.currentTimeMillis();
        PrintWriter results = null;
        PrintWriter labels = null;
        boolean stopped = false;
        try {
            if (params.containsKey("results")) {
                try {
                    results = new PrintWriter(new FileWriter(params.get("results").get(0), true));
                } catch (java.io.IOException e) {
                    // The results file is the harness's progress record — failing to
                    // open it must be fatal and loud, never swallowed (a worker that
                    // plays games nobody can account for is worse than one that dies).
                    System.err.println("FATAL: cannot open results file: " + e);
                    System.exit(2);
                }
            }
            if (params.containsKey("census")) {
                Census.open(params.get("census").get(0));
                if (rolloutK > 0) {
                    // Fork copies share the census stream with their mainline
                    // game index — rollout decisions would inflate per-run
                    // veto/rung telemetry. Labeler runs go without census.
                    System.err.println("WARNING: -census with -rollout pollutes "
                            + "telemetry with fork decisions; prefer omitting it");
                }
            }
            if (params.containsKey("labels")) {
                try {
                    labels = new PrintWriter(new FileWriter(params.get("labels").get(0), true));
                } catch (java.io.IOException e) {
                    System.err.println("FATAL: cannot open labels file: " + e);
                    System.exit(2);
                }
            }
            if (rolloutK > 0 && labels == null) {
                System.err.println("FATAL: -rollout requires -labels <out.jsonl>");
                System.exit(2);
            }
            if (params.containsKey("obs")) {
                try {
                    Obs.open(params.get("obs").get(0));
                    if (forkObs) {
                        String p = params.get("obs").get(0);
                        Obs.openForks(p.endsWith(".zst")
                                ? p.substring(0, p.length() - 4) + "-forks.zst"
                                : p + "-forks.zst");
                    }
                } catch (java.io.IOException e) {
                    // The observation log is the corpus artifact — same rule as
                    // the results file: fail loud, never play unaccounted games.
                    System.err.println("FATAL: cannot open obs file: " + e);
                    System.exit(2);
                }
            }
            // Headless worker: never route uncaught throwables to Forge's GUI
            // bug-report dialog — Main registers ExceptionHandler, and an
            // OutOfMemoryError that escaped the per-game catch opened a modal
            // Swing dialog that wedged workers forever (model-mirror run,
            // 2026-07-12). Log and let the JVM policy decide (the harness runs
            // workers with -XX:+ExitOnOutOfMemoryError; chunk re-issue covers
            // a dead worker).
            Thread.setDefaultUncaughtExceptionHandler((t, e) -> {
                System.err.println("[anvil] uncaught " + e.getClass().getName()
                        + " on thread " + t.getName() + " (headless: no dialog)");
                e.printStackTrace();
            });
            for (int g = 0; g < nGames; g++) {
                if (stopFile != null && stopFile.exists()) {
                    stopped = true;
                    System.out.println("stop-file present; exiting gracefully after "
                            + g + "/" + nGames + " games");
                    break;
                }
                if (bridge.poisoned()) {
                    // Protocol-v0 mismatch clause (issue #9): the stream has
                    // lost request/response correspondence; the game that hit
                    // it was recorded as crashed. No further games on this
                    // stream — exit so the harness recycles the worker (the
                    // relaunch replays the failed game on a fresh stream).
                    System.err.println("[AnvilRun] bridge poisoned; draining after "
                            + g + "/" + nGames + " games — harness will recycle");
                    break;
                }
                int idx = rangeStart + g;
                long seed = seedBase != null
                        ? splitmix64(seedBase + idx * 0x9E3779B97F4A7C15L) : legacyBaseSeed + idx;
                if (drillTargets != null && !drillTargets.containsKey(idx)) {
                    tally.merge("drill_skip", 1, Integer::sum);
                    if (results != null) {
                        results.println("{\"i\":" + idx + ",\"seed\":" + seed
                                + ",\"status\":\"drill_skip\",\"winner\":null"
                                + ",\"turns\":0,\"ms\":0,\"draw_clock\":false"
                                + ",\"decks\":[],\"profiles\":[]}");
                        results.flush();
                    }
                    continue;
                }
                MyRandom.setRandom(new Random(seed));

                String[] pair = pairNames.get((int) ((idx / (long) gamesPerPair) % pairNames.size()));
                List<Deck> decks = new ArrayList<>();
                for (String deckName : pair) {
                    Deck d = deckCache.computeIfAbsent(deckName,
                            n -> SimulateMatch.deckFromCommandLineParameter(n, type));
                    if (d == null) {
                        System.err.println("FATAL: could not load deck: " + deckName);
                        System.exit(2);
                    }
                    decks.add(d);
                }

                // Per-seat personality: pure function of the game seed, so the
                // corpus expert mix reproduces from (seedbase, index) alone.
                String[] seatProfiles = new String[decks.size()];
                List<RegisteredPlayer> pp = new ArrayList<>();
                for (int j = 0; j < decks.size(); j++) {
                    Deck d = decks.get(j);
                    seatProfiles[j] = profiles.get((int) Long.remainderUnsigned(
                            splitmix64(seed + (j + 1) * 0x9E3779B97F4A7C15L), profiles.size()));
                    RegisteredPlayer rp = type.equals(GameType.Commander)
                            ? RegisteredPlayer.forCommander(d) : new RegisteredPlayer(d);
                    // Mixed-seat arms (M1 D8): seats outside -bridgeseats get an
                    // empty tag set — every decision falls through to the
                    // inherited heuristic (provenance rule intact), and the name
                    // prefix makes games.jsonl winners parseable per arm.
                    boolean seatBridged = bridgeSeats == null || bridgeSeats.contains(j);
                    AnvilLobbyPlayer lp = new AnvilLobbyPlayer(
                            (seatBridged ? "Anvil(" : "Heur(") + (j + 1) + ")-" + d.getName(),
                            bridge, seatBridged ? tags : java.util.Collections.emptySet());
                    lp.setAiProfile(seatProfiles[j]);
                    rp.setPlayer(lp);
                    pp.add(rp);
                }

                Match mc = new Match(rules, pp, "Anvil");
                Game game = mc.createGame();
                Census.startGame(idx, seed);
                Obs.startGame(idx, seed, game, type.toString());
                long gameT0 = System.currentTimeMillis();
                bridge.gameStart("g" + idx, seed);
                int[] drillTurns = drillTargets != null ? drillTargets.get(idx) : null;
                RolloutMonitor rolloutMonitor = null;
                if (rolloutK > 0) {
                    rolloutMonitor = new RolloutMonitor(game, idx, seed,
                            rolloutK, rolloutPoints, rolloutReshuffle, bridge,
                            type.toString(), labels, watchdogs, drillTurns, drillStop,
                            forkObs, forceBranch, forceSeq, seqNatOnly,
                            schedJobs != null ? schedJobs.get(idx) : null,
                            choiceJobs != null ? choiceJobs.get(idx) : null,
                            candidateJobs != null ? candidateJobs.get(idx) : null);
                    game.subscribeToEvents(rolloutMonitor);
                }
                // Rollout forks run inside the game's wall — budget the clocks
                // for them (45 s/rollout is far above the 4.4 s median but
                // below the per-rollout timeout, so a pathological point can't
                // eat the whole game budget). Forced-branch mode runs 2xK;
                // sequence mode 3xK (1xK single-natural-arm).
                int fpBudget = drillTurns != null ? drillTurns.length : rolloutPoints;
                int candidateArms = 0;
                if (candidateJobs != null && candidateJobs.get(idx) != null) {
                    for (CandidatePoint cp : candidateJobs.get(idx).values()) {
                        candidateArms = Math.max(candidateArms, cp.arms.size());
                    }
                }
                int perPoint = (candidateJobs != null ? (1 + candidateArms)
                        : schedJobs != null ? (1 + schedMaxArms)
                        : choiceJobs != null ? (1 + choiceMaxArms)
                        : forceSeq > 0 ? (seqNatOnly ? 1 : 3) : (forceBranch ? 2 : 1))
                        * rolloutK;
                int extraS = rolloutK > 0 ? fpBudget * perPoint * 45 : 0;
                final boolean[] drawClockHit = {false};
                ScheduledFuture<?> drawClock = watchdogs.schedule(() -> {
                    drawClockHit[0] = true;
                    game.setGameOver(GameEndReason.Draw);
                }, DRAW_CLOCK_S + extraS, TimeUnit.SECONDS);
                String status;
                try {
                    TimeLimitedCodeBlock.runWithTimeout(() -> mc.startGame(game),
                            GAME_HARD_CAP_S + extraS, TimeUnit.SECONDS);
                    status = game.getOutcome() == null ? "no_outcome"
                            : game.getOutcome().isDraw() ? "draw" : "won";
                } catch (Throwable e) {
                    // Throwable, not Exception|StackOverflowError: any Error
                    // class escaping here reaches the uncaught handler and
                    // used to wedge the worker in a GUI dialog. OOM is the
                    // exception — -XX:+ExitOnOutOfMemoryError kills the JVM
                    // before this catch matters.
                    game.setGameOver(GameEndReason.Draw);
                    status = "crash_or_hang:" + e.getClass().getSimpleName();
                    if (Boolean.getBoolean("anvil.crash.trace")) {
                        e.printStackTrace();
                    }
                } finally {
                    drawClock.cancel(false);
                }
                if (rolloutMonitor != null) {
                    rolloutMonitor.finishCandidateSkips();
                }
                long wallMs = System.currentTimeMillis() - gameT0;
                String winner = game.getOutcome() != null && !game.getOutcome().isDraw()
                        ? game.getOutcome().getWinningLobbyPlayer().getName() : null;
                int turns = game.getOutcome() != null ? game.getOutcome().getLastTurnNumber() : -1;
                Census.endGame(winner, turns);
                int winnerIdx = -1;
                if (winner != null) {
                    // registered players, not getPlayers(): the live list drops
                    // eliminated players, so at game end the winner is ~always
                    // index 0 (same reindex class as the M1 D1 header fix).
                    // Obs records index seats by getRegisteredPlayers throughout.
                    for (int wi = 0; wi < game.getRegisteredPlayers().size(); wi++) {
                        if (game.getRegisteredPlayers().get(wi).getName().equals(winner)) {
                            winnerIdx = wi;
                            break;
                        }
                    }
                }
                Obs.endGame(status, winnerIdx, turns, wallMs, drawClockHit[0]);
                bridge.gameEnd("g" + idx, winner, turns, wallMs);
                // Between mainline games, same rationale as the sched-mode
                // per-completion clear above: bridged seats never run the
                // AI-window clear, so long-lived workers accumulate one game
                // graph per game (historically masked by chunk recycling).
                forge.ai.AiCache.clear();
                tally.merge(status, 1, Integer::sum);
                if (results != null) {
                    results.println("{\"i\":" + idx + ",\"seed\":" + seed
                            + ",\"status\":\"" + status + "\""
                            + ",\"winner\":" + (winner == null ? "null" : "\"" + winner.replace("\"", "'") + "\"")
                            + ",\"turns\":" + turns + ",\"ms\":" + wallMs
                            + ",\"draw_clock\":" + drawClockHit[0]
                            + ",\"decks\":[\"" + jstr(pair[0]) + "\",\"" + jstr(pair[1]) + "\"]"
                            + ",\"profiles\":[\"" + jstr(seatProfiles[0]) + "\",\"" + jstr(seatProfiles[1]) + "\"]}");
                    results.flush();
                }
                System.out.printf("game %d seed=%d -> %s (%d turns)%n", idx, seed, status, turns);
            }
        } catch (Throwable e) {
            e.printStackTrace();
        } finally {
            if (results != null) {
                results.close();
            }
            if (labels != null) {
                labels.close();
            }
            Census.close();
            Obs.close();
            bridge.close();
            watchdogs.shutdownNow();
        }

        long wallS = (System.currentTimeMillis() - t0) / 1000;
        System.out.println("=== anvil tally ===" + (stopped ? " (stopped)" : ""));
        tally.forEach((k, v) -> System.out.printf("%-24s %d%n", k, v));
        System.out.printf("wall=%ds%n", wallS);
        System.out.flush();
    }

    // JSON string escape. Control chars matter: modal spell text carries
    // literal newlines ("Choose one —\n• ..."), which split labels rows
    // into unparseable fragments (caught live in the M8 D1 probe; the
    // M7 act_first path had the same latent hole).
    private static String jstr(String s) {
        StringBuilder b = new StringBuilder(s.length() + 8);
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '\\' || c == '"') {
                b.append('\\').append(c);
            } else if (c == '\n') {
                b.append("\\n");
            } else if (c == '\r') {
                b.append("\\r");
            } else if (c == '\t') {
                b.append("\\t");
            } else if (c < 0x20) {
                b.append(String.format("\\u%04x", (int) c));
            } else {
                b.append(c);
            }
        }
        return b.toString();
    }

    private static Map<String, List<String>> parseParams(String[] args) {
        Map<String, List<String>> params = new HashMap<>();
        List<String> current = null;
        for (int i = 1; i < args.length; i++) {
            String a = args[i];
            if (a.startsWith("-") && a.length() > 1 && !Character.isDigit(a.charAt(1))) {
                current = new ArrayList<>();
                params.put(a.substring(1), current);
            } else if (current != null) {
                current.add(a);
            } else {
                return null;
            }
        }
        return params;
    }

    // ------------------------------------------------------------------
    // M10 sched mode data (m10-ceiling-spec): one SchedPoint per sampled
    // (game, turn); each directed arm is an ordered schedule + a payment
    // mode. The file is a TSV contract with the Python planner.
    // ------------------------------------------------------------------

    static final class SchedArm {
        final int id;
        final boolean joint;
        final List<String> labels;

        SchedArm(int id, boolean joint, List<String> labels) {
            this.id = id;
            this.joint = joint;
            this.labels = labels;
        }
    }

    static final class SchedPoint {
        final int turn;
        final int horizon; // 0 = run completions to natural game end
        final int seat;    // registered-player index expected at the fork
        final List<SchedArm> arms = new ArrayList<>();

        SchedPoint(int turn, int horizon, int seat) {
            this.turn = turn;
            this.horizon = horizon;
            this.seat = seat;
        }
    }

    private static Map<Integer, Map<Integer, SchedPoint>> readSchedFile(String path) {
        Map<Integer, Map<Integer, SchedPoint>> out = new HashMap<>();
        try {
            for (String line : Files.readAllLines(Paths.get(path), StandardCharsets.UTF_8)) {
                if (line.isEmpty() || line.startsWith("#")) {
                    continue;
                }
                String[] f = line.split("\t", -1);
                if (f.length < 6) {
                    throw new IllegalArgumentException("bad sched line (need >= 6 tab fields): " + line);
                }
                int idx = Integer.parseInt(f[0]);
                int turn = Integer.parseInt(f[1]);
                int horizon = Integer.parseInt(f[2]);
                int seat = Integer.parseInt(f[3]);
                int armId = Integer.parseInt(f[4]);
                boolean joint;
                if ("joint".equals(f[5])) {
                    joint = true;
                } else if ("auto".equals(f[5])) {
                    joint = false;
                } else {
                    throw new IllegalArgumentException("bad paymode (joint|auto): " + line);
                }
                if (armId < 1) {
                    throw new IllegalArgumentException("armId must be >= 1 (0 = natural): " + line);
                }
                List<String> armLabels = new ArrayList<>();
                for (int i = 6; i < f.length; i++) {
                    if (!f[i].isEmpty()) {
                        armLabels.add(f[i]);
                    }
                }
                SchedPoint p = out.computeIfAbsent(idx, k -> new TreeMap<>())
                        .computeIfAbsent(turn, t -> new SchedPoint(t, horizon, seat));
                if (p.horizon != horizon || p.seat != seat) {
                    throw new IllegalArgumentException(
                            "horizon/seat mismatch within g" + idx + " t" + turn);
                }
                for (SchedArm a : p.arms) {
                    if (a.id == armId) {
                        throw new IllegalArgumentException(
                                "duplicate armId " + armId + " at g" + idx + " t" + turn);
                    }
                }
                p.arms.add(new SchedArm(armId, joint, armLabels));
            }
        } catch (Exception e) {
            System.err.println("FATAL: cannot read sched file " + path + ": " + e);
            System.exit(2);
        }
        return out;
    }

    // ------------------------------------------------------------------
    // M11 choice mode data (m11-routing-probes-spec.md): one ChoicePoint
    // per sampled (game, turn); each arm forces one candidate index
    // (tutor) or pay/decline (prevent) via ChoiceDirective. TSV contract
    // with the Python planner.
    // ------------------------------------------------------------------

    static final class ChoiceArm {
        final int id;
        final int kind;    // ChoiceDirective.KIND_*
        final int action;  // tutor: candidate index; prevent: ACT_PAY/ACT_DECLINE

        ChoiceArm(int id, int kind, int action) {
            this.id = id;
            this.kind = kind;
            this.action = action;
        }
    }

    static final class ChoicePoint {
        final int turn;
        final int horizon; // 0 = run completions to natural game end
        final int seat;    // registered-player index expected at the fork
        final List<ChoiceArm> arms = new ArrayList<>();

        ChoicePoint(int turn, int horizon, int seat) {
            this.turn = turn;
            this.horizon = horizon;
            this.seat = seat;
        }
    }

    private static Map<Integer, Map<Integer, ChoicePoint>> readChoiceFile(String path) {
        Map<Integer, Map<Integer, ChoicePoint>> out = new HashMap<>();
        try {
            for (String line : Files.readAllLines(Paths.get(path), StandardCharsets.UTF_8)) {
                if (line.isEmpty() || line.startsWith("#")) {
                    continue;
                }
                String[] f = line.split("\t", -1);
                if (f.length != 7) {
                    throw new IllegalArgumentException("bad choice line (need 7 tab fields): " + line);
                }
                int idx = Integer.parseInt(f[0]);
                int turn = Integer.parseInt(f[1]);
                int seat = Integer.parseInt(f[2]);
                int horizon = Integer.parseInt(f[3]);
                int armId = Integer.parseInt(f[4]);
                int kind;
                if ("tutor".equals(f[5])) {
                    kind = ChoiceDirective.KIND_TUTOR;
                } else if ("prevent".equals(f[5])) {
                    kind = ChoiceDirective.KIND_PREVENT;
                } else {
                    throw new IllegalArgumentException("bad kind (tutor|prevent): " + line);
                }
                int action;
                if (kind == ChoiceDirective.KIND_PREVENT) {
                    if ("pay".equals(f[6])) {
                        action = ChoiceDirective.ACT_PAY;
                    } else if ("decline".equals(f[6])) {
                        action = ChoiceDirective.ACT_DECLINE;
                    } else {
                        throw new IllegalArgumentException("bad prevent action (pay|decline): " + line);
                    }
                } else {
                    action = Integer.parseInt(f[6]);
                    if (action < 0) {
                        throw new IllegalArgumentException("tutor action must be >= 0: " + line);
                    }
                }
                if (armId < 1) {
                    throw new IllegalArgumentException("armId must be >= 1 (0 = natural): " + line);
                }
                ChoicePoint p = out.computeIfAbsent(idx, k -> new TreeMap<>())
                        .computeIfAbsent(turn, t -> new ChoicePoint(t, horizon, seat));
                if (p.horizon != horizon || p.seat != seat) {
                    throw new IllegalArgumentException(
                            "horizon/seat mismatch within g" + idx + " t" + turn);
                }
                for (ChoiceArm a : p.arms) {
                    if (a.id == armId) {
                        throw new IllegalArgumentException(
                                "duplicate armId " + armId + " at g" + idx + " t" + turn);
                    }
                }
                p.arms.add(new ChoiceArm(armId, kind, action));
            }
        } catch (Exception e) {
            System.err.println("FATAL: cannot read choice file " + path + ": " + e);
            System.exit(2);
        }
        return out;
    }

    // ------------------------------------------------------------------
    // Generic priority candidate drills.  Contract:
    // gameIdx windowId ordinal turn phase seat entityId normalizedSA
    // ------------------------------------------------------------------

    static final class CandidateArm {
        final int ordinal;
        final int entity;
        final String sa;

        CandidateArm(int ordinal, int entity, String sa) {
            this.ordinal = ordinal;
            this.entity = entity;
            this.sa = sa;
        }
    }

    static final class CandidatePoint {
        final String source;
        final int window;
        final int turn;
        final String phase;
        final int seat;
        final List<CandidateArm> arms = new ArrayList<>();

        CandidatePoint(String source, int window, int turn, String phase, int seat) {
            this.source = source;
            this.window = window;
            this.turn = turn;
            this.phase = phase;
            this.seat = seat;
        }
    }

    private static Map<Integer, Map<Integer, CandidatePoint>> readCandidateFile(String path) {
        Map<Integer, Map<Integer, CandidatePoint>> out = new HashMap<>();
        final String[] source = { "" };
        try {
            for (String line : Files.readAllLines(Paths.get(path), StandardCharsets.UTF_8)) {
                if (line.startsWith("# source-store=")) {
                    source[0] = line.substring("# source-store=".length()).trim();
                    continue;
                }
                if (line.isEmpty() || line.startsWith("#") || line.startsWith("gameIdx\t")) {
                    continue;
                }
                String[] f = line.split("\t", -1);
                if (f.length != 8) {
                    throw new IllegalArgumentException("bad candidate line (need 8 tab fields): " + line);
                }
                int game = Integer.parseInt(f[0]);
                int window = Integer.parseInt(f[1]);
                int ordinal = Integer.parseInt(f[2]);
                int turn = Integer.parseInt(f[3]);
                String phase = f[4];
                int seat = Integer.parseInt(f[5]);
                int entity = Integer.parseInt(f[6]);
                String sa = f[7];
                if (ordinal < 0 || entity < 0 || sa.isEmpty()) {
                    throw new IllegalArgumentException("invalid candidate descriptor: " + line);
                }
                CandidatePoint p = out.computeIfAbsent(game, k -> new TreeMap<>())
                        .computeIfAbsent(window, w -> new CandidatePoint(source[0], w, turn, phase, seat));
                if (p.turn != turn || p.seat != seat || !p.phase.equals(phase)) {
                    throw new IllegalArgumentException(
                            "turn/phase/seat mismatch within g" + game + " window " + window);
                }
                for (CandidateArm a : p.arms) {
                    if (a.ordinal == ordinal) {
                        throw new IllegalArgumentException(
                                "duplicate candidate ordinal " + ordinal + " at g" + game
                                        + " window " + window);
                    }
                }
                p.arms.add(new CandidateArm(ordinal, entity, sa));
            }
        } catch (Exception e) {
            System.err.println("FATAL: cannot read candidate file " + path + ": " + e);
            System.exit(2);
        }
        return out;
    }

    /** Bounded-horizon stop for sched completions: end-of-turn stopTurn =
     *  the first TurnBegan with a higher number; forced end is a Draw so
     *  the row stays obviously non-decisive (the CensusRun/certify
     *  convention). */
    static final class HorizonStop {
        final Game game;
        final int stopTurn;
        volatile boolean stopped = false;

        HorizonStop(Game game, int stopTurn) {
            this.game = game;
            this.stopTurn = stopTurn;
        }

        @Subscribe
        public void onTurnBegan(GameEventTurnBegan ev) {
            if (!game.isGameOver() && ev.turnNumber() > stopTurn) {
                stopped = true;
                game.setGameOver(GameEndReason.Draw);
            }
        }
    }

    // ------------------------------------------------------------------
    // Rollout-label mode (M2 D4): fork the live game at sampled quiescent
    // MAIN1 priority windows, complete K copies to game end under the
    // bridge, one labels-JSONL record per fork point. Fork discipline is
    // ForkFidelityCheck's (quiescence drain, active-player priority only,
    // RNG snapshot/restore around the block, wire-only obs sessions).
    // ------------------------------------------------------------------

    private static final int ROLLOUT_TIMEOUT_S = 120;
    // Fork-store synthetic game ids live in their own namespace above any
    // reachable mainline index: base + ns*STRIDE + (gameIdx*100 + fp)*100 + r.
    // Without the base, a drilled source game with gameIdx=0 encodes forks
    // 0..k-1, colliding with mainline store indices (the run13 iteration-0
    // crash). Without ns (-forkns, assigned per source store by the planner
    // and recorded in the drill manifest), two generation runs drilling the
    // same source g from DIFFERENT stores collide with each other (the run17
    // iteration-2 MultiStore crash). The id stays a pure join key — nothing
    // decodes it; provenance travels in the fork header's pg/fp/r + manifest.
    // Store-format change — era-scoped: stores written before -forkns existed
    // use ns=0 ids; never mix eras in one MultiStore join.
    private static final long FORK_G_BASE = 1_000_000_000_000L;
    private static final long FORK_NS_STRIDE = 1_000_000_000L;
    private static long forkGBase = FORK_G_BASE;

    private static final class RolloutMonitor {
        final Game game;
        final int gameIdx;
        final long seed;
        final int k;
        final boolean reshuffle;
        final AnvilBridge bridge;
        final String fmt;
        final PrintWriter labels;
        final ScheduledExecutorService watchdogs;
        final boolean stopAfter;
        final boolean forkObs;
        final boolean forceBranch;
        final int forceSeq;
        final boolean seqNatOnly;
        /** M10 sched mode: this game's fork points by turn; null = not sched. */
        final Map<Integer, SchedPoint> sched;
        /** M11 choice mode: this game's fork points by turn; null = not choice. */
        final Map<Integer, ChoicePoint> choice;
        /** Generic candidate points keyed by stable priority window id. */
        final Map<Integer, CandidatePoint> candidate;
        final Set<Integer> candidateDone = new HashSet<>();
        /** One printed stack per distinct throwable class per lane run. */
        private final Set<String> crashClassesPrinted = new HashSet<>();
        final java.util.TreeSet<Integer> targets = new java.util.TreeSet<>();
        int fp = 0;

        RolloutMonitor(Game game, int gameIdx, long seed, int k, int points,
                boolean reshuffle, AnvilBridge bridge, String fmt,
                PrintWriter labels, ScheduledExecutorService watchdogs,
                int[] drillTurns, boolean stopAfter, boolean forkObs,
                boolean forceBranch, int forceSeq, boolean seqNatOnly,
                Map<Integer, SchedPoint> sched, Map<Integer, ChoicePoint> choice,
                Map<Integer, CandidatePoint> candidate) {
            this.game = game;
            this.gameIdx = gameIdx;
            this.seed = seed;
            this.k = k;
            this.reshuffle = reshuffle;
            this.bridge = bridge;
            this.fmt = fmt;
            this.labels = labels;
            this.watchdogs = watchdogs;
            this.stopAfter = stopAfter;
            this.forkObs = forkObs;
            this.forceBranch = forceBranch;
            this.forceSeq = forceSeq;
            this.seqNatOnly = seqNatOnly;
            this.sched = sched;
            this.choice = choice;
            this.candidate = candidate;
            if (drillTurns != null) {
                // Drill mode: explicit fork turns from the manifest.
                for (int t : drillTurns) {
                    targets.add(t);
                }
            } else {
                // Target turns from a meta-RNG (pure function of the game seed —
                // never perturbs game randomness); distinct turns in [2, 16].
                Random meta = new Random(splitmix64(seed ^ 0xD4D4D4D4D4D4D4D4L));
                while (targets.size() < Math.min(points, 15)) {
                    targets.add(2 + meta.nextInt(15));
                }
            }
        }

        @Subscribe
        public void onPriority(GameEventPlayerPriority ev) {
            if (game.isGameOver()) {
                // A hard-capped game is setGameOver(Draw) by the runner but
                // its ABANDONED thread may still be playing; never fork more
                // completions from it (d6-run10 iter-9 cascade).
                return;
            }
            if (targets.isEmpty()) {
                return;
            }
            if (candidate == null && ev.phase() != PhaseType.MAIN1) {
                return;
            }
            PhaseHandler ph = game.getPhaseHandler();
            if (ph.getTurn() < targets.first() || !game.getStack().isEmpty()) {
                return;
            }
            // Legacy drill modes remain active-seat-only. Candidate mode is
            // explicitly all-phase/all-priority-seat, with stack-empty and
            // state-effect drain guards retained below.
            if (candidate == null && ph.getPriorityPlayer() != ph.getPlayerTurn()) {
                return;
            }
            java.util.Set<Card> affected = new HashSet<>();
            do {
                game.getAction().checkStateEffects(false, affected);
                if (game.isGameOver()) {
                    return;
                }
            } while (game.getStack().addAllTriggeredAbilitiesToStack());
            if (!game.getStack().isEmpty()) {
                return;
            }
            int turn = ph.getTurn();
            if (candidate != null) {
                long window = Obs.nextDecisionSequence(game);
                CandidatePoint point = candidatePointFor(
                        turn, ev.phase(), ph.getPriorityPlayer(), window);
                if (point == null) {
                    return;
                }
                candidateDone.add(point.window);
                doCandidateRollouts(turn, point);
            } else {
                int targetTurn = targets.first();
                while (!targets.isEmpty() && targets.first() <= turn) {
                    targets.pollFirst();
                }
                doRollouts(turn, targetTurn);
            }
            if (stopAfter && (candidate != null ? candidateDone.size() >= candidate.size() : targets.isEmpty())) {
                // Drill mode: the completions are the product; don't replay
                // the rest of the mainline. Draw end keeps the results row
                // obviously non-decisive.
                game.setGameOver(GameEndReason.Draw);
            }
        }

        private CandidatePoint candidatePointFor(
                int turn, PhaseType phase, Player priority, long window) {
            if (candidate == null) {
                return null;
            }
            int seat = -1;
            for (int i = 0; i < game.getRegisteredPlayers().size(); i++) {
                if (game.getRegisteredPlayers().get(i).getName().equals(priority.getName())) {
                    seat = i;
                    break;
                }
            }
            for (CandidatePoint p : candidate.values()) {
                if (candidateDone.contains(p.window) || p.turn != turn
                        || (window >= 0 && p.window != window)) {
                    continue;
                }
                if (p.seat != seat) {
                    continue;
                }
                if (p.phase != null && !p.phase.isEmpty() && !p.phase.equals(String.valueOf(phase))) {
                    continue;
                }
                return p;
            }
            return null;
        }

        private void doRollouts(int turn, int targetTurn) {
            if (sched != null) {
                doSchedRollouts(turn, targetTurn);
                return;
            }
            if (choice != null) {
                doChoiceRollouts(turn, targetTurn);
                return;
            }
            if (candidate != null) {
                // candidate mode is dispatched directly from onPriority with
                // its stable window descriptor; this guard prevents an old
                // turn-only path from accidentally consuming it.
                return;
            }
            if (forceSeq > 0) {
                doSeqRollouts(turn, targetTurn);
                return;
            }
            if (forceBranch) {
                doForcedRollouts(turn, targetTurn);
                return;
            }
            int myFp = fp++;
            // The mark keys this fork point to the NEXT mainline priority
            // window in the obs stream (the label's training window).
            Obs.mark(game, "fork", "fp", myFp, "kr", k);
            byte[] rngState = snapshotRng();
            int[] wins = new int[game.getRegisteredPlayers().size()];
            int draws = 0;
            int crashes = 0;
            long block0 = System.nanoTime();
            long copyMsTotal = 0;
            for (int r = 0; r < k; r++) {
                Game copy;
                long c0 = System.nanoTime();
                try {
                    copy = new GameCopier(game).makeCopy();
                } catch (Throwable t) {
                    crashes++;
                    if (Boolean.getBoolean("anvil.crash.trace")) {
                        System.err.println("[rollout] copy crash g" + gameIdx
                                + " fp" + myFp + " r" + r + ":");
                        t.printStackTrace();
                    }
                    MyRandom.setRandom(restoreRng(rngState));
                    continue;
                }
                copyMsTotal += (System.nanoTime() - c0) / 1_000_000;
                long rollSeed = splitmix64(
                        seed ^ (myFp * 0x9E3779B97F4A7C15L) ^ (r * 0xBF58476D1CE4E5B9L));
                Random rollRng = new Random(rollSeed);
                if (reshuffle) {
                    // Determinization: silently re-randomize both libraries
                    // (Zone.setCards — no shuffle events/triggers) so the K
                    // rollouts average over unseen order instead of replaying
                    // the one concrete order nobody has observed. Known-order
                    // states (scry tops, tucked bottoms) are knowingly
                    // approximated — label quality, not ledger unbiasedness.
                    for (Player p : copy.getPlayers()) {
                        List<Card> lib = new ArrayList<>();
                        for (Card c : p.getZone(ZoneType.Library)) {
                            lib.add(c);
                        }
                        Collections.shuffle(lib, rollRng);
                        p.getZone(ZoneType.Library).setCards(lib);
                    }
                }
                copy.getPhaseHandler().devResumeAtPriority();
                copy.copyLastState();
                String wid = "g" + gameIdx + ".f" + myFp + "r" + r;
                if (forkObs) {
                    // Per-completion identity + seed: synthetic unique game id
                    // for the store/mu joins; the completion's own seed so
                    // server-side sampled noise decorrelates across the K.
                    long off = ((long) gameIdx * 100 + myFp) * 100 + r;
                    if (off >= FORK_NS_STRIDE) {
                        // would bleed into the next -forkns namespace slice
                        throw new IllegalStateException(
                                "fork id offset " + off + " >= FORK_NS_STRIDE (gameIdx " + gameIdx + ")");
                    }
                    Obs.startForkGame(copy, wid, forkGBase + off,
                            rollSeed, fmt, game, gameIdx, myFp, r, targetTurn);
                    bridge.gameStart(wid, rollSeed, Obs.lastHeaderForBridge(copy));
                } else {
                    Obs.startWireGame(copy, wid, seed, fmt, game);
                    bridge.gameStart(wid, seed, Obs.lastHeaderForBridge(copy));
                }
                MyRandom.setRandom(rollRng);
                long compT0 = System.currentTimeMillis();
                int wi = -1;
                boolean crashed = false;
                ScheduledFuture<?> clock = watchdogs.schedule(
                        () -> copy.setGameOver(GameEndReason.Draw),
                        ROLLOUT_TIMEOUT_S, TimeUnit.SECONDS);
                try {
                    copy.getPhaseHandler().mainGameLoop();
                    if (copy.getOutcome() == null || copy.getOutcome().isDraw()) {
                        draws++;
                    } else {
                        String w = copy.getOutcome().getWinningLobbyPlayer().getName();
                        for (int j = 0; j < copy.getRegisteredPlayers().size(); j++) {
                            if (copy.getRegisteredPlayers().get(j).getName().equals(w)) {
                                wi = j;
                                break;
                            }
                        }
                        if (wi >= 0) {
                            wins[wi]++;
                        } else {
                            draws++;
                        }
                    }
                } catch (Throwable t) {
                    crashes++;
                    crashed = true;
                    if (Boolean.getBoolean("anvil.crash.trace")) {
                        System.err.println("[rollout] completion crash g" + gameIdx
                                + " fp" + myFp + " r" + r + ":");
                        t.printStackTrace();
                    }
                } finally {
                    clock.cancel(false);
                    if (!copy.isGameOver()) {
                        copy.setGameOver(GameEndReason.Draw);
                    }
                    MyRandom.setRandom(restoreRng(rngState));
                    if (forkObs) {
                        int turns = -1;
                        try {
                            turns = copy.getPhaseHandler().getTurn();
                        } catch (Exception ignored) {
                        }
                        Obs.endForkGame(copy,
                                crashed ? "crash" : (wi >= 0 ? "won" : "draw"),
                                wi, turns, System.currentTimeMillis() - compT0);
                    }
                    Obs.endWireGame(copy);
                }
            }
            // Re-announce the mainline: the fork wire sessions re-bound the
            // server's per-stream header.
            bridge.gameStart("g" + gameIdx, seed, Obs.lastHeaderForBridge(game));
            if (labels != null) {
                StringBuilder sb = new StringBuilder(192);
                sb.append("{\"i\":").append(gameIdx)
                        .append(",\"seed\":").append(seed)
                        .append(",\"fp\":").append(myFp)
                        .append(",\"t\":").append(turn)
                        .append(",\"tt\":").append(targetTurn)
                        .append(",\"k\":").append(k)
                        .append(",\"w\":[");
                for (int j = 0; j < wins.length; j++) {
                    sb.append(j > 0 ? "," : "").append(wins[j]);
                }
                sb.append("],\"draw\":").append(draws)
                        .append(",\"crash\":").append(crashes)
                        .append(",\"copy_ms\":").append(copyMsTotal)
                        .append(",\"ms\":").append((System.nanoTime() - block0) / 1_000_000)
                        .append('}');
                synchronized (labels) {
                    labels.println(sb);
                    labels.flush();
                }
            }
        }

        /** M7 forced-branch paired rollouts (m7-plan D2): two branches x k
         *  completions per fork point, branch pairs (fp, r) sharing rollSeed
         *  (identical determinization + downstream RNG + announced server
         *  noise seed — common random numbers; divergence comes only from the
         *  forced first decision). One labels row per fork point with both
         *  branches; a pair with a crashed/skipped member drops whole. */
        private void doForcedRollouts(int turn, int targetTurn) {
            int myFp = fp++;
            PhaseHandler ph = game.getPhaseHandler();
            Player prio = ph.getPriorityPlayer();
            boolean bridgeSeat = prio.getController() instanceof PlayerControllerAnvil
                    && ((PlayerControllerAnvil) prio.getController()).bridgesPriority();
            if (!bridgeSeat) {
                // Guard (pin 5ii): forcing a heuristic seat measures nothing.
                // Loud row keeps drill accounting exact; no Obs.mark (nothing
                // will join here).
                if (labels != null) {
                    synchronized (labels) {
                        labels.println("{\"i\":" + gameIdx + ",\"seed\":" + seed
                                + ",\"fp\":" + myFp + ",\"t\":" + turn
                                + ",\"tt\":" + targetTurn
                                + ",\"forced\":true,\"seat_skip\":true}");
                        labels.flush();
                    }
                }
                return;
            }
            Obs.mark(game, "fork", "fp", myFp, "kr", k);
            byte[] rngState = snapshotRng();
            int np = game.getRegisteredPlayers().size();
            // outcome[b][r]: -3 forced-skip, -2 crash, -1 draw, >=0 winner idx
            int[][] outcome = new int[2][k];
            java.util.Map<String, Integer> skips = new java.util.TreeMap<>();
            String seatName = prio.getName();
            long block0 = System.nanoTime();
            long copyMsTotal = 0;
            for (int r = 0; r < k; r++) {
                long rollSeed = splitmix64(
                        seed ^ (myFp * 0x9E3779B97F4A7C15L) ^ (r * 0xBF58476D1CE4E5B9L));
                for (int b = 0; b < 2; b++) {
                    Game copy;
                    long c0 = System.nanoTime();
                    try {
                        copy = new GameCopier(game).makeCopy();
                    } catch (Throwable t) {
                        outcome[b][r] = -2;
                        if (Boolean.getBoolean("anvil.crash.trace")) {
                            System.err.println("[forced] copy crash g" + gameIdx
                                    + " fp" + myFp + " r" + r + (b == 0 ? "a" : "h") + ":");
                            t.printStackTrace();
                        }
                        MyRandom.setRandom(restoreRng(rngState));
                        continue;
                    }
                    copyMsTotal += (System.nanoTime() - c0) / 1_000_000;
                    // Fresh Random per branch from the SHARED seed: identical
                    // shuffle consumption -> identical post-shuffle RNG state.
                    Random rollRng = new Random(rollSeed);
                    if (reshuffle) {
                        for (Player p : copy.getPlayers()) {
                            List<Card> lib = new ArrayList<>();
                            for (Card c : p.getZone(ZoneType.Library)) {
                                lib.add(c);
                            }
                            Collections.shuffle(lib, rollRng);
                            p.getZone(ZoneType.Library).setCards(lib);
                        }
                    }
                    copy.getPhaseHandler().devResumeAtPriority();
                    copy.copyLastState();
                    String wid = "g" + gameIdx + ".f" + myFp + "r" + r + (b == 0 ? "a" : "h");
                    // Wire session announces the COMPLETION seed (not the
                    // parent's): instrument-mode sampled serving decorrelates
                    // across r while branch pairs stay correlated (pin 4).
                    Obs.startWireGame(copy, wid, rollSeed, fmt, game);
                    bridge.gameStart(wid, rollSeed, Obs.lastHeaderForBridge(copy));
                    PlayerControllerAnvil.armForcedFirst(copy, seatName,
                            b == 0 ? PlayerControllerAnvil.ForcedFirst.ACT
                                   : PlayerControllerAnvil.ForcedFirst.HOLD);
                    MyRandom.setRandom(rollRng);
                    int wi = -1;
                    boolean crashed = false;
                    ScheduledFuture<?> clock = watchdogs.schedule(
                            () -> copy.setGameOver(GameEndReason.Draw),
                            ROLLOUT_TIMEOUT_S, TimeUnit.SECONDS);
                    try {
                        copy.getPhaseHandler().mainGameLoop();
                        if (copy.getOutcome() != null && !copy.getOutcome().isDraw()) {
                            String w = copy.getOutcome().getWinningLobbyPlayer().getName();
                            for (int j = 0; j < copy.getRegisteredPlayers().size(); j++) {
                                if (copy.getRegisteredPlayers().get(j).getName().equals(w)) {
                                    wi = j;
                                    break;
                                }
                            }
                        }
                    } catch (Throwable t) {
                        crashed = true;
                        if (Boolean.getBoolean("anvil.crash.trace")) {
                            System.err.println("[forced] completion crash g" + gameIdx
                                    + " fp" + myFp + " r" + r + (b == 0 ? "a" : "h") + ":");
                            t.printStackTrace();
                        }
                    } finally {
                        clock.cancel(false);
                        if (!copy.isGameOver()) {
                            copy.setGameOver(GameEndReason.Draw);
                        }
                        MyRandom.setRandom(restoreRng(rngState));
                        PlayerControllerAnvil.ForcedResult fres =
                                PlayerControllerAnvil.forcedResult(copy);
                        PlayerControllerAnvil.clearForced(copy);
                        if (crashed) {
                            outcome[b][r] = -2;
                        } else if (b == 0
                                && fres != PlayerControllerAnvil.ForcedResult.CAST) {
                            // Act branch didn't cast: skip, reason counted.
                            outcome[b][r] = -3;
                            skips.merge(fres.name(), 1, Integer::sum);
                        } else if (b == 1
                                && fres != PlayerControllerAnvil.ForcedResult.HELD) {
                            // Hold anomaly (seat never asked, e.g. lethal on
                            // resume): skip, distinct key.
                            outcome[b][r] = -3;
                            skips.merge("HOLD_" + fres.name(), 1, Integer::sum);
                        } else {
                            outcome[b][r] = wi; // -1 draw or winner idx
                        }
                        Obs.endWireGame(copy);
                    }
                }
            }
            // Re-announce the mainline: the fork wire sessions re-bound the
            // server's per-stream header.
            bridge.gameStart("g" + gameIdx, seed, Obs.lastHeaderForBridge(game));
            if (labels == null) {
                return;
            }
            // Pair accounting: r contributes iff BOTH branches completed
            // (draw or decisive). Draws stay in the denominator (pairs).
            int pairs = 0;
            int[] wAct = new int[np];
            int[] wHold = new int[np];
            int drawAct = 0;
            int drawHold = 0;
            int[] crash = new int[2];
            int skipAct = 0;
            int holdAnom = 0;
            for (int r = 0; r < k; r++) {
                for (int b = 0; b < 2; b++) {
                    if (outcome[b][r] == -2) {
                        crash[b]++;
                    }
                }
                if (outcome[0][r] == -3) {
                    skipAct++;
                }
                if (outcome[1][r] == -3) {
                    holdAnom++;
                }
                if (outcome[0][r] >= -1 && outcome[1][r] >= -1) {
                    pairs++;
                    if (outcome[0][r] >= 0) {
                        wAct[outcome[0][r]]++;
                    } else {
                        drawAct++;
                    }
                    if (outcome[1][r] >= 0) {
                        wHold[outcome[1][r]]++;
                    } else {
                        drawHold++;
                    }
                }
            }
            StringBuilder sb = new StringBuilder(256);
            sb.append("{\"i\":").append(gameIdx)
                    .append(",\"seed\":").append(seed)
                    .append(",\"fp\":").append(myFp)
                    .append(",\"t\":").append(turn)
                    .append(",\"tt\":").append(targetTurn)
                    .append(",\"k\":").append(k)
                    .append(",\"forced\":true")
                    .append(",\"seat\":\"").append(jstr(seatName)).append('"')
                    .append(",\"pairs\":").append(pairs)
                    .append(",\"w_act\":[");
            for (int j = 0; j < np; j++) {
                sb.append(j > 0 ? "," : "").append(wAct[j]);
            }
            sb.append("],\"w_hold\":[");
            for (int j = 0; j < np; j++) {
                sb.append(j > 0 ? "," : "").append(wHold[j]);
            }
            sb.append("],\"draw_act\":").append(drawAct)
                    .append(",\"draw_hold\":").append(drawHold)
                    .append(",\"crash_act\":").append(crash[0])
                    .append(",\"crash_hold\":").append(crash[1])
                    .append(",\"skip_act\":").append(skipAct)
                    .append(",\"hold_anom\":").append(holdAnom);
            if (!skips.isEmpty()) {
                sb.append(",\"skips\":{");
                boolean first = true;
                for (java.util.Map.Entry<String, Integer> e : skips.entrySet()) {
                    sb.append(first ? "" : ",").append('"').append(jstr(e.getKey()))
                            .append("\":").append(e.getValue());
                    first = false;
                }
                sb.append('}');
            }
            sb.append(",\"copy_ms\":").append(copyMsTotal)
                    .append(",\"ms\":").append((System.nanoTime() - block0) / 1_000_000)
                    .append('}');
            synchronized (labels) {
                labels.println(sb);
                labels.flush();
            }
        }

        /** Generic candidate campaign: one shared NATURAL completion and one
         * exact forced completion per candidate arm for every paired rollout
         * seed.  The seed is keyed by (game, stable window id, rollout), never
         * by the transient fp counter, so a subset rerun preserves pairing. */
        private void doCandidateRollouts(int turn, CandidatePoint point) {
            int myFp = fp++;
            PhaseHandler ph = game.getPhaseHandler();
            Player prio = ph.getPriorityPlayer();
            boolean bridgeSeat = prio.getController() instanceof PlayerControllerAnvil
                    && ((PlayerControllerAnvil) prio.getController()).bridgesPriority();
            if (!bridgeSeat || point.seat < 0 || point.seat >= game.getRegisteredPlayers().size()) {
                writeCandidateSkips(point, "SEAT_MISMATCH", myFp, turn);
                return;
            }
            String seatName = game.getRegisteredPlayers().get(point.seat).getName();
            // Candidate targets are joined by the source decision sequence.
            // Do not persist a mainline Obs.mark here: mark() consumes a
            // sequence number before chooseSpellAbilityToPlay emits the
            // current decision and would shift every later candidate window
            // in this replay.  The labels row carries the fork metadata.
            byte[] rngState = snapshotRng();
            int[] natural = new int[k];
            Arrays.fill(natural, -2);
            int[][] forced = new int[point.arms.size()][k];
            for (int[] row : forced) {
                Arrays.fill(row, -3); // intervention skipped until realized
            }
            int[][] natWins = new int[point.arms.size()][1];
            int[][] forcedWins = new int[point.arms.size()][1];
            int[] paired = new int[point.arms.size()];
            int[] realized = new int[point.arms.size()];
            Map<String, Integer>[] skips = new Map[point.arms.size()];
            for (int i = 0; i < skips.length; i++) {
                skips[i] = new TreeMap<>();
            }
            long block0 = System.nanoTime();
            long copyMsTotal = 0;
            for (int r = 0; r < k; r++) {
                long rollSeed = splitmix64(
                        seed ^ (point.window * 0x9E3779B97F4A7C15L)
                                ^ (r * 0xBF58476D1CE4E5B9L));
                Game naturalCopy;
                long c0 = System.nanoTime();
                try {
                    naturalCopy = new GameCopier(game).makeCopy();
                } catch (Throwable t) {
                    MyRandom.setRandom(restoreRng(rngState));
                    continue;
                }
                copyMsTotal += (System.nanoTime() - c0) / 1_000_000;
                natural[r] = completeCandidateCopy(naturalCopy,
                        "g" + gameIdx + ".w" + point.window + ".r" + r + ".n",
                        rollSeed, seatName, null, rngState);

                for (int a = 0; a < point.arms.size(); a++) {
                    CandidateArm arm = point.arms.get(a);
                    Game forcedCopy;
                    c0 = System.nanoTime();
                    try {
                        forcedCopy = new GameCopier(game).makeCopy();
                    } catch (Throwable t) {
                        skips[a].merge("COPY_FAILURE", 1, Integer::sum);
                        MyRandom.setRandom(restoreRng(rngState));
                        continue;
                    }
                    copyMsTotal += (System.nanoTime() - c0) / 1_000_000;
                    forced[a][r] = completeCandidateCopy(forcedCopy,
                            "g" + gameIdx + ".w" + point.window + ".r" + r + ".c" + arm.ordinal,
                            rollSeed, seatName, arm, rngState);
                    PlayerControllerAnvil.ForcedCandidate fc =
                            PlayerControllerAnvil.forcedCandidate(forcedCopy);
                    PlayerControllerAnvil.CandidateResult result = fc == null
                            ? PlayerControllerAnvil.CandidateResult.PENDING : fc.result;
                    PlayerControllerAnvil.clearForcedCandidate(forcedCopy);
                    if (result == PlayerControllerAnvil.CandidateResult.PENDING) {
                        // The copied game ended or drifted before consuming
                        // the directive; expose it as the public missing-arm
                        // status rather than leaking an internal pending
                        // state into the label vocabulary.
                        result = PlayerControllerAnvil.CandidateResult.NO_MATCH;
                    }
                    if (result == PlayerControllerAnvil.CandidateResult.CAST) {
                        realized[a]++;
                    } else {
                        skips[a].merge(result.name(), 1, Integer::sum);
                        forced[a][r] = -3;
                    }
                    if (natural[r] >= -1 && forced[a][r] >= -1) {
                        paired[a]++;
                        if (natural[r] >= 0) {
                            natWins[a][0] += natural[r] == point.seat ? 1 : 0;
                        }
                        if (forced[a][r] >= 0) {
                            forcedWins[a][0] += forced[a][r] == point.seat ? 1 : 0;
                        }
                    }
                }
            }
            bridge.gameStart("g" + gameIdx, seed, Obs.lastHeaderForBridge(game));
            if (labels == null) {
                return;
            }
            for (int a = 0; a < point.arms.size(); a++) {
                CandidateArm arm = point.arms.get(a);
                StringBuilder sb = new StringBuilder(384);
                sb.append("{\"ev\":\"candidate\",\"source\":\"")
                        .append(jstr(point.source)).append('"')
                        .append(",\"i\":").append(gameIdx)
                        .append(",\"window\":").append(point.window)
                        .append(",\"fp\":").append(myFp)
                        .append(",\"t\":").append(turn)
                        .append(",\"phase\":\"").append(jstr(point.phase)).append('"')
                        .append(",\"seat\":").append(point.seat)
                        .append(",\"candidate\":{\"entity\":").append(arm.entity)
                        .append(",\"sa\":\"").append(jstr(arm.sa)).append("\"}")
                        .append(",\"paired\":").append(paired[a])
                        .append(",\"natural_wins\":").append(natWins[a][0])
                        .append(",\"forced_wins\":").append(forcedWins[a][0])
                        .append(",\"paired_win_delta\":")
                        .append(forcedWins[a][0] - natWins[a][0])
                        .append(",\"forced_realized\":").append(realized[a])
                        .append(",\"forced_total\":").append(k)
                        .append(",\"skip_counts\":{");
                boolean first = true;
                for (Map.Entry<String, Integer> e : skips[a].entrySet()) {
                    sb.append(first ? "" : ",").append('"').append(jstr(e.getKey()))
                            .append("\":").append(e.getValue());
                    first = false;
                }
                sb.append("},\"copy_ms\":").append(copyMsTotal)
                        .append(",\"ms\":").append((System.nanoTime() - block0) / 1_000_000)
                        .append('}');
                synchronized (labels) {
                    labels.println(sb);
                    labels.flush();
                }
            }
        }

        private void writeCandidateSkips(CandidatePoint point, String reason, int myFp, int turn) {
            if (labels == null) {
                return;
            }
            for (CandidateArm arm : point.arms) {
                synchronized (labels) {
                    labels.println("{\"ev\":\"candidate\",\"source\":\""
                            + jstr(point.source) + "\",\"i\":" + gameIdx
                            + ",\"window\":" + point.window + ",\"fp\":" + myFp
                            + ",\"t\":" + turn + ",\"seat\":" + point.seat
                            + ",\"candidate\":{\"entity\":" + arm.entity
                            + ",\"sa\":\"" + jstr(arm.sa) + "\"}"
                            + ",\"status\":\"" + reason + "\",\"paired\":0}");
                    labels.flush();
                }
            }
        }

        /** Emit an explicit miss for a target that the replay never reached
         *  (for example, sampled source and replay streams diverged before
         *  the recorded window).  A missing row must not look like a clean
         *  zero-sample arm to the Python label join. */
        void finishCandidateSkips() {
            if (candidate == null) {
                return;
            }
            int nowTurn = game.getPhaseHandler().getTurn();
            for (CandidatePoint point : candidate.values()) {
                if (!candidateDone.contains(point.window)) {
                    candidateDone.add(point.window);
                    writeCandidateSkips(point, "NO_MATCH", fp++, nowTurn);
                }
            }
        }

        /** Return registered-player winner index, -1 for draw/no outcome,
         * -2 for a completion crash. */
        private int completeCandidateCopy(Game copy, String wid, long rollSeed,
                String seatName, CandidateArm arm, byte[] rngState) {
            copy.getPhaseHandler().devResumeAtPriority();
            copy.copyLastState();
            Obs.startWireGame(copy, wid, rollSeed, fmt, game);
            bridge.gameStart(wid, rollSeed, Obs.lastHeaderForBridge(copy));
            if (arm != null) {
                PlayerControllerAnvil.armForcedCandidate(copy, seatName, arm.entity, arm.sa);
            }
            Random rollRng = new Random(rollSeed);
            if (reshuffle) {
                for (Player p : copy.getPlayers()) {
                    List<Card> lib = new ArrayList<>();
                    for (Card c : p.getZone(ZoneType.Library)) {
                        lib.add(c);
                    }
                    Collections.shuffle(lib, rollRng);
                    p.getZone(ZoneType.Library).setCards(lib);
                }
            }
            MyRandom.setRandom(rollRng);
            boolean crashed = false;
            ScheduledFuture<?> clock = watchdogs.schedule(
                    () -> copy.setGameOver(GameEndReason.Draw),
                    ROLLOUT_TIMEOUT_S, TimeUnit.SECONDS);
            int winner = -1;
            try {
                copy.getPhaseHandler().mainGameLoop();
                if (copy.getOutcome() != null && !copy.getOutcome().isDraw()) {
                    String w = copy.getOutcome().getWinningLobbyPlayer().getName();
                    for (int j = 0; j < copy.getRegisteredPlayers().size(); j++) {
                        if (copy.getRegisteredPlayers().get(j).getName().equals(w)) {
                            winner = j;
                            break;
                        }
                    }
                }
            } catch (Throwable t) {
                crashed = true;
            } finally {
                clock.cancel(false);
                if (!copy.isGameOver()) {
                    copy.setGameOver(GameEndReason.Draw);
                }
                MyRandom.setRandom(restoreRng(rngState));
                Obs.endWireGame(copy);
            }
            return crashed ? -2 : winner;
        }

        /** M10 sched rollouts (m10-ceiling-spec instrument): NATURAL + each
         *  directed schedule arm x K completions per fork point, rollSeeds
         *  PAIRED across arms per (point, roll) — the ADR-0073
         *  same-determinization pattern; horizon-stopped (h > 0) or run to
         *  natural game end (h = 0, stage 2). ONE labels row per completion
         *  carrying the directive trace + the certify-style end snapshot.
         *  Labels-only; drift / seat mismatch emits one loud skip row and no
         *  completions (the Python reader counts these against replay
         *  fidelity). */
        private void doSchedRollouts(int turn, int targetTurn) {
            int myFp = fp++;
            SchedPoint point = sched.get(targetTurn);
            PhaseHandler ph = game.getPhaseHandler();
            Player prio = ph.getPriorityPlayer();
            boolean bridgeSeat = prio.getController() instanceof PlayerControllerAnvil
                    && ((PlayerControllerAnvil) prio.getController()).bridgesPriority();
            int prioSeat = -1;
            for (int j = 0; j < game.getRegisteredPlayers().size(); j++) {
                if (game.getRegisteredPlayers().get(j).getName().equals(prio.getName())) {
                    prioSeat = j;
                }
            }
            String skip = point == null ? "no_point"
                    : turn != targetTurn ? "drift"
                    : !bridgeSeat ? "seat_unbridged"
                    : prioSeat != point.seat ? "seat_mismatch" : null;
            if (skip != null) {
                if (labels != null) {
                    synchronized (labels) {
                        labels.println("{\"ev\":\"sched\",\"i\":" + gameIdx
                                + ",\"seed\":" + seed + ",\"fp\":" + myFp
                                + ",\"t\":" + turn + ",\"tt\":" + targetTurn
                                + ",\"skip\":\"" + skip + "\"}");
                        labels.flush();
                    }
                }
                return;
            }
            Obs.mark(game, "fork", "fp", myFp, "kr", k);
            byte[] rngState = snapshotRng();
            String seatName = prio.getName();
            for (int r = 0; r < k; r++) {
                // Keyed on the TARGET TURN, not the fp counter: stage 2 re-runs
                // a SUBSET of stage-1 points (positives only), so fp numbering
                // shifts between runs — the rollSeed identity the spec's
                // both-horizon trick depends on must survive that.
                long rollSeed = splitmix64(
                        seed ^ (targetTurn * 0x9E3779B97F4A7C15L) ^ (r * 0xBF58476D1CE4E5B9L));
                for (int ai = -1; ai < point.arms.size(); ai++) {
                    SchedArm arm = ai < 0 ? null : point.arms.get(ai);
                    long c0 = System.nanoTime();
                    Game copy;
                    try {
                        copy = new GameCopier(game).makeCopy();
                    } catch (Throwable t) {
                        writeSchedRow(myFp, targetTurn, arm, r, rollSeed, null, null,
                                null, true, false, 0);
                        MyRandom.setRandom(restoreRng(rngState));
                        continue;
                    }
                    Random rollRng = new Random(rollSeed);
                    if (reshuffle) {
                        for (Player p : copy.getPlayers()) {
                            List<Card> lib = new ArrayList<>();
                            for (Card c : p.getZone(ZoneType.Library)) {
                                lib.add(c);
                            }
                            Collections.shuffle(lib, rollRng);
                            p.getZone(ZoneType.Library).setCards(lib);
                        }
                    }
                    copy.getPhaseHandler().devResumeAtPriority();
                    copy.copyLastState();
                    String wid = "g" + gameIdx + ".f" + myFp + "r" + r + "s"
                            + (arm == null ? 0 : arm.id);
                    Obs.startWireGame(copy, wid, rollSeed, fmt, game);
                    bridge.gameStart(wid, rollSeed, Obs.lastHeaderForBridge(copy));
                    ScheduleDirective dir = null;
                    if (arm != null) {
                        dir = ScheduleDirective.arm(copy, seatName, targetTurn,
                                arm.labels, arm.joint);
                    }
                    HorizonStop stop = null;
                    if (point.horizon > 0) {
                        stop = new HorizonStop(copy, targetTurn + point.horizon);
                        copy.subscribeToEvents(stop);
                    }
                    MyRandom.setRandom(rollRng);
                    boolean crashed = false;
                    final boolean[] clockHit = {false};
                    ScheduledFuture<?> clock = watchdogs.schedule(() -> {
                        clockHit[0] = true;
                        copy.setGameOver(GameEndReason.Draw);
                    }, ROLLOUT_TIMEOUT_S, TimeUnit.SECONDS);
                    try {
                        copy.getPhaseHandler().mainGameLoop();
                    } catch (Throwable t) {
                        crashed = true;
                    } finally {
                        clock.cancel(false);
                        if (!copy.isGameOver()) {
                            copy.setGameOver(GameEndReason.Draw);
                        }
                        MyRandom.setRandom(restoreRng(rngState));
                        writeSchedRow(myFp, targetTurn, arm, r, rollSeed, copy, dir,
                                stop, crashed, clockHit[0],
                                (System.nanoTime() - c0) / 1_000_000);
                        ScheduleDirective.clear(copy);
                        Obs.endWireGame(copy);
                        // AiCache is a GLOBAL static memo that heuristic play
                        // clears at every AI priority window — but bridged
                        // seats never take that path, so it grows one game
                        // graph per completion (the M10 sweep OOM: 144/146
                        // retained Games rooted at AiCache.dataMap, heap-dump
                        // proven). Clearing BETWEEN completions is outside any
                        // game's play: entity-keyed entries can't hit across
                        // copies (identity args), deck-keyed entries recompute
                        // identically — trace-invisible, and proven so by the
                        // serve-smoke row-identity re-check.
                        forge.ai.AiCache.clear();
                    }
                }
            }
            bridge.gameStart("g" + gameIdx, seed, Obs.lastHeaderForBridge(game));
        }

        /** One sched labels row — the schema is a CONTRACT with the Python
         *  reader; fields and their conditionality must not drift. copy ==
         *  null encodes a GameCopier crash (crash:true, no snapshot). */
        private void writeSchedRow(int myFp, int targetTurn, SchedArm arm, int roll,
                long rollSeed, Game copy, ScheduleDirective dir, HorizonStop stop,
                boolean crashed, boolean clockHit, long ms) {
            if (labels == null) {
                return;
            }
            StringBuilder sb = new StringBuilder(480);
            sb.append("{\"ev\":\"sched\",\"i\":").append(gameIdx)
                    .append(",\"seed\":").append(seed)
                    .append(",\"fp\":").append(myFp)
                    .append(",\"t\":").append(targetTurn)
                    .append(",\"arm\":").append(arm == null ? 0 : arm.id)
                    .append(",\"roll\":").append(roll)
                    .append(",\"rollseed\":").append(rollSeed)
                    .append(",\"crash\":").append(crashed);
            if (arm != null) {
                sb.append(",\"joint\":").append(arm.joint)
                        .append(",\"sched_n\":").append(arm.labels.size());
            }
            if (dir != null) {
                sb.append(",\"exec\":").append(dir.executed)
                        .append(",\"void\":").append(dir.isVoid())
                        .append(",\"deferred\":").append(dir.deferred)
                        .append(",\"degraded_at\":").append(dir.degradedAt);
                if (dir.degradeWhy != null) {
                    sb.append(",\"degrade_why\":\"").append(jstr(dir.degradeWhy)).append('"');
                }
                if (dir.landPlayed != null) {
                    sb.append(",\"land\":\"").append(jstr(dir.landPlayed)).append('"');
                }
                sb.append(",\"steps\":\"").append(jstr(dir.traceSummary())).append('"')
                        .append(",\"pay\":{\"win\":").append(dir.payWindows)
                        .append(",\"dir\":").append(dir.payDirected)
                        .append(",\"salvage\":").append(dir.paySalvage)
                        .append(",\"fail\":").append(dir.payFail)
                        .append(",\"auto\":").append(dir.payAuto)
                        .append(",\"costmod\":").append(dir.payCostmod)
                        .append(",\"err\":").append(dir.payErr).append('}');
            }
            if (copy != null) {
                int tEnd = -1;
                try {
                    tEnd = copy.getPhaseHandler().getTurn();
                } catch (Exception ignored) {
                }
                // Unique-winner extraction, NOT getWinningLobbyPlayer: a
                // forced draw (horizon stop / rollout clock) runs
                // Player.onGameOver, which marks EVERY surviving player as
                // "won" — the winning-player accessor then returns an
                // arbitrary map-order pick (JVM-varying; the smoke's
                // determinism diff caught exactly this). Exactly one won =
                // a real winner; anything else = -1.
                int winner = -1;
                int nWon = 0;
                if (copy.getOutcome() != null) {
                    for (int j = 0; j < copy.getRegisteredPlayers().size(); j++) {
                        forge.game.player.PlayerOutcome po =
                                copy.getRegisteredPlayers().get(j).getOutcome();
                        if (po != null && po.hasWon()) {
                            winner = j;
                            nWon++;
                        }
                    }
                    if (nWon != 1) {
                        winner = -1;
                    }
                }
                boolean stopped = stop != null && stop.stopped;
                sb.append(",\"stopped\":").append(stopped)
                        .append(",\"ended\":").append(!crashed && !stopped && !clockHit
                                && copy.getOutcome() != null)
                        .append(",\"t_end\":").append(tEnd)
                        .append(",\"winner\":").append(winner);
                // certify-style end snapshot, registered-player seat order
                int np = copy.getRegisteredPlayers().size();
                int[] life = new int[np];
                int[] creatures = new int[np];
                int[] power = new int[np];
                int[] hand = new int[np];
                int[] lands = new int[np];
                try {
                    for (int j = 0; j < np; j++) {
                        Player gp = null;
                        for (Player q : copy.getPlayers()) {
                            if (q.getName().equals(copy.getRegisteredPlayers().get(j).getName())) {
                                gp = q;
                            }
                        }
                        if (gp == null) {
                            continue;
                        }
                        life[j] = gp.getLife();
                        hand[j] = gp.getCardsIn(ZoneType.Hand).size();
                        for (Card c : gp.getCardsIn(ZoneType.Battlefield)) {
                            if (c.isCreature()) {
                                creatures[j]++;
                                power[j] += c.getNetPower();
                            }
                            if (c.isLand()) {
                                lands[j]++;
                            }
                        }
                    }
                } catch (Exception ignored) {
                }
                sb.append(",\"snap\":{");
                appendIntArr(sb, "life", life).append(',');
                appendIntArr(sb, "creatures", creatures).append(',');
                appendIntArr(sb, "power", power).append(',');
                appendIntArr(sb, "hand", hand).append(',');
                appendIntArr(sb, "lands", lands).append('}');
            }
            sb.append(",\"ms\":").append(ms).append('}');
            synchronized (labels) {
                labels.println(sb);
                labels.flush();
            }
        }

        private StringBuilder appendIntArr(StringBuilder sb, String key, int[] v) {
            sb.append('"').append(key).append("\":[");
            for (int i = 0; i < v.length; i++) {
                sb.append(i > 0 ? "," : "").append(v[i]);
            }
            return sb.append(']');
        }

        /** M11 choice rollouts (m11-routing-probes-spec.md): NATURAL + each
         *  forced-choice arm x K completions per fork point, rollSeeds
         *  PAIRED across arms per (point, roll) — the doSchedRollouts
         *  pattern verbatim (fork at the target turn's quiescent MAIN1
         *  own-priority window; windows in earlier phases / on turns where
         *  the seat never holds a MAIN1 window are OUT of coverage, which
         *  the planner prices — the spec's ~85% coverage cap, loudly
         *  counted by fired:false rows). One labels row per completion
         *  carrying the ChoiceDirective trace + the certify-style end
         *  snapshot. */
        private void doChoiceRollouts(int turn, int targetTurn) {
            int myFp = fp++;
            ChoicePoint point = choice.get(targetTurn);
            PhaseHandler ph = game.getPhaseHandler();
            Player prio = ph.getPriorityPlayer();
            boolean bridgeSeat = prio.getController() instanceof PlayerControllerAnvil
                    && ((PlayerControllerAnvil) prio.getController()).bridgesPriority();
            int prioSeat = -1;
            for (int j = 0; j < game.getRegisteredPlayers().size(); j++) {
                if (game.getRegisteredPlayers().get(j).getName().equals(prio.getName())) {
                    prioSeat = j;
                }
            }
            String skip = point == null ? "no_point"
                    : turn != targetTurn ? "drift"
                    : !bridgeSeat ? "seat_unbridged"
                    : prioSeat != point.seat ? "seat_mismatch" : null;
            if (skip != null) {
                if (labels != null) {
                    synchronized (labels) {
                        labels.println("{\"ev\":\"choice\",\"i\":" + gameIdx
                                + ",\"seed\":" + seed + ",\"fp\":" + myFp
                                + ",\"t\":" + turn + ",\"tt\":" + targetTurn
                                + ",\"skip\":\"" + skip + "\"}");
                        labels.flush();
                    }
                }
                return;
            }
            Obs.mark(game, "fork", "fp", myFp, "kr", k);
            byte[] rngState = snapshotRng();
            String seatName = prio.getName();
            for (int r = 0; r < k; r++) {
                // Target-turn-keyed rollSeed (the sched-mode identity rule):
                // paired across arms per (point, roll), stable across re-runs
                // of a subset of points.
                long rollSeed = splitmix64(
                        seed ^ (targetTurn * 0x9E3779B97F4A7C15L) ^ (r * 0xBF58476D1CE4E5B9L));
                for (int ai = -1; ai < point.arms.size(); ai++) {
                    ChoiceArm arm = ai < 0 ? null : point.arms.get(ai);
                    long c0 = System.nanoTime();
                    Game copy;
                    try {
                        copy = new GameCopier(game).makeCopy();
                    } catch (Throwable t) {
                        writeChoiceRow(myFp, targetTurn, arm, r, rollSeed, null, null,
                                null, true, false,
                                "copier:" + t.getClass().getSimpleName(), 0);
                        MyRandom.setRandom(restoreRng(rngState));
                        continue;
                    }
                    Random rollRng = new Random(rollSeed);
                    if (reshuffle) {
                        for (Player p : copy.getPlayers()) {
                            List<Card> lib = new ArrayList<>();
                            for (Card c : p.getZone(ZoneType.Library)) {
                                lib.add(c);
                            }
                            Collections.shuffle(lib, rollRng);
                            p.getZone(ZoneType.Library).setCards(lib);
                        }
                    }
                    copy.getPhaseHandler().devResumeAtPriority();
                    copy.copyLastState();
                    String wid = "g" + gameIdx + ".f" + myFp + "r" + r + "c"
                            + (arm == null ? 0 : arm.id);
                    Obs.startWireGame(copy, wid, rollSeed, fmt, game);
                    bridge.gameStart(wid, rollSeed, Obs.lastHeaderForBridge(copy));
                    ChoiceDirective dir = null;
                    if (arm != null) {
                        dir = ChoiceDirective.arm(copy, seatName, targetTurn,
                                arm.kind, arm.action);
                    }
                    HorizonStop stop = null;
                    if (point.horizon > 0) {
                        stop = new HorizonStop(copy, targetTurn + point.horizon);
                        copy.subscribeToEvents(stop);
                    }
                    MyRandom.setRandom(rollRng);
                    boolean crashed = false;
                    String crashWhy = null;
                    final boolean[] clockHit = {false};
                    ScheduledFuture<?> clock = watchdogs.schedule(() -> {
                        clockHit[0] = true;
                        copy.setGameOver(GameEndReason.Draw);
                    }, ROLLOUT_TIMEOUT_S, TimeUnit.SECONDS);
                    try {
                        copy.getPhaseHandler().mainGameLoop();
                    } catch (Throwable t) {
                        crashed = true;
                        // The M11 crash forensics gap (2026-08-27): swallowed
                        // throwables left a 0.6%-of-completions crash class
                        // unattributable (fast early-resume deaths, whole
                        // points wiped). Class+message ride the row; one
                        // stack per lane run goes to stderr.
                        crashWhy = t.getClass().getSimpleName() + ": "
                                + String.valueOf(t.getMessage());
                        if (crashClassesPrinted.add(t.getClass().getName())) {
                            System.err.println("[choice] completion crash at g"
                                    + gameIdx + " t" + targetTurn + ":");
                            t.printStackTrace();
                        }
                    } finally {
                        clock.cancel(false);
                        if (!copy.isGameOver()) {
                            copy.setGameOver(GameEndReason.Draw);
                        }
                        MyRandom.setRandom(restoreRng(rngState));
                        writeChoiceRow(myFp, targetTurn, arm, r, rollSeed, copy, dir,
                                stop, crashed, clockHit[0], crashWhy,
                                (System.nanoTime() - c0) / 1_000_000);
                        ChoiceDirective.clear(copy);
                        Obs.endWireGame(copy);
                        // AiCache: the global memo bridged seats never clear
                        // (the M10 sweep OOM lesson) — clear between
                        // completions, outside any game's play.
                        forge.ai.AiCache.clear();
                    }
                }
            }
            bridge.gameStart("g" + gameIdx, seed, Obs.lastHeaderForBridge(game));
        }

        /** One choice labels row — schema is a CONTRACT with the Python
         *  reader (the writeSchedRow idiom; end-snapshot block kept
         *  field-identical). copy == null encodes a GameCopier crash. */
        private void writeChoiceRow(int myFp, int targetTurn, ChoiceArm arm, int roll,
                long rollSeed, Game copy, ChoiceDirective dir, HorizonStop stop,
                boolean crashed, boolean clockHit, String crashWhy, long ms) {
            if (labels == null) {
                return;
            }
            StringBuilder sb = new StringBuilder(480);
            sb.append("{\"ev\":\"choice\",\"i\":").append(gameIdx)
                    .append(",\"seed\":").append(seed)
                    .append(",\"fp\":").append(myFp)
                    .append(",\"t\":").append(targetTurn)
                    .append(",\"arm\":").append(arm == null ? 0 : arm.id)
                    .append(",\"roll\":").append(roll)
                    .append(",\"rollseed\":").append(rollSeed)
                    .append(",\"crash\":").append(crashed);
            if (crashWhy != null) {
                sb.append(",\"crash_why\":\"").append(jstr(crashWhy)).append('"');
            }
            if (arm != null) {
                sb.append(",\"kind\":\"")
                        .append(arm.kind == ChoiceDirective.KIND_TUTOR ? "tutor" : "prevent")
                        .append("\",\"act\":").append(arm.action);
            }
            if (dir != null) {
                sb.append(",\"fired\":").append(dir.fired)
                        .append(",\"windows\":").append(dir.windowsSeen)
                        .append(",\"ncand\":").append(dir.ncand);
                if (dir.chosen != null) {
                    sb.append(",\"chosen\":\"").append(jstr(dir.chosen)).append('"');
                }
                if (dir.miss != null) {
                    sb.append(",\"miss\":\"").append(jstr(dir.miss)).append('"');
                }
                if (dir.kind == ChoiceDirective.KIND_PREVENT) {
                    sb.append(",\"pay_ok\":").append(dir.payOk);
                }
            }
            if (copy != null) {
                int tEnd = -1;
                try {
                    tEnd = copy.getPhaseHandler().getTurn();
                } catch (Exception ignored) {
                }
                // Unique-winner extraction, NOT getWinningLobbyPlayer (the
                // writeSchedRow forced-draw lesson: every survivor "won").
                int winner = -1;
                int nWon = 0;
                if (copy.getOutcome() != null) {
                    for (int j = 0; j < copy.getRegisteredPlayers().size(); j++) {
                        forge.game.player.PlayerOutcome po =
                                copy.getRegisteredPlayers().get(j).getOutcome();
                        if (po != null && po.hasWon()) {
                            winner = j;
                            nWon++;
                        }
                    }
                    if (nWon != 1) {
                        winner = -1;
                    }
                }
                boolean stopped = stop != null && stop.stopped;
                sb.append(",\"stopped\":").append(stopped)
                        .append(",\"ended\":").append(!crashed && !stopped && !clockHit
                                && copy.getOutcome() != null)
                        .append(",\"t_end\":").append(tEnd)
                        .append(",\"winner\":").append(winner);
                int np = copy.getRegisteredPlayers().size();
                int[] life = new int[np];
                int[] creatures = new int[np];
                int[] power = new int[np];
                int[] hand = new int[np];
                int[] lands = new int[np];
                try {
                    for (int j = 0; j < np; j++) {
                        Player gp = null;
                        for (Player q : copy.getPlayers()) {
                            if (q.getName().equals(copy.getRegisteredPlayers().get(j).getName())) {
                                gp = q;
                            }
                        }
                        if (gp == null) {
                            continue;
                        }
                        life[j] = gp.getLife();
                        hand[j] = gp.getCardsIn(ZoneType.Hand).size();
                        for (Card c : gp.getCardsIn(ZoneType.Battlefield)) {
                            if (c.isCreature()) {
                                creatures[j]++;
                                power[j] += c.getNetPower();
                            }
                            if (c.isLand()) {
                                lands[j]++;
                            }
                        }
                    }
                } catch (Exception ignored) {
                }
                sb.append(",\"snap\":{");
                appendIntArr(sb, "life", life).append(',');
                appendIntArr(sb, "creatures", creatures).append(',');
                appendIntArr(sb, "power", power).append(',');
                appendIntArr(sb, "hand", hand).append(',');
                appendIntArr(sb, "lands", lands).append('}');
            }
            sb.append(",\"ms\":").append(ms).append('}');
            synchronized (labels) {
                labels.println(sb);
                labels.flush();
            }
        }

        /** M7 D2 sequence probe (routing pin 2026-08-11): THREE arms per fork
         *  point — natural (no directive) / hold-N / act-N — x k completions,
         *  arms of an (fp, r) triple sharing rollSeed (common random numbers,
         *  divergence only from the directive over the N-turn horizon). A
         *  triple with a crashed member drops whole. Labels-only, same seat
         *  guard as the one-shot mode. */
        private void doSeqRollouts(int turn, int targetTurn) {
            if (seqNatOnly) {
                doNatObserveRollouts(turn, targetTurn);
                return;
            }
            int myFp = fp++;
            PhaseHandler ph = game.getPhaseHandler();
            Player prio = ph.getPriorityPlayer();
            boolean bridgeSeat = prio.getController() instanceof PlayerControllerAnvil
                    && ((PlayerControllerAnvil) prio.getController()).bridgesPriority();
            if (!bridgeSeat) {
                if (labels != null) {
                    synchronized (labels) {
                        labels.println("{\"i\":" + gameIdx + ",\"seed\":" + seed
                                + ",\"fp\":" + myFp + ",\"t\":" + turn
                                + ",\"tt\":" + targetTurn
                                + ",\"seq\":true,\"seat_skip\":true}");
                        labels.flush();
                    }
                }
                return;
            }
            Obs.mark(game, "fork", "fp", myFp, "kr", k);
            byte[] rngState = snapshotRng();
            int np = game.getRegisteredPlayers().size();
            // outcome[arm][r]: -2 crash, -1 draw, >=0 winner idx
            // arms: 0 = natural, 1 = hold-N, 2 = act-N
            int[][] outcome = new int[3][k];
            long[] holds = new long[3];
            long[] casts = new long[3];
            long[] exhausts = new long[3];
            // ADR-0054 labels-row extension: the act arm's first realized
            // cast per completion (candidate-label SA string; null = the
            // completion never cast — all windows exhausted).
            String[] actFirst = new String[k];
            String seatName = prio.getName();
            int untilTurn = targetTurn + forceSeq - 1;
            long block0 = System.nanoTime();
            long copyMsTotal = 0;
            for (int r = 0; r < k; r++) {
                long rollSeed = splitmix64(
                        seed ^ (myFp * 0x9E3779B97F4A7C15L) ^ (r * 0xBF58476D1CE4E5B9L));
                for (int arm = 0; arm < 3; arm++) {
                    Game copy;
                    long c0 = System.nanoTime();
                    try {
                        copy = new GameCopier(game).makeCopy();
                    } catch (Throwable t) {
                        outcome[arm][r] = -2;
                        MyRandom.setRandom(restoreRng(rngState));
                        continue;
                    }
                    copyMsTotal += (System.nanoTime() - c0) / 1_000_000;
                    Random rollRng = new Random(rollSeed);
                    if (reshuffle) {
                        for (Player p : copy.getPlayers()) {
                            List<Card> lib = new ArrayList<>();
                            for (Card c : p.getZone(ZoneType.Library)) {
                                lib.add(c);
                            }
                            Collections.shuffle(lib, rollRng);
                            p.getZone(ZoneType.Library).setCards(lib);
                        }
                    }
                    copy.getPhaseHandler().devResumeAtPriority();
                    copy.copyLastState();
                    String wid = "g" + gameIdx + ".f" + myFp + "r" + r
                            + (arm == 0 ? "n" : (arm == 1 ? "h" : "a"));
                    Obs.startWireGame(copy, wid, rollSeed, fmt, game);
                    bridge.gameStart(wid, rollSeed, Obs.lastHeaderForBridge(copy));
                    if (arm == 1) {
                        PlayerControllerAnvil.armSeq(copy, seatName,
                                PlayerControllerAnvil.SeqMode.HOLD, untilTurn);
                    } else if (arm == 2) {
                        PlayerControllerAnvil.armSeq(copy, seatName,
                                PlayerControllerAnvil.SeqMode.ACT, untilTurn);
                    }
                    MyRandom.setRandom(rollRng);
                    int wi = -1;
                    boolean crashed = false;
                    ScheduledFuture<?> clock = watchdogs.schedule(
                            () -> copy.setGameOver(GameEndReason.Draw),
                            ROLLOUT_TIMEOUT_S, TimeUnit.SECONDS);
                    try {
                        copy.getPhaseHandler().mainGameLoop();
                        if (copy.getOutcome() != null && !copy.getOutcome().isDraw()) {
                            String w = copy.getOutcome().getWinningLobbyPlayer().getName();
                            for (int j = 0; j < copy.getRegisteredPlayers().size(); j++) {
                                if (copy.getRegisteredPlayers().get(j).getName().equals(w)) {
                                    wi = j;
                                    break;
                                }
                            }
                        }
                    } catch (Throwable t) {
                        crashed = true;
                    } finally {
                        clock.cancel(false);
                        if (!copy.isGameOver()) {
                            copy.setGameOver(GameEndReason.Draw);
                        }
                        MyRandom.setRandom(restoreRng(rngState));
                        PlayerControllerAnvil.SeqDirective sdd =
                                PlayerControllerAnvil.seqDirective(copy);
                        if (sdd != null) {
                            holds[arm] += sdd.holds;
                            casts[arm] += sdd.casts;
                            exhausts[arm] += sdd.exhausts;
                            if (arm == 2) {
                                actFirst[r] = sdd.firstCastSa;
                            }
                        }
                        PlayerControllerAnvil.clearSeq(copy);
                        outcome[arm][r] = crashed ? -2 : wi;
                        Obs.endWireGame(copy);
                    }
                }
            }
            bridge.gameStart("g" + gameIdx, seed, Obs.lastHeaderForBridge(game));
            if (labels == null) {
                return;
            }
            // Triple accounting: r contributes iff ALL THREE arms completed.
            int triples = 0;
            int[][] w = new int[3][np];
            int[] draws = new int[3];
            int[] crash = new int[3];
            for (int r = 0; r < k; r++) {
                for (int arm = 0; arm < 3; arm++) {
                    if (outcome[arm][r] == -2) {
                        crash[arm]++;
                    }
                }
                if (outcome[0][r] >= -1 && outcome[1][r] >= -1 && outcome[2][r] >= -1) {
                    triples++;
                    for (int arm = 0; arm < 3; arm++) {
                        if (outcome[arm][r] >= 0) {
                            w[arm][outcome[arm][r]]++;
                        } else {
                            draws[arm]++;
                        }
                    }
                }
            }
            StringBuilder sb = new StringBuilder(320);
            sb.append("{\"i\":").append(gameIdx)
                    .append(",\"seed\":").append(seed)
                    .append(",\"fp\":").append(myFp)
                    .append(",\"t\":").append(turn)
                    .append(",\"tt\":").append(targetTurn)
                    .append(",\"k\":").append(k)
                    .append(",\"seq\":true")
                    .append(",\"n\":").append(forceSeq)
                    .append(",\"seat\":\"").append(jstr(seatName)).append('"')
                    .append(",\"triples\":").append(triples);
            String[] armName = {"nat", "hold", "act"};
            for (int arm = 0; arm < 3; arm++) {
                sb.append(",\"w_").append(armName[arm]).append("\":[");
                for (int j = 0; j < np; j++) {
                    sb.append(j > 0 ? "," : "").append(w[arm][j]);
                }
                sb.append("],\"draw_").append(armName[arm]).append("\":").append(draws[arm])
                        .append(",\"crash_").append(armName[arm]).append("\":").append(crash[arm]);
            }
            sb.append(",\"holds\":").append(holds[1])
                    .append(",\"acts\":").append(casts[2])
                    .append(",\"exhausts\":").append(exhausts[2])
                    .append(",\"nat_anom\":").append(holds[0] + casts[0] + exhausts[0]);
            // ADR-0054: act-arm first-cast distribution over counted triples
            // (act_first counts by SA string; act_none = completions that
            // never cast; act_first_modal/agree = the target's cast* and its
            // agreement fraction among completions that did cast).
            java.util.Map<String, Integer> firstCounts = new java.util.TreeMap<>();
            int actNone = 0;
            for (int r = 0; r < k; r++) {
                if (outcome[0][r] < -1 || outcome[1][r] < -1 || outcome[2][r] < -1) {
                    continue; // same triple filter as the win counts
                }
                if (actFirst[r] == null) {
                    actNone++;
                } else {
                    firstCounts.merge(actFirst[r], 1, Integer::sum);
                }
            }
            String modal = null;
            int modalN = 0, castN = 0;
            for (java.util.Map.Entry<String, Integer> e : firstCounts.entrySet()) {
                castN += e.getValue();
                if (e.getValue() > modalN) {
                    modalN = e.getValue();
                    modal = e.getKey();
                }
            }
            sb.append(",\"act_first\":{");
            boolean firstEntry = true;
            for (java.util.Map.Entry<String, Integer> e : firstCounts.entrySet()) {
                sb.append(firstEntry ? "" : ",").append('"').append(jstr(e.getKey()))
                        .append("\":").append(e.getValue());
                firstEntry = false;
            }
            sb.append("},\"act_none\":").append(actNone);
            if (modal != null) {
                sb.append(",\"act_first_modal\":\"").append(jstr(modal)).append('"')
                        .append(",\"act_first_agree\":")
                        .append(String.format(java.util.Locale.ROOT, "%.4f",
                                modalN / (double) castN));
            }
            sb.append(",\"copy_ms\":").append(copyMsTotal)
                    .append(",\"ms\":").append((System.nanoTime() - block0) / 1_000_000)
                    .append('}');
            synchronized (labels) {
                labels.println(sb);
                labels.flush();
            }
        }

        // M8 D1 (m8-plan D1): single NATURAL arm, K completions per drilled
        // fork point under an OBSERVE directive — records the drilled seat's
        // first realized non-land cast (SA + absolute game turn), first
        // land-play turn, and the per-completion outcome, so the timing read
        // can join bins to wins. Labels-only; the forced arms never run.
        private void doNatObserveRollouts(int turn, int targetTurn) {
            int myFp = fp++;
            PhaseHandler ph = game.getPhaseHandler();
            Player prio = ph.getPriorityPlayer();
            boolean bridgeSeat = prio.getController() instanceof PlayerControllerAnvil
                    && ((PlayerControllerAnvil) prio.getController()).bridgesPriority();
            if (!bridgeSeat) {
                if (labels != null) {
                    synchronized (labels) {
                        labels.println("{\"i\":" + gameIdx + ",\"seed\":" + seed
                                + ",\"fp\":" + myFp + ",\"t\":" + turn
                                + ",\"tt\":" + targetTurn
                                + ",\"seq\":true,\"arms\":\"nat\",\"seat_skip\":true}");
                        labels.flush();
                    }
                }
                return;
            }
            Obs.mark(game, "fork", "fp", myFp, "kr", k);
            byte[] rngState = snapshotRng();
            int np = game.getRegisteredPlayers().size();
            int[] out = new int[k]; // -2 crash, -1 draw, >=0 winner idx
            String[] firstSa = new String[k];
            int[] firstT = new int[k];
            int[] landT = new int[k];
            long anom = 0;
            String seatName = prio.getName();
            long block0 = System.nanoTime();
            long copyMsTotal = 0;
            for (int r = 0; r < k; r++) {
                long rollSeed = splitmix64(
                        seed ^ (myFp * 0x9E3779B97F4A7C15L) ^ (r * 0xBF58476D1CE4E5B9L));
                firstT[r] = -1;
                landT[r] = -1;
                Game copy;
                long c0 = System.nanoTime();
                try {
                    copy = new GameCopier(game).makeCopy();
                } catch (Throwable t) {
                    out[r] = -2;
                    MyRandom.setRandom(restoreRng(rngState));
                    continue;
                }
                copyMsTotal += (System.nanoTime() - c0) / 1_000_000;
                Random rollRng = new Random(rollSeed);
                if (reshuffle) {
                    for (Player p : copy.getPlayers()) {
                        List<Card> lib = new ArrayList<>();
                        for (Card c : p.getZone(ZoneType.Library)) {
                            lib.add(c);
                        }
                        Collections.shuffle(lib, rollRng);
                        p.getZone(ZoneType.Library).setCards(lib);
                    }
                }
                copy.getPhaseHandler().devResumeAtPriority();
                copy.copyLastState();
                String wid = "g" + gameIdx + ".f" + myFp + "r" + r + "n";
                Obs.startWireGame(copy, wid, rollSeed, fmt, game);
                bridge.gameStart(wid, rollSeed, Obs.lastHeaderForBridge(copy));
                // OBSERVE never expires: the >=+3-or-never bin needs the
                // whole completion, not the N-turn window.
                PlayerControllerAnvil.armSeq(copy, seatName,
                        PlayerControllerAnvil.SeqMode.OBSERVE, Integer.MAX_VALUE);
                MyRandom.setRandom(rollRng);
                int wi = -1;
                boolean crashed = false;
                ScheduledFuture<?> clock = watchdogs.schedule(
                        () -> copy.setGameOver(GameEndReason.Draw),
                        ROLLOUT_TIMEOUT_S, TimeUnit.SECONDS);
                try {
                    copy.getPhaseHandler().mainGameLoop();
                    if (copy.getOutcome() != null && !copy.getOutcome().isDraw()) {
                        String w = copy.getOutcome().getWinningLobbyPlayer().getName();
                        for (int j = 0; j < copy.getRegisteredPlayers().size(); j++) {
                            if (copy.getRegisteredPlayers().get(j).getName().equals(w)) {
                                wi = j;
                                break;
                            }
                        }
                    }
                } catch (Throwable t) {
                    crashed = true;
                } finally {
                    clock.cancel(false);
                    if (!copy.isGameOver()) {
                        copy.setGameOver(GameEndReason.Draw);
                    }
                    MyRandom.setRandom(restoreRng(rngState));
                    PlayerControllerAnvil.SeqDirective sdd =
                            PlayerControllerAnvil.seqDirective(copy);
                    if (sdd != null) {
                        anom += sdd.holds + sdd.casts + sdd.exhausts;
                        firstSa[r] = sdd.firstSpellSa;
                        firstT[r] = sdd.firstSpellTurn;
                        landT[r] = sdd.firstLandTurn;
                    }
                    PlayerControllerAnvil.clearSeq(copy);
                    out[r] = crashed ? -2 : wi;
                    Obs.endWireGame(copy);
                }
            }
            bridge.gameStart("g" + gameIdx, seed, Obs.lastHeaderForBridge(game));
            if (labels == null) {
                return;
            }
            int comps = 0;
            int[] w = new int[np];
            int draws = 0;
            int crash = 0;
            for (int r = 0; r < k; r++) {
                if (out[r] == -2) {
                    crash++;
                } else {
                    comps++;
                    if (out[r] >= 0) {
                        w[out[r]]++;
                    } else {
                        draws++;
                    }
                }
            }
            StringBuilder sb = new StringBuilder(1024);
            sb.append("{\"i\":").append(gameIdx)
                    .append(",\"seed\":").append(seed)
                    .append(",\"fp\":").append(myFp)
                    .append(",\"t\":").append(turn)
                    .append(",\"tt\":").append(targetTurn)
                    .append(",\"k\":").append(k)
                    .append(",\"seq\":true,\"arms\":\"nat\"")
                    .append(",\"n\":").append(forceSeq)
                    .append(",\"seat\":\"").append(jstr(seatName)).append('"')
                    .append(",\"comps\":").append(comps)
                    .append(",\"w_nat\":[");
            for (int j = 0; j < np; j++) {
                sb.append(j > 0 ? "," : "").append(w[j]);
            }
            sb.append("],\"draw_nat\":").append(draws)
                    .append(",\"crash_nat\":").append(crash)
                    .append(",\"nat_anom\":").append(anom)
                    .append(",\"out\":[");
            for (int r = 0; r < k; r++) {
                sb.append(r > 0 ? "," : "").append(out[r]);
            }
            sb.append("],\"first_sa\":[");
            for (int r = 0; r < k; r++) {
                sb.append(r > 0 ? "," : "");
                if (firstSa[r] == null) {
                    sb.append("null");
                } else {
                    sb.append('"').append(jstr(firstSa[r])).append('"');
                }
            }
            sb.append("],\"first_t\":[");
            for (int r = 0; r < k; r++) {
                sb.append(r > 0 ? "," : "").append(firstT[r]);
            }
            sb.append("],\"land_t\":[");
            for (int r = 0; r < k; r++) {
                sb.append(r > 0 ? "," : "").append(landT[r]);
            }
            sb.append("],\"copy_ms\":").append(copyMsTotal)
                    .append(",\"ms\":").append((System.nanoTime() - block0) / 1_000_000)
                    .append('}');
            synchronized (labels) {
                labels.println(sb);
                labels.flush();
            }
        }
    }

    private static byte[] snapshotRng() {
        try {
            java.io.ByteArrayOutputStream bos = new java.io.ByteArrayOutputStream();
            try (java.io.ObjectOutputStream oos = new java.io.ObjectOutputStream(bos)) {
                oos.writeObject(MyRandom.getRandom());
            }
            return bos.toByteArray();
        } catch (Exception e) {
            throw new RuntimeException("RNG snapshot failed", e);
        }
    }

    private static Random restoreRng(byte[] state) {
        try (java.io.ObjectInputStream ois = new java.io.ObjectInputStream(
                new java.io.ByteArrayInputStream(state))) {
            return (Random) ois.readObject();
        } catch (Exception e) {
            throw new RuntimeException("RNG restore failed", e);
        }
    }
}
