package forge.view;

import java.io.File;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.commons.lang3.time.StopWatch;

import forge.LobbyPlayer;
import forge.ai.LobbyPlayerAi;
import forge.ai.rl.LobbyPlayerRL;
import forge.ai.rl.ModelServerException;
import forge.ai.rl.PlayerControllerRL;
import forge.ai.rl.ReplayConfig;
import forge.ai.rl.RLConfig;
import forge.ai.rl.RLModelMode;
import forge.deck.Deck;
import forge.deck.io.DeckSerializer;
import forge.game.*;
import forge.game.player.Player;
import forge.game.player.RegisteredPlayer;
import forge.localinstance.properties.ForgeConstants;
import forge.model.FModel;
import forge.player.GamePlayerUtil;
import forge.util.MyRandom;
import forge.util.ReplayRandom;

/**
 * Headless runner for RL training data collection and evaluation.
 *
 * Modes:
 *   collect  - Run heuristic AI vs heuristic AI, recording trajectories for imitation learning
 *   evaluate - Run RL AI vs heuristic AI, measuring win rate
 *   selfplay - Run RL AI vs RL AI for self-play training
 *
 * Usage:
 *   forge rltrain collect -d deck1.dck -d deck2.dck -n 1000
 *   forge rltrain evaluate -d deck1.dck -d deck2.dck -n 100
 *   forge rltrain selfplay -d deck1.dck -d deck2.dck -n 1000
 */
public class SimulateRLTraining {

    public static void simulate(String[] args) {
        FModel.initialize(null, null);

        System.out.println("=== Forge RL Training Mode ===");

        // Parse arguments: rltrain <mode> -d deck1 -d deck2 -n N ...
        final Map<String, List<String>> params = new HashMap<>();
        String mode = "collect";
        int startIdx = 1; // skip "rltrain" at args[0]

        // First non-flag arg is the mode
        if (args.length > 1 && !args[1].startsWith("-")) {
            mode = args[1].toLowerCase();
            startIdx = 2;
        }

        String currentKey = null;
        for (int i = startIdx; i < args.length; i++) {
            final String a = args[i];
            if (a.startsWith("-")) {
                currentKey = a.substring(1);
                if (!params.containsKey(currentKey)) {
                    params.put(currentKey, new ArrayList<>());
                }
            } else if (currentKey != null) {
                params.get(currentKey).add(a);
            }
        }

        int nGames = params.containsKey("n")
                ? Integer.parseInt(params.get("n").get(0)) : 100;
        int timeout = params.containsKey("c")
                ? Integer.parseInt(params.get("c").get(0)) : 180;
        String outputDir = params.containsKey("o")
                ? params.get("o").get(0) : "rl_data/trajectories";
        boolean quiet = params.containsKey("q");
        boolean useOnnx = params.containsKey("onnx");
        boolean deterministic = params.containsKey("deterministic");
        boolean zeroIntermediateReward = params.containsKey("zero-intermediate-reward");
        int threads = params.containsKey("t")
                ? Integer.parseInt(params.get("t").get(0))
                : Runtime.getRuntime().availableProcessors();
        if (deterministic && threads != 1) {
            threads = 1;
            System.out.println("Deterministic replay forces single-threaded execution.");
        }
        String grpcHost = params.containsKey("host")
                ? params.get("host").get(0) : "localhost";
        Long seed = params.containsKey("seed")
                ? Long.parseLong(params.get("seed").get(0))
                : null;
        String replayId = params.containsKey("replay-id")
                ? params.get("replay-id").get(0)
                : null;
        String traceOut = params.containsKey("trace-out")
                ? params.get("trace-out").get(0)
                : "rl_data/replays";
        String baselinePolicy = params.containsKey("baseline")
                ? params.get("baseline").get(0).toLowerCase()
                : "heuristic";
        String comparePolicy = params.containsKey("compare")
                ? params.get("compare").get(0).toLowerCase()
                : "rl";
        // Support comma-separated ports for multi-server parallelism
        int[] grpcPorts;
        if (params.containsKey("port")) {
            String portStr = params.get("port").get(0);
            String[] portParts = portStr.split(",");
            grpcPorts = new int[portParts.length];
            for (int pi = 0; pi < portParts.length; pi++) {
                grpcPorts[pi] = Integer.parseInt(portParts[pi].trim());
            }
        } else {
            grpcPorts = new int[]{50051};
        }
        int grpcPort = grpcPorts[0];

        // Load decks
        List<Deck> decks = new ArrayList<>();
        if (params.containsKey("d")) {
            for (String deckParam : params.get("d")) {
                Deck d = loadDeck(deckParam);
                if (d == null) {
                    System.out.println("Could not load deck: " + deckParam);
                    return;
                }
                decks.add(d);
                System.out.println("Loaded deck: " + d.getName() + " (" + d.getMain().countAll() + " cards)");
            }
        }

        // Load all decks from directory
        if (params.containsKey("D")) {
            String dirPath = params.get("D").get(0);
            File dir = new File(dirPath);
            if (dir.isDirectory()) {
                for (File f : dir.listFiles((d, name) -> name.endsWith(".dck"))) {
                    Deck d = DeckSerializer.fromFile(f);
                    if (d != null) {
                        decks.add(d);
                    }
                }
                System.out.println("Loaded " + decks.size() + " decks from " + dirPath);
            }
        }

        if (decks.size() < 2) {
            System.out.println("Need at least 2 decks. Use -d <deck.dck> or -D <directory>");
            printHelp();
            return;
        }

        System.out.println("Mode: " + mode + " | Games: " + nGames
                + " | Threads: " + threads + " | Timeout: " + timeout + "s");
        System.out.println("Output: " + outputDir);
        System.out.println();

        // Head-to-head mode arguments
        String onnxDir1 = params.containsKey("m1") ? params.get("m1").get(0) : null;
        String onnxDir2 = params.containsKey("m2") ? params.get("m2").get(0) : null;

        // Parse port2 for league play (opponent ports)
        int[] grpcPorts2;
        if (params.containsKey("port2")) {
            String portStr2 = params.get("port2").get(0);
            String[] portParts2 = portStr2.split(",");
            grpcPorts2 = new int[portParts2.length];
            for (int pi = 0; pi < portParts2.length; pi++) {
                grpcPorts2[pi] = Integer.parseInt(portParts2[pi].trim());
            }
        } else {
            grpcPorts2 = grpcPorts; // default: same as player 1
        }

        switch (mode) {
            case "collect":
                runCollectionMode(decks, nGames, timeout, outputDir, quiet, threads,
                        zeroIntermediateReward);
                break;
            case "evaluate":
                runEvaluationMode(decks, nGames, timeout, outputDir, quiet, grpcHost, grpcPorts, threads, useOnnx);
                break;
            case "selfplay":
                runSelfPlayMode(decks, nGames, timeout, outputDir, quiet, grpcHost, grpcPorts, threads);
                break;
            case "leagueplay":
                runLeaguePlayMode(decks, nGames, timeout, outputDir, quiet, grpcHost, grpcPorts, grpcPorts2, threads);
                break;
            case "headtohead":
                if (onnxDir1 == null || onnxDir2 == null) {
                    System.out.println("Head-to-head requires -m1 <model_dir> -m2 <model_dir>");
                    return;
                }
                runHeadToHeadMode(decks, nGames, timeout, outputDir, quiet, threads,
                        onnxDir1, onnxDir2);
                break;
            case "replaydiff":
                if (seed == null) {
                    System.out.println("Replay diff mode requires -seed <long>");
                    return;
                }
                runReplayDiffMode(decks, timeout, quiet, grpcHost, grpcPort,
                        useOnnx, seed, replayId, traceOut,
                        baselinePolicy, comparePolicy, onnxDir1);
                break;
            default:
                System.out.println("Unknown mode: " + mode);
                printHelp();
        }
    }

    /**
     * Collection mode: heuristic AI vs heuristic AI, recording all decisions as trajectories.
     * This produces training data for imitation learning (Phase 2 of the plan).
     * Runs games in parallel across multiple threads.
     */
    private static void runCollectionMode(List<Deck> decks, int nGames, int timeout,
                                           String outputDir, boolean quiet, int threads,
                                           boolean zeroIntermediateReward) {
        System.out.println("=== Imitation Learning Data Collection ===");
        System.out.println("Using " + threads + " threads");
        System.out.println("Intermediate reward recording: "
                + (zeroIntermediateReward ? "ZEROED" : "NORMAL"));

        AtomicInteger completed = new AtomicInteger(0);
        AtomicInteger p1Wins = new AtomicInteger(0);
        AtomicInteger p2Wins = new AtomicInteger(0);
        AtomicInteger draws = new AtomicInteger(0);
        AtomicInteger errors = new AtomicInteger(0);
        java.util.concurrent.atomic.AtomicLong totalTurns = new java.util.concurrent.atomic.AtomicLong(0);
        java.util.concurrent.atomic.AtomicLong totalFiles = new java.util.concurrent.atomic.AtomicLong(0);
        long startTime = System.currentTimeMillis();

        java.util.concurrent.ExecutorService executor =
                java.util.concurrent.Executors.newFixedThreadPool(threads);
        List<java.util.concurrent.Future<?>> futures = new ArrayList<>();

        System.out.printf("  Game │ Done/Total │ Games/s │ P1  │ P2  │ Draw│ Err │"
                + " Turns │  Files │ ETA%n");
        System.out.printf("  ─────┼────────────┼─────────┼─────┼─────┼─────┼─────┼"
                + "───────┼────────┼──────%n");

        java.util.Random deckRng = new java.util.Random();
        for (int i = 0; i < nGames; i++) {
            final int gameIdx = i;
            // Randomize deck pairing — any deck vs any deck, including mirrors
            final int d1Idx = deckRng.nextInt(decks.size());
            int d2Idx = deckRng.nextInt(decks.size());
            final Deck deck1 = decks.get(d1Idx);
            final Deck deck2 = decks.get(d2Idx);
            final boolean p1First = (i % 2 == 0);

            futures.add(executor.submit(() -> {
                RLConfig config = new RLConfig();
                config.setMode(RLModelMode.RECORD_HEURISTIC);
                config.setRecordTrajectories(true);
                config.setZeroIntermediateReward(zeroIntermediateReward);
                config.setTrajectoryOutputDir(outputDir);

                try {
                    GameResult result = runSingleGame(
                            deck1, deck2, config, config,
                            "P1_" + gameIdx, "P2_" + gameIdx,
                            timeout, p1First);

                    if (result.isDraw) {
                        draws.incrementAndGet();
                    } else if (result.player1Won) {
                        p1Wins.incrementAndGet();
                    } else {
                        p2Wins.incrementAndGet();
                    }
                    totalTurns.addAndGet(result.turns);
                } catch (Exception e) {
                    errors.incrementAndGet();
                }

                int done = completed.incrementAndGet();
                // Count trajectory files
                File dir = new File(outputDir);
                long files = 0;
                if (dir.isDirectory()) {
                    File[] list = dir.listFiles((d, name) -> name.endsWith(".jsonl"));
                    if (list != null) files = list.length;
                }
                totalFiles.set(files);

                if (done % 10 == 0 || done == nGames) {
                    long elapsed = System.currentTimeMillis() - startTime;
                    double gps = done * 1000.0 / elapsed;
                    int remaining = nGames - done;
                    double etaSec = gps > 0 ? remaining / gps : 0;
                    double avgTurns = totalTurns.get() / (double) Math.max(done, 1);
                    System.out.printf(
                            "  %4d │ %4d/%-4d  │ %5.1f   │ %3d │ %3d │ %3d │ %3d │"
                            + " %5.1f │ %6d │ %4.0fs%n",
                            gameIdx, done, nGames, gps,
                            p1Wins.get(), p2Wins.get(),
                            draws.get(), errors.get(),
                            avgTurns, files, etaSec);
                    System.out.flush();
                }
            }));
        }

        // Wait for all games to complete
        for (java.util.concurrent.Future<?> f : futures) {
            try {
                f.get(timeout + 30, TimeUnit.SECONDS);
            } catch (Exception e) {
                errors.incrementAndGet();
            }
        }
        executor.shutdown();

        long totalMs = System.currentTimeMillis() - startTime;
        double gamesPerSec = nGames * 1000.0 / totalMs;
        double avgTurns = totalTurns.get() / (double) Math.max(nGames, 1);

        System.out.println();
        System.out.println("=== Collection Complete ===");
        System.out.printf("Games: %d | P1: %d | P2: %d | Draw: %d | Errors: %d%n",
                nGames, p1Wins.get(), p2Wins.get(), draws.get(), errors.get());
        System.out.printf("Avg turns: %.1f | Files: %d%n", avgTurns, totalFiles.get());
        System.out.printf("Total time: %.1fs | %.1f games/sec (%d threads)%n",
                totalMs / 1000.0, gamesPerSec, threads);
    }

    /**
     * Evaluation mode: RL AI vs heuristic AI, measuring win rate.
     */
    private static void runEvaluationMode(List<Deck> decks, int nGames, int timeout,
                                            String outputDir, boolean quiet,
                                            String grpcHost, int[] grpcPorts, int threads,
                                            boolean useOnnx) {
        System.out.println("=== RL Evaluation Mode (" + threads + " threads"
                + (useOnnx ? ", ONNX" : ", GRPC" + (grpcPorts.length > 1 ? " x" + grpcPorts.length + " servers" : ""))
                + ") ===");

        AtomicInteger rlWins = new AtomicInteger(0);
        AtomicInteger heuristicWins = new AtomicInteger(0);
        AtomicInteger draws = new AtomicInteger(0);
        AtomicInteger completed = new AtomicInteger(0);
        AtomicInteger serverErrors = new AtomicInteger(0);
        AtomicInteger consecutiveErrors = new AtomicInteger(0);
        final int MAX_SERVER_ERRORS = 3;
        long startTime = System.currentTimeMillis();

        java.util.concurrent.ExecutorService executor =
                java.util.concurrent.Executors.newFixedThreadPool(threads);
        List<java.util.concurrent.Future<?>> futures = new ArrayList<>();

        java.util.Random evalDeckRng = new java.util.Random();
        for (int i = 0; i < nGames; i++) {
            final int gameIdx = i;
            // Round-robin port assignment across available servers
            final int assignedPort = grpcPorts[i % grpcPorts.length];
            // Randomize deck pairing for eval too
            final Deck rlDeck = decks.get(evalDeckRng.nextInt(decks.size()));
            final Deck aiDeck = decks.get(evalDeckRng.nextInt(decks.size()));
            final boolean rlFirst = (i % 2 == 0);

            futures.add(executor.submit(() -> {
                // Each thread gets its own RLConfig to avoid contention
                RLConfig rlConfig = new RLConfig();
                if (useOnnx) {
                    rlConfig.setMode(RLModelMode.ONNX);
                } else {
                    rlConfig.setMode(RLModelMode.GRPC);
                    rlConfig.setGrpcHost(grpcHost);
                    rlConfig.setGrpcPort(assignedPort);
                }
                rlConfig.setRecordTrajectories(true);
                rlConfig.setTrajectoryOutputDir(outputDir);

                RLConfig heuristicConfig = new RLConfig();
                heuristicConfig.setMode(RLModelMode.HEURISTIC_FALLBACK);
                heuristicConfig.setRecordTrajectories(false);

                try {
                    GameResult result;
                    if (rlFirst) {
                        result = runSingleGame(rlDeck, aiDeck, rlConfig, heuristicConfig,
                                "RL_" + gameIdx, "Heuristic_" + gameIdx, timeout, true);
                    } else {
                        result = runSingleGame(aiDeck, rlDeck, heuristicConfig, rlConfig,
                                "Heuristic_" + gameIdx, "RL_" + gameIdx, timeout, true);
                    }

                    consecutiveErrors.set(0);

                    if (result.isDraw) {
                        draws.incrementAndGet();
                    } else {
                        boolean rlWon = rlFirst ? result.player1Won : !result.player1Won;
                        if (rlWon) rlWins.incrementAndGet();
                        else heuristicWins.incrementAndGet();
                    }
                } catch (ModelServerException e) {
                    serverErrors.incrementAndGet();
                    int consec = consecutiveErrors.incrementAndGet();
                    System.out.printf("Game %d MODEL_SERVER_ERROR (%d/%d): %s%n",
                            gameIdx, consec, MAX_SERVER_ERRORS, e.getMessage());
                } catch (Exception e) {
                    System.out.printf("Game %d FAILED: %s%n", gameIdx, e.getMessage());
                }

                int done = completed.incrementAndGet();
                if (!quiet && (done % 10 == 0 || done == nGames)) {
                    int total = rlWins.get() + heuristicWins.get();
                    double winRate = total > 0 ? (double) rlWins.get() / total : 0;
                    long elapsed = System.currentTimeMillis() - startTime;
                    double gps = done * 1000.0 / elapsed;
                    System.out.printf("Game %d/%d: RL win rate: %.1f%% (%d/%d) [%.1f games/s]%n",
                            done, nGames, winRate * 100, rlWins.get(), total, gps);
                    System.out.flush();
                }
            }));
        }

        // Wait for all games
        for (java.util.concurrent.Future<?> f : futures) {
            try {
                f.get(timeout + 30, TimeUnit.SECONDS);
            } catch (Exception e) {
                // timeout or execution error
            }
        }
        executor.shutdown();

        int totalDecisive = rlWins.get() + heuristicWins.get();
        double winRate = totalDecisive > 0 ? (double) rlWins.get() / totalDecisive : 0;
        System.out.println();
        if (consecutiveErrors.get() >= MAX_SERVER_ERRORS) {
            System.out.println("=== Evaluation ABORTED (model server down) ===");
        } else {
            System.out.println("=== Evaluation Complete ===");
        }
        System.out.printf("RL Wins: %d (%.1f%%) | Heuristic Wins: %d | Draws: %d | Server Errors: %d%n",
                rlWins.get(), winRate * 100, heuristicWins.get(), draws.get(), serverErrors.get());
    }

    /**
     * Self-play mode: RL AI vs RL AI, multi-threaded with multi-port support.
     */
    private static void runSelfPlayMode(List<Deck> decks, int nGames, int timeout,
                                          String outputDir, boolean quiet,
                                          String grpcHost, int[] grpcPorts, int threads) {
        System.out.println("=== Self-Play Mode (" + threads + " threads, GRPC"
                + (grpcPorts.length > 1 ? " x" + grpcPorts.length + " servers" : "") + ") ===");

        AtomicInteger completed = new AtomicInteger(0);
        AtomicInteger serverErrors = new AtomicInteger(0);
        AtomicInteger consecutiveErrors = new AtomicInteger(0);
        final int MAX_SERVER_ERRORS = 3;
        long startTime = System.currentTimeMillis();

        java.util.concurrent.ExecutorService executor =
                java.util.concurrent.Executors.newFixedThreadPool(threads);
        List<java.util.concurrent.Future<?>> futures = new ArrayList<>();

        java.util.Random spDeckRng = new java.util.Random();
        for (int i = 0; i < nGames; i++) {
            final int gameIdx = i;
            final int assignedPort = grpcPorts[i % grpcPorts.length];
            final Deck deck1 = decks.get(spDeckRng.nextInt(decks.size()));
            final Deck deck2 = decks.get(spDeckRng.nextInt(decks.size()));
            final boolean p1First = (i % 2 == 0);

            futures.add(executor.submit(() -> {
                // Each player gets its own RLConfig to avoid contention
                RLConfig configA = new RLConfig();
                configA.setMode(RLModelMode.GRPC);
                configA.setGrpcHost(grpcHost);
                configA.setGrpcPort(assignedPort);
                configA.setRecordTrajectories(true);
                configA.setTrajectoryOutputDir(outputDir);

                RLConfig configB = new RLConfig();
                configB.setMode(RLModelMode.GRPC);
                configB.setGrpcHost(grpcHost);
                configB.setGrpcPort(assignedPort);
                configB.setRecordTrajectories(true);
                configB.setTrajectoryOutputDir(outputDir);

                try {
                    runSingleGame(deck1, deck2, configA, configB,
                            "RL_A_" + gameIdx, "RL_B_" + gameIdx, timeout, p1First);
                    consecutiveErrors.set(0);
                } catch (ModelServerException e) {
                    serverErrors.incrementAndGet();
                    int consec = consecutiveErrors.incrementAndGet();
                    System.out.printf("Game %d MODEL_SERVER_ERROR (%d/%d): %s%n",
                            gameIdx, consec, MAX_SERVER_ERRORS, e.getMessage());
                } catch (Exception e) {
                    System.out.printf("Game %d FAILED: %s%n", gameIdx, e.getMessage());
                }

                int done = completed.incrementAndGet();
                if (!quiet && done % 10 == 0) {
                    double elapsed = (System.currentTimeMillis() - startTime) / 1000.0;
                    System.out.printf("Game %d/%d [%.1f games/s]%n",
                            done, nGames, done / elapsed);
                }
            }));
        }

        // Wait for all games
        for (java.util.concurrent.Future<?> f : futures) {
            try {
                f.get(timeout + 30, TimeUnit.SECONDS);
            } catch (TimeoutException e) {
                f.cancel(true);
            } catch (Exception e) {
                // ignore
            }
        }
        executor.shutdown();

        System.out.printf("Self-play complete: %d games, %d server errors%n",
                completed.get(), serverErrors.get());
        System.out.println("Trajectories saved to: " + outputDir);
    }

    /**
     * League play mode: current model (port) vs opponent model (port2).
     * Records trajectories for the current model (player 1) only.
     * Usage: rltrain leagueplay -d deck1.dck -d deck2.dck -n 100 -port 50051 -port2 50052
     */
    private static void runLeaguePlayMode(List<Deck> decks, int nGames, int timeout,
                                           String outputDir, boolean quiet,
                                           String grpcHost, int[] currentPorts, int[] opponentPorts,
                                           int threads) {
        System.out.println("=== League Play Mode (" + threads + " threads, "
                + currentPorts.length + " current server(s), "
                + opponentPorts.length + " opponent server(s)) ===");

        AtomicInteger currentWins = new AtomicInteger(0);
        AtomicInteger opponentWins = new AtomicInteger(0);
        AtomicInteger draws = new AtomicInteger(0);
        AtomicInteger completed = new AtomicInteger(0);
        AtomicInteger serverErrors = new AtomicInteger(0);
        AtomicInteger consecutiveErrors = new AtomicInteger(0);
        final int MAX_SERVER_ERRORS = 3;
        long startTime = System.currentTimeMillis();

        java.util.concurrent.ExecutorService executor =
                java.util.concurrent.Executors.newFixedThreadPool(threads);
        List<java.util.concurrent.Future<?>> futures = new ArrayList<>();

        java.util.Random deckRng = new java.util.Random();
        for (int i = 0; i < nGames; i++) {
            final int gameIdx = i;
            final int curPort = currentPorts[i % currentPorts.length];
            final int oppPort = opponentPorts[i % opponentPorts.length];
            final Deck deck1 = decks.get(deckRng.nextInt(decks.size()));
            final Deck deck2 = decks.get(deckRng.nextInt(decks.size()));
            final boolean currentFirst = (i % 2 == 0);

            futures.add(executor.submit(() -> {
                // Current model — record trajectories
                RLConfig currentConfig = new RLConfig();
                currentConfig.setMode(RLModelMode.GRPC);
                currentConfig.setGrpcHost(grpcHost);
                currentConfig.setGrpcPort(curPort);
                currentConfig.setRecordTrajectories(true);
                currentConfig.setTrajectoryOutputDir(outputDir);

                // Opponent model — no trajectory recording
                RLConfig opponentConfig = new RLConfig();
                opponentConfig.setMode(RLModelMode.GRPC);
                opponentConfig.setGrpcHost(grpcHost);
                opponentConfig.setGrpcPort(oppPort);
                opponentConfig.setRecordTrajectories(false);

                try {
                    GameResult result;
                    if (currentFirst) {
                        result = runSingleGame(deck1, deck2, currentConfig, opponentConfig,
                                "Current_" + gameIdx, "Opponent_" + gameIdx, timeout, true);
                    } else {
                        result = runSingleGame(deck2, deck1, opponentConfig, currentConfig,
                                "Opponent_" + gameIdx, "Current_" + gameIdx, timeout, true);
                    }

                    consecutiveErrors.set(0);

                    if (result.isDraw) {
                        draws.incrementAndGet();
                    } else {
                        boolean currentWon = currentFirst ? result.player1Won : !result.player1Won;
                        if (currentWon) currentWins.incrementAndGet();
                        else opponentWins.incrementAndGet();
                    }
                } catch (ModelServerException e) {
                    serverErrors.incrementAndGet();
                    int consec = consecutiveErrors.incrementAndGet();
                    System.out.printf("Game %d MODEL_SERVER_ERROR (%d/%d): %s%n",
                            gameIdx, consec, MAX_SERVER_ERRORS, e.getMessage());
                } catch (Exception e) {
                    System.out.printf("Game %d FAILED: %s%n", gameIdx, e.getMessage());
                }

                int done = completed.incrementAndGet();
                if (!quiet && (done % 10 == 0 || done == nGames)) {
                    int total = currentWins.get() + opponentWins.get();
                    double winRate = total > 0 ? (double) currentWins.get() / total : 0;
                    long elapsed = System.currentTimeMillis() - startTime;
                    double gps = done * 1000.0 / elapsed;
                    System.out.printf("Game %d/%d: current win rate: %.1f%% (%d/%d) [%.1f games/s]%n",
                            done, nGames, winRate * 100, currentWins.get(), total, gps);
                    System.out.flush();
                }
            }));
        }

        for (java.util.concurrent.Future<?> f : futures) {
            try {
                f.get(timeout + 30, TimeUnit.SECONDS);
            } catch (TimeoutException e) {
                f.cancel(true);
            } catch (Exception e) {
                // ignore
            }
        }
        executor.shutdown();

        int totalDecisive = currentWins.get() + opponentWins.get();
        double winRate = totalDecisive > 0 ? (double) currentWins.get() / totalDecisive : 0;
        System.out.printf("League play complete: %d games, current win rate: %.1f%%, server errors: %d%n",
                completed.get(), winRate * 100, serverErrors.get());
    }

    /**
     * Head-to-head mode: two ONNX models play against each other.
     * Usage: rltrain headtohead -m1 path/to/model1 -m2 path/to/model2 -d deck1.dck -d deck2.dck -n 100
     */
    private static void runHeadToHeadMode(List<Deck> decks, int nGames, int timeout,
                                            String outputDir, boolean quiet, int threads,
                                            String onnxDir1, String onnxDir2) {
        System.out.println("=== Head-to-Head Mode ===");
        System.out.println("Model 1: " + onnxDir1);
        System.out.println("Model 2: " + onnxDir2);

        AtomicInteger m1Wins = new AtomicInteger(0);
        AtomicInteger m2Wins = new AtomicInteger(0);
        AtomicInteger draws = new AtomicInteger(0);
        AtomicInteger completed = new AtomicInteger(0);

        java.util.concurrent.ExecutorService executor =
                java.util.concurrent.Executors.newFixedThreadPool(threads);
        List<java.util.concurrent.Future<?>> futures = new ArrayList<>();

        java.util.Random deckRng = new java.util.Random();
        for (int i = 0; i < nGames; i++) {
            final int gameIdx = i;
            final Deck deck1 = decks.get(deckRng.nextInt(decks.size()));
            final Deck deck2 = decks.get(deckRng.nextInt(decks.size()));
            // Alternate who goes first
            final boolean m1First = (i % 2 == 0);

            futures.add(executor.submit(() -> {
                RLConfig config1 = new RLConfig();
                config1.setMode(RLModelMode.ONNX);
                config1.setOnnxModelDir(onnxDir1);
                config1.setRecordTrajectories(false);

                RLConfig config2 = new RLConfig();
                config2.setMode(RLModelMode.ONNX);
                config2.setOnnxModelDir(onnxDir2);
                config2.setRecordTrajectories(false);

                try {
                    GameResult result;
                    if (m1First) {
                        result = runSingleGame(deck1, deck2, config1, config2,
                                "M1_" + gameIdx, "M2_" + gameIdx, timeout, true);
                    } else {
                        result = runSingleGame(deck2, deck1, config2, config1,
                                "M2_" + gameIdx, "M1_" + gameIdx, timeout, true);
                    }

                    if (result.isDraw) {
                        draws.incrementAndGet();
                    } else {
                        boolean m1Won = m1First ? result.player1Won : !result.player1Won;
                        if (m1Won) m1Wins.incrementAndGet();
                        else m2Wins.incrementAndGet();
                    }
                } catch (Exception e) {
                    System.out.printf("Game %d FAILED: %s%n", gameIdx, e.getMessage());
                }

                int done = completed.incrementAndGet();
                if (!quiet && (done % 10 == 0 || done == nGames)) {
                    int w1 = m1Wins.get(), w2 = m2Wins.get();
                    int total = w1 + w2 + draws.get();
                    double pct = total > 0 ? 100.0 * w1 / total : 0;
                    System.out.printf("Game %d/%d: M1 win rate: %.1f%% (%d/%d)%n",
                            done, nGames, pct, w1, total);
                }
            }));
        }

        for (java.util.concurrent.Future<?> f : futures) {
            try { f.get(); } catch (Exception e) { /* ignore */ }
        }
        executor.shutdown();

        System.out.println();
        System.out.println("=== Head-to-Head Complete ===");
        System.out.printf("Model 1 Wins: %d (%.1f%%) | Model 2 Wins: %d (%.1f%%) | Draws: %d%n",
                m1Wins.get(), 100.0 * m1Wins.get() / Math.max(nGames, 1),
                m2Wins.get(), 100.0 * m2Wins.get() / Math.max(nGames, 1),
                draws.get());
    }

    private static void runReplayDiffMode(List<Deck> decks, int timeout,
                                          boolean quiet, String grpcHost,
                                          int grpcPort, boolean useOnnx,
                                          long seed, String replayId,
                                          String traceOut,
                                          String baselinePolicy,
                                          String comparePolicy,
                                          String onnxDir1) {
        if (decks.size() < 2) {
            System.out.println("Replay diff mode requires exactly two decks.");
            return;
        }
        String resolvedReplayId = replayId != null ? replayId
                : "replay_" + seed;
        Deck deck1 = decks.get(0);
        Deck deck2 = decks.get(1);

        System.out.println("=== Deterministic Replay Diff Mode ===");
        System.out.println("Replay ID: " + resolvedReplayId);
        System.out.println("Seed: " + seed);
        System.out.println("Baseline: " + baselinePolicy + " vs heuristic");
        System.out.println("Compare: " + comparePolicy + " vs heuristic");

        RLConfig baselineP1 = buildPolicyConfig(
                baselinePolicy, grpcHost, grpcPort, useOnnx, onnxDir1,
                buildReplayConfig(seed, resolvedReplayId, "baseline", traceOut));
        RLConfig baselineP2 = buildPolicyConfig(
                "heuristic", grpcHost, grpcPort, useOnnx, onnxDir1,
                buildReplayConfig(seed, resolvedReplayId, "baseline", traceOut));
        RLConfig compareP1 = buildPolicyConfig(
                comparePolicy, grpcHost, grpcPort, useOnnx, onnxDir1,
                buildReplayConfig(seed, resolvedReplayId, "compare", traceOut));
        RLConfig compareP2 = buildPolicyConfig(
                "heuristic", grpcHost, grpcPort, useOnnx, onnxDir1,
                buildReplayConfig(seed, resolvedReplayId, "compare", traceOut));

        GameResult baseline = runSingleGame(deck1, deck2, baselineP1, baselineP2,
                "PlayerA", "PlayerB", timeout, true);
        GameResult compare = runSingleGame(deck1, deck2, compareP1, compareP2,
                "PlayerA", "PlayerB", timeout, true);

        if (!quiet) {
            System.out.println("Baseline result: " + baseline.summary);
            System.out.println("Compare result: " + compare.summary);
            System.out.println("Replay traces saved to: " + traceOut);
        }
    }

    // ===== Core game execution =====

    private static GameResult runSingleGame(Deck deck1, Deck deck2,
                                              RLConfig config1, RLConfig config2,
                                              String name1, String name2,
                                              int timeoutSec, boolean p1First) {
        java.util.Random originalRandom = MyRandom.getRandom();
        ReplayConfig replay = config1.getReplayConfig() != null
                ? config1.getReplayConfig()
                : config2.getReplayConfig();
        if (replay != null) {
            ReplayRandom.startSession(replay.getSeed(), replay.getReplayId(),
                    replay.getRunLabel(), replay.isDeterministicPolicy(),
                    replay.isRngAuditEnabled());
            MyRandom.setRandom(new java.util.Random(replay.getSeed()));
        }

        try {
        // Create players
        LobbyPlayer lobby1 = createPlayer(name1, config1);
        LobbyPlayer lobby2 = createPlayer(name2, config2);

        RegisteredPlayer rp1 = new RegisteredPlayer(deck1);
        rp1.setPlayer(lobby1);
        RegisteredPlayer rp2 = new RegisteredPlayer(deck2);
        rp2.setPlayer(lobby2);

        List<RegisteredPlayer> players = p1First
                ? Arrays.asList(rp1, rp2) : Arrays.asList(rp2, rp1);

        GameRules rules = new GameRules(GameType.Constructed);
        rules.setAppliedVariants(EnumSet.of(GameType.Constructed));
        rules.setGamesPerMatch(1);
        rules.setSimTimeout(timeoutSec);

        Match match = new Match(rules, players, "RL Training");
        Game game = match.createGame();

        // Increase AI timeout for RL — evaluating all spell candidates
        // takes longer than the default 5s on complex boards
        game.AI_TIMEOUT = 15;

        // Store player refs before game (lost players get
        // removed from game.getPlayers() after game ends)
        Map<LobbyPlayer, Player> lobbyToPlayer = new HashMap<>();
        String gameId = name1 + "_vs_" + name2;
        String deck1Name = deck1 != null ? deck1.getName() : null;
        String deck2Name = deck2 != null ? deck2.getName() : null;
        for (Player p : game.getPlayers()) {
            lobbyToPlayer.put(p.getLobbyPlayer(), p);
            if (p.getController() instanceof PlayerControllerRL) {
                boolean isPlayerOnePerspective = name1.equals(p.getName());
                ((PlayerControllerRL) p.getController())
                    .getRLController()
                    .onGameStart(
                            gameId + "_" + p.getName(),
                            gameId,
                            p.getName(),
                            isPlayerOnePerspective ? name2 : name1,
                            isPlayerOnePerspective ? deck1Name : deck2Name,
                            isPlayerOnePerspective ? deck2Name : deck1Name,
                            p.getLobbyPlayer().equals(lobby1) ? 1 : 2);
            }
        }

        // Attach decision listeners and state recorders
        // Uses plain LobbyPlayerAi (subclassing breaks game)
        // so we create recorders directly here
        Map<Player, forge.ai.rl.training.TrajectoryRecorder>
                recorders = new HashMap<>();
        RLConfig anyConfig = config1.isRecordTrajectories()
                ? config1 : config2;

        // Combat recording is handled by PlayerControllerRL
        // (captures pre-decision state + heuristic choice).
        // No GameStateRecorder needed — avoids post-decision
        // state encoding issues (tapped flag leak etc.).

        // Run the game with timeout
        try {
            TimeLimitedCodeBlock.runWithTimeout(() -> {
                match.startGame(game);
            }, timeoutSec, TimeUnit.SECONDS);
        } catch (TimeoutException e) {
            if (!game.isGameOver()) {
                game.setGameOver(GameEndReason.Draw);
            }
        } catch (ModelServerException e) {
            throw e;
        } catch (Exception | StackOverflowError e) {
            // Check if a ModelServerException is wrapped inside
            Throwable cause = e.getCause();
            while (cause != null) {
                if (cause instanceof ModelServerException) {
                    throw (ModelServerException) cause;
                }
                cause = cause.getCause();
            }
            if (!game.isGameOver()) {
                game.setGameOver(GameEndReason.Draw);
            }
        }

        // Finalize RLController trajectory recorders
        for (Player p : lobbyToPlayer.values()) {
            if (p.getController() instanceof PlayerControllerRL) {
                RegisteredPlayer rp = p.getRegisteredPlayer();
                boolean won = game.getOutcome() != null
                        && !game.getOutcome().isDraw()
                        && rp != null
                        && game.getOutcome().isWinner(rp);
                ((PlayerControllerRL) p.getController())
                    .getRLController()
                    .onGameEnd(won);
            }
        }

        // Build result
        GameResult result = new GameResult();
        if (game.getOutcome() == null || game.getOutcome().isDraw()) {
            result.isDraw = true;
            result.summary = "Draw";
        } else {
            LobbyPlayer winner = game.getOutcome().getWinningLobbyPlayer();
            result.player1Won = winner.equals(lobby1);
            result.summary = winner.getName() + " won (turn " + game.getPhaseHandler().getTurn() + ")";
        }
        result.turns = game.getPhaseHandler().getTurn();

        return result;
        } finally {
            MyRandom.setRandom(originalRandom);
            ReplayRandom.endSession();
        }
    }

    private static LobbyPlayer createPlayer(String name, RLConfig config) {
        if (config.getMode() == RLModelMode.HEURISTIC_FALLBACK) {
            LobbyPlayerAi aiPlayer;
            aiPlayer = new LobbyPlayerAi(name, null);
            aiPlayer.setAiProfile("Default");
            return aiPlayer;
        } else {
            // RL mode — use our RL controller
            return new LobbyPlayerRL(name, config);
        }
    }

    private static Deck loadDeck(String deckParam) {
        // Check if it's a file path
        int dotpos = deckParam.lastIndexOf('.');
        if (dotpos > 0 && dotpos == deckParam.length() - 4) {
            // Try as absolute path first
            File f = new File(deckParam);
            if (f.exists()) {
                return DeckSerializer.fromFile(f);
            }
            // Try in constructed deck directory
            f = new File(ForgeConstants.DECK_CONSTRUCTED_DIR + deckParam);
            if (f.exists()) {
                return DeckSerializer.fromFile(f);
            }
        }

        // Try by name in deck storage
        try {
            return FModel.getDecks().getConstructed().get(deckParam);
        } catch (Exception e) {
            return null;
        }
    }

    private static void printHelp() {
        System.out.println();
        System.out.println("Usage: forge rltrain <mode> [options]");
        System.out.println();
        System.out.println("Modes:");
        System.out.println("  collect   - Record heuristic AI games for imitation learning");
        System.out.println("  evaluate  - Evaluate RL AI against heuristic AI");
        System.out.println("  selfplay  - Run RL AI self-play games");
        System.out.println("  replaydiff - Run deterministic baseline/compare games with paired traces");
        System.out.println();
        System.out.println("Options:");
        System.out.println("  -d <deck>     Deck file (.dck) or deck name (repeat for multiple)");
        System.out.println("  -D <dir>      Load all .dck files from directory");
        System.out.println("  -n <N>        Number of games (default: 100)");
        System.out.println("  -o <dir>      Output directory for trajectories (default: rl_data/trajectories)");
        System.out.println("  -t <threads>  Parallel threads (default: all CPUs)");
        System.out.println("  -c <seconds>  Timeout per game (default: 180)");
        System.out.println("  -host <host>  Model server host (default: localhost)");
        System.out.println("  -port <port>  Model server port (default: 50051)");
        System.out.println("  -seed <long>  Seed for deterministic replay");
        System.out.println("  -replay-id <id> Replay identifier for paired trace files");
        System.out.println("  -trace-out <dir> Replay trace output directory");
        System.out.println("  -deterministic Enable deterministic replay mode");
        System.out.println("  -zero-intermediate-reward Zero stored intermediateReward values in trajectory JSONL");
        System.out.println("  -baseline <heuristic|rl> Baseline player-one policy for replaydiff");
        System.out.println("  -compare <heuristic|rl> Compare player-one policy for replaydiff");
        System.out.println("  -q            Quiet mode");
    }

    private static ReplayConfig buildReplayConfig(long seed, String replayId,
                                                  String runLabel,
                                                  String traceOut) {
        return new ReplayConfig(seed, replayId, runLabel,
                true, true, traceOut);
    }

    private static RLConfig buildPolicyConfig(String policyMode,
                                              String grpcHost,
                                              int grpcPort,
                                              boolean useOnnx,
                                              String onnxDir,
                                              ReplayConfig replayConfig) {
        RLConfig config = new RLConfig();
        config.setReplayConfig(replayConfig);
        if ("rl".equalsIgnoreCase(policyMode)) {
            if (useOnnx) {
                config.setMode(RLModelMode.ONNX);
                if (onnxDir != null) {
                    config.setOnnxModelDir(onnxDir);
                }
            } else {
                config.setMode(RLModelMode.GRPC);
                config.setGrpcHost(grpcHost);
                config.setGrpcPort(grpcPort);
            }
        } else {
            config.setMode(RLModelMode.RECORD_HEURISTIC);
        }
        return config;
    }

    private static float normalize(double v, double min, double max) {
        if (max <= min) return 0f;
        return (float) Math.max(0, Math.min(1, (v - min) / (max - min)));
    }

    private static class GameResult {
        boolean player1Won = false;
        boolean isDraw = false;
        int turns = 0;
        String summary = "";
    }
}
