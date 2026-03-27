package forge.ai.rl.training;

import forge.ai.LobbyPlayerAi;
import forge.ai.rl.LobbyPlayerRL;
import forge.ai.rl.PlayerControllerRL;
import forge.ai.rl.RLConfig;
import forge.ai.rl.RLModelMode;
import forge.deck.Deck;
import forge.game.*;
import forge.game.player.Player;
import forge.game.player.RegisteredPlayer;
import org.tinylog.Logger;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Runs AI vs AI games at scale for training data collection and evaluation.
 *
 * Supports:
 * - RL vs Heuristic AI (evaluation)
 * - RL vs RL (self-play)
 * - Heuristic vs Heuristic (imitation learning data collection)
 * - Parallel game execution
 * - Game statistics tracking
 */
public class GameRunner {

    private final RLConfig config;
    private final AtomicInteger gamesCompleted = new AtomicInteger(0);
    private final AtomicInteger rlWins = new AtomicInteger(0);
    private final AtomicInteger heuristicWins = new AtomicInteger(0);
    private final AtomicInteger draws = new AtomicInteger(0);
    private final AtomicInteger errors = new AtomicInteger(0);

    public GameRunner(RLConfig config) {
        this.config = config;
    }

    /**
     * Run a batch of RL vs Heuristic AI games.
     *
     * @param rlDeck     deck for the RL player
     * @param aiDeck     deck for the heuristic AI
     * @param numGames   number of games to play
     * @param threads    number of parallel games
     * @return summary statistics
     */
    public GameBatchResult runRLvsHeuristic(Deck rlDeck, Deck aiDeck, int numGames, int threads) {
        ExecutorService executor = Executors.newFixedThreadPool(threads);
        List<Future<SingleGameResult>> futures = new ArrayList<>();

        for (int i = 0; i < numGames; i++) {
            final int gameIndex = i;
            // Alternate who goes first
            final boolean rlGoesFirst = (i % 2 == 0);
            futures.add(executor.submit(() -> runSingleGame(rlDeck, aiDeck, gameIndex, rlGoesFirst)));
        }

        List<SingleGameResult> results = new ArrayList<>();
        for (Future<SingleGameResult> f : futures) {
            try {
                results.add(f.get(300, TimeUnit.SECONDS)); // 5 min timeout per game
            } catch (Exception e) {
                Logger.error("Game execution failed: {}", e.getMessage());
                errors.incrementAndGet();
            }
        }

        executor.shutdown();
        return summarize(results);
    }

    /**
     * Run heuristic vs heuristic games for imitation learning data collection.
     * Both sides record trajectories.
     */
    public GameBatchResult runImitationGames(Deck deck1, Deck deck2, int numGames, int threads) {
        RLConfig recordConfig = new RLConfig();
        recordConfig.setMode(RLModelMode.RECORD_HEURISTIC);
        recordConfig.setRecordTrajectories(true);
        recordConfig.setTrajectoryOutputDir(config.getTrajectoryOutputDir());

        ExecutorService executor = Executors.newFixedThreadPool(threads);
        List<Future<SingleGameResult>> futures = new ArrayList<>();

        for (int i = 0; i < numGames; i++) {
            final int gameIndex = i;
            futures.add(executor.submit(() -> {
                try {
                    return runRecordedGame(deck1, deck2, gameIndex, recordConfig);
                } catch (Exception e) {
                    Logger.error("Imitation game {} failed: {}", gameIndex, e.getMessage());
                    errors.incrementAndGet();
                    return new SingleGameResult(gameIndex, false, false, 0, "ERROR: " + e.getMessage());
                }
            }));
        }

        List<SingleGameResult> results = new ArrayList<>();
        for (Future<SingleGameResult> f : futures) {
            try {
                results.add(f.get(300, TimeUnit.SECONDS));
            } catch (Exception e) {
                Logger.error("Imitation game failed: {}", e.getMessage());
                errors.incrementAndGet();
            }
        }

        executor.shutdown();
        return summarize(results);
    }

    private SingleGameResult runSingleGame(Deck rlDeck, Deck aiDeck, int gameIndex, boolean rlGoesFirst) {
        long startTime = System.currentTimeMillis();
        String gameId = "game_" + gameIndex + "_" + startTime;

        try {
            // Create players
            LobbyPlayerRL rlLobby = new LobbyPlayerRL("RL_Player", config);
            LobbyPlayerAi aiLobby = new LobbyPlayerAi("Heuristic_AI", null);

            RegisteredPlayer rlReg = new RegisteredPlayer(rlDeck);
            rlReg.setPlayer(rlLobby);
            RegisteredPlayer aiReg = new RegisteredPlayer(aiDeck);
            aiReg.setPlayer(aiLobby);

            List<RegisteredPlayer> players;
            if (rlGoesFirst) {
                players = List.of(rlReg, aiReg);
            } else {
                players = List.of(aiReg, rlReg);
            }

            GameRules rules = new GameRules(GameType.Constructed);
            rules.setPlayForAnte(false);
            Match match = new Match(rules, players, "RL Training Game " + gameIndex);

            // Run game
            Game game = match.createGame();

            // Get RL player controller and notify game start
            Player rlPlayer = null;
            for (Player p : game.getPlayers()) {
                if (p.getController() instanceof PlayerControllerRL) {
                    rlPlayer = p;
                    boolean isRlPerspective = "RL_Player".equals(p.getName());
                    ((PlayerControllerRL) p.getController()).getRLController().onGameStart(
                            gameId,
                            gameId,
                            p.getName(),
                            isRlPerspective ? "Heuristic_AI" : "RL_Player",
                            isRlPerspective && rlDeck != null ? rlDeck.getName() : aiDeck != null ? aiDeck.getName() : null,
                            isRlPerspective && aiDeck != null ? aiDeck.getName() : rlDeck != null ? rlDeck.getName() : null);
                    break;
                }
            }

            match.startGame(game);

            // Determine outcome
            boolean rlWon = false;
            if (game.getOutcome() != null && rlPlayer != null) {
                rlWon = game.getOutcome().isWinner(rlPlayer.getRegisteredPlayer());
            }

            // Notify RL controller of game end
            if (rlPlayer != null && rlPlayer.getController() instanceof PlayerControllerRL) {
                PlayerControllerRL ctrl = (PlayerControllerRL) rlPlayer.getController();
                ctrl.getRLController().onGameEnd(rlWon);
                ctrl.logDiagnostics();
            }

            long duration = System.currentTimeMillis() - startTime;
            if (rlWon) rlWins.incrementAndGet();
            else heuristicWins.incrementAndGet();
            gamesCompleted.incrementAndGet();

            return new SingleGameResult(gameIndex, rlWon, true, duration,
                    String.format("RL %s (turn %d)", rlWon ? "WIN" : "LOSS", game.getPhaseHandler().getTurn()));

        } catch (Exception e) {
            Logger.error("Game {} failed: {}", gameIndex, e.getMessage());
            errors.incrementAndGet();
            return new SingleGameResult(gameIndex, false, false, System.currentTimeMillis() - startTime,
                    "ERROR: " + e.getMessage());
        }
    }

    private SingleGameResult runRecordedGame(Deck deck1, Deck deck2, int gameIndex, RLConfig recordConfig) {
        long startTime = System.currentTimeMillis();
        String gameId = "imitation_" + gameIndex + "_" + startTime;

        // Both players use RL controller in RECORD_HEURISTIC mode
        LobbyPlayerRL p1Lobby = new LobbyPlayerRL("Player1", recordConfig);
        LobbyPlayerRL p2Lobby = new LobbyPlayerRL("Player2", recordConfig);

        RegisteredPlayer p1Reg = new RegisteredPlayer(deck1);
        p1Reg.setPlayer(p1Lobby);
        RegisteredPlayer p2Reg = new RegisteredPlayer(deck2);
        p2Reg.setPlayer(p2Lobby);

        GameRules rules = new GameRules(GameType.Constructed);
        rules.setPlayForAnte(false);
        Match match = new Match(rules, List.of(p1Reg, p2Reg), "Imitation Game " + gameIndex);

        Game game = match.createGame();

        // Notify both players
        for (Player p : game.getPlayers()) {
            if (p.getController() instanceof PlayerControllerRL) {
                boolean isPlayerOnePerspective = "Player1".equals(p.getName());
                ((PlayerControllerRL) p.getController()).getRLController().onGameStart(
                        gameId + "_" + p.getName(),
                        gameId,
                        p.getName(),
                        isPlayerOnePerspective ? "Player2" : "Player1",
                        isPlayerOnePerspective && deck1 != null ? deck1.getName() : deck2 != null ? deck2.getName() : null,
                        isPlayerOnePerspective && deck2 != null ? deck2.getName() : deck1 != null ? deck1.getName() : null);
            }
        }

        match.startGame(game);

        // Notify end
        for (Player p : game.getPlayers()) {
            if (p.getController() instanceof PlayerControllerRL) {
                boolean won = game.getOutcome() != null && game.getOutcome().isWinner(p.getRegisteredPlayer());
                ((PlayerControllerRL) p.getController()).getRLController().onGameEnd(won);
            }
        }

        long duration = System.currentTimeMillis() - startTime;
        gamesCompleted.incrementAndGet();
        return new SingleGameResult(gameIndex, false, true, duration, "Imitation game completed");
    }

    private GameBatchResult summarize(List<SingleGameResult> results) {
        GameBatchResult summary = new GameBatchResult();
        summary.totalGames = results.size();
        summary.rlWins = (int) results.stream().filter(r -> r.rlWon && r.completed).count();
        summary.heuristicWins = (int) results.stream().filter(r -> !r.rlWon && r.completed).count();
        summary.errors = (int) results.stream().filter(r -> !r.completed).count();
        summary.avgDurationMs = results.stream().filter(r -> r.completed).mapToLong(r -> r.durationMs).average().orElse(0);
        summary.rlWinRate = summary.totalGames > 0
                ? (double) summary.rlWins / (summary.rlWins + summary.heuristicWins) : 0;
        return summary;
    }

    public int getGamesCompleted() { return gamesCompleted.get(); }
    public int getRLWins() { return rlWins.get(); }
    public int getHeuristicWins() { return heuristicWins.get(); }

    // Result data classes

    public static class SingleGameResult {
        public final int gameIndex;
        public final boolean rlWon;
        public final boolean completed;
        public final long durationMs;
        public final String summary;

        public SingleGameResult(int gameIndex, boolean rlWon, boolean completed, long durationMs, String summary) {
            this.gameIndex = gameIndex;
            this.rlWon = rlWon;
            this.completed = completed;
            this.durationMs = durationMs;
            this.summary = summary;
        }
    }

    public static class GameBatchResult {
        public int totalGames;
        public int rlWins;
        public int heuristicWins;
        public int errors;
        public double avgDurationMs;
        public double rlWinRate;

        @Override
        public String toString() {
            return String.format("Games: %d | RL Wins: %d (%.1f%%) | Heuristic Wins: %d | Errors: %d | Avg Duration: %.0fms",
                    totalGames, rlWins, rlWinRate * 100, heuristicWins, errors, avgDurationMs);
        }
    }
}
