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
import com.google.common.collect.Lists;
import forge.ai.anvil.AnvilOptions;
import forge.ai.anvil.Census;
import forge.game.spellability.SpellAbility;
import forge.ai.anvil.ChoiceDirective;
import forge.ai.anvil.LocalRandomBridge;
import forge.ai.anvil.Obs;
import forge.ai.anvil.SurfaceDirective;
import forge.ai.anvil.Surfaces;
import forge.ai.anvil.PlayerControllerAnvil;
import forge.ai.anvil.ScheduleDirective;
import forge.ai.anvil.SearchDirective;
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
    /** Exactly one registered player with hasWon() -> its index; else -1
     *  (a forced Draw marks every survivor as won). Shared by the mainline
     *  status and the rollout drivers (ADR-0102). */
    static int uniqueWinner(Game g) {
        int winner = -1;
        int nWon = 0;
        if (g.getOutcome() != null) {
            for (int j = 0; j < g.getRegisteredPlayers().size(); j++) {
                forge.game.player.PlayerOutcome po = g.getRegisteredPlayers().get(j).getOutcome();
                if (po != null && po.hasWon()) {
                    winner = j;
                    nWon++;
                }
            }
            if (nWon != 1) {
                winner = -1;
            }
        }
        return winner;
    }

    private static final int DRAW_CLOCK_S = 300;
    /** ADR-0102 deterministic caps (see the rules setup below). */
    static final int DEFAULT_TURN_CAP = 52;
    static final int DEFAULT_WINDOW_CAP = 1650;
    private static final int GAME_HARD_CAP_S = 360;

    private static final Set<String> DEFAULT_TAGS = new HashSet<>(Arrays.asList(
            PlayerControllerAnvil.TAG_PRIORITY, PlayerControllerAnvil.TAG_MULLIGAN,
            PlayerControllerAnvil.TAG_TUCK, PlayerControllerAnvil.TAG_TRIGGER,
            PlayerControllerAnvil.TAG_BINARY, PlayerControllerAnvil.TAG_NUMBER,
            PlayerControllerAnvil.TAG_SURFACE_ONE, PlayerControllerAnvil.TAG_SURFACE_SET,
            PlayerControllerAnvil.TAG_SURFACE_MODE, PlayerControllerAnvil.TAG_SURFACE_ORDER,
            PlayerControllerAnvil.TAG_SURFACE_DAMAGE, PlayerControllerAnvil.TAG_SURFACE_TARGET));

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
        // M12 Build 3 (ADR-0105): -abilities <names.txt> <out.jsonl> — every
        // pool card's abilities as {h, host, kind, txt} through AbilityKey
        // (the same canonical text the observation keys), no game played.
        if (params != null && params.containsKey("abilities") && params.get("abilities").size() == 2) {
            dumpAbilities(params.get("abilities").get(0), params.get("abilities").get(1));
            return;
        }
        boolean fixedPair = params != null && params.containsKey("d") && params.get("d").size() == 2;
        boolean pairFile = params != null && params.containsKey("pairs");
        if (params == null || (fixedPair == pairFile && !params.containsKey("replay"))) {
            System.out.println("Syntax: forge anvil (-d <deck1> <deck2> | -pairs <file> [-gpp <n>]) [-f <format>] "
                    + "[-b local-random|grpc:host:port] [-tags <csv>] [-bridgeseats <csv>] [-reask] "
                    + "[-census <out.jsonl>] [-obs <out.zst>] [-paytelemetry] "
                    + "[-range <start> <count> -seedbase <long> [-results <jsonl>] [-stopfile <path>]] "
                    + "[-rollout <k> -points <m> -labels <jsonl> [-noreshuffle]] "
                    + "[-drillfile <txt> [-drillstop]] [-forkobs] [-forcebranch] [-forceseq <n>] "
                    + "[-forcecandidate <tsv>] "
                    + "[-seqarms nat|all] [-forceschedule <tsv>] [-forcechoice <tsv>] "
                    + "[-certify <horizon>] [-n <games>] [-s <baseSeed>] "
                    + "[-turncap <n>] [-windowcap <n>] [-pool <id>] [-forkcommit <hash>] "
                    + "[-search [-searchrate <p>] [-searchrolls <k>] [-searchopts <cap>] [-searchmana] "
                    + "[-searchsurf <B> [-searchsurfcap <C>]] [-searchact <bar> [-searchtemp <T>]] "
                    + "[-searchseats <csv>] [-searchactkinds <csv|all>] [-searchleaf next|eot|h<N>|end] [-searchrollsalt <long>] [-searchpay <B> [-searchpayleaf eot|next|h<N>|end] [-searchpaybridge]] [-searchdeep <B> [-searchdeepleaf eot|h<N>|end] [-searchdeeprolls <k>] [-searchdeeplo <m>] [-searchdeepfloor <p>] [-searchdeepbar <bar>]] [-searchclock <s>] [-searchvoidrescue] [-searchalloc <tau> [-searchfloor <p>]]] "
                    + "[-payrescue] [-targetmask legal-plans] [-replay <jobs.jsonl>]");
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
        // Provenance on the obs game header + the bridge hello (ADR-0102 item 4).
        if (params.containsKey("pool")) {
            Obs.poolId = params.get("pool").get(0);
        }
        String forkCommit = params.containsKey("forkcommit") ? params.get("forkcommit").get(0) : "";
        if (!forkCommit.isEmpty()) {
            Obs.forkCommit = forkCommit;
        }
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
        // M12 Build 0 (ADR-0101 §1 / ADR-0102): the one-ply search instrument.
        // At the active bridged seat's quiescent MAIN1/MAIN2 windows (rate-
        // sampled, seeded), every option incl. pass is applied on a
        // determinized copy, played by the network to the seat's next
        // quiescent window and valued by anvil.value; one labels row per
        // searched window (per-option leaf values, forward calls, leaf kind)
        // completed with the natural pick. Telemetry only — never acts.
        final boolean search = params.containsKey("search");
        final double searchRate = params.containsKey("searchrate")
                ? Double.parseDouble(params.get("searchrate").get(0)) : 1.0;
        final int searchRolls = params.containsKey("searchrolls")
                ? Integer.parseInt(params.get("searchrolls").get(0)) : 1;
        final int searchOpts = params.containsKey("searchopts")
                ? Integer.parseInt(params.get("searchopts").get(0)) : 0;
        // Pure mana abilities stay in the MASK (legal actions) but are not
        // search candidates by default: the first smoke (09-06) spent 58% of
        // its candidates on "{T}: Add {B}"-class activations whose leaf is the
        // pass leaf with the seat tapped down (13% void). -searchmana restores
        // them for comparison.
        final boolean searchMana = params.containsKey("searchmana");
        // M12 Build 3 (Surfaces): -searchsurf B expands the FIRST traced
        // surface callback (tutor/discard/order/scry/mode/name/damage) on the
        // top-B first-ply candidates' paths — one copy per enumerated answer
        // (the natural answer is the first-ply copy under CRN). 0 = off.
        final int searchSurf = params.containsKey("searchsurf")
                ? Integer.parseInt(params.get("searchsurf").get(0)) : 0;
        final int searchSurfCap = params.containsKey("searchsurfcap")
                ? Integer.parseInt(params.get("searchsurfcap").get(0)) : Surfaces.DEFAULT_CAP;
        // M12 Build 3 evening 4 (ADR-0105): -searchpay B expands the first
        // traced PAYMENT window on the top-B candidates' paths — its own slot
        // beside the surface slot (a payment fires at cast, before any other
        // surface, and would otherwise crowd them out) — one copy per goal
        // option (auto = answer 0), under the END-OF-TURN leaf by default
        // (-searchpayleaf eot: a payment's consequence is what stayed untapped
        // for the rest of the turn and the opponent's; next = fork A's leaf,
        // the calibration comparison). -searchpaybridge lets copies bridge the
        // pay tag as their natural line (default off: the copy-side gate).
        final int searchPay = params.containsKey("searchpay")
                ? Integer.parseInt(params.get("searchpay").get(0)) : 0;
        final String searchPayLeaf = params.containsKey("searchpayleaf")
                ? params.get("searchpayleaf").get(0) : "eot";
        // The payment target's calibration (ADR-0105 evening 4 close): the
        // pay slot's leaf family — next (fork A's leaf), eot (= h0: the seat's
        // first quiescent window of a later turn), h<N> (of a turn > t+N, the
        // certify horizon), end (natural game end: the outcome is the value,
        // no head call). Each pay answer copy under a horizon leaf also
        // snapshots the certify axes at its stop (the sub row's "snap").
        // Search-copy / recording only; the mainline is untouched.
        if (SearchMonitor.payLeafHorizon(searchPayLeaf) == Integer.MIN_VALUE) {
            throw new IllegalArgumentException("-searchpayleaf eot|next|h<N>|end");
        }
        // Evening 5 (ADR-0106 C1): the PRIORITY slot's leaf — the same family,
        // on the first-ply candidate copies (and the surface slot's answer
        // copies, which share the first ply's leaf so the natural answer stays
        // the first-ply copy under CRN). Default next = fork A's leaf (every
        // read so far). Under a horizon leaf every first-ply copy snapshots
        // the certify axes at its stop (the opts row's per-roll "snap") — the
        // priority-slot calibration read's rollout side. Search-copy /
        // recording only; the mainline is untouched.
        final String searchLeaf = params.containsKey("searchleaf")
                ? params.get("searchleaf").get(0) : "next";
        if (SearchMonitor.payLeafHorizon(searchLeaf) == Integer.MIN_VALUE) {
            throw new IllegalArgumentException("-searchleaf next|eot|h<N>|end");
        }
        // -searchclock <s>: the wall-clock allowance under search (the crash
        // guard around a searched game; 900 s since the 09-07 loop game). A
        // rollout leaf (h<N> / end) plays copies to a far horizon, so its
        // runs raise it explicitly.
        // Evening 5 (ADR-0106 C1 follow-up): -searchrollsalt <long> salts the
        // per-roll determinization seed of every copy. The calibration arms
        // share roll seeds (CRN), so a deeper leaf's copy and the end copy of
        // the same roll share their first turns and the deeper leaf's agreement
        // with the outcome is optimistic by construction; an end arm under a
        // salt is the de-confounded verdict. The rate draw (which windows are
        // searched) is unsalted, so the arms still join per window.
        final long searchRollSalt = params.containsKey("searchrollsalt")
                ? Long.parseLong(params.get("searchrollsalt").get(0)) : 0L;
        final int searchClock = params.containsKey("searchclock")
                ? Integer.parseInt(params.get("searchclock").get(0)) : 900;
        // The partial-expansion slot (ADR-0106 C3): -searchdeep B re-expands
        // the natural pick + the top-B first-ply candidates (by their lifted
        // value) to a DEEPER leaf (-searchdeepleaf, default h2 — C1's deep
        // arm) at -searchdeeprolls (default 4: h2's roll σ is ≈ 8× next's)
        // under CRN, and the option stage then decides on the deep values
        // over that set (the rest pruned) under -searchdeepbar (default the
        // acting bar). Gated on the shallow margin: the deep round runs where
        // max V − V(natural) lies in [-searchdeeplo, bar) — the first ply
        // sees something but not enough to act on (default 0.02 ≈ two roll
        // σ at next; 11% of heuristic windows, more on network arms) — plus
        // a seeded floor draw at -searchdeepfloor on every other window
        // (default 0.1: fork L's labels need every shape on ungated
        // windows). It runs at decide time (after the controller's ask), so
        // the natural is always in the deep set; the deep copies play the
        // (option, lifted answer) pair the shallow stage proposed. Requires
        // -searchact. Search-copy / recording only; the mainline is
        // untouched. Measured, not assumed: an h2 copy is ≈ 10× a next copy
        // on the calibration arms (09-15).
        final int searchDeep = params.containsKey("searchdeep")
                ? Integer.parseInt(params.get("searchdeep").get(0)) : 0;
        final String searchDeepLeaf = params.containsKey("searchdeepleaf")
                ? params.get("searchdeepleaf").get(0) : "h2";
        if (SearchMonitor.payLeafHorizon(searchDeepLeaf) == Integer.MIN_VALUE || "next".equals(searchDeepLeaf)) {
            throw new IllegalArgumentException("-searchdeepleaf eot|h<N>|end");
        }
        final int searchDeepRolls = params.containsKey("searchdeeprolls")
                ? Integer.parseInt(params.get("searchdeeprolls").get(0)) : 4;
        final double searchDeepLo = params.containsKey("searchdeeplo")
                ? Double.parseDouble(params.get("searchdeeplo").get(0)) : 0.02;
        final double searchDeepFloor = params.containsKey("searchdeepfloor")
                ? Double.parseDouble(params.get("searchdeepfloor").get(0)) : 0.1;
        final double searchDeepBar = params.containsKey("searchdeepbar")
                ? Double.parseDouble(params.get("searchdeepbar").get(0)) : Double.NaN;
        final boolean searchPayBridge = params.containsKey("searchpaybridge");
        PlayerControllerAnvil.copyPayBridge = searchPayBridge;
        // Build 4 (ADR-0109 item 2, 09-17): THE ALLOCATION HEAD. At every
        // candidate window (after the -searchrate draw) an "anvil.alloc" ask
        // on the value-ask wire (observation = Obs.peekPriority, the leaf
        // value's own record shape) returns the head's P(the search would act
        // here: its margin >= the acting bar). The window is searched where
        // p >= tau, or on a seeded uniform-floor draw at -searchfloor (ungated
        // windows keep producing labels for the head: the self-sealing
        // hazard), or where the ask is unserved (NaN: the uniform rate, as
        // before). Skipped windows leave an "alloc" row (p, by: skip) so the
        // allocation is auditable. Off (NaN) = the rate draw alone: the
        // mainline is byte-identical (no ask, no scan).
        final double searchAlloc = params.containsKey("searchalloc")
                ? Double.parseDouble(params.get("searchalloc").get(0)) : Double.NaN;
        final double searchFloor = params.containsKey("searchfloor")
                ? Double.parseDouble(params.get("searchfloor").get(0)) : 0.1;
        // ADR-0114 (the void-rescue instrument): -searchvoidrescue — a first-ply
        // candidate whose roll-0 copy voided gets ONE more copy on the same roll
        // seed with the forced option realized by the heuristic's planner; its
        // leaf value, the plan and the void's reason are recorded on the row's
        // option ("vr", "h") and never enter the acting rule. Off = byte-identical.
        SearchMonitor.VOID_RESCUE = params.containsKey("searchvoidrescue");
        // Build 4 (09-17): -vetofallback heuristic — the model's vetoed pick realized by the heuristic's planner
        PlayerControllerAnvil.vetoFallbackHeuristic = params.containsKey("vetofallback")
                && !params.get("vetofallback").isEmpty() && "heuristic".equals(params.get("vetofallback").get(0));
        // Build 4 (ADR-0109): -modegate off lifts the mode playability gate (the targets read's third arm)
        forge.ai.anvil.Surfaces.modeGate = !(params.containsKey("modegate") && !params.get("modegate").isEmpty()
                && "off".equals(params.get("modegate").get(0)));
        // ADR-0110 (the 09-16 merge): -searchvoidskip 0|1 (default 1) — a first-ply
        // candidate whose roll-0 copy voided is not re-rolled (kind "skip").
        SearchMonitor.VOID_SKIP = !params.containsKey("searchvoidskip")
                || !"0".equals(params.get("searchvoidskip").get(0));
        // Evening 4 (ADR-0105): the ADR-0102 rescue class admitted + paid
        // directed (AnvilOptions.PAYRESCUE); a game-path change under the
        // flag only, on every header as provenance.
        if (params.containsKey("payrescue")) {
            forge.ai.anvil.AnvilOptions.PAYRESCUE = true;
            Obs.payRescue = true;
        }
        if (params.containsKey("targetmask")) {
            List<String> values = params.get("targetmask");
            if (values.isEmpty() || !"legal-plans".equals(values.get(0))) {
                throw new IllegalArgumentException("-targetmask legal-plans");
            }
            Obs.targetPlanMask = true;
        }
        // M12 Build 2 (m12-plan canonical shape §2): the ACTING rule. -searchact
        // <bar> turns the instrument into the behavior policy: at a searched
        // window with margin = max V − V(natural) ≥ bar the controller samples
        // the option from the leaf-value softmax at -searchtemp <T> (0 =
        // argmax; default 0.025 — an option one 0.05-bar below the best keeps
        // ~13% weight), below the bar the natural pick stands. Absent = the
        // Build 0 telemetry-only instrument. -searchseats <csv> names the
        // searched seats explicitly (default: every bridged seat); a heuristic
        // seat named there is searched and acted for — the control arm.
        final double searchAct = params.containsKey("searchact")
                ? Double.parseDouble(params.get("searchact").get(0)) : Double.NaN;
        final double searchTemp = params.containsKey("searchtemp")
                ? Double.parseDouble(params.get("searchtemp").get(0)) : 0.025;
        if (searchDeep > 0 && Double.isNaN(searchAct)) {
            throw new IllegalArgumentException("-searchdeep needs -searchact (the deep round runs at decide time)");
        }
        // Evening 5 (ADR-0106 A): -searchactkinds <csv|all> turns the acting
        // rule's second stage on for these surface kinds: on the acted option
        // the second round's answers (-searchsurf B) are sampled the same way
        // the option is (bar / temp), a non-natural sample armed on the
        // mainline for the action's callback (SurfaceDirective.armMainline).
        // Absent = options only (Build 2). Modes first; pay is never acted here.
        final String searchActKinds = params.containsKey("searchactkinds")
                ? params.get("searchactkinds").get(0) : null;
        final boolean[] actKinds = new boolean[Surfaces.KIND_NAMES.length];
        if (searchActKinds != null) {
            for (String kn : searchActKinds.split(",")) {
                kn = kn.trim();
                if (kn.isEmpty()) {
                    continue;
                }
                boolean hit = false;
                for (int k = 0; k < Surfaces.KIND_NAMES.length; k++) {
                    if (k != Surfaces.PAY && ("all".equals(kn) || Surfaces.KIND_NAMES[k].equals(kn))) {
                        actKinds[k] = true;
                        hit = true;
                    }
                }
                if (!hit) {
                    throw new IllegalArgumentException("-searchactkinds: unknown kind " + kn);
                }
            }
        }
        Set<Integer> searchSeats = null;
        if (params.containsKey("searchseats")) {
            searchSeats = new HashSet<>();
            for (String sIdx : params.get("searchseats").get(0).split(",")) {
                searchSeats.add(Integer.parseInt(sIdx.trim()));
            }
        }
        if (search) {
            // The behavior policy's pins on every game header (provenance).
            Obs.searchPins = String.format(java.util.Locale.ROOT,
                    "{\"rate\":%s,\"rolls\":%d,\"opts\":%d,\"mana\":%b,\"surf\":%d,\"surfcap\":%d,"
                    + "\"bar\":%s,\"temp\":%s,\"seats\":%s,\"pay\":%d,\"payleaf\":\"%s\",\"paybridge\":%b,"
                    + "\"leaf\":\"%s\",\"actkinds\":%s,\"rollsalt\":%d,"
                    + "\"deep\":%d,\"deepleaf\":\"%s\",\"deeprolls\":%d,\"deeplo\":%s,\"deepfloor\":%s,\"deepbar\":%s,\"voidskip\":%b,\"voidrescue\":%b,\"alloc\":%s,\"floor\":%s}",
                    searchRate, searchRolls, searchOpts, searchMana, searchSurf, searchSurfCap,
                    Double.isNaN(searchAct) ? "null" : String.valueOf(searchAct),
                    String.valueOf(searchTemp),
                    searchSeats == null ? "null" : "\"" + params.get("searchseats").get(0) + "\"",
                    searchPay, searchPayLeaf, searchPayBridge, searchLeaf,
                    searchActKinds == null ? "null" : "\"" + searchActKinds + "\"", searchRollSalt,
                    searchDeep, searchDeepLeaf, searchDeepRolls, String.valueOf(searchDeepLo),
                    String.valueOf(searchDeepFloor),
                    Double.isNaN(searchDeepBar) ? "null" : String.valueOf(searchDeepBar),
                    SearchMonitor.VOID_SKIP, SearchMonitor.VOID_RESCUE,
                    Double.isNaN(searchAlloc) ? "null" : String.valueOf(searchAlloc),
                    String.valueOf(searchFloor));
        }

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
            // M11 Build 0 (critic-lookahead read): -forkobs is ALLOWED here —
            // every completion's decision windows stream to the fork store
            // (arm-aware synthetic ids, "a" in the fork header) so a critic
            // can value the post-arm states. Recording only; game path untouched.
            if (forceBranch || forceSeq > 0 || drillTargets != null) {
                System.err.println("FATAL: -forceschedule excludes -forcebranch/-forceseq/-drillfile");
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

        // M10 reset Fork 3 (m10-reset-draft §D.3): -certify <horizon> = INLINE
        // certification. At each -points-sampled quiescent MAIN1 fork point of a
        // bridged active seat, the monitor asks the bridge for schedule arms
        // (anvil.certify: option labels + Obs.peekPriority); an empty answer =
        // no rollouts (the bridge's accept rate x -points = the rate knob, the
        // server's --certify-rate 0 = off). A non-empty answer runs the sched
        // rollouts (NATURAL + arms x K, horizon-stopped) exactly as
        // -forceschedule would, plus one "sched_arms" labels row carrying the
        // arm definitions (there is no TSV). Store frames keep logging the
        // mainline; completions are wire-only sessions (the mint's shape).
        int certifyHorizon = -1;
        if (params.containsKey("certify")) {
            if (rolloutK <= 0 || !params.containsKey("labels")) {
                System.err.println("FATAL: -certify requires -rollout <k> + -labels");
                System.exit(2);
            }
            if (forkObs || forceBranch || forceSeq > 0 || drillTargets != null || schedJobs != null) {
                System.err.println("FATAL: -certify excludes "
                        + "-forkobs/-forcebranch/-forceseq/-drillfile/-forceschedule");
                System.exit(2);
            }
            certifyHorizon = Integer.parseInt(params.get("certify").get(0));
            if (certifyHorizon < 0) {
                System.err.println("FATAL: -certify horizon must be >= 0 (0 = natural end)");
                System.exit(2);
            }
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
                    || schedJobs != null || certifyHorizon >= 0) {
                System.err.println("FATAL: -forcechoice excludes "
                        + "-forkobs/-forcebranch/-forceseq/-drillfile/-forceschedule/-certify");
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

        // The certifier merge (09-23; m12-plan "the certifier merge", ADR-0108
        // routed): -replay <jobs.jsonl> = replay a stored window and adjudicate
        // its payment answers on SEARCH COPIES — the one fork-and-adjudicate
        // primitive, the M9 CensusRun -certify's second entry point. One job
        // per line (CensusRun.CertJob's flat contract + seat / profiles): job,
        // seed, deck1, deck2, p (the seat's name, "(N)" = seat N-1) or seat, t,
        // sa (an option-label substring), ord, arms, k, horizon (turns past t
        // the copies play; < 0 = to the outcome), mode ("observe" = arm 0 roll
        // 0 only, the window's dec record on the row), profile1 / profile2
        // (else the seed-derived pair; recorded on every row). The mainline
        // plays the job's game (idx = job) to the ord-th quiescent priority
        // window on turn t where the seat holds an option matching sa; there
        // every (arm, roll) forks a search copy with that option forced and a
        // PAY SurfaceDirective answering `arm` at the forced option's own
        // payment window (0 = auto, the paired baseline), played to the
        // horizon leaf, one ev:"certify" row per copy (CensusRun.certRow's
        // schema + kind / calls / i / seat / profiles); a job whose window
        // never comes gets one never_fired row per arm. Roll 0 = the TRUE
        // continuation (the mainline's RNG, no determinization), rolls >= 1
        // determinized per fork J — the M9 semantics. The mainline ends at the
        // fork point (the completions are the product). Its own mode: excludes
        // every rollout / search genre; needs -labels; needs no bridge (every
        // seat heuristic under -b local-random; the outcome leaf makes no head
        // call) but runs on a bridged arm as well. The prefix trajectory is
        // AnvilRun's (seed-derived profiles, the caps): a coordinate mined from
        // an AnvilRun game replays exactly under the same flags; an M9
        // CensusRun coordinate reaches its window only where the two runners'
        // play agrees (the 09-23 parity smoke measures that rate).
        List<ReplayJob> replayJobs = null;
        if (params.containsKey("replay")) {
            if (!params.containsKey("labels")) {
                System.err.println("FATAL: -replay requires -labels <out.jsonl>");
                System.exit(2);
            }
            if (rolloutK > 0 || search || drillTargets != null || schedJobs != null || choiceJobs != null
                    || certifyHorizon >= 0 || forkObs) {
                System.err.println("FATAL: -replay excludes "
                        + "-rollout/-search/-drillfile/-forceschedule/-forcechoice/-certify/-forkobs");
                System.exit(2);
            }
            replayJobs = ReplayJob.read(params.get("replay").get(0));
            nGames = replayJobs.size();
            rangeStart = 0;
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
                    || choiceJobs != null || certifyHorizon >= 0 || search) {
                System.err.println("FATAL: -forcecandidate excludes other rollout/search modes");
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
                    hp[0], Integer.parseInt(hp[1]), "anvil-worker-r" + rangeStart, forkCommit);
            grpc.setFormatTag("mtg." + type.name().toLowerCase());
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
        // M12 Build 0 (ADR-0102): deterministic caps, defaults pinned from the
        // m9-rebaseline distribution at the 99.5th percentile (priority windows
        // per game p99.5 = 1,614; turns p99.5 = 51); 0 disables.
        rules.setAnvilTurnCap(params.containsKey("turncap")
                ? Integer.parseInt(params.get("turncap").get(0)) : DEFAULT_TURN_CAP);
        rules.setAnvilWindowCap(params.containsKey("windowcap")
                ? Integer.parseInt(params.get("windowcap").get(0)) : DEFAULT_WINDOW_CAP);

        // Deck schedule: fixed pair (-d) or index-mapped pairs file (-pairs).
        List<String[]> pairNames = new ArrayList<>();
        int gamesPerPair = params.containsKey("gpp")
                ? Integer.parseInt(params.get("gpp").get(0)) : 5;
        if (fixedPair) {
            pairNames.add(new String[] { params.get("d").get(0), params.get("d").get(1) });
            gamesPerPair = Integer.MAX_VALUE;
        } else if (params.containsKey("pairs")) {
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
            if (search && labels == null) {
                System.err.println("FATAL: -search requires -labels <out.jsonl>");
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
                final ReplayJob rjob = replayJobs != null ? replayJobs.get(g) : null;
                int idx = rjob != null ? rjob.job : rangeStart + g;
                long seed = rjob != null ? rjob.seed : seedBase != null
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

                String[] pair = rjob != null ? new String[] { rjob.deck1, rjob.deck2 }
                        : pairNames.get((int) ((idx / (long) gamesPerPair) % pairNames.size()));
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
                    if (rjob != null && rjob.profiles != null) {
                        seatProfiles[j] = rjob.profiles[j]; // the coordinate's own pair
                    }
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
                AnvilGames.noGui(game); // -Danvil.nogui=on: upstream #11780 DummyCardView (ADR-0110)
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
                            candidateJobs != null ? candidateJobs.get(idx) : null,
                            certifyHorizon);
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
                        : certifyHorizon >= 0 ? (1 + CERTIFY_MAX_ARMS)
                        : choiceJobs != null ? (1 + choiceMaxArms)
                        : forceSeq > 0 ? (seqNatOnly ? 1 : 3) : (forceBranch ? 2 : 1))
                        * rolloutK;
                int extraS = rolloutK > 0 ? fpBudget * perPoint * 45 : 0;
                if (search) {
                    game.subscribeToEvents(new SearchMonitor(game, idx, seed, bridge,
                            type.toString(), labels, watchdogs, searchRate, searchRolls, searchOpts,
                            searchMana, searchSurf, searchSurfCap, searchAct, searchTemp, searchSeats,
                            searchPay, searchPayLeaf, searchLeaf, actKinds, searchRollSalt,
                            searchDeep, searchDeepLeaf, searchDeepRolls, searchDeepLo, searchDeepFloor,
                            searchDeepBar, searchAlloc, searchFloor));
                    // The deterministic caps bound the game; the wall clock is
                    // a crash guard only under search (copies run inside it).
                    // 900 s (was 3,600): the widest boards the smokes showed
                    // sit under 15 s per window; the hour-long allowance let
                    // the dzla10 arm's loop game run 65 min (09-07). A rollout
                    // leaf's run names its own allowance (-searchclock).
                    extraS += searchClock;
                }
                ReplayMonitor replay = null;
                if (rjob != null) {
                    replay = new ReplayMonitor(game, rjob, seed, seatProfiles, bridge, type.toString(),
                            labels, watchdogs);
                    game.subscribeToEvents(replay); // the turn-passed miss
                    PlayerControllerAnvil.armPickHook(game, replay); // the fork point
                    // the copies run inside the mainline's wall: a rollout-leaf
                    // allowance per copy (ROLLOUT_END_TIMEOUT_S bounds each)
                    extraS += (rjob.maxArm() + 1) * Math.max(1, rjob.k) * 45;
                }
                final boolean[] drawClockHit = {false};
                ScheduledFuture<?> drawClock = watchdogs.schedule(() -> {
                    drawClockHit[0] = true;
                    game.setGameOver(GameEndReason.Draw);
                }, DRAW_CLOCK_S + extraS, TimeUnit.SECONDS);
                String status;
                try {
                    TimeLimitedCodeBlock.runWithTimeout(() -> mc.startGame(game),
                            GAME_HARD_CAP_S + extraS, TimeUnit.SECONDS);
                    // ADR-0102: a Draw end reason (cap, draw clock) marks
                    // every survivor as "won" in GameOutcome, so isDraw() is
                    // false and the clocked game was recorded as a win with
                    // an arbitrary winner (found at the Build 0 sanity run;
                    // the 300 s clock had this bug since M0 — ~1 game per
                    // 2,000). The mainline now uses the rollout code's
                    // unique-winner rule: exactly one winner or it is a draw.
                    status = game.getOutcome() == null ? "no_outcome"
                            : uniqueWinner(game) < 0 ? "draw" : "won";
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
                if (replay != null) {
                    PlayerControllerAnvil.clearPickHook(game);
                    replay.finish("never_fired"); // the window never came (game over / drift)
                }
                if (rolloutMonitor != null) {
                    rolloutMonitor.finishCandidateSkips();
                }
                long wallMs = System.currentTimeMillis() - gameT0;
                final int uw = uniqueWinner(game);
                String winner = uw >= 0 ? game.getRegisteredPlayers().get(uw).getName() : null;
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
                final String capReason = game.getAnvilCapReason();
                Obs.endGame(status, winnerIdx, turns, wallMs, drawClockHit[0], capReason);
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
                            + ",\"cap\":" + (capReason == null ? "null" : "\"" + capReason + "\"")
                            + ",\"windows\":" + game.getAnvilPriorityGrants()
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

    /** ADR-0105: the offline ability dump. Names one per line (the pool
     *  manifest's canonical names); unknown names are reported, not fatal. */
    static void dumpAbilities(String namesFile, String outFile) {
        int cards = 0, missing = 0, entries = 0;
        // A throwaway game: CardFactory builds a card's abilities against a
        // Game (AbilityFactory needs one); the null-owner path used by the
        // UI leaves spells and keyword traits out (1,424 abilities for
        // 1,701 cards, 484 cards empty — measured 09-07).
        java.util.List<forge.game.player.RegisteredPlayer> rps = new java.util.ArrayList<>();
        forge.deck.Deck empty = new forge.deck.Deck();
        rps.add(new forge.game.player.RegisteredPlayer(empty).setPlayer(new forge.ai.LobbyPlayerAi("a", null)));
        rps.add(new forge.game.player.RegisteredPlayer(empty).setPlayer(new forge.ai.LobbyPlayerAi("b", null)));
        GameRules dumpRules = new GameRules(GameType.Commander);
        Match dumpMatch = new Match(dumpRules, rps, "AbilityDump");
        Game dumpGame = new Game(rps, dumpRules, dumpMatch);
        forge.game.player.Player owner = dumpGame.getPlayers().get(0);
        forge.ai.anvil.AbilityKey.enumerateErrors = 0;
        try (java.io.BufferedReader in = java.nio.file.Files.newBufferedReader(
                java.nio.file.Paths.get(namesFile), java.nio.charset.StandardCharsets.UTF_8);
             java.io.PrintWriter out = new java.io.PrintWriter(java.nio.file.Files.newBufferedWriter(
                java.nio.file.Paths.get(outFile), java.nio.charset.StandardCharsets.UTF_8))) {
            String name;
            while ((name = in.readLine()) != null) {
                name = name.trim();
                if (name.isEmpty() || name.startsWith("#")) {
                    continue;
                }
                forge.item.PaperCard pc = forge.StaticData.instance().getCommonCards().getCard(name);
                if (pc == null) {
                    pc = forge.StaticData.instance().getVariantCards().getCard(name);
                }
                if (pc == null) {
                    missing++;
                    out.println("{\"miss\":" + jq(name) + "}");
                    continue;
                }
                cards++;
                try {
                    forge.game.card.Card c = forge.game.card.Card.fromPaperCard(pc, owner);
                    for (forge.ai.anvil.AbilityKey.Entry e : forge.ai.anvil.AbilityKey.enumerate(c)) {
                        entries++;
                        out.println("{\"h\":\"" + e.key + "\",\"host\":" + jq(e.host) + ",\"kind\":\"" + e.kind
                                + "\",\"txt\":" + jq(e.text) + "}");
                    }
                } catch (Exception ex) {
                    out.println("{\"err\":" + jq(name) + ",\"msg\":" + jq(String.valueOf(ex)) + "}");
                }
            }
        } catch (java.io.IOException e) {
            System.out.println("[AnvilRun] -abilities failed: " + e);
            return;
        }
        System.out.println("[AnvilRun] -abilities: " + cards + " cards, " + entries + " abilities, "
                + missing + " missing, " + forge.ai.anvil.AbilityKey.enumerateErrors
                + " enumeration errors -> " + outFile);
    }

    private static String jq(String s) {
        if (s == null) {
            return "null";
        }
        StringBuilder sb = new StringBuilder(s.length() + 8).append('"');
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"': sb.append("\\\""); break;
                case '\\': sb.append("\\\\"); break;
                case '\n': sb.append("\\n"); break;
                case '\r': sb.append("\\r"); break;
                case '\t': sb.append("\\t"); break;
                default:
                    if (c < 0x20) {
                        sb.append(String.format("\\u%04x", (int) c));
                    } else {
                        sb.append(c);
                    }
            }
        }
        return sb.append('"').toString();
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
                if (armId < 0) {
                    throw new IllegalArgumentException("armId must be >= 0 (0 = natural): " + line);
                }
                if (armId == 0) {
                    // M10 reset (ADR-0094) paired strength read: a NATURAL-ONLY
                    // fork point — the row creates the point with no directed
                    // arm; the implicit natural arm (ai = -1) runs its K
                    // completions alone. Labels after the paymode are ignored.
                    SchedPoint p0 = out.computeIfAbsent(idx, k -> new TreeMap<>())
                            .computeIfAbsent(turn, t -> new SchedPoint(t, horizon, seat));
                    if (p0.horizon != horizon || p0.seat != seat) {
                        throw new IllegalArgumentException(
                                "horizon/seat mismatch within g" + idx + " t" + turn);
                    }
                    continue;
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

    /**
     * M12 Build 0 (ADR-0101 §1, ADR-0102 fork J): the one-ply search
     * instrument. Listens for the active bridged seat's priority at a
     * quiescent MAIN1/MAIN2 window (GameCopier resumes copies at the active
     * player's priority, so only those windows fork faithfully); with
     * probability rate (seeded on (game seed, window ordinal)) evaluates every
     * candidate — pass + each mask option — on its own determinized copy
     * (fork J: libraries reshuffled, the opponent's hand resampled from its
     * unknown set; rollSeed PAIRED across candidates per roll = CRN), the
     * network playing every intermediate decision on the copy's wire session,
     * to the seat's next quiescent window (fork A), valued by anvil.value.
     * One labels row per searched window: {ev:"search", i, seed, t, ph, sw,
     * seat, n_opts, opts:[{o, label, v:[per roll], kind:[...], calls:[...],
     * ms}], copy_ms, ms, nat} — nat = the mainline's natural pick, completed
     * by the controller after its own ask. M12 Build 2: with -searchact the
     * row also carries the acting rule's verdict (by, margin, act, act_o,
     * logp, p, applied) and the controller ACTS on it (SearchDirective.Pending).
     */
    static final class SearchMonitor {
        final Game game;
        final int gameIdx;
        final long seed;
        final AnvilBridge bridge;
        final String fmt;
        final PrintWriter labels;
        final ScheduledExecutorService watchdogs;
        final double rate;
        final int rolls;
        final int optCap;
        final boolean includeMana;
        final int surfTop;
        final int surfCap;
        /** M12 Build 2: the acting bar (NaN = telemetry only), the softmax
         *  temperature and the searched seats (null = every bridged seat). */
        final double actBar;
        final double actTemp;
        final Set<Integer> seats;
        /** Evening 4: the payment expansion slot (top-B paths) and its leaf. */
        final int payTop;
        /** The pay slot's leaf mode (next | eot | h<N> | end) and its horizon
         *  in turns: -1 = next (fork A's leaf), 0 = eot, N = h<N>,
         *  Integer.MAX_VALUE = end. */
        final String payLeaf;
        final int payLeafH;
        /** Evening 5 (ADR-0106 C1): the priority slot's leaf mode and horizon
         *  (the same encoding as payLeafH); the surface slot follows it. */
        final String prioLeaf;
        final int prioLeafH;
        /** Evening 5: the surface kinds the acting rule's answer stage
         *  runs on (null = none). */
        final boolean[] actKinds;
        /** -searchrollsalt: XORed into every roll seed (0 = the CRN baseline). */
        final long rollSalt;
        /** The partial-expansion slot (ADR-0106 C3): the deep set size (0 =
         *  off), its leaf mode + horizon, rolls, the band's low edge, the
         *  floor rate and the deep bar (NaN = the acting bar). */
        final int deepTop;
        final String deepLeaf;
        final int deepLeafH;
        final int deepRolls;
        final double deepLo;
        final double deepFloor;
        final double deepBar;
        /** Build 4 (ADR-0109 item 2): the allocation head's threshold (NaN =
         *  off: the rate draw alone) and the uniform exploration floor. */
        final double alloc;
        final double allocFloor;
        /** ADR-0110: skip rolls ≥ 1 of a first-ply candidate whose roll-0 copy
         *  voided (the forced option absent on the copy — a copy-fidelity
         *  artifact, deterministic per candidate, so it repeats on every roll;
         *  its value is NaN either way; 29.6% of first-ply copies on the 09-15
         *  bench cell, ≈ 15% of them at rolls 2). Search-copy / recording only. */
        static boolean VOID_SKIP = true;
        /** ADR-0114: the void-rescue instrument — one heuristic-realized copy per
         *  voided first-ply candidate, recorded on the row, never acted on. */
        static boolean VOID_RESCUE = false;
        int sw = 0;
        private static final java.util.Set<String> crashClassesPrinted =
                java.util.Collections.synchronizedSet(new HashSet<>());

        /** Parses a -searchpayleaf mode; Integer.MIN_VALUE = not a mode. */
        static int payLeafHorizon(String mode) {
            if ("next".equals(mode)) {
                return -1;
            }
            if ("eot".equals(mode)) {
                return 0;
            }
            if ("end".equals(mode)) {
                return Integer.MAX_VALUE;
            }
            if (mode != null && mode.length() > 1 && mode.charAt(0) == 'h') {
                try {
                    int n = Integer.parseInt(mode.substring(1));
                    return n >= 0 ? n : Integer.MIN_VALUE;
                } catch (NumberFormatException e) {
                    return Integer.MIN_VALUE;
                }
            }
            return Integer.MIN_VALUE;
        }

        /** SearchDirective.leafAfterTurn for a pay copy searched at `turn`:
         *  -1 under next; the last turn of natural play otherwise (the copy's
         *  leaf is the seat's first quiescent window of a later turn; under
         *  end no turn qualifies and the copy plays to its outcome). */
        int payLeafAfter(int turn) {
            return leafAfter(turn, payLeafH);
        }

        /** The priority slot's leafAfterTurn at `turn` (-searchleaf). */
        int prioLeafAfter(int turn) {
            return leafAfter(turn, prioLeafH);
        }

        static int leafAfter(int turn, int h) {
            if (h < 0) {
                return -1;
            }
            if (h == Integer.MAX_VALUE) {
                return Integer.MAX_VALUE;
            }
            return turn + h;
        }

        SearchMonitor(Game game, int gameIdx, long seed, AnvilBridge bridge, String fmt,
                PrintWriter labels, ScheduledExecutorService watchdogs, double rate, int rolls,
                int optCap, boolean includeMana, int surfTop, int surfCap,
                double actBar, double actTemp, Set<Integer> seats, int payTop, String payLeaf,
                String prioLeaf) {
            this(game, gameIdx, seed, bridge, fmt, labels, watchdogs, rate, rolls, optCap, includeMana, surfTop,
                    surfCap, actBar, actTemp, seats, payTop, payLeaf, prioLeaf, null, 0L);
        }

        SearchMonitor(Game game, int gameIdx, long seed, AnvilBridge bridge, String fmt,
                PrintWriter labels, ScheduledExecutorService watchdogs, double rate, int rolls,
                int optCap, boolean includeMana, int surfTop, int surfCap,
                double actBar, double actTemp, Set<Integer> seats, int payTop, String payLeaf,
                String prioLeaf, boolean[] actKinds, long rollSalt) {
            this(game, gameIdx, seed, bridge, fmt, labels, watchdogs, rate, rolls, optCap, includeMana, surfTop,
                    surfCap, actBar, actTemp, seats, payTop, payLeaf, prioLeaf, actKinds, rollSalt,
                    0, "h2", 4, 0.02, 0.1, Double.NaN, Double.NaN, 0.1);
        }

        SearchMonitor(Game game, int gameIdx, long seed, AnvilBridge bridge, String fmt,
                PrintWriter labels, ScheduledExecutorService watchdogs, double rate, int rolls,
                int optCap, boolean includeMana, int surfTop, int surfCap,
                double actBar, double actTemp, Set<Integer> seats, int payTop, String payLeaf,
                String prioLeaf, boolean[] actKinds, long rollSalt, int deepTop, String deepLeaf,
                int deepRolls, double deepLo, double deepFloor, double deepBar,
                double alloc, double allocFloor) {
            this.alloc = alloc;
            this.allocFloor = allocFloor;
            this.deepTop = Math.max(0, deepTop);
            this.deepLeaf = deepLeaf == null ? "h2" : deepLeaf;
            this.deepLeafH = payLeafHorizon(this.deepLeaf);
            this.deepRolls = Math.max(1, deepRolls);
            this.deepLo = deepLo;
            this.deepFloor = deepFloor;
            this.deepBar = deepBar;
            this.actKinds = actKinds;
            this.rollSalt = rollSalt;
            this.payTop = Math.max(0, payTop);
            this.payLeaf = payLeaf;
            this.payLeafH = payLeafHorizon(payLeaf);
            this.prioLeaf = prioLeaf == null ? "next" : prioLeaf;
            this.prioLeafH = payLeafHorizon(this.prioLeaf);
            this.surfTop = Math.max(0, surfTop);
            this.surfCap = Math.max(1, surfCap);
            this.actBar = actBar;
            this.actTemp = actTemp;
            this.seats = seats;
            this.game = game;
            this.gameIdx = gameIdx;
            this.seed = seed;
            this.bridge = bridge;
            this.fmt = fmt;
            this.labels = labels;
            this.watchdogs = watchdogs;
            this.rate = rate;
            this.rolls = Math.max(1, rolls);
            this.optCap = optCap;
            this.includeMana = includeMana;
        }

        @Subscribe
        public void onPriority(GameEventPlayerPriority ev) {
            if (game.isGameOver()) {
                return;
            }
            if (ev.phase() != PhaseType.MAIN1 && ev.phase() != PhaseType.MAIN2) {
                return;
            }
            PhaseHandler ph = game.getPhaseHandler();
            if (!game.getStack().isEmpty() || ph.getPriorityPlayer() != ph.getPlayerTurn()) {
                return;
            }
            Player prio = ph.getPriorityPlayer();
            if (!(prio.getController() instanceof PlayerControllerAnvil)) {
                return;
            }
            // Default: every seat that bridges priority. An explicit -searchseats
            // list overrides the bridged requirement: a HEURISTIC seat named
            // there is searched too — the day-zero read's heuristic + lookahead
            // control arm (ADR-0104 item 5): the heuristic plays the mainline
            // and every intermediate decision on the copies, the masked head
            // values the leaves, the acting rule forces its pick on the
            // heuristic's own realization (no network plan).
            boolean bridgedPrio = ((PlayerControllerAnvil) prio.getController()).bridgesPriority();
            int seatIdx = game.getRegisteredPlayers().indexOf(prio);
            if (seats == null ? !bridgedPrio : !seats.contains(seatIdx)) {
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
            int mySw = sw++;
            if (rate < 1.0) {
                long h = splitmix64(seed ^ (mySw * 0x9E3779B97F4A7C15L) ^ 0x5EA4C4L);
                if (((h >>> 11) * 0x1.0p-53) >= rate) {
                    return;
                }
            }
            List<SpellAbility> options = null;
            String allocJson = null;
            if (!Double.isNaN(alloc)) {
                // The allocation ask: the window's own peek record (the leaf
                // value's shape, with the session's history ring) to the
                // head; the scan + the ask under an RNG snapshot (the scan is
                // RNG-neutral by Build 0's proof; the snapshot is insurance),
                // the options handed to doSearch so the searched window scans once.
                final long a0 = System.nanoTime();
                byte[] rs = snapshotRng();
                double p;
                try {
                    options = Lists.newArrayList(AnvilOptions.priorityOptions(game, prio));
                    p = bridge.value(TAG_ALLOC, Obs.peekPriority(game, prio, options, true));
                } finally {
                    MyRandom.setRandom(restoreRng(rs));
                }
                long fh = splitmix64(seed ^ (mySw * 0x9E3779B97F4A7C15L) ^ 0xA110CL);
                boolean floorHit = ((fh >>> 11) * 0x1.0p-53) < allocFloor;
                String by = Double.isNaN(p) ? "unserved" : p >= alloc ? "head" : floorHit ? "floor" : "skip";
                allocJson = "{\"p\":" + (Double.isNaN(p) ? "null" : String.format(java.util.Locale.ROOT, "%.4f", p))
                        + ",\"by\":\"" + by + "\",\"ms\":" + (System.nanoTime() - a0) / 1_000_000 + "}";
                Census.rec(game, prio, "searchAlloc", "by", by, "n_opts", options.size());
                if ("skip".equals(by)) {
                    if (labels != null) {
                        String row = "{\"ev\":\"alloc\",\"i\":" + gameIdx + ",\"seed\":" + seed
                                + ",\"t\":" + ph.getTurn() + ",\"ph\":\"" + ev.phase() + "\",\"sw\":" + mySw
                                + ",\"seat\":" + game.getRegisteredPlayers().indexOf(prio)
                                + ",\"n_opts\":" + options.size() + ",\"alloc\":" + allocJson + "}";
                        synchronized (labels) {
                            labels.println(row);
                            labels.flush();
                        }
                    }
                    return;
                }
            }
            doSearch(prio, ph.getTurn(), String.valueOf(ev.phase()), mySw, options, allocJson);
        }

        /** One candidate copy's outcome. */
        static final class CopyResult {
            double v = Double.NaN;
            String kind;
            long asks;
            long ms;
            long copyMs;
            List<SearchDirective.Surface> surfaces = Collections.emptyList();
            /** SurfaceDirective outcome: null = fired clean / unarmed; else unfired | idx | sum | neg. */
            String surfMiss = null;
            /** The fired surface window's dec record (SurfaceDirective.frame); null = unarmed / unfired. */
            String surfFrame = null;
            /** Pay answer copies under a horizon leaf: the certify axes at the
             *  copy's stop (JSON array), the calibration read's rollout side. */
            String snap = null;
            /** ADR-0114: why the forced ask voided; on a rescue copy the
             *  heuristic's plan (Obs.planJson) or its refusal. */
            String voidReason = null;
            String plan = null;
            String refuse = null;
            /** The certifier merge (09-23): the certify row's fields off the
             *  copy — the unique winner at an end / draw (registered index,
             *  -1 otherwise), the snapshot's ints, and the fired PAY answer's
             *  execution record (SurfaceDirective.exec / goals / kinds / ncand / turn). */
            int winner = -1;
            int[] snapInts = null;
            String surfExec = null;
            List<String> surfGoals = null;
            List<Integer> surfKinds = null;
            int surfN = -1;
            int surfTurn = -1;
            /** The copy's wire game header (the observe row's header). */
            String wireHeader = null;
        }

        /** The certify-style end snapshot of a copy (CensusRun.certRow's
         *  axes, seat order = registered order): [t_end, ended, life0, life1,
         *  creatures0, creatures1, power0, power1, hand0, hand1, lands0, lands1]. */
        private static String certSnap(Game copy, boolean ended) {
            return certSnapJson(certSnapInts(copy, ended));
        }

        static String certSnapJson(int[] a) {
            StringBuilder sb = new StringBuilder(64).append('[');
            for (int i = 0; i < a.length; i++) {
                if (i > 0) {
                    sb.append(',');
                }
                sb.append(a[i]);
            }
            return sb.append(']').toString();
        }

        /** The snapshot's ints in certSnap's order. */
        static int[] certSnapInts(Game copy, boolean ended) {
            int tEnd = -1;
            try {
                tEnd = copy.getPhaseHandler().getTurn();
            } catch (Exception ignored) {
            }
            int[] life = new int[2], creatures = new int[2], power = new int[2], hand = new int[2],
                    lands = new int[2];
            try {
                List<Player> ps = copy.getRegisteredPlayers();
                for (int j = 0; j < Math.min(2, ps.size()); j++) {
                    Player gp = ps.get(j);
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
            return new int[] { tEnd, ended ? 1 : 0, life[0], life[1], creatures[0], creatures[1], power[0],
                    power[1], hand[0], hand[1], lands[0], lands[1] };
        }

        /** One candidate copy: the forced option (label; null = pass) at the
         *  seat's first window, optionally a SurfaceDirective answer at the
         *  ordinal-th surface callback of surfKind on the path, natural play
         *  to the leaf, valued by anvil.value. Never throws except for a
         *  poisoned bridge (protocol law). */
        private CopyResult runCopy(String label, long rollSeed, String wid, int prioSeat, String seatName,
                byte[] rngState, int surfKind, int surfOrd, int[] surfAnswer) {
            return runCopy(label, rollSeed, wid, prioSeat, seatName, rngState, surfKind, surfOrd, surfAnswer, -1);
        }

        /** @param leafAfterTurn ≥ 0: the end-of-turn leaf (SearchDirective.leafAfterTurn) */
        private CopyResult runCopy(String label, long rollSeed, String wid, int prioSeat, String seatName,
                byte[] rngState, int surfKind, int surfOrd, int[] surfAnswer, int leafAfterTurn) {
            return runCopy(label, rollSeed, wid, prioSeat, seatName, rngState, surfKind, surfOrd, surfAnswer,
                    leafAfterTurn, false);
        }

        /** @param rescue ADR-0114: the forced option realized by the heuristic's planner */
        private CopyResult runCopy(String label, long rollSeed, String wid, int prioSeat, String seatName,
                byte[] rngState, int surfKind, int surfOrd, int[] surfAnswer, int leafAfterTurn,
                boolean rescue) {
            return runCopy(label, rollSeed, wid, prioSeat, seatName, rngState, surfKind, surfOrd, surfAnswer,
                    leafAfterTurn, rescue, false);
        }

        /** @param trueLine the certifier merge (09-23): the TRUE continuation —
         *                  no determinization (the M9 certify's roll 0); every
         *                  search copy passes false (fork J) */
        private CopyResult runCopy(String label, long rollSeed, String wid, int prioSeat, String seatName,
                byte[] rngState, int surfKind, int surfOrd, int[] surfAnswer, int leafAfterTurn,
                boolean rescue, boolean trueLine) {
            return runCopy(label, rollSeed, wid, prioSeat, seatName, rngState, surfKind, surfOrd, surfAnswer,
                    leafAfterTurn, rescue, trueLine, null);
        }

        /** @param playRng the certifier merge (09-23, ADR-0117): a REPLAY copy —
         *                 the copy plays under this RNG state (the mainline's
         *                 state before the seat's decision) and its seat
         *                 continues the mainline's decision through its own
         *                 natural chooser (SearchDirective.replayNatural); the
         *                 roll seed determinizes rolls ≥ 1 only. null = the
         *                 search's copy (the roll seed's stream, the option
         *                 forced) */
        private CopyResult runCopy(String label, long rollSeed, String wid, int prioSeat, String seatName,
                byte[] rngState, int surfKind, int surfOrd, int[] surfAnswer, int leafAfterTurn,
                boolean rescue, boolean trueLine, byte[] playRng) {
            CopyResult res = new CopyResult();
            long c0 = System.nanoTime();
            Game copy;
            try {
                copy = new GameCopier(game).makeCopy();
            } catch (Throwable t) {
                MyRandom.setRandom(restoreRng(rngState));
                res.kind = "copy_crash";
                return res;
            }
            res.copyMs = (System.nanoTime() - c0) / 1_000_000;
            Random rollRng = new Random(rollSeed);
            if (!trueLine) {
                determinize(copy, seatName, rollRng);
            }
            if (playRng != null) {
                rollRng = restoreRng(playRng); // the mainline's pre-decision stream
            } else if (trueLine) {
                rollRng = restoreRng(rngState);
            }
            copy.getPhaseHandler().devResumeAtPriority();
            copy.copyLastState();
            Obs.startWireGame(copy, wid, rollSeed, fmt, game);
            res.wireHeader = Obs.lastHeaderForBridge(copy);
            bridge.gameStart(wid, rollSeed, res.wireHeader);
            SearchDirective dir = SearchDirective.arm(copy, seatName, label);
            dir.leafAfterTurn = leafAfterTurn;
            dir.heuristicForce = rescue;
            dir.replayNatural = playRng != null;
            SurfaceDirective sdir = surfAnswer == null ? null
                    : SurfaceDirective.arm(copy, seatName, surfKind, surfOrd, surfAnswer);
            long asks0 = bridge.asksSoFar();
            MyRandom.setRandom(rollRng);
            boolean crashed = false;
            final boolean[] clockHit = {false};
            // A rollout leaf (h<N> / end) plays a copy well past the next
            // window; its clock is the game-end allowance, not fork A's.
            int copyTurn = 0;
            try {
                copyTurn = copy.getPhaseHandler().getTurn();
            } catch (Exception ignored) {
            }
            final int copyClockS = leafAfterTurn > copyTurn ? ROLLOUT_END_TIMEOUT_S : ROLLOUT_TIMEOUT_S;
            ScheduledFuture<?> clock = watchdogs.schedule(() -> {
                clockHit[0] = true;
                copy.setGameOver(GameEndReason.Draw);
            }, copyClockS, TimeUnit.SECONDS);
            try {
                copy.getPhaseHandler().mainGameLoop();
            } catch (Throwable t) {
                crashed = true;
                // One printed stack per throwable class per JVM: a silent
                // crash class is unattributable (the M11 choice-mode finding).
                if (crashClassesPrinted.add(t.getClass().getName())) {
                    System.err.println("[search] copy crash " + wid + " (" + label + "): " + t);
                    t.printStackTrace();
                }
            } finally {
                clock.cancel(false);
                if (!copy.isGameOver()) {
                    copy.setGameOver(GameEndReason.Draw);
                }
            }
            try {
                if (crashed) {
                    res.kind = "crash";
                } else if (clockHit[0]) {
                    res.kind = "timeout";
                } else if ("leaf".equals(dir.outcome) && dir.leafPeek != null) {
                    res.kind = "leaf";
                    res.v = bridge.value(TAG_VALUE, dir.leafPeek);
                    if (Double.isNaN(res.v)) {
                        res.kind = "unserved";
                    }
                } else if ("void".equals(dir.outcome)) {
                    res.kind = "void";
                } else {
                    int wi = uniqueWinner(copy);
                    res.kind = wi >= 0 ? "end" : "draw";
                    res.v = wi < 0 ? 0.5 : (wi == prioSeat ? 1.0 : 0.0);
                    res.winner = wi;
                }
                res.surfaces = new ArrayList<>(dir.surfaces);
                res.voidReason = dir.voidReason;
                res.plan = dir.plan;
                res.refuse = dir.refuse;
                if (sdir != null) {
                    res.surfMiss = !sdir.fired ? "unfired" : sdir.miss;
                    res.surfFrame = sdir.frame;
                    res.surfExec = sdir.exec;
                    res.surfGoals = sdir.goals;
                    res.surfKinds = sdir.kinds;
                    res.surfN = sdir.ncand;
                    res.surfTurn = sdir.turn;
                }
                if (leafAfterTurn >= 0) {
                    // the calibration reads' rollout side (the pay slot's,
                    // evening 4; the priority slot's, evening 5): where the
                    // copy stopped and what the board looked like there
                    res.snapInts = certSnapInts(copy, "end".equals(res.kind) || "draw".equals(res.kind));
                    res.snap = certSnapJson(res.snapInts);
                }
            } catch (RuntimeException e) {
                throw e; // a poisoned bridge ends the game (protocol law)
            } finally {
                res.asks = bridge.asksSoFar() - asks0;
                MyRandom.setRandom(restoreRng(rngState));
                SearchDirective.clear(copy);
                SurfaceDirective.clear(copy);
                Obs.endWireGame(copy);
                forge.ai.AiCache.clear();
                res.ms = (System.nanoTime() - c0) / 1_000_000;
            }
            return res;
        }

        private long rollSeedOf(int turn, int mySw, int r) {
            return splitmix64(seed ^ rollSalt ^ (turn * 0x9E3779B97F4A7C15L)
                    ^ (mySw * 0xBF58476D1CE4E5B9L) ^ (r * 0x94D049BB133111EBL));
        }

        private static void appendInts(StringBuilder sb, int[] a) {
            sb.append('[');
            if (a != null) {
                for (int i = 0; i < a.length; i++) {
                    if (i > 0) {
                        sb.append(',');
                    }
                    sb.append(a[i]);
                }
            }
            sb.append(']');
        }

        /** The first traced surface of the class (pay / not pay) on a path, or null. */
        private static SearchDirective.Surface firstOfClass(List<SearchDirective.Surface> l, boolean pay) {
            for (SearchDirective.Surface sf : l) {
                if ((sf.kind == Surfaces.PAY) == pay) {
                    return sf;
                }
            }
            return null;
        }

        /** One expansion slot: the top-`top` valued candidates whose path traced
         *  a surface of the class, each expanded over its enumerated answers
         *  (one copy per answer per roll). Appends `sub` entries (a comma
         *  before each when `already` + the entries so far > 0); returns the
         *  number appended. The payment slot runs under the end-of-turn leaf
         *  when payLeafEot, where the natural answer is re-run too. */
        private int expandRound(StringBuilder sb, boolean pay, int top, int already, int nCand, int[] nV,
                double[] meanV, List<List<SearchDirective.Surface>> firstSurf, List<String> cands, int turn,
                int mySw, int prioSeat, String seatName, byte[] rngState, String[][] firstKind,
                double[][] firstV, long[] copyMsBox) {
            return expandRound(sb, pay, top, already, nCand, nV, meanV, firstSurf, cands, turn, mySw, prioSeat,
                    seatName, rngState, firstKind, firstV, copyMsBox, null);
        }

        /** @param collect evening 5: when non-null, the slot's answers per
         *                 candidate (index = candidate) for the acting rule's
         *                 second stage — the enumerated answers with their
         *                 mean leaf values and the natural answer's index */
        private int expandRound(StringBuilder sb, boolean pay, int top, int already, int nCand, int[] nV,
                double[] meanV, List<List<SearchDirective.Surface>> firstSurf, List<String> cands, int turn,
                int mySw, int prioSeat, String seatName, byte[] rngState, String[][] firstKind,
                double[][] firstV, long[] copyMsBox, SearchDirective.Pending.SurfAnswers[] collect) {
            if (top <= 0) {
                return 0;
            }
            List<Integer> order = new ArrayList<>();
            for (int c = 0; c < nCand; c++) {
                if (nV[c] > 0 && firstOfClass(firstSurf.get(c), pay) != null) {
                    order.add(c);
                }
            }
            order.sort((a, b) -> Double.compare(meanV[b] / nV[b], meanV[a] / nV[a]));
            final int leafAfter = pay ? payLeafAfter(turn) : prioLeafAfter(turn);
            // the natural answer is the first-ply copy under CRN only when the
            // slot's leaf is the first ply's; the pay slot re-runs it otherwise
            final boolean rerunNatural = pay && leafAfter != prioLeafAfter(turn);
            int nSub = Math.min(top, order.size());
            for (int si = 0; si < nSub; si++) {
                int c = order.get(si);
                SearchDirective.Surface sf = firstOfClass(firstSurf.get(c), pay);
                Random erng = new Random(splitmix64(seed ^ (turn * 0x9E3779B97F4A7C15L)
                        ^ (mySw * 0xBF58476D1CE4E5B9L) ^ ((sf.kind + 1) * 0xD1B54A32D192ED03L)));
                List<int[]> answers = Surfaces.enumerate(sf.kind, sf.n, sf.min, sf.max, sf.natural, sf.aux,
                        surfCap, erng, sf.repeat);
                String frame = null; // the surface window's dec record, from the first answer copy that fired
                if (already + si > 0) {
                    sb.append(',');
                }
                sb.append("{\"o\":").append(c)
                        .append(",\"kind\":\"").append(Surfaces.KIND_NAMES[sf.kind]).append('"')
                        .append(",\"ord\":").append(sf.ordinal)
                        .append(",\"label\":\"").append(jstr(sf.label)).append('"')
                        .append(",\"n\":").append(sf.n)
                        .append(",\"min\":").append(sf.min)
                        .append(",\"max\":").append(sf.max)
                        .append(",\"rep\":").append(sf.repeat);
                if (pay) {
                    sb.append(",\"leaf\":\"").append(payLeaf).append('"');
                }
                sb.append(",\"nat\":");
                appendInts(sb, sf.natural);
                sb.append(",\"ans\":[");
                double[] ansMean = new double[answers.size()];
                int[] ansN = new int[answers.size()];
                int natAi = -1;
                for (int ai = 0; ai < answers.size(); ai++) {
                    int[] a = answers.get(ai);
                    if (ai > 0) {
                        sb.append(',');
                    }
                    sb.append("{\"a\":");
                    appendInts(sb, a);
                    sb.append(",\"v\":[");
                    StringBuilder kinds = new StringBuilder();
                    StringBuilder calls = new StringBuilder();
                    StringBuilder miss = new StringBuilder();
                    StringBuilder snaps = new StringBuilder();
                    long ansMs = 0;
                    boolean natural = sf.natural != null && Arrays.equals(a, sf.natural);
                    if (natural) {
                        natAi = ai;
                    }
                    for (int r = 0; r < rolls; r++) {
                        if (r > 0) {
                            sb.append(',');
                            kinds.append(',');
                            calls.append(',');
                            miss.append(',');
                            snaps.append(',');
                        }
                        if (natural && !rerunNatural) {
                            // the natural answer IS the first-ply copy under CRN
                            double v = firstV[c][r];
                            if (!Double.isNaN(v)) {
                                ansMean[ai] += v;
                                ansN[ai]++;
                            }
                            sb.append(Double.isNaN(v) ? "null" : String.format(java.util.Locale.ROOT, "%.5f", v));
                            kinds.append('"').append(firstKind[c][r]).append('"');
                            calls.append('0');
                            miss.append("null");
                            snaps.append("null");
                            continue;
                        }
                        long rollSeed = rollSeedOf(turn, mySw, r);
                        String wid = "g" + gameIdx + ".s" + mySw + "r" + r + "o" + c + (pay ? "p" : "a") + ai;
                        CopyResult cr = runCopy(cands.get(c), rollSeed, wid, prioSeat, seatName, rngState,
                                sf.kind, sf.ordinal, a, leafAfter);
                        copyMsBox[0] += cr.copyMs;
                        ansMs += cr.ms;
                        if (frame == null && cr.surfFrame != null) {
                            frame = cr.surfFrame;
                        }
                        if (!Double.isNaN(cr.v)) {
                            ansMean[ai] += cr.v;
                            ansN[ai]++;
                        }
                        sb.append(Double.isNaN(cr.v) ? "null" : String.format(java.util.Locale.ROOT, "%.5f", cr.v));
                        kinds.append('"').append(cr.kind).append('"');
                        calls.append(cr.asks);
                        miss.append(cr.surfMiss == null ? "null" : "\"" + cr.surfMiss + "\"");
                        snaps.append(cr.snap == null ? "null" : cr.snap);
                    }
                    sb.append("],\"kind\":[").append(kinds).append("],\"calls\":[").append(calls)
                            .append("],\"miss\":[").append(miss).append(']');
                    if (pay && rerunNatural) {
                        // the horizon leaf's rollout side (per roll) + the copies' wall
                        sb.append(",\"snap\":[").append(snaps).append("],\"ms\":").append(ansMs);
                    }
                    sb.append('}');
                }
                sb.append(']');
                if (frame != null) {
                    // evening 2: the state the answers were chosen in (a dec
                    // record: opts + obs + hist), the distillation loader's frame
                    sb.append(",\"frame\":").append(frame);
                }
                sb.append('}');
                if (collect != null && actKinds != null && actKinds[sf.kind]) {
                    double[] vals = new double[answers.size()];
                    for (int ai = 0; ai < answers.size(); ai++) {
                        vals[ai] = ansN[ai] > 0 ? ansMean[ai] / ansN[ai] : Double.NaN;
                    }
                    collect[c] = new SearchDirective.Pending.SurfAnswers(sf.kind, sf.ordinal, sf.label,
                            answers.toArray(new int[0][]), vals, natAi);
                }
            }
            return nSub;
        }

        /** The partial-expansion slot's runner (SearchDirective.Pending.DeepRound):
         *  at decide time — the controller's ask done, the natural known —
         *  every candidate in `set` is re-expanded to the deep leaf, one copy
         *  per roll (rollSeedOf(turn, mySw, r), r < deepRolls: CRN-paired
         *  across the set; the first `rolls` share the first ply's
         *  determinizations), each playing its forced option with the answer
         *  the shallow stage sampled for it (ansIdx[c] ≥ 0 → a
         *  SurfaceDirective on surf[c]'s callback) else its natural line.
         *  Returns the mean deep values (NaN = every roll void / crashed)
         *  and the row's copies fragment: [{o, leaf, v, kind, calls, snap,
         *  ms}]. The game RNG is snapshotted / restored around the copies. */
        private SearchDirective.Pending.DeepRound.Result deepRound(int[] set, int[] ansIdx, List<String> cands,
                SearchDirective.Pending.SurfAnswers[] surf, int turn, int mySw, int prioSeat, String seatName) {
            byte[] rngState = snapshotRng();
            final int leafAfter = leafAfter(turn, deepLeafH);
            double[] v = new double[cands.size()];
            java.util.Arrays.fill(v, Double.NaN);
            StringBuilder sb = new StringBuilder(512);
            sb.append('[');
            for (int si = 0; si < set.length; si++) {
                int c = set[si];
                SearchDirective.Pending.SurfAnswers sa = surf != null && ansIdx[c] >= 0 ? surf[c] : null;
                if (si > 0) {
                    sb.append(',');
                }
                sb.append("{\"o\":").append(c).append(",\"leaf\":\"").append(deepLeaf).append("\",\"v\":[");
                StringBuilder kinds = new StringBuilder();
                StringBuilder calls = new StringBuilder();
                StringBuilder snaps = new StringBuilder();
                double sum = 0;
                int n = 0;
                long ms = 0;
                for (int r = 0; r < deepRolls; r++) {
                    if (r > 0) {
                        sb.append(',');
                        kinds.append(',');
                        calls.append(',');
                        snaps.append(',');
                    }
                    long rollSeed = rollSeedOf(turn, mySw, r);
                    String wid = "g" + gameIdx + ".s" + mySw + "r" + r + "o" + c + "d";
                    CopyResult cr = sa == null
                            ? runCopy(cands.get(c), rollSeed, wid, prioSeat, seatName, rngState, -1, -1, null,
                                    leafAfter)
                            : runCopy(cands.get(c), rollSeed, wid, prioSeat, seatName, rngState, sa.kind, sa.ordinal,
                                    sa.answers[ansIdx[c]], leafAfter);
                    ms += cr.ms;
                    if (!Double.isNaN(cr.v)) {
                        sum += cr.v;
                        n++;
                    }
                    sb.append(Double.isNaN(cr.v) ? "null" : String.format(java.util.Locale.ROOT, "%.5f", cr.v));
                    kinds.append('"').append(cr.kind).append('"');
                    calls.append(cr.asks);
                    snaps.append(cr.snap == null ? "null" : cr.snap);
                }
                if (n > 0) {
                    v[c] = sum / n;
                }
                sb.append("],\"kind\":[").append(kinds).append("],\"calls\":[").append(calls)
                        .append("],\"snap\":[").append(snaps).append("],\"ms\":").append(ms);
                if (sa != null) {
                    sb.append(",\"a_i\":").append(ansIdx[c]);
                }
                sb.append('}');
            }
            sb.append(']');
            // the mainline's bridge session is restored by runCopy's caller
            // in doSearch; the deep round runs after it, so re-announce it
            bridge.gameStart("g" + gameIdx, seed, Obs.lastHeaderForBridge(game));
            return new SearchDirective.Pending.DeepRound.Result(v, sb.toString());
        }

        private void doSearch(Player prio, int turn, String phase, int mySw, List<SpellAbility> scanned,
                String allocJson) {
            final long block0 = System.nanoTime();
            int prioSeat = game.getRegisteredPlayers().indexOf(prio);
            List<SpellAbility> options = scanned != null ? scanned
                    : Lists.newArrayList(AnvilOptions.priorityOptions(game, prio));
            List<String> cands = new ArrayList<>(options.size() + 1);
            cands.add(null); // pass
            int manaSkipped = 0;
            for (SpellAbility sa : options) {
                if (!includeMana && sa.isManaAbility()) {
                    manaSkipped++;
                    continue;
                }
                cands.add(Census.str(sa));
            }
            int nCand = optCap > 0 ? Math.min(cands.size(), optCap + 1) : cands.size();
            byte[] rngState = snapshotRng();
            String seatName = prio.getName();
            StringBuilder sb = new StringBuilder(2048);
            sb.append("{\"ev\":\"search\",\"i\":").append(gameIdx)
                    .append(",\"seed\":").append(seed)
                    .append(",\"t\":").append(turn)
                    .append(",\"ph\":\"").append(phase).append('"')
                    .append(",\"sw\":").append(mySw)
                    .append(",\"seat\":").append(prioSeat)
                    .append(",\"n_opts\":").append(options.size())
                    .append(",\"mana_skipped\":").append(manaSkipped)
                    .append(",\"rolls\":").append(rolls)
                    .append(",\"leaf\":\"").append(prioLeaf).append('"')
                    .append(",\"opts\":[");
            long copyMsTotal = 0;
            // ---- first ply: every candidate on its own determinized copy per roll
            double[] meanV = new double[nCand];
            int[] nV = new int[nCand];
            String[][] firstKind = new String[nCand][rolls];
            double[][] firstV = new double[nCand][rolls];
            List<List<SearchDirective.Surface>> firstSurf = new ArrayList<>(nCand);
            for (int c = 0; c < nCand; c++) {
                final String label = cands.get(c);
                if (c > 0) {
                    sb.append(',');
                }
                sb.append("{\"o\":").append(c).append(",\"label\":\"").append(jstr(label == null ? "pass" : label)).append('"')
                        .append(",\"v\":[");
                StringBuilder kinds = new StringBuilder();
                StringBuilder calls = new StringBuilder();
                StringBuilder snaps = new StringBuilder();
                long optMs = 0;
                List<SearchDirective.Surface> surf0 = Collections.emptyList();
                String voidReason0 = null;
                final int prioLeafAfter = prioLeafAfter(turn);
                for (int r = 0; r < rolls; r++) {
                    // PAIRED across candidates: same determinization per roll.
                    long rollSeed = rollSeedOf(turn, mySw, r);
                    if (r > 0) {
                        sb.append(',');
                        kinds.append(',');
                        calls.append(',');
                        snaps.append(',');
                    }
                    if (r > 0 && VOID_SKIP && "void".equals(firstKind[c][0])) {
                        firstKind[c][r] = "skip";
                        firstV[c][r] = Double.NaN;
                        sb.append("null");
                        kinds.append("\"skip\"");
                        calls.append('0');
                        snaps.append("null");
                        continue;
                    }
                    String wid = "g" + gameIdx + ".s" + mySw + "r" + r + "o" + c;
                    CopyResult cr = runCopy(label, rollSeed, wid, prioSeat, seatName, rngState, -1, -1, null,
                            prioLeafAfter);
                    snaps.append(cr.snap == null ? "null" : cr.snap);
                    copyMsTotal += cr.copyMs;
                    optMs += cr.ms;
                    firstKind[c][r] = cr.kind;
                    firstV[c][r] = cr.v;
                    if (r == 0) {
                        surf0 = cr.surfaces;
                        voidReason0 = cr.voidReason;
                    }
                    if (!Double.isNaN(cr.v)) {
                        meanV[c] += cr.v;
                        nV[c]++;
                    }
                    sb.append(Double.isNaN(cr.v) ? "null" : String.format(java.util.Locale.ROOT, "%.5f", cr.v));
                    kinds.append('"').append(cr.kind).append('"');
                    calls.append(cr.asks);
                }
                firstSurf.add(surf0);
                sb.append("],\"kind\":[").append(kinds).append("],\"calls\":[").append(calls)
                        .append("],\"ms\":").append(optMs).append(",\"n_surf\":").append(surf0.size());
                if (prioLeafAfter >= 0) {
                    sb.append(",\"snap\":[").append(snaps).append(']');
                }
                if (label != null && "void".equals(firstKind[c][0]) && voidReason0 != null) {
                    // 09-21 (ADR-0114 routed): the void reason is a standing census,
                    // recorded on every voided first-ply candidate, flag or no flag.
                    sb.append(",\"vr\":\"").append(jstr(voidReason0)).append('"');
                }
                if (VOID_RESCUE && label != null && "void".equals(firstKind[c][0])) {
                    // ADR-0114: the void-rescue instrument — one more copy on roll 0's
                    // seed (CRN with every other candidate's roll 0), the forced
                    // option realized by the heuristic's planner; recorded under
                    // "h", outside the value arrays the acting rule reads.
                    CopyResult hr = runCopy(label, rollSeedOf(turn, mySw, 0),
                            "g" + gameIdx + ".s" + mySw + "r0o" + c + "h", prioSeat, seatName, rngState,
                            -1, -1, null, prioLeafAfter, true);
                    copyMsTotal += hr.copyMs;
                    sb.append(",\"h\":{\"kind\":\"").append(hr.kind).append('"')
                            .append(",\"v\":").append(Double.isNaN(hr.v) ? "null"
                                    : String.format(java.util.Locale.ROOT, "%.5f", hr.v))
                            .append(",\"calls\":").append(hr.asks).append(",\"ms\":").append(hr.ms);
                    if (hr.plan != null) {
                        sb.append(",\"plan\":").append(hr.plan);
                    }
                    if (hr.refuse != null) {
                        sb.append(",\"refuse\":\"").append(jstr(hr.refuse)).append('"');
                    }
                    sb.append('}');
                    Census.rec(game, prio, "searchRescue", "kind", hr.kind,
                            "refused", hr.refuse != null, "calls", hr.asks, "ms", hr.ms);
                }
                sb.append('}');
            }
            sb.append(']');
            // ---- expansion round (-searchsurf B): the first traced NON-PAYMENT
            // surface callback on the top-B candidates' paths, one copy per
            // answer; (-searchpay B, evening 4) the first traced PAYMENT window
            // on the top-B paths in its own slot, under the end-of-turn leaf
            // (every answer re-run there, the natural one included: a
            // different leaf from the first ply's)
            final long[] copyMsBox = {copyMsTotal};
            final SearchDirective.Pending.SurfAnswers[] surfAns =
                    actKinds != null && surfTop > 0 && !Double.isNaN(actBar)
                            ? new SearchDirective.Pending.SurfAnswers[nCand] : null;
            if (surfTop > 0 || payTop > 0) {
                sb.append(",\"sub\":[");
                int nSub = expandRound(sb, false, surfTop, 0, nCand, nV, meanV, firstSurf, cands, turn, mySw,
                        prioSeat, seatName, rngState, firstKind, firstV, copyMsBox, surfAns);
                expandRound(sb, true, payTop, nSub, nCand, nV, meanV, firstSurf, cands, turn, mySw,
                        prioSeat, seatName, rngState, firstKind, firstV, copyMsBox);
                sb.append(']');
            }
            copyMsTotal = copyMsBox[0];
            sb.append(",\"copy_ms\":").append(copyMsTotal)
                    .append(",\"ms\":").append((System.nanoTime() - block0) / 1_000_000);
            if (allocJson != null) {
                sb.append(",\"alloc\":").append(allocJson);
            }
            bridge.gameStart("g" + gameIdx, seed, Obs.lastHeaderForBridge(game));
            final PrintWriter out = labels;
            final java.util.function.Consumer<String> sink = out == null ? null : row -> {
                synchronized (out) {
                    out.println(row);
                    out.flush();
                }
            };
            // M12 Build 2: the acting rule's inputs — first-ply candidate
            // labels and mean leaf values (NaN = unvalued), the bar, the
            // temperature and a private sample seed keyed on (game seed,
            // turn, window) so a replay samples the same option.
            String[] candArr = new String[nCand];
            double[] valArr = new double[nCand];
            for (int c = 0; c < nCand; c++) {
                candArr[c] = cands.get(c);
                valArr[c] = nV[c] > 0 ? meanV[c] / nV[c] : Double.NaN;
            }
            long sampleSeed = splitmix64(seed ^ (turn * 0x9E3779B97F4A7C15L)
                    ^ (mySw * 0xBF58476D1CE4E5B9L) ^ 0xAC71A6L);
            final SearchDirective.Pending.DeepRound deepRun = deepTop > 0 && !Double.isNaN(actBar)
                    ? (set, ansIdx) -> deepRound(set, ansIdx, cands, surfAns, turn, mySw, prioSeat, seatName)
                    : null;
            SearchDirective.expectNatural(game, new SearchDirective.Pending(sb.toString(), sink,
                    candArr, valArr, actBar, actTemp, sampleSeed, surfAns, deepRun, deepTop, deepLo, deepFloor,
                    deepBar));
        }
    }

    /** Fork J (ADR-0102): determinize a copy to the acting seat's information
     *  set — every library reshuffled (the pre-existing rollout default) and
     *  each OTHER seat's hand resampled uniformly from hand ∪ library (the
     *  decklist is public in this pool; revealed-from-hand cards are
     *  approximated as unknown). Zone.setCards: no shuffle events/triggers. */
    static void determinize(Game copy, String actingName, Random rng) {
        for (Player p : copy.getPlayers()) {
            List<Card> lib = new ArrayList<>();
            for (Card c : p.getZone(ZoneType.Library)) {
                lib.add(c);
            }
            if (!p.getName().equals(actingName)) {
                List<Card> hand = new ArrayList<>();
                for (Card c : p.getZone(ZoneType.Hand)) {
                    hand.add(c);
                }
                int n = hand.size();
                List<Card> pool = new ArrayList<>(hand);
                pool.addAll(lib);
                Collections.shuffle(pool, rng);
                p.getZone(ZoneType.Hand).setCards(new ArrayList<>(pool.subList(0, n)));
                lib = new ArrayList<>(pool.subList(n, pool.size()));
            }
            Collections.shuffle(lib, rng);
            p.getZone(ZoneType.Library).setCards(lib);
        }
    }

    /** The certifier merge (09-23): one -replay job — the M9 certify jobs
     *  contract (CensusRun.CertJob) plus seat / profiles. Immutable. */
    static final class ReplayJob {
        final int job;
        final long seed;
        final String deck1, deck2, sa;
        final int seat, t, ord, arms, k, horizon;
        final boolean observe;
        /** The coordinate's own AI profiles (profile1 / profile2), or null =
         *  AnvilRun's seed-derived pair. */
        final String[] profiles;
        /** The window's phase (the census row's ph: MAIN1 / MAIN2 / UPKEEP …),
         *  or null = any phase. The 09-23 smoke: a main-2 cast forked at main 1
         *  is a window the AI declines (heur_refuse), so the phase is part of
         *  the coordinate whenever the miner has it. */
        final String ph;

        ReplayJob(Map<String, String> m) {
            job = Integer.parseInt(m.get("job"));
            seed = Long.parseLong(m.get("seed"));
            deck1 = m.get("deck1");
            deck2 = m.get("deck2");
            sa = m.get("sa") == null ? "" : m.get("sa");
            t = Integer.parseInt(m.get("t"));
            ord = m.containsKey("ord") ? Integer.parseInt(m.get("ord")) : 0;
            arms = m.containsKey("arms") ? Integer.parseInt(m.get("arms")) : 8;
            k = m.containsKey("k") ? Integer.parseInt(m.get("k")) : 1;
            horizon = m.containsKey("horizon") ? Integer.parseInt(m.get("horizon")) : 2;
            observe = "observe".equals(m.get("mode"));
            seat = m.containsKey("seat") ? Integer.parseInt(m.get("seat")) : seatOf(m.get("p"));
            String p1 = m.get("profile1");
            String p2 = m.get("profile2");
            profiles = p1 != null && p2 != null ? new String[] { p1, p2 } : null;
            ph = m.get("ph");
            if (deck1 == null || deck2 == null) {
                throw new IllegalArgumentException("job " + job + ": deck1 / deck2 required");
            }
        }

        /** Arms 0..maxArm run (observe: arm 0 only). */
        int maxArm() {
            return observe ? 0 : Math.max(0, arms);
        }

        /** SearchDirective.leafAfterTurn for the copies forked at `turn`:
         *  turn + horizon, or the outcome (horizon < 0). */
        int leafAfter(int turn) {
            return horizon < 0 ? Integer.MAX_VALUE : turn + horizon;
        }

        /** The seat index from a player name of the census / Anvil shape
         *  ("Census(2)-deck" / "Anvil(1)-deck" / "Heur(1)-deck" = the (N)
         *  group minus one); 0 when the name carries none. */
        static int seatOf(String p) {
            if (p == null) {
                return 0;
            }
            int a = p.indexOf('(');
            int b = a < 0 ? -1 : p.indexOf(')', a);
            if (a < 0 || b < 0) {
                return 0;
            }
            try {
                return Math.max(0, Integer.parseInt(p.substring(a + 1, b)) - 1);
            } catch (NumberFormatException e) {
                return 0;
            }
        }

        static List<ReplayJob> read(String path) {
            List<ReplayJob> jobs = new ArrayList<>();
            java.util.Set<Integer> ids = new HashSet<>();
            try {
                for (String line : Files.readAllLines(Paths.get(path), StandardCharsets.UTF_8)) {
                    if (line.isBlank()) {
                        continue;
                    }
                    ReplayJob j = new ReplayJob(CensusRun.flatJson(line));
                    if (!ids.add(j.job)) {
                        System.err.println("FATAL: -replay: duplicate job id " + j.job);
                        System.exit(2);
                    }
                    jobs.add(j);
                }
            } catch (Exception e) {
                System.err.println("FATAL: cannot read -replay jobs " + path + ": " + e);
                System.exit(2);
            }
            if (jobs.isEmpty()) {
                System.err.println("FATAL: -replay: no jobs in " + path);
                System.exit(2);
            }
            return jobs;
        }
    }

    /**
     * The certifier merge (09-23): the -replay job's fork point on its
     * mainline = the window where the seat's NATURAL PICK is the stored
     * option (PlayerControllerAnvil.PickHook: the pick before the play), on
     * the job's turn (and phase when named), the active player's priority on
     * an empty stack (the fork-fidelity rule), the ord-th such pick. Forks
     * one search copy per (arm, roll) through the SearchMonitor's copy runner
     * (never subscribed: rate 0) with the option forced and a PAY
     * SurfaceDirective answering the arm at the option's own payment window,
     * and writes one certify row per copy. The mainline ends at the fork
     * point; a window that never comes (the turn passes, the game ends, the
     * pick was made on the opponent's turn or with the stack non-empty —
     * no faithful fork point) gets one never_fired row per arm from finish().
     */
    static final class ReplayMonitor implements PlayerControllerAnvil.PickHook {
        final Game game;
        final ReplayJob job;
        final long seed;
        final String[] profiles;
        final AnvilBridge bridge;
        final PrintWriter labels;
        final SearchMonitor copies;
        int seen = 0;
        volatile boolean done = false;
        /** The RNG state at the seat's latest priority event on the job's
         *  turn = the state before the decision the pick hook reports (the
         *  copies continue the mainline's decision under it). */
        private byte[] preAskRng = null;

        ReplayMonitor(Game game, ReplayJob job, long seed, String[] profiles, AnvilBridge bridge, String fmt,
                PrintWriter labels, ScheduledExecutorService watchdogs) {
            this.game = game;
            this.job = job;
            this.seed = seed;
            this.profiles = profiles;
            this.bridge = bridge;
            this.labels = labels;
            this.copies = new SearchMonitor(game, job.job, seed, bridge, fmt, null, watchdogs, 0.0, 1, 0, false,
                    0, 1, Double.NaN, 0.025, null, 0, "next", "next");
        }

        @Subscribe
        public void onTurnBegan(GameEventTurnBegan ev) {
            if (!done && !game.isGameOver() && ev.turnNumber() > job.t) {
                finish("never_fired");
                game.setGameOver(GameEndReason.Draw);
            }
        }

        @Subscribe
        public void onPriority(GameEventPlayerPriority ev) {
            if (done || game.isGameOver()) {
                return;
            }
            PhaseHandler ph = game.getPhaseHandler();
            if (ph.getTurn() != job.t) {
                return;
            }
            Player prio = ph.getPriorityPlayer();
            if (prio != null && game.getRegisteredPlayers().indexOf(prio) == job.seat) {
                preAskRng = snapshotRng();
            }
        }

        /** The seat's natural pick at a mainline ask (before the play). */
        @Override
        public void picked(Player p, List<SpellAbility> picked) {
            if (done || game.isGameOver() || picked == null || picked.isEmpty()) {
                return;
            }
            PhaseHandler ph = game.getPhaseHandler();
            if (ph.getTurn() != job.t) {
                return;
            }
            // Any phase where the ACTIVE player holds priority on an empty
            // stack (the fork-fidelity rule: GameCopier resumes copies at the
            // active player's priority); the job's phase when it names one.
            // A cast on the opponent's turn or in response (stack non-empty)
            // has no faithful fork point — its job reads never_fired.
            if (job.ph != null && !job.ph.equals(String.valueOf(ph.getPhase()))) {
                return;
            }
            if (!game.getStack().isEmpty() || ph.getPriorityPlayer() != p || ph.getPlayerTurn() != p) {
                return;
            }
            int prioSeat = game.getRegisteredPlayers().indexOf(p);
            if (prioSeat != job.seat) {
                return;
            }
            String label = Census.str(picked.get(0));
            if (label == null || !label.contains(job.sa) || seen++ != job.ord) {
                return;
            }
            fork(p, prioSeat, label, ph.getTurn());
            done = true;
            game.setGameOver(GameEndReason.Draw);
        }

        private void fork(Player prio, int prioSeat, String label, int turn) {
            Obs.mark(game, "fork", "fp", 0, "kr", Math.max(1, job.k));
            byte[] rngState = snapshotRng();
            String seatName = prio.getName();
            int leafAfter = job.leafAfter(turn);
            int maxArm = job.maxArm();
            for (int arm = 0; arm <= maxArm; arm++) {
                for (int roll = 0; roll < Math.max(1, job.k); roll++) {
                    // CensusRun.certifyGame's roll seed: shared across arms so
                    // completions pair; roll 0 = the true continuation
                    long rollSeed = splitmix64(job.seed ^ (job.job * 0x9E3779B97F4A7C15L)
                            ^ (roll * 0xBF58476D1CE4E5B9L));
                    String wid = "g" + job.job + ".c" + arm + "r" + roll;
                    // every roll continues the mainline's decision (replayNatural
                    // under the pre-decision RNG); roll 0 on the true line,
                    // rolls >= 1 on a determinized copy
                    SearchMonitor.CopyResult cr = copies.runCopy(label, rollSeed, wid, prioSeat, seatName,
                            rngState, Surfaces.PAY, 0, new int[] { arm }, leafAfter, false, roll == 0,
                            preAskRng != null ? preAskRng : rngState);
                    boolean fired = writeRow(arm, roll, turn, label, cr);
                    System.out.printf("job %d arm %d roll %d -> %s%n", job.job, arm, roll,
                            fired ? cr.surfExec : "miss:" + cr.kind);
                    if (roll == 0 && !fired) {
                        break; // the identical prefix cannot fire later (CensusRun's rule)
                    }
                    if (job.observe) {
                        break;
                    }
                }
            }
            bridge.gameStart("g" + job.job, seed, Obs.lastHeaderForBridge(game));
        }

        /** One certify row (CensusRun.certRow's contract, additive fields);
         *  returns the fired bit. */
        private boolean writeRow(int arm, int roll, int turn, String label, SearchMonitor.CopyResult cr) {
            boolean fired = cr.surfMiss == null && cr.surfN >= 0;
            String reason = fired ? null
                    : "void".equals(cr.kind) ? "void:" + (cr.voidReason == null ? "?" : cr.voidReason)
                    : "crash".equals(cr.kind) || "timeout".equals(cr.kind) || "copy_crash".equals(cr.kind) ? cr.kind
                    : "idx".equals(cr.surfMiss) ? "no_such_option"
                    : cr.surfMiss == null ? "unfired" : cr.surfMiss;
            StringBuilder sb = new StringBuilder(400);
            sb.append("{\"ev\":\"certify\",\"job\":").append(job.job)
                    .append(",\"arm\":").append(arm)
                    .append(",\"roll\":").append(roll)
                    .append(",\"fired\":").append(fired);
            if (!fired) {
                sb.append(",\"reason\":\"").append(jstr(reason)).append('"');
            }
            if (fired && arm > 0 && cr.surfGoals != null) {
                sb.append(",\"goals\":[");
                for (int i = 0; i < cr.surfGoals.size(); i++) {
                    if (i > 0) {
                        sb.append(',');
                    }
                    sb.append('"').append(jstr(cr.surfGoals.get(i))).append('"');
                }
                sb.append("],\"gk\":[");
                for (int i = 0; cr.surfKinds != null && i < cr.surfKinds.size(); i++) {
                    if (i > 0) {
                        sb.append(',');
                    }
                    sb.append(cr.surfKinds.get(i));
                }
                sb.append(']');
            }
            if (fired) {
                sb.append(",\"exec\":\"").append(job.observe ? "observed" : jstr(cr.surfExec)).append('"')
                        .append(",\"t_fired\":").append(cr.surfTurn >= 0 ? cr.surfTurn : turn);
            }
            int[] a = cr.snapInts != null ? cr.snapInts : new int[12];
            if (cr.snapInts == null) {
                a[0] = -1;
            }
            sb.append(",\"t_end\":").append(a[0])
                    .append(",\"ended\":").append(a[1] == 1)
                    .append(",\"winner\":").append(cr.winner)
                    .append(",\"snap\":{\"life\":[").append(a[2]).append(',').append(a[3])
                    .append("],\"creatures\":[").append(a[4]).append(',').append(a[5])
                    .append("],\"power\":[").append(a[6]).append(',').append(a[7])
                    .append("],\"hand\":[").append(a[8]).append(',').append(a[9])
                    .append("],\"lands\":[").append(a[10]).append(',').append(a[11])
                    .append("],\"avail_options\":").append(fired ? cr.surfN - 1 : -1)
                    .append("},\"kind\":\"").append(cr.kind).append('"')
                    .append(",\"calls\":").append(cr.asks)
                    .append(",\"ms\":").append(cr.ms)
                    .append(",\"i\":").append(job.job)
                    .append(",\"seed\":").append(seed)
                    .append(",\"seat\":").append(job.seat)
                    .append(",\"t\":").append(turn)
                    .append(",\"ph\":\"").append(String.valueOf(game.getPhaseHandler().getPhase())).append('"')
                    .append(",\"label\":\"").append(jstr(label)).append('"')
                    .append(",\"profiles\":[\"").append(jstr(profiles[0])).append("\",\"")
                    .append(jstr(profiles[1])).append("\"]");
            if (job.observe && cr.surfFrame != null) {
                // the scorer's input (payment_drill_score --rows): the window's
                // dec record (the wire shape, hist spliced) + the copy's game header
                sb.append(",\"frame\":").append(cr.surfFrame);
                String header = cr.wireHeader != null ? cr.wireHeader : Obs.lastHeaderForBridge(game);
                if (header != null) {
                    sb.append(",\"header\":").append(header);
                }
            }
            sb.append('}');
            synchronized (labels) {
                labels.println(sb);
                labels.flush();
            }
            return fired;
        }

        /** The window never came: one miss row per arm (the reader's
         *  no-baseline / miss accounting), once. */
        void finish(String reason) {
            if (done) {
                return;
            }
            done = true;
            System.out.printf("job %d -> miss:%s (seen %d)%n", job.job, reason, seen);
            for (int arm = 0; arm <= job.maxArm(); arm++) {
                String row = "{\"ev\":\"certify\",\"job\":" + job.job + ",\"arm\":" + arm + ",\"roll\":0"
                        + ",\"fired\":false,\"reason\":\"" + jstr(reason) + "\",\"t_end\":-1,\"ended\":false"
                        + ",\"winner\":-1,\"snap\":{\"life\":[0,0],\"creatures\":[0,0],\"power\":[0,0]"
                        + ",\"hand\":[0,0],\"lands\":[0,0],\"avail_options\":-1},\"kind\":\"miss\""
                        + ",\"i\":" + job.job + ",\"seed\":" + seed + ",\"seat\":" + job.seat
                        + ",\"profiles\":[\"" + jstr(profiles[0]) + "\",\"" + jstr(profiles[1]) + "\"]}";
                synchronized (labels) {
                    labels.println(row);
                    labels.flush();
                }
            }
        }
    }

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
    /** A pay copy under a rollout leaf (-searchpayleaf h<N> / end): the
     *  allowance for a heuristic half-game (the calibration read). */
    private static final int ROLLOUT_END_TIMEOUT_S = 600;
    /** Inline certification: arms per point cap (sched_pins.ARM_CAP) — the clock budget. */
    private static final int CERTIFY_MAX_ARMS = 16;
    static final String TAG_CERTIFY = "anvil.certify";
    /** M12 Build 0: the search-leaf value ask (AnvilBridge.value). */
    static final String TAG_VALUE = "anvil.value";
    /** Build 4 (ADR-0109 item 2): the allocation ask on the value wire —
     *  P(the search acts at this window), micro-units; NaN = unserved. */
    static final String TAG_ALLOC = "anvil.alloc";
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
        /** M10 reset Fork 3 inline certification horizon; -1 = off. */
        final int certifyHorizon;
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
                Map<Integer, CandidatePoint> candidate, int certifyHorizon) {
            this.game = game;
            this.certifyHorizon = certifyHorizon;
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
                doSchedRollouts(turn, targetTurn, sched.get(targetTurn));
                return;
            }
            if (certifyHorizon >= 0) {
                doCertifyRollouts(turn, targetTurn);
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
        /** M10 reset Fork 3: the inline-certification fork point. The arm set
         *  is decided at the window by the bridge (anvil.certify); an empty
         *  answer is a counted skip. Non-empty -> a transient SchedPoint runs
         *  through doSchedRollouts unchanged (same rows, same directive, same
         *  rollSeed identity keyed on the target turn). */
        private void doCertifyRollouts(int turn, int targetTurn) {
            PhaseHandler ph = game.getPhaseHandler();
            Player prio = ph.getPriorityPlayer();
            boolean bridgeSeat = prio.getController() instanceof PlayerControllerAnvil
                    && ((PlayerControllerAnvil) prio.getController()).bridgesPriority();
            if (!bridgeSeat || turn != targetTurn) {
                return; // unbridged seat / drifted target: not a certification window
            }
            int prioSeat = -1;
            for (int j = 0; j < game.getRegisteredPlayers().size(); j++) {
                if (game.getRegisteredPlayers().get(j).getName().equals(prio.getName())) {
                    prioSeat = j;
                }
            }
            List<SpellAbility> options = Lists.newArrayList(AnvilOptions.priorityOptions(game, prio));
            List<String> optLabels = Lists.newArrayListWithCapacity(options.size());
            for (SpellAbility sa : options) {
                optLabels.add(Census.str(sa));
            }
            List<int[]> arms;
            try {
                arms = bridge.certifyArms(TAG_CERTIFY, optLabels, Obs.peekPriority(game, prio, options));
            } catch (RuntimeException e) {
                throw e; // a poisoned bridge ends the game (protocol law); nothing else throws
            }
            if (arms == null || arms.isEmpty()) {
                if (labels != null) {
                    synchronized (labels) {
                        labels.println("{\"ev\":\"sched\",\"i\":" + gameIdx
                                + ",\"seed\":" + seed + ",\"fp\":" + fp
                                + ",\"t\":" + turn + ",\"tt\":" + targetTurn
                                + ",\"skip\":\"" + (arms == null ? "certify_unserved" : "declined") + "\"}");
                        labels.flush();
                    }
                }
                return;
            }
            SchedPoint point = new SchedPoint(targetTurn, certifyHorizon, prioSeat);
            StringBuilder armsJson = new StringBuilder(512);
            for (int ai = 0; ai < arms.size() && ai < CERTIFY_MAX_ARMS; ai++) {
                List<String> armLabels = new ArrayList<>();
                for (int idx : arms.get(ai)) {
                    if (idx >= 0 && idx < optLabels.size()) {
                        armLabels.add(optLabels.get(idx));
                    }
                }
                point.arms.add(new SchedArm(ai + 1, true, armLabels));
                if (ai > 0) {
                    armsJson.append(',');
                }
                armsJson.append('[');
                for (int li = 0; li < armLabels.size(); li++) {
                    if (li > 0) {
                        armsJson.append(',');
                    }
                    armsJson.append('"').append(jstr(armLabels.get(li))).append('"');
                }
                armsJson.append(']');
            }
            if (labels != null) {
                // the arm definitions (there is no schedfile): the Python
                // finish step's read_sched equivalent
                synchronized (labels) {
                    labels.println("{\"ev\":\"sched_arms\",\"i\":" + gameIdx
                            + ",\"seed\":" + seed + ",\"fp\":" + fp
                            + ",\"t\":" + targetTurn + ",\"seat\":" + prioSeat
                            + ",\"horizon\":" + certifyHorizon
                            + ",\"n_opts\":" + optLabels.size()
                            + ",\"arms\":[" + armsJson + "]}");
                    labels.flush();
                }
            }
            doSchedRollouts(turn, targetTurn, point);
        }

        private void doSchedRollouts(int turn, int targetTurn, SchedPoint point) {
            int myFp = fp++;
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
                    final int armId = arm == null ? 0 : arm.id;
                    if (forkObs) {
                        // M11 Build 0: per-completion fork-store frame. The
                        // synthetic id folds the arm in (0 = natural, 1..16 =
                        // arms, 20 slots) under the plain path's (g, fp, r)
                        // layout; the fork header carries "a" for the join.
                        long off = (((long) gameIdx * 100 + myFp) * 100 + r) * 20 + armId;
                        if (off >= FORK_NS_STRIDE) {
                            throw new IllegalStateException(
                                    "fork id offset " + off + " >= FORK_NS_STRIDE (gameIdx " + gameIdx + ")");
                        }
                        Obs.startForkGame(copy, wid, forkGBase + off, rollSeed, fmt, game,
                                gameIdx, myFp, r, targetTurn, armId);
                    } else {
                        Obs.startWireGame(copy, wid, rollSeed, fmt, game);
                    }
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
                        if (forkObs) {
                            int tEnd = -1;
                            try {
                                tEnd = copy.getPhaseHandler().getTurn();
                            } catch (Exception ignored) {
                            }
                            boolean stoppedHere = stop != null && stop.stopped;
                            int wi = (crashed || stoppedHere || clockHit[0]) ? -1 : uniqueWinner(copy);
                            Obs.endForkGame(copy, crashed ? "crash" : (wi >= 0 ? "won" : "draw"),
                                    wi, tEnd, (System.nanoTime() - c0) / 1_000_000);
                        }
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

        /** Unique-winner extraction, NOT getWinningLobbyPlayer: a forced
         *  draw (horizon stop / rollout clock) runs Player.onGameOver, which
         *  marks EVERY surviving player as "won" — the winning-player accessor
         *  then returns an arbitrary map-order pick (JVM-varying; the smoke's
         *  determinism diff caught exactly this). Exactly one won = a real
         *  winner (registered-players index); anything else = -1. */
        private static int uniqueWinner(Game copy) {
            return AnvilRun.uniqueWinner(copy);
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
                int winner = uniqueWinner(copy);
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
