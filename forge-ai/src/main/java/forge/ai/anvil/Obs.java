package forge.ai.anvil;

import com.github.luben.zstd.ZstdOutputStream;
import forge.LobbyPlayer;
import forge.ai.LobbyPlayerAi;
import forge.card.ColorSet;
import forge.card.MagicColor;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.keyword.Keyword;
import forge.game.phase.PhaseHandler;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.player.RegisteredPlayer;
import forge.game.spellability.AbilitySub;
import forge.game.spellability.AlternativeCost;
import forge.game.spellability.OptionalCost;
import forge.game.spellability.SpellAbility;
import forge.game.spellability.TargetChoices;
import forge.item.PaperCard;

import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.util.Map;

/**
 * Observation/trajectory writer (Anvil M1 D1, docs/design/observation-schema-v1.md).
 * One zstd frame per game appended to a single file, JSONL records inside the
 * frame, sidecar .idx.jsonl with one line per game (offset/lengths/seed/recs).
 *
 * Record kinds: "game" header (schema version, seat order — all player fields
 * elsewhere are indices into it), "dec" (one per PlayerController callback,
 * observation captured at entry, BEFORE delegation can mutate state), "ret"
 * (the answer, written at exit, joined on "s" — separate record because
 * mid-resolution callbacks nest), "end".
 *
 * Static synchronized like Census: one worker plays one game at a time. Writes
 * are additionally gated on Game identity — after a hard-cap timeout the
 * abandoned game thread can keep firing callbacks while the runner has moved
 * on (the no-interrupts rule means it is never killed); those records must not
 * leak into the next game's frame.
 *
 * M2 D1 (fork-side re-binding): record building is per-Game "sessions" so a
 * forked game can produce wire observations (the model server featurizes from
 * them) without touching the store. The store frame stays a single static
 * binding (one worker, one corpus game at a time); startWireGame opens a
 * wire-only session for a fork, seeding its history ring from the parent's
 * (valid because GameCopier now preserves card ids). Execution is strictly
 * nested on the game thread (mainline blocks while a fork plays out), so the
 * zero-arg lastDecForBridge — read immediately after each dec by the bridge —
 * can safely resolve to the most recent dec across sessions.
 *
 * Runaway guard: a wedged post-hard-cap thread can write observations
 * indefinitely (pilot: one game wrote 24.6 GB raw in ~6 min against a legit
 * max of 571 MB before the worker was killed mid-write, leaving a truncated
 * frame). RAW_CAP truncates the frame with a loud "obs_cap" end record —
 * policy labels up to the cut stay usable; the status keeps the game out of
 * the value loss. The cap firing also releases the Obs lock promptly so the
 * runner's endGame/close are never starved by a wedge.
 */
public final class Obs {
    // v2 (boundary bundle 2026-08-21): entity choice-state kv ("cho",
    // ObsSnapshot.choiceState) + blesses the additive payment-window kv the
    // pay_mana_class bridge emits (goal-shaped labels, m9-payment-surface
    // spec §6). Readers gate on this — never mix sv eras in one store join.
    public static final int SCHEMA_VERSION = 2;
    private static final int ZSTD_LEVEL = 3;
    /** Per-game raw-byte ceiling; 2x the 50K-pilot's largest legit frame. */
    private static final long RAW_CAP = Long.getLong("anvil.obs.rawcap", 1L << 30);

    private static FileOutputStream file;
    private static PrintWriter idx;
    private static long fileOffset;

    private static CountingStream counting;
    private static OutputStream frame;
    private static Game currentGame;
    private static int gameIdx = -1;
    private static long gameSeed;
    private static long recs;
    private static long rawBytes;
    private static long obsErrors;

    // ---- M4 D3 fork-session store: drill completions as frames of their
    // own, in a SEPARATE file — the mainline's zstd frame is open while a
    // fork plays out, so fork records cannot interleave into the main
    // stream. One fork session runs at a time (the mainline blocks), so a
    // single set of statics suffices, mirroring the mainline set above.
    private static FileOutputStream forkFile;
    private static PrintWriter forkIdx;
    private static long forkFileOffset;
    private static CountingStream forkCounting;
    private static OutputStream forkFrame;
    private static Game forkCurrentGame; // frame owner — the mainline's
    // currentGame analog. A game hard-cap ABANDONS its thread mid-completion
    // (TimeLimitedCodeBlock); the abandoned thread keeps running its
    // completions, and without this binding its fork-frame calls interleave
    // with the live game's — found live in d6-run10 iter-9 (its interrupted
    // thread's getChannel().position() closed the fd: every later frame was
    // a clen-0 phantom).
    private static long forkGameIdx = -1;
    private static long forkGameSeed;
    private static long forkRecs;
    private static long forkRawBytes;

    // ---- M1 D8 bridge support: wire-record capture, oi labels, history ring ----
    /** = anvil.encoder.transform.HISTORY_K; the wire observation carries the
     *  last K prior decisions so the server featurizes identically to
     *  training (info-set rule applied Python-side, where it is leak-tested).
     *  Entries append at dec time and back-fill the host at ret time,
     *  matching the training loader's view (it joins rets into decs over the
     *  whole game, so "prior" decs carry their answers even when the ret
     *  lands later in the stream). Residual skew: a window nested inside a
     *  still-open entity-answering parent trains against that parent's
     *  future host but serves -1 — loader-side fix queued with the M2
     *  schema work. */
    private static final int HIST_CAP = 8;

    private static final class HistEntry {
        final String m;
        final int p;
        long host = -1;

        HistEntry(String m, int p) {
            this.m = m;
            this.p = p;
        }
    }

    /** Per-game record-building state. store=true is the (single) corpus
     *  game whose records go to the zstd frame; store=false is a wire-only
     *  session (forked rollout) whose records exist only for the bridge —
     *  unless forkStore is set (M4 D3): then its records ALSO stream to the
     *  fork frame file as a store frame of their own. */
    private static final class Session {
        final boolean store;
        final boolean forkStore;
        final String wireId; // null for store sessions ("g<idx>" on the wire)
        long seq;            // per-game; "s" joins dec/ret
        final java.util.ArrayDeque<HistEntry> histRing = new java.util.ArrayDeque<>();
        final java.util.LinkedHashMap<Long, HistEntry> pendingDec = new java.util.LinkedHashMap<>();
        long prioSeq = -1;
        java.util.List<SpellAbility> prioOptions;
        String lastDecRecord;
        String lastHistJson;
        String headerRecord;

        Session(boolean store, String wireId) {
            this(store, wireId, false);
        }

        Session(boolean store, String wireId, boolean forkStore) {
            this.store = store;
            this.forkStore = forkStore;
            this.wireId = wireId;
        }
    }

    /** Weak keys: sessions are removed at end-of-game, but crash paths must
     *  not pin dead Game graphs (Game has identity equals). */
    private static final java.util.WeakHashMap<Game, Session> sessions = new java.util.WeakHashMap<>();
    /** Most recent dec across sessions; valid because execution is strictly
     *  nested and the bridge reads it immediately after the dec it belongs to. */
    private static Session lastDecSession;
    /** Most recently started session (store or wire); backs the zero-arg
     *  lastHeaderForBridge used right after startGame. */
    private static Session lastStartedSession;

    private Obs() {
    }

    /** State snapshot JSON (the dec records' "obs" object) for tests/tooling;
     *  no store needs to be open. */
    public static String stateJson(Game g) {
        StringBuilder sb = new StringBuilder(4096);
        ObsSnapshot.write(sb, g);
        return sb.toString();
    }

    public static synchronized boolean isOpen() {
        return file != null;
    }

    /** True when records for this Game are being built — store frame open or
     *  a wire-only session active; false for a stale post-hard-cap thread
     *  (its session is removed at frame end). Lets callers skip expensive
     *  materialization work that only feeds the log/bridge. */
    public static synchronized boolean isLogging(Game g) {
        return sessions.get(g) != null;
    }

    public static synchronized void open(String path) throws IOException {
        file = new FileOutputStream(path, true);
        fileOffset = file.getChannel().position();
        String idxPath = path.endsWith(".zst")
                ? path.substring(0, path.length() - 4) + ".idx.jsonl" : path + ".idx.jsonl";
        idx = new PrintWriter(new FileWriter(idxPath, true));
    }

    public static synchronized void close() {
        endGameFrame();
        if (idx != null) {
            idx.flush();
            idx.close();
            idx = null;
        }
        if (file != null) {
            try {
                file.close();
            } catch (IOException ignored) {
            }
            file = null;
        }
        closeForks();
        if (obsErrors > 0) {
            System.err.println("Obs: " + obsErrors + " observation serialization errors (obs:null records)");
        }
    }

    /** Opens the fork-session frame file (M4 D3, -forkobs); same layout as
     *  the mainline obs file, sibling index sidecar. */
    public static synchronized void openForks(String path) throws IOException {
        forkFile = new FileOutputStream(path, true);
        forkFileOffset = forkFile.getChannel().position();
        String idxPath = path.endsWith(".zst")
                ? path.substring(0, path.length() - 4) + ".idx.jsonl" : path + ".idx.jsonl";
        forkIdx = new PrintWriter(new FileWriter(idxPath, true));
    }

    public static synchronized void closeForks() {
        endForkFrame();
        if (forkIdx != null) {
            forkIdx.flush();
            forkIdx.close();
            forkIdx = null;
        }
        if (forkFile != null) {
            try {
                forkFile.close();
            } catch (IOException ignored) {
            }
            forkFile = null;
        }
    }

    public static synchronized void startGame(int idx0, long seed, Game g, String fmt) {
        if (file == null) {
            return;
        }
        endGameFrame(); // defensive: a crashed game may not have reached endGame
        gameIdx = idx0;
        gameSeed = seed;
        currentGame = g;
        recs = 0;
        rawBytes = 0;
        counting = new CountingStream(file);
        try {
            frame = new ZstdOutputStream(counting, ZSTD_LEVEL);
        } catch (IOException e) {
            System.err.println("Obs: cannot open zstd frame for game " + idx0 + ": " + e);
            frame = null;
            currentGame = null;
            return;
        }
        Session s = new Session(true, null);
        StringBuilder sb = buildHeader(idx0, seed, g, fmt, null);
        s.headerRecord = sb.toString(); // GameStart.header for the M1 bridge
        sessions.put(g, s);
        lastStartedSession = s;
        write(sb);
    }

    /**
     * Wire-only session (M2 D1): records for this Game are built for the
     * bridge (lastDecForBridge/lastHeaderForBridge) but never written to the
     * store — the fork-side re-binding that lets the model policy drive a
     * forked game without polluting the corpus. The history ring seeds from
     * the parent's session (deep copy) so the fork's first window carries the
     * same last-K view the mainline had at the fork point; host ids stay
     * valid because GameCopier preserves card ids. wireId is the derived
     * game id the driver announces via AnvilBridge.gameStart.
     */
    public static synchronized void startWireGame(Game g, String wireId, long seed, String fmt,
            Game parent) {
        Session s = new Session(false, wireId);
        Session ps = parent == null ? null : sessions.get(parent);
        if (ps != null) {
            for (HistEntry h : ps.histRing) {
                HistEntry c = new HistEntry(h.m, h.p);
                c.host = h.host;
                s.histRing.addLast(c);
            }
        }
        s.headerRecord = buildHeader(-1, seed, g, fmt, wireId).toString();
        sessions.put(g, s);
        lastStartedSession = s;
    }

    /** Ends a wire-only session (fork completed/abandoned). Store sessions
     *  end via endGame/endGameFrame. */
    public static synchronized void endWireGame(Game g) {
        Session s = sessions.get(g);
        if (s == null || s.store) {
            return;
        }
        sessions.remove(g);
        if (lastDecSession == s) {
            lastDecSession = null;
        }
        if (lastStartedSession == s) {
            lastStartedSession = null;
        }
    }

    /**
     * Fork-session store variant of startWireGame (M4 D3, -forkobs): the
     * session behaves exactly like a wire session for the bridge (parent
     * hist-ring seed, wid announce), and ADDITIONALLY streams its records to
     * the fork frame file as a store frame keyed by a synthetic unique game
     * index. The header carries fork provenance ({"fork":{pg,fp,r,tt}}) and
     * the completion's OWN seed (the rollout RNG seed — per-completion
     * unique, so server-side sampled noise decorrelates across the K
     * completions). Dec records in fork frames carry the serve-time "hist"
     * verbatim (the wire composite): the first windows' history includes
     * parent-game entries a loader-side reconstruction could never see.
     */
    public static synchronized void startForkGame(Game g, String wireId, long synthG,
            long rollSeed, String fmt, Game parent, int parentG, int fp, int r, int tt) {
        if (forkFile == null || (parent != null && sessions.get(parent) == null)) {
            // No parent session = the parent's mainline frame already closed
            // (hard-cap abandoned thread still running its completions):
            // wire-only, never the shared frame file. Under --drill-sample
            // the g=-1 header makes the server refuse, so nothing trains.
            startWireGame(g, wireId, rollSeed, fmt, parent);
            return;
        }
        endForkFrame(); // defensive: a crashed completion may not have closed
        Session s = new Session(false, wireId, true);
        Session ps = parent == null ? null : sessions.get(parent);
        if (ps != null) {
            for (HistEntry h : ps.histRing) {
                HistEntry c = new HistEntry(h.m, h.p);
                c.host = h.host;
                s.histRing.addLast(c);
            }
        }
        StringBuilder sb = buildHeader(synthG, rollSeed, g, fmt, wireId);
        sb.setLength(sb.length() - 1);
        sb.append(",\"fork\":{\"pg\":").append(parentG).append(",\"fp\":").append(fp)
                .append(",\"r\":").append(r).append(",\"tt\":").append(tt).append("}}");
        s.headerRecord = sb.toString();
        sessions.put(g, s);
        lastStartedSession = s;
        forkCurrentGame = g;
        forkGameIdx = synthG;
        forkGameSeed = rollSeed;
        forkRecs = 0;
        forkRawBytes = 0;
        forkCounting = new CountingStream(forkFile);
        try {
            forkFrame = new ZstdOutputStream(forkCounting, ZSTD_LEVEL);
        } catch (IOException e) {
            System.err.println("Obs: cannot open fork frame for game " + synthG + ": " + e);
            forkFrame = null;
            forkCounting = null;
            return;
        }
        writeFork(sb);
    }

    /** Ends a fork-store session: writes the end record (status/winner from
     *  the driver, which computed the completion outcome) and closes the
     *  frame. Follow with endWireGame(g) for session cleanup — safe to call
     *  from a finally that also covers the plain-wire path. */
    public static synchronized void endForkGame(Game g, String status, int winnerIdx,
            int turns, long ms) {
        Session s = sessions.get(g);
        if (s == null || !s.forkStore) {
            return;
        }
        if (g != forkCurrentGame) {
            // Stale completion (its frame was defensively closed by a later
            // startForkGame): session cleanup only — never the live frame.
            endWireGame(g);
            return;
        }
        if (forkFrame != null) {
            StringBuilder sb = new StringBuilder(128);
            sb.append("{\"k\":\"end\",\"status\":").append(q(status))
                    .append(",\"winner\":").append(winnerIdx)
                    .append(",\"turns\":").append(turns)
                    .append(",\"ms\":").append(ms).append('}');
            writeFork(sb);
        }
        endForkFrame();
        endWireGame(g);
    }

    /** The "game" header record; wire sessions carry a "wid" field (the
     *  derived wire game id) and g=-1 (no store index). */
    private static StringBuilder buildHeader(long idx0, long seed, Game g, String fmt,
            String wireId) {
        StringBuilder sb = new StringBuilder(512);
        sb.append("{\"k\":\"game\",\"sv\":").append(SCHEMA_VERSION)
                .append(",\"g\":").append(idx0)
                .append(",\"seed\":").append(seed);
        if (wireId != null) {
            sb.append(",\"wid\":").append(q(wireId));
        }
        sb.append(",\"fmt\":").append(q(fmt))
                .append(",\"players\":[");
        int i = 0;
        for (Player p : g.getRegisteredPlayers()) {
            if (i++ > 0) {
                sb.append(',');
            }
            sb.append("{\"name\":").append(q(p.getName()));
            RegisteredPlayer rp = p.getRegisteredPlayer();
            if (rp != null && rp.getDeck() != null) {
                sb.append(",\"deck\":").append(q(rp.getDeck().getName()));
                if (!rp.getCommanders().isEmpty()) {
                    sb.append(",\"cmd\":[");
                    int j = 0;
                    for (PaperCard pc : rp.getCommanders()) {
                        if (j++ > 0) {
                            sb.append(',');
                        }
                        sb.append(q(pc.getName()));
                    }
                    sb.append(']');
                }
            }
            LobbyPlayer lp = p.getLobbyPlayer();
            if (lp instanceof LobbyPlayerAi) {
                String profile = ((LobbyPlayerAi) lp).getAiProfile();
                if (profile != null && !profile.isEmpty()) {
                    sb.append(",\"profile\":").append(q(profile));
                }
            }
            sb.append('}');
        }
        sb.append("]}");
        return sb;
    }

    /**
     * Store-frame marker record (M2 D4 rollout labels): {"k":"mark","s":seq,
     * "m":kind,"t":turn, ...numeric kv}. Shares the dec seq counter so marks
     * order strictly against decision records; readers that don't know the
     * kind skip it (decode_frame handles dec/ret/end only). Store sessions
     * only — wire sessions have nowhere to persist a mark.
     */
    public static synchronized void mark(Game g, String kind, Object... kv) {
        Session ses = sessions.get(g);
        if (ses == null || !ses.store || frame == null || g != currentGame) {
            return;
        }
        long s = ses.seq++;
        StringBuilder sb = new StringBuilder(192);
        sb.append("{\"k\":\"mark\",\"s\":").append(s)
                .append(",\"m\":").append(q(kind));
        int turn = -1;
        try {
            PhaseHandler ph = g.getPhaseHandler();
            if (ph != null) {
                turn = ph.getTurn();
            }
        } catch (Exception ignored) {
        }
        sb.append(",\"t\":").append(turn);
        for (int i = 0; i + 1 < kv.length; i += 2) {
            String key = String.valueOf(kv[i]);
            if (key.equals("k") || key.equals("s") || key.equals("m") || key.equals("t")) {
                // duplicate JSON keys resolve to the LAST value in most
                // parsers — a kv named "k" silently turned every mark into
                // an unknown-kind record (found in the first labeler smoke)
                System.err.println("Obs.mark: reserved key '" + key + "' skipped");
                continue;
            }
            sb.append(",\"").append(key).append("\":").append(kv[i + 1]);
        }
        sb.append('}');
        write(sb);
    }

    public static synchronized void endGame(String status, int winnerIdx, int turns, long ms, boolean drawClock) {
        if (frame == null) {
            return;
        }
        StringBuilder sb = new StringBuilder(128);
        sb.append("{\"k\":\"end\",\"status\":").append(q(status))
                .append(",\"winner\":").append(winnerIdx)
                .append(",\"turns\":").append(turns)
                .append(",\"ms\":").append(ms);
        if (drawClock) {
            sb.append(",\"draw_clock\":true");
        }
        sb.append('}');
        write(sb);
        endGameFrame();
    }

    /**
     * Decision record at callback entry; returns the seq to pass to ret(), or
     * -1 when not logging. The observation snapshots state before the
     * heuristic/bridge answers (some callbacks resolve their own effects).
     */
    public static long dec(Game g, Player p, String m, Object... kv) {
        return decInternal(g, p, m, null, null, false, kv);
    }

    /** Unprovenanced decision variant that records plain option labels. */
    public static long decWithOptions(Game g, Player p, String m,
            java.util.List<String> opts, Object... kv) {
        return decInternal(g, p, m, null, opts, false, kv);
    }

    /** Canonical WUBRG labels for single-color callbacks. */
    public static java.util.List<String> colorOptions(ColorSet colors) {
        java.util.List<String> out = new java.util.ArrayList<>();
        for (byte color : MagicColor.WUBRG) {
            if ((colors.getColor() & color) != 0) {
                out.add(MagicColor.toLongString(color));
            }
        }
        return out;
    }

    /** Bridged variant: provenance tag + materialized option labels. */
    public static long decBridged(Game g, Player p, String m, java.util.List<String> opts, Object... kv) {
        return decInternal(g, p, m, "bridge", opts, false, kv);
    }

    /**
     * Priority-window dec (M1 D2): materializes the engine-legal option set
     * (same scan as the bridged path — legal-actions-only invariant) into
     * structured opts entries {"e":hostId,"sa":str,"kind":...} so the label
     * has a logged legality basis (replay drift forbids recomputing it later;
     * the gate metric's single-legal-option exclusion needs it). Pass is not
     * an entry: a null ret IS the pass. Options are scanned only when this
     * game is actually being logged.
     */
    public static long decPriority(Game g, Player p) {
        return decPriority(g, p, null);
    }

    /** Bridged variant carries provenance; both stash the scanned options so
     *  ret() can label the chosen option index ("oi" — exact SA-level labels
     *  for future corpora; the sa-string join is ambiguous for ~31% of
     *  multi-SA hosts, measured 2026-07-10). */
    public static long decPriority(Game g, Player p, String by) {
        return decPriority(g, p, by, null);
    }

    /** preScanned lets the bridged path reuse its own option scan (the scan
     *  is the D3-measured haircut; scanning twice per window would double it). */
    public static long decPriority(Game g, Player p, String by,
            java.util.List<SpellAbility> preScanned) {
        if (!isLogging(g)) {
            return -1;
        }
        java.util.List<String> opts = null;
        java.util.List<Card> extraEnts = null;
        java.util.List<SpellAbility> options = null;
        try {
            options = preScanned != null ? preScanned : AnvilOptions.priorityOptions(g, p);
            opts = new java.util.ArrayList<>(options.size());
            for (SpellAbility sa : options) {
                StringBuilder ob = new StringBuilder(96);
                Card h = sa.getHostCard();
                ob.append("{\"e\":").append(h == null ? -1 : h.getId())
                        .append(",\"sa\":").append(q(trunc(String.valueOf(sa))))
                        .append(",\"kind\":\"").append(kind(sa)).append("\"}");
                opts.add(ob.toString());
                // Hosts castable from an unwalked zone (library top): the
                // snapshot must contain them or the label references nothing.
                if (h != null && h.getZone() != null
                        && h.getZone().getZoneType() == forge.game.zone.ZoneType.Library
                        && (extraEnts == null || !extraEnts.contains(h))) {
                    if (extraEnts == null) {
                        extraEnts = new java.util.ArrayList<>(2);
                    }
                    extraEnts.add(h);
                }
            }
        } catch (Exception e) {
            opts = null; // options are diagnostic-critical but not corpus-fatal
            options = null;
            obsErrors++;
        }
        long s = decInternal(g, p, "chooseSpellAbilityToPlay", by, opts, true, extraEnts);
        if (s >= 0) {
            synchronized (Obs.class) {
                Session ses = sessions.get(g);
                if (ses != null) {
                    ses.prioSeq = s;
                    ses.prioOptions = options;
                }
            }
        }
        return s;
    }

    private static synchronized long decInternal(Game g, Player p, String m, String by,
            java.util.List<String> opts, boolean optsRaw, Object... kv) {
        return decInternal(g, p, m, by, opts, optsRaw, null, kv);
    }

    private static synchronized long decInternal(Game g, Player p, String m, String by,
            java.util.List<String> opts, boolean optsRaw, java.util.List<Card> extraEnts,
            Object... kv) {
        Session ses = sessions.get(g);
        if (ses == null || (ses.store && frame == null)) {
            return -1;
        }
        long s = ses.seq++;
        StringBuilder sb = new StringBuilder(8192);
        sb.append("{\"k\":\"dec\",\"s\":").append(s);
        int turn = -1;
        String phase = null;
        try {
            PhaseHandler ph = g.getPhaseHandler();
            if (ph != null) {
                turn = ph.getTurn();
                PhaseType pt = ph.getPhase();
                phase = pt == null ? null : pt.toString();
            }
        } catch (Exception ignored) {
        }
        int pIdx = p == null ? -1 : g.getRegisteredPlayers().indexOf(p);
        sb.append(",\"t\":").append(turn)
                .append(",\"ph\":").append(q(phase))
                .append(",\"p\":").append(pIdx)
                .append(",\"m\":\"").append(m).append('"')
                .append(",\"d\":").append(Thread.currentThread().getStackTrace().length);
        if (by != null) {
            sb.append(",\"by\":").append(q(by));
        }
        if (kv.length > 1) {
            sb.append(",\"args\":{");
            for (int i = 0; i + 1 < kv.length; i += 2) {
                if (i > 0) {
                    sb.append(',');
                }
                sb.append('"').append(kv[i]).append("\":").append(scalar(kv[i + 1]));
            }
            sb.append('}');
        }
        if (opts != null) {
            sb.append(",\"opts\":[");
            for (int i = 0; i < opts.size(); i++) {
                if (i > 0) {
                    sb.append(',');
                }
                // raw entries are pre-rendered JSON objects (decPriority);
                // otherwise plain labels, quoted
                sb.append(optsRaw ? opts.get(i) : q(opts.get(i)));
            }
            sb.append(']');
        }
        sb.append(",\"obs\":");
        int obsStart = sb.length();
        try {
            ObsSnapshot.write(sb, g, extraEnts);
        } catch (Exception e) {
            sb.setLength(obsStart);
            sb.append("null,\"err\":").append(q(e.toString()));
            obsErrors++;
        }
        sb.append('}');
        ses.lastDecRecord = sb.toString();
        ses.lastHistJson = histJson(ses); // prior decs only: snapshot BEFORE appending this one
        lastDecSession = ses;
        HistEntry entry = new HistEntry(m, pIdx);
        ses.histRing.addLast(entry);
        if (ses.histRing.size() > HIST_CAP) {
            ses.histRing.removeFirst();
        }
        ses.pendingDec.put(s, entry);
        if (ses.pendingDec.size() > 64) { // crash-path leak bound; normal nesting is shallow
            ses.pendingDec.remove(ses.pendingDec.keySet().iterator().next());
        }
        if (ses.store) {
            write(sb);
        } else if (ses.forkStore && forkFrame != null && g == forkCurrentGame) {
            // Fork frames store the WIRE composite (dec + serve-time hist):
            // the first windows' history includes parent-game entries a
            // loader-side reconstruction from frame records could never see.
            // Owner check: a hard-cap-abandoned thread's completion must not
            // write into the live game's frame.
            writeFork(new StringBuilder(lastDecOf(ses)));
        }
        return s;
    }

    private static String histJson(Session ses) {
        StringBuilder sb = new StringBuilder(64 * ses.histRing.size() + 2);
        sb.append('[');
        int i = 0;
        for (HistEntry h : ses.histRing) {
            if (i++ > 0) {
                sb.append(',');
            }
            sb.append("{\"m\":").append(q(h.m)).append(",\"p\":").append(h.p)
                    .append(",\"e\":").append(h.host).append('}');
        }
        return sb.append(']').toString();
    }

    /** Answer record at callback exit; joined to its dec on "s". */
    public static synchronized void ret(Game g, long s, Object v) {
        Session ses = sessions.get(g);
        if (ses == null || s < 0) {
            return;
        }
        boolean toStore = ses.store && frame != null;
        boolean toFork = ses.forkStore && forkFrame != null && g == forkCurrentGame;
        if (toStore || toFork) {
            StringBuilder sb = new StringBuilder(160);
            sb.append("{\"k\":\"ret\",\"s\":").append(s).append(",\"v\":");
            value(sb, v, 0);
            if (s == ses.prioSeq && ses.prioOptions != null && v != null) {
                sb.append(",\"oi\":").append(optionIndexOf(ses.prioOptions, v));
            }
            sb.append('}');
            if (toStore) {
                write(sb);
            } else {
                writeFork(sb);
            }
        }
        HistEntry entry = ses.pendingDec.remove(s);
        if (entry != null) {
            entry.host = retHostId(v); // back-fill; may already be ring-evicted
        }
        if (s == ses.prioSeq) {
            ses.prioSeq = -1;
            ses.prioOptions = null;
        }
    }

    /**
     * Index of the chosen SA in the stashed option scan, or -1. Identity
     * first; alt-cost variants are fresh copies each scan, so fall back to a
     * composite key (host + rendered string + alternative cost) and stay
     * honest on ambiguity (-1) rather than guess.
     */
    private static int optionIndexOf(java.util.List<SpellAbility> prioOptions, Object v) {
        SpellAbility chosen = null;
        if (v instanceof SpellAbility) {
            chosen = (SpellAbility) v;
        } else if (v instanceof java.util.List && !((java.util.List<?>) v).isEmpty()
                && ((java.util.List<?>) v).get(0) instanceof SpellAbility) {
            chosen = (SpellAbility) ((java.util.List<?>) v).get(0);
        }
        if (chosen == null) {
            return -1;
        }
        for (int i = 0; i < prioOptions.size(); i++) {
            if (prioOptions.get(i) == chosen) {
                return i;
            }
        }
        int hit = -1;
        int n = 0;
        for (int i = 0; i < prioOptions.size(); i++) {
            SpellAbility o = prioOptions.get(i);
            if (o.getHostCard() == chosen.getHostCard()
                    && o.getAlternativeCost() == chosen.getAlternativeCost()
                    && String.valueOf(o).equals(String.valueOf(chosen))) {
                hit = i;
                n++;
            }
        }
        return n == 1 ? hit : -1;
    }

    /** Chosen-host id for a history entry, mirroring what the Python side
     *  reads off ret[0] (SpellAbility/Card answers; -1 otherwise). */
    private static long retHostId(Object v) {
        Object first = v;
        if (v instanceof java.util.List) {
            java.util.List<?> l = (java.util.List<?>) v;
            first = l.isEmpty() ? null : l.get(0);
        }
        if (first instanceof SpellAbility) {
            Card h = ((SpellAbility) first).getHostCard();
            return h == null ? -1 : h.getId();
        }
        if (first instanceof Card) {
            return ((Card) first).getId();
        }
        return -1;
    }

    /**
     * The dec record just written, with the last-K completed-decision history
     * spliced in — the M1 bridge observation payload. The server featurizes
     * it with the same transform as training; the information-set rule is
     * applied Python-side, where it is leak-tested. Null when not logging
     * (D8 eval games run with --obs on; the wire payload rides the same
     * serialization the corpus uses).
     */
    public static synchronized String lastDecForBridge() {
        return lastDecOf(lastDecSession);
    }

    /** Per-game variant: the given Game's last dec (session-routed; use when
     *  the caller holds the Game — unambiguous under fork nesting). */
    public static synchronized String lastDecForBridge(Game g) {
        return lastDecOf(sessions.get(g));
    }

    private static String lastDecOf(Session ses) {
        if (ses == null || ses.lastDecRecord == null) {
            return null;
        }
        StringBuilder sb = new StringBuilder(
                ses.lastDecRecord.length() + ses.lastHistJson.length() + 16);
        sb.append(ses.lastDecRecord, 0, ses.lastDecRecord.length() - 1)
                .append(",\"hist\":").append(ses.lastHistJson).append('}');
        return sb.toString();
    }

    /** The game header record (GameStart.header wire field) of the most
     *  recently started session. Null when not logging. */
    public static synchronized String lastHeaderForBridge() {
        return lastStartedSession == null ? null : lastStartedSession.headerRecord;
    }

    /** Per-game header (store or wire session). Null when no session. */
    public static synchronized String lastHeaderForBridge(Game g) {
        Session ses = sessions.get(g);
        return ses == null ? null : ses.headerRecord;
    }

    // ---------- answer-value serialization (best-effort v1; see schema doc) ----------

    private static void value(StringBuilder sb, Object v, int depth) {
        if (v == null) {
            sb.append("null");
            return;
        }
        if (depth > 3) {
            sb.append(q(trunc(String.valueOf(v))));
            return;
        }
        if (v instanceof Boolean || v instanceof Integer || v instanceof Long) {
            sb.append(v);
        } else if (v instanceof String) {
            sb.append(q((String) v));
        } else if (v instanceof Card) {
            sb.append("{\"e\":").append(((Card) v).getId()).append('}');
        } else if (v instanceof Player) {
            Player p = (Player) v;
            sb.append("{\"pi\":").append(p.getGame().getRegisteredPlayers().indexOf(p)).append('}');
        } else if (v instanceof SpellAbility) {
            castPlan(sb, (SpellAbility) v, depth);
        } else if (v.getClass().isEnum()) {
            sb.append(q(((Enum<?>) v).name()));
        } else if (v instanceof Map) {
            sb.append("{\"map\":[");
            int i = 0;
            for (Map.Entry<?, ?> e : ((Map<?, ?>) v).entrySet()) {
                if (i++ > 0) {
                    sb.append(',');
                }
                if (i > 64) {
                    sb.append("\"...\"");
                    break;
                }
                sb.append('[');
                value(sb, e.getKey(), depth + 1);
                sb.append(',');
                value(sb, e.getValue(), depth + 1);
                sb.append(']');
            }
            sb.append("]}");
        } else if (v instanceof Iterable) {
            sb.append('[');
            int i = 0;
            for (Object o : (Iterable<?>) v) {
                if (i++ > 0) {
                    sb.append(',');
                }
                if (i > 128) {
                    sb.append("\"...\"");
                    break;
                }
                value(sb, o, depth + 1);
            }
            sb.append(']');
        } else if (v instanceof org.apache.commons.lang3.tuple.Pair) {
            org.apache.commons.lang3.tuple.Pair<?, ?> pr = (org.apache.commons.lang3.tuple.Pair<?, ?>) v;
            sb.append('[');
            value(sb, pr.getLeft(), depth + 1);
            sb.append(',');
            value(sb, pr.getRight(), depth + 1);
            sb.append(']');
        } else {
            sb.append("{\"str\":").append(q(trunc(String.valueOf(v)))).append('}');
        }
    }

    /**
     * CastPlan-shaped serialization of a chosen SpellAbility (M1 D2, schema
     * doc "CastPlan ret"): everything the heuristic decided at cast-decision
     * time that never surfaces as a callback (census): targets and X read off
     * the SA; optional costs already baked in (the SA the AI returns is the
     * copy GameActionUtil.addOptionalCosts built); modes only when already
     * bound — for cast spells they usually bind later at chooseModeForAbility,
     * which is logged as its own dec/ret. Sub-abilities contribute their own
     * pre-set targets.
     */
    private static void castPlan(StringBuilder sb, SpellAbility sa, int depth) {
        sb.append('{');
        if (sa.getHostCard() != null) {
            sb.append("\"e\":").append(sa.getHostCard().getId()).append(',');
        }
        sb.append("\"sa\":").append(q(trunc(String.valueOf(sa))))
                .append(",\"kind\":\"").append(kind(sa)).append('"');
        targets(sb, sa);
        Integer x = sa.getXManaCostPaid();
        if (x != null) {
            sb.append(",\"x\":").append(x);
        }
        AlternativeCost alt = sa.getAlternativeCost();
        if (alt != null) {
            sb.append(",\"alt\":").append(q(alt.name()));
        }
        int nOpt = 0;
        for (OptionalCost oc : sa.getOptionalCosts()) {
            sb.append(nOpt++ == 0 ? ",\"opt\":[" : ",").append(q(oc.name()));
        }
        if (nOpt > 0) {
            sb.append(']');
        }
        int mk = sa.getOptionalKeywordAmount(Keyword.MULTIKICKER);
        if (mk > 0) {
            sb.append(",\"mk\":").append(mk);
        }
        java.util.List<AbilitySub> modes = sa.getChosenList();
        if (modes != null && !modes.isEmpty() && depth < 3) {
            sb.append(",\"modes\":[");
            int j = 0;
            for (AbilitySub mode : modes) {
                if (j++ > 0) {
                    sb.append(',');
                }
                castPlan(sb, mode, depth + 1);
            }
            sb.append(']');
        }
        // sub-ability chain: only links that carry their own targets, indexed
        // by chain position (bounded — chains are hand-authored card script)
        int link = 0;
        int emitted = 0;
        for (SpellAbility sub = sa.getSubAbility(); sub != null && link < 16; sub = sub.getSubAbility(), link++) {
            StringBuilder tsb = new StringBuilder(64);
            targets(tsb, sub);
            if (tsb.length() == 0) {
                continue; // no live targets (all stale-filtered or none set)
            }
            sb.append(emitted++ == 0 ? ",\"sub\":[" : ",").append("{\"i\":").append(link)
                    .append(tsb).append('}');
        }
        if (emitted > 0) {
            sb.append(']');
        }
        sb.append('}');
    }

    private static String kind(SpellAbility sa) {
        return sa.isLandAbility() ? "land" : sa.isSpell() ? "spell"
                : sa.isActivatedAbility() ? "ability" : "other";
    }

    /** Root-level targets in the observation idiom: {"e":id} / {"pi":seat} /
     *  {"e":hostId,"stk":1} for a targeted stack SA (stack entries are keyed
     *  by host card id in ObsSnapshot, so this joins). */
    private static void targets(StringBuilder sb, SpellAbility sa) {
        TargetChoices tc = sa.getTargets();
        if (tc == null || tc.isEmpty()) {
            return;
        }
        java.util.List<String> refs = new java.util.ArrayList<>(4);
        for (Object t : tc) {
            if (t instanceof Card) {
                Card c = (Card) t;
                // Stale-evaluation guard (D3 validation batch): a modal
                // spell's sub-chain can retain TargetChoices from an earlier
                // AI pass, pointing at an entity that no longer exists (dead
                // token). Resolution re-binds targets, so a ref to a card in
                // no zone is noise that would dangle in the label.
                if (c.getZone() == null || !c.getZone().contains(c)) {
                    continue;
                }
                refs.add("{\"e\":" + c.getId() + '}');
            } else if (t instanceof Player) {
                Player tp = (Player) t;
                refs.add("{\"pi\":" + tp.getGame().getRegisteredPlayers().indexOf(tp) + '}');
            } else if (t instanceof SpellAbility) {
                // A stack-SA target joins on its host card id; stale if that
                // spell is no longer on the stack.
                Card h = ((SpellAbility) t).getHostCard();
                if (h == null || h.getZone() == null
                        || h.getZone().getZoneType() != forge.game.zone.ZoneType.Stack
                        || !h.getZone().contains(h)) {
                    continue;
                }
                refs.add("{\"e\":" + h.getId() + ",\"stk\":1}");
            } else {
                refs.add("{\"str\":" + q(trunc(String.valueOf(t))) + '}');
            }
        }
        if (refs.isEmpty()) {
            return;
        }
        sb.append(",\"tgt\":[");
        for (int i = 0; i < refs.size(); i++) {
            if (i > 0) {
                sb.append(',');
            }
            sb.append(refs.get(i));
        }
        sb.append(']');
    }

    private static String scalar(Object o) {
        if (o == null) {
            return "null";
        }
        if (o instanceof Integer || o instanceof Long || o instanceof Boolean) {
            return o.toString();
        }
        return q(String.valueOf(o));
    }

    private static String trunc(String s) {
        return s.length() > 120 ? s.substring(0, 120) : s;
    }

    // ---------- frame plumbing ----------

    private static void write(StringBuilder sb) {
        sb.append('\n');
        byte[] bytes = sb.toString().getBytes(StandardCharsets.UTF_8);
        try {
            frame.write(bytes);
            rawBytes += bytes.length;
            recs++;
        } catch (IOException e) {
            // close + account the partial frame; dropping it with no idx row
            // would leave fileOffset stale and misalign every later frame
            System.err.println("Obs: write failed for game " + gameIdx + ", closing frame: " + e);
            endGameFrame();
            return;
        }
        if (rawBytes > RAW_CAP) {
            System.err.println("Obs: game " + gameIdx + " hit raw cap ("
                    + rawBytes + " > " + RAW_CAP + " bytes), truncating frame");
            try {
                frame.write("{\"k\":\"end\",\"status\":\"obs_cap\",\"winner\":-1,\"turns\":-1,\"ms\":0}\n"
                        .getBytes(StandardCharsets.UTF_8));
                recs++;
            } catch (IOException ignored) {
            }
            endGameFrame();
        }
    }

    private static void endGameFrame() {
        if (frame == null) {
            return;
        }
        try {
            frame.close(); // finishes the zstd frame; CountingStream shields the file
        } catch (IOException e) {
            System.err.println("Obs: frame close failed for game " + gameIdx + ": " + e);
        }
        if (idx != null) {
            idx.println("{\"g\":" + gameIdx + ",\"off\":" + fileOffset + ",\"clen\":" + counting.count
                    + ",\"rlen\":" + rawBytes + ",\"seed\":" + gameSeed + ",\"recs\":" + recs + "}");
            idx.flush();
        }
        fileOffset += counting.count;
        if (file != null) {
            try {
                // channel position is ground truth (append mode, single writer);
                // self-heals drift when a failed write landed partial bytes
                fileOffset = file.getChannel().position();
            } catch (IOException ignored) {
            }
        }
        frame = null;
        counting = null;
        if (currentGame != null) {
            Session s = sessions.remove(currentGame);
            if (s != null && lastDecSession == s) {
                lastDecSession = null;
            }
            if (s != null && lastStartedSession == s) {
                lastStartedSession = null;
            }
        }
        currentGame = null;
    }

    private static void writeFork(StringBuilder sb) {
        if (forkFrame == null) {
            return;
        }
        sb.append('\n');
        byte[] bytes = sb.toString().getBytes(StandardCharsets.UTF_8);
        try {
            forkFrame.write(bytes);
            forkRawBytes += bytes.length;
            forkRecs++;
        } catch (IOException e) {
            System.err.println("Obs: fork write failed for game " + forkGameIdx
                    + ", closing frame: " + e);
            endForkFrame();
            return;
        }
        if (forkRawBytes > RAW_CAP) {
            System.err.println("Obs: fork game " + forkGameIdx + " hit raw cap ("
                    + forkRawBytes + " > " + RAW_CAP + " bytes), truncating frame");
            try {
                forkFrame.write("{\"k\":\"end\",\"status\":\"obs_cap\",\"winner\":-1,\"turns\":-1,\"ms\":0}\n"
                        .getBytes(StandardCharsets.UTF_8));
                forkRecs++;
            } catch (IOException ignored) {
            }
            endForkFrame();
        }
    }

    private static void endForkFrame() {
        if (forkFrame == null) {
            return;
        }
        try {
            forkFrame.close();
        } catch (IOException e) {
            System.err.println("Obs: fork frame close failed for game " + forkGameIdx + ": " + e);
        }
        if (forkIdx != null) {
            forkIdx.println("{\"g\":" + forkGameIdx + ",\"off\":" + forkFileOffset
                    + ",\"clen\":" + forkCounting.count + ",\"rlen\":" + forkRawBytes
                    + ",\"seed\":" + forkGameSeed + ",\"recs\":" + forkRecs + "}");
            forkIdx.flush();
        }
        forkFileOffset += forkCounting.count;
        if (forkFile != null) {
            try {
                forkFileOffset = forkFile.getChannel().position();
            } catch (IOException ignored) {
            }
        }
        forkFrame = null;
        forkCounting = null;
        forkCurrentGame = null;
        forkGameIdx = -1;
    }

    static String q(String s) {
        if (s == null) {
            return "null";
        }
        StringBuilder sb = new StringBuilder(s.length() + 8).append('"');
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '"' || c == '\\') {
                sb.append('\\').append(c);
            } else if (c < 0x20) {
                sb.append(String.format("\\u%04x", (int) c));
            } else {
                sb.append(c);
            }
        }
        return sb.append('"').toString();
    }

    /** Counts bytes through to the file; close() flushes but never closes it. */
    private static final class CountingStream extends OutputStream {
        private final OutputStream out;
        long count;

        CountingStream(OutputStream out) {
            this.out = out;
        }

        @Override
        public void write(int b) throws IOException {
            out.write(b);
            count++;
        }

        @Override
        public void write(byte[] b, int off, int len) throws IOException {
            out.write(b, off, len);
            count += len;
        }

        @Override
        public void flush() throws IOException {
            out.flush();
        }

        @Override
        public void close() throws IOException {
            out.flush();
        }
    }
}
