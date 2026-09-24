package forge.ai.anvil;

import com.google.common.collect.Lists;

import forge.LobbyPlayer;
import forge.ai.AiPlayDecision;
import forge.card.ColorSet;
import forge.card.MagicColor;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CardCollection;
import forge.game.card.CardCollectionView;
import forge.game.combat.Combat;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.WrappedAbility;

import java.util.List;
import java.util.Set;

/**
 * Anvil's controller (override plan, M0 form): the bridged tag set is answered
 * through an AnvilBridge; every other decision inherits the heuristic AI
 * (via CensusPlayerController, so when Census logging is open every callback
 * is recorded — bridged ones tagged by="bridge", the rest implicitly
 * heuristic-fallback; provenance rule of the override plan).
 *
 * Priority semantics (M0 random-legal): options are materialized engine-side
 * (legal-actions-only invariant) as pass + engine-legal, payable spell
 * abilities + legal land drops; the bridge picks an index. A picked spell is
 * then run through the AI's canPlaySa so targets/X are pre-set the AI-path
 * way (census finding: targets and X are injected, never callbacks); if the
 * AI evaluation vetoes it (no valid targets etc.), the window passes. So M0
 * plays "random over engine-legal, AI-targeted" — documented delta from pure
 * random-legal, revisited when CastPlan lands at M1.
 */
public class PlayerControllerAnvil extends CensusPlayerController {
    public static final String TAG_PRIORITY = "mtg.priority";
    public static final String TAG_MULLIGAN = "mtg.mulligan_keep";
    public static final String TAG_TUCK = "mtg.mulligan_tuck";
    public static final String TAG_TRIGGER = "mtg.trigger";
    public static final String TAG_BINARY = "mtg.binary";
    public static final String TAG_NUMBER = "mtg.number";
    public static final String TAG_ATTACK = "mtg.attack";   // M2 D5
    public static final String TAG_BLOCK = "mtg.block";     // M2 D5
    public static final String TAG_PAY_CLASS = "mtg.pay_mana_class"; // M9 D3 §3c
    // M12 Build 3 (ADR-0105): the decision surfaces, one tag per answer shape;
    // answered through the Surfaces force hooks when the seat bridges the tag
    public static final String TAG_SURFACE_ONE = "mtg.surface.entity_one";
    public static final String TAG_SURFACE_SET = "mtg.surface.entity_set";
    public static final String TAG_SURFACE_MODE = "mtg.surface.mode";
    public static final String TAG_SURFACE_ORDER = "mtg.surface.order";   // evening 3
    public static final String TAG_SURFACE_DAMAGE = "mtg.surface.damage"; // evening 3
    public static final String TAG_SURFACE_TARGET = "mtg.surface.target"; // Build 4 (ADR-0109)
    /** Evening 4 (ADR-0105): may a SEARCH COPY bridge its payment windows?
     *  Default false — the copy-side gate: a copy pays by the engine's auto
     *  payer (the natural line) unless a PAY SurfaceDirective directs the
     *  window; the served pay head's answer on a copy was leaf noise (an
     *  untrained head deviating at random inside every copy of every read
     *  since Build 2). AnvilRun -searchpaybridge restores the old behaviour. */
    public static volatile boolean copyPayBridge = false;
    public static final String TAG_COLOR = "mtg.choose_color"; // M11

    private final AnvilBridge bridge;
    private final Set<String> bridgedTags;

    public PlayerControllerAnvil(Game game, Player p, LobbyPlayer lp, AnvilBridge bridge, Set<String> bridgedTags) {
        super(game, p, lp);
        this.bridge = bridge;
        this.bridgedTags = bridgedTags;
    }

    private boolean bridged(String tag) {
        return bridgedTags.contains(tag);
    }

    /** ADR-0105: the bridge for a surface tag this seat bridges, else null. */
    public AnvilBridge bridgeFor(String tag) {
        return bridged(tag) && bridge != null && !bridge.poisoned() ? bridge : null;
    }

    @Override
    public byte chooseColor(String message, SpellAbility sa, ColorSet colors) {
        if (!bridged(TAG_COLOR)) {
            return super.chooseColor(message, sa, colors);
        }
        List<String> options = Obs.colorOptions(colors);
        if (options.isEmpty()) {
            return super.chooseColor(message, sa, colors);
        }
        long obsSeq = Obs.decBridged(getGame(), getPlayer(), "chooseColor", options,
                "message", message, "sa", Census.str(sa));
        int pick = bridge.selectOne(TAG_COLOR, options);
        if (pick < 0 || pick >= options.size()) {
            pick = 0;
        }
        byte chosen = MagicColor.fromName(options.get(pick));
        Census.rec(getGame(), getPlayer(), "chooseColor", "by", "bridge",
                "options", options.size(), "pick", options.get(pick),
                "color", MagicColor.toLongString(chosen));
        Obs.ret(getGame(), obsSeq, MagicColor.toLongString(chosen));
        return chosen;
    }

    /**
     * Brave the Elements and other one-color effects use the plural Forge
     * callback even when exactly one color is required.  Keep the existing
     * SELECT_ONE bridge shape for that case; multi-color selection remains on
     * the inherited heuristic path until it has a dedicated action surface.
     */
    @Override
    public ColorSet chooseColors(String message, SpellAbility sa, int min, int max,
            ColorSet colors) {
        if (!bridged(TAG_COLOR) || min != 1 || max != 1) {
            return super.chooseColors(message, sa, min, max, colors);
        }
        List<String> options = Obs.colorOptions(colors);
        if (options.isEmpty()) {
            return super.chooseColors(message, sa, min, max, colors);
        }
        long obsSeq = Obs.decBridged(getGame(), getPlayer(), "chooseColors", options,
                "message", message, "sa", Census.str(sa), "min", min, "max", max);
        int pick = bridge.selectOne(TAG_COLOR, options);
        if (pick < 0 || pick >= options.size()) {
            pick = 0;
        }
        String label = options.get(pick);
        byte chosen = MagicColor.fromName(label);
        ColorSet chosenSet = ColorSet.fromNames(List.of(label));
        Census.rec(getGame(), getPlayer(), "chooseColors", "by", "bridge",
                "options", options.size(), "pick", label,
                "color", MagicColor.toLongString(chosen));
        Obs.ret(getGame(), obsSeq, MagicColor.toLongString(chosen));
        return chosenSet;
    }

    /** Does this seat answer priority over the bridge? (M7 forced-branch
     *  seat guard: both seats carry this controller class, but non-bridged
     *  seats have empty tag sets and play heuristic — forcing them would
     *  measure nothing.) */
    public boolean bridgesPriority() {
        return bridged(TAG_PRIORITY);
    }

    /**
     * D6 run-2 re-ask-on-veto (d6-vtrace-loop §6b): on an M1 CastPlan veto,
     * re-issue the priority decision with the vetoed candidate removed instead
     * of converting the window to a pass. Off = pre-amendment behavior.
     * Static because config is per-worker-JVM (AnvilRun sets it once from
     * -reask) and fork-created controllers must inherit it.
     */
    private static volatile boolean reaskOnVeto = false;
    /** Options shrink every re-ask, so termination is structural; the cap is
     *  insurance against pathologically wide windows re-vetoing in chains. */
    private static final int REASK_CAP = 8;

    /** Build 4 (09-17, ADR-0111 addendum): on a CastPlan veto of the model's pick, realize the
     *  pick through the heuristic's own planner (heuristicRealize — exactly how a search copy
     *  realizes a directed option) instead of re-asking. The model chooses the option, the
     *  engine's AI plans its details; the search's leaf values were computed under that
     *  realization. AnvilRun -vetofallback heuristic; default off = the re-ask path unchanged. */
    public static volatile boolean vetoFallbackHeuristic = false;

    public static void setReaskOnVeto(boolean v) {
        reaskOnVeto = v;
    }

    // ---- M7 forced-branch first decision (m7-plan D2) ----------------------
    // A one-shot directive on a fork copy: the drilled seat's FIRST
    // chooseSpellAbilityToPlay is forced to ACT (bridge ask with the pass
    // answer masked; §6b-style re-ask on veto with pass still masked) or HOLD
    // (pass without asking; play free afterwards). Keyed on Game identity
    // (the Obs stale-thread pattern): an abandoned hard-capped rollout thread
    // must never consume a directive armed for a later copy. WeakHashMap so
    // dead copies don't pin entries.

    public enum ForcedFirst { ACT, HOLD }

    /** Outcome of a consumed (or never-consumed) directive. CAST/HELD are the
     *  two branch-defining results; every SKIP_* drops the completion from
     *  pairing, loudly, with the reason in the labels row. */
    public enum ForcedResult {
        PENDING,            // armed, seat never asked (e.g. game ended first)
        CAST, HELD,
        SKIP_NO_OPTIONS,    // window had no castable candidates at all
        SKIP_EXHAUSTED,     // every candidate vetoed (or re-ask cap hit)
        SKIP_PASS_RESPONSE, // server answered pass despite the mask
        SKIP_NO_ONESHOT     // bridge lacks the composite path (M0 shape)
    }

    private static final class Forced {
        final ForcedFirst first;
        final String playerName;
        volatile ForcedResult result = ForcedResult.PENDING;

        Forced(ForcedFirst f, String p) {
            first = f;
            playerName = p;
        }
    }

    private static final java.util.Map<Game, Forced> forced =
            java.util.Collections.synchronizedMap(new java.util.WeakHashMap<>());

    public static void armForcedFirst(Game g, String playerName, ForcedFirst f) {
        forced.put(g, new Forced(f, playerName));
    }

    // ------------------------------------------------------------------
    // M7 D2 sequence probe (m7-plan routing pin, 2026-08-11): PERSISTENT
    // directive over an N-turn horizon — the sequence-granularity sibling
    // of the one-shot Forced above. HOLD = force-pass every bridged
    // priority cast window while turn <= untilTurn; ACT = forbid_decline
    // every window, mid-sequence exhaustion DEGRADES TO PASS (counted,
    // never a skip — a sequence arm cannot drop out mid-game the way a
    // one-shot branch can). Same Game-identity keying as Forced.

    public enum SeqMode { HOLD, ACT, OBSERVE }

    public static final class SeqDirective {
        public final SeqMode mode;
        final String playerName;
        final int untilTurn; // active while game turn <= untilTurn
        public volatile int holds = 0;    // windows force-passed (HOLD)
        public volatile int casts = 0;    // realized forced casts (ACT)
        public volatile int exhausts = 0; // ACT windows degraded to pass
        // First realized cast of the completion, as the candidate-label
        // SA string (Census.str — the model's sa_vocab basis). ADR-0054:
        // the sequence-contrastive target rewards the EVALUATED cast, so
        // the labels row must say which cast the act arm actually led with.
        public volatile String firstCastSa = null;
        // M8 D1 (m8-plan): OBSERVE mode never forces — it records the
        // seat's natural timing. First realized SPELL cast (isSpell();
        // lands and activated abilities — fetch cracks, equips — are mana
        // development, not the spell-timing axis, and would pull
        // completions into the in-window bin) + its absolute game turn,
        // and the first land-play turn as the confound check. -1 = never.
        public volatile String firstSpellSa = null;
        public volatile int firstSpellTurn = -1;
        public volatile int firstLandTurn = -1;

        SeqDirective(SeqMode m, String p, int u) {
            mode = m;
            playerName = p;
            untilTurn = u;
        }
    }

    private static final java.util.Map<Game, SeqDirective> seq =
            java.util.Collections.synchronizedMap(new java.util.WeakHashMap<>());

    public static void armSeq(Game g, String playerName, SeqMode m, int untilTurn) {
        seq.put(g, new SeqDirective(m, playerName, untilTurn));
    }

    /** Null when unarmed; counters live on the returned object. */
    public static SeqDirective seqDirective(Game g) {
        return seq.get(g);
    }

    public static void clearSeq(Game g) {
        seq.remove(g);
    }

    /** The seat's active sequence directive for the current window, or null
     *  (unarmed / other seat / horizon expired). */
    private SeqDirective activeSeq() {
        SeqDirective sd = seq.get(getGame());
        if (sd == null || !sd.playerName.equals(player.getName())
                || getGame().getPhaseHandler().getTurn() > sd.untilTurn) {
            return null;
        }
        return sd;
    }

    /** PENDING if armed but never consumed; null-safe (PENDING when unarmed —
     *  callers only read games they armed). */
    public static ForcedResult forcedResult(Game g) {
        Forced f = forced.get(g);
        return f == null ? ForcedResult.PENDING : f.result;
    }

    public static void clearForced(Game g) {
        forced.remove(g);
    }

    /** Consume the directive iff it targets this seat and is still pending.
     *  Non-matching seat leaves it armed (defensive: the first ask should
     *  always be the drilled seat — GameCopier resumes at its priority). */
    private Forced consumeForced() {
        Forced f = forced.get(getGame());
        if (f == null || f.result != ForcedResult.PENDING
                || !f.playerName.equals(player.getName())) {
            return null;
        }
        return f;
    }

    /** The certifier merge (09-23): a mainline pick hook — told the seat's
     *  natural pick at every priority ask, before the play, so a replay
     *  runner can fork at the window where the mainline ACTUALLY chose the
     *  stored option (the 09-23 smoke: the first window holding the option
     *  is often one the AI declines — heur_refuse — the cast came later in
     *  the phase). Keyed on Game identity (the directive idiom); never on a
     *  search copy (a copy is its own Game). */
    public interface PickHook {
        void picked(Player p, List<SpellAbility> picked);
    }

    private static final java.util.Map<Game, PickHook> pickHooks =
            java.util.Collections.synchronizedMap(new java.util.WeakHashMap<>());

    public static void armPickHook(Game g, PickHook h) {
        pickHooks.put(g, h);
    }

    public static void clearPickHook(Game g) {
        pickHooks.remove(g);
    }

    @Override
    public List<SpellAbility> chooseSpellAbilityToPlay() {
        // Evening 5: a mainline surface arm lives until the seat's next
        // QUIESCENT priority window — the copy's leaf, the bound the sub row's
        // ordinal was counted to (a cast spell's entity choices fire at
        // resolution, after the play returns; the first smoke took the arm
        // at the play boundary and every arm was "unfired").
        if (getGame().getStack().isEmpty()) {
            SurfaceDirective stale = SurfaceDirective.takeMainline(getGame(), player.getName());
            if (stale != null) {
                recordSurfaceArm(stale, "next_window");
            }
        }
        List<SpellAbility> picked = chooseSpellAbilityToPlayInner();
        final PickHook hook = pickHooks.get(getGame());
        if (hook != null) {
            try {
                hook.picked(player, picked);
            } catch (RuntimeException e) {
                throw e; // a poisoned bridge ends the game (protocol law)
            } catch (Exception e) {
                System.err.println("[anvil] pick hook: " + e);
            }
        }
        // M12 Build 0: a searched mainline window's row waits for the natural
        // pick (what the policy did here); complete it once, after the answer.
        SearchDirective.Pending pend = SearchDirective.takePending(getGame());
        if (pend == null) {
            return picked;
        }
        final String natural = picked == null || picked.isEmpty() ? "pass" : Census.str(picked.get(0));
        if (!pend.acts()) {
            pend.complete(natural);
            return picked;
        }
        // M12 Build 2 (m12-plan canonical shape §2, the acting rule): margin =
        // max V − V(natural); at or above the bar the behavior policy samples
        // from the leaf-value softmax — the natural line stays in the
        // distribution (exploration + a behavior logp for PG); below the bar
        // the natural pick stands. Never worse than the fallback by
        // construction: a sampled option that cannot be applied here (the
        // shape-fit class the copies did not see) falls back to the natural
        // line, counted.
        SearchDirective.Pending.Decision d = pend.decide(natural);
        if (d.actIdx < 0 || d.actIdx == d.natIdx) {
            pend.complete(natural, d, "natural");
            armSurface(pend, d);
            return picked;
        }
        final String label = pend.cands[d.actIdx];
        if (label == null) {
            Census.rec(getGame(), getPlayer(), "chooseSpellAbilityToPlay",
                    "by", d.by, "pick", "pass", "nat", natural, "margin", d.margin);
            pend.complete(natural, d, "pass");
            return null;
        }
        ForcedAsk fa = searchForcedAsk(label);
        List<SpellAbility> forced = fa.sas;
        if (forced == null || forced.isEmpty()) {
            final String why = fa.voidReason == null ? "empty" : fa.voidReason;
            Census.rec(getGame(), getPlayer(), "chooseSpellAbilityToPlay",
                    "by", d.by, "pick", label, "nat", natural, "margin", d.margin, "void", true, "vr", why);
            pend.complete(natural, d, "act_void", why);
            return picked;
        }
        Census.rec(getGame(), getPlayer(), "chooseSpellAbilityToPlay",
                "by", d.by, "pick", Census.str(forced.get(0)), "nat", natural, "margin", d.margin);
        pend.complete(natural, d, "act");
        armSurface(pend, d);
        return forced;
    }

    /** Evening 5 (ADR-0106 A): the acting rule's sampled surface answer on
     *  the acted option is armed for this seat's path to its next quiescent
     *  window (its ordinal-th callback of the kind, label-guarded); the arm
     *  is taken — fired, missed or unfired — at that window and counted in
     *  the census per kind (surfaceAct). An arm still pending here (a
     *  non-quiescent re-ask in between) is counted as stale. */
    private void armSurface(SearchDirective.Pending pend, SearchDirective.Pending.Decision d) {
        SearchDirective.Pending.SurfAnswers s = d.arm(pend);
        if (s == null) {
            return;
        }
        SurfaceDirective prev = SurfaceDirective.takeMainline(getGame(), player.getName());
        if (prev != null) {
            recordSurfaceArm(prev, "stale");
        }
        SurfaceDirective.armMainline(getGame(), player.getName(), s.kind, s.ordinal, s.answers[d.ansIdx], s.label);
    }

    private void recordSurfaceArm(SurfaceDirective d, String when) {
        Census.rec(getGame(), getPlayer(), "surfaceAct", "kind", Surfaces.KIND_NAMES[d.kind], "ord", d.ordinal,
                "outcome", d.outcome(), "n", d.ncand, "at", when, "label", d.label == null ? "" : d.label);
    }


    /** M12 Build 2: realize the search's sampled option on the mainline — a
     *  single-option forbid-decline ask (the search copy's W_FORCE shape) so
     *  the network fills the plan (targets, X, modes, payment). One attempt:
     *  a veto at apply, a pass despite the mask or an absent option is null
     *  (the caller plays the natural line and counts it). The dec is logged
     *  under by="search" so the store can tell the forced re-ask from the
     *  window's natural ask that precedes it. */
    /** The mainline's forced ask (the acting rule's sampled option): the
     *  answer, or why it voided (09-21, ADR-0114 routed — the act_void class
     *  by reason; the mainline analogue of the copies' {@code vr}). */
    private static final class ForcedAsk {
        final List<SpellAbility> sas;
        final String voidReason;

        ForcedAsk(List<SpellAbility> sas, String voidReason) {
            this.sas = sas;
            this.voidReason = voidReason;
        }
    }

    private ForcedAsk searchForcedAsk(String label) {
        SpellAbility target = null;
        for (SpellAbility sa : AnvilOptions.priorityOptions(getGame(), player)) {
            if (label.equals(Census.str(sa))) {
                target = sa;
                break;
            }
        }
        if (target == null) {
            return new ForcedAsk(null, "no_option");
        }
        if (!bridged(TAG_PRIORITY)) {
            List<SpellAbility> h = heuristicRealize(target); // the control arm: no network plan
            return new ForcedAsk(h, h == null ? "heur_refuse" : null);
        }
        List<SpellAbility> options = Lists.newArrayList(target);
        List<String> labels = Lists.newArrayList("pass", label);
        long obsSeq = Obs.decPriority(getGame(), getPlayer(), "search", options);
        CastPlanAnswer plan = bridge.priorityCastPlan(TAG_PRIORITY, labels,
                Obs.lastDecForBridge(getGame()), 0, true);
        if (plan == null) {
            Obs.ret(getGame(), obsSeq, null);
            return new ForcedAsk(null, "no_plan");
        }
        OneShot r = oneShotCast(options, plan, obsSeq, 0);
        AnvilOptions.invalidate(getGame(), player);
        if (r.vetoedOption > 0) {
            return new ForcedAsk(null, r.veto == null ? "veto" : r.veto);
        }
        return new ForcedAsk(r.sas, r.sas == null ? "pass" : null);
    }

    private List<SpellAbility> chooseSpellAbilityToPlayInner() {
        if (!bridged(TAG_PRIORITY)) {
            // M12 Build 2 control arm (ADR-0104 item 5): a HEURISTIC seat on a
            // search copy honours the directive with the heuristic's own
            // realization — the forced option is played if the AI can set it
            // up (canPlaySa), the leaf / void / pass rules are the bridged
            // seat's. Unarmed = the heuristic as ever.
            SearchDirective sr = SearchDirective.active(getGame(), player);
            if (sr == null) {
                return super.chooseSpellAbilityToPlay();
            }
            List<SpellAbility> options = Lists.newArrayList(AnvilOptions.priorityOptions(getGame(), player));
            final SearchDirective.Window w = sr.window(options, getGame().getStack().isEmpty(), getGame().getPhaseHandler().getTurn());
            if (w.kind == SearchDirective.W_PASS) {
                Census.rec(getGame(), getPlayer(), "chooseSpellAbilityToPlay",
                        "by", "search", "pick", "pass");
                return null;
            }
            if (w.kind == SearchDirective.W_LEAF || w.kind == SearchDirective.W_VOID) {
                if (w.kind == SearchDirective.W_LEAF) {
                    sr.leafPeek = Obs.peekPriority(getGame(), player, options, true);
                }
                getGame().setAnvilCapReason(w.kind == SearchDirective.W_LEAF ? "leaf" : "search_void");
                getGame().setGameOver(forge.game.GameEndReason.Draw);
                return null;
            }
            if (w.kind == SearchDirective.W_FORCE) {
                if (sr.replayNatural) {
                    // ADR-0117: the mainline's decision continued — the AI's
                    // own chooser (the same state and RNG stream as the
                    // mainline's ask), the pick verified against the coordinate
                    List<SpellAbility> nat = super.chooseSpellAbilityToPlay();
                    String got = nat == null || nat.isEmpty() ? "pass" : Census.str(nat.get(0));
                    if (!sr.optionLabel.equals(got)) {
                        searchVoid(sr, "diverged:" + got); // the copy's own pick, for the census
                        return null;
                    }
                    return nat;
                }
                List<SpellAbility> r = heuristicRealize(w.ask.get(0));
                if (r == null) {
                    searchVoid(sr, "heur_refuse");
                }
                return r;
            }
            return super.chooseSpellAbilityToPlay();
        }
        // Mutable copy: re-ask removes vetoed candidates between attempts.
        List<SpellAbility> options =
                Lists.newArrayList(AnvilOptions.priorityOptions(getGame(), player));

        Forced fd = consumeForced();
        if (fd != null && fd.first == ForcedFirst.HOLD) {
            // Forced hold: pass this window without asking; free afterwards.
            fd.result = ForcedResult.HELD;
            Census.rec(getGame(), getPlayer(), "chooseSpellAbilityToPlay",
                    "by", "forced", "pick", "pass");
            return null;
        }
        boolean forcedAct = fd != null; // fd.first == ACT
        if (forcedAct && options.isEmpty()) {
            fd.result = ForcedResult.SKIP_NO_OPTIONS;
            return null;
        }

        // Sequence directive (persistent, N-turn horizon). One-shot Forced
        // takes precedence if both are somehow armed (they never are).
        SeqDirective sd = fd == null ? activeSeq() : null;
        if (sd != null && sd.mode == SeqMode.HOLD) {
            sd.holds++;
            Census.rec(getGame(), getPlayer(), "chooseSpellAbilityToPlay",
                    "by", "seq", "pick", "pass");
            return null;
        }
        boolean seqAct = sd != null && sd.mode == SeqMode.ACT;
        if (seqAct && options.isEmpty()) {
            sd.exhausts++; // nothing to force this window; arm plays on
            return null;
        }

        // M10 schedule directive (m10-ceiling-spec "Engine build owed"): the
        // within-turn -forceschedule arm. Only one directive genre is armed
        // per copy; Forced/Seq take structural precedence. The window rule
        // (land-first / next-item force / degrade-and-count / hold) lives in
        // ScheduleDirective.window; this path only masks the ask and reports
        // outcomes back.
        ScheduleDirective sc = (fd == null && sd == null)
                ? ScheduleDirective.active(getGame(), player) : null;
        boolean schedForce = false;
        if (sc != null) {
            final PhaseType phase = getGame().getPhaseHandler().getPhase();
            final boolean quiescentMain = getGame().getStack().isEmpty()
                    && (phase == PhaseType.MAIN1 || phase == PhaseType.MAIN2);
            final ScheduleDirective.Window w = sc.window(options, quiescentMain);
            if (w.kind == ScheduleDirective.W_PASS) {
                Census.rec(getGame(), getPlayer(), "chooseSpellAbilityToPlay",
                        "by", "sched", "pick", "pass");
                return null;
            }
            if (w.kind == ScheduleDirective.W_NATURAL) {
                sc = null; // degraded at this window: natural play from here
            } else {
                options = w.ask; // masked forbid-decline ask (land or item)
                schedForce = true;
            }
        }

        // M12 Build 0 search copy (SearchDirective): one genre per copy, so
        // this only ever fires when nothing above is armed. Forces the
        // searched option at the copy's first window of this seat, ends the
        // copy at the seat's next quiescent window (the leaf — fork A) with
        // the leaf's peek captured, natural play in between.
        SearchDirective sr = (fd == null && sd == null && sc == null)
                ? SearchDirective.active(getGame(), player) : null;
        if (sr != null) {
            final SearchDirective.Window w = sr.window(options, getGame().getStack().isEmpty(), getGame().getPhaseHandler().getTurn());
            if (w.kind == SearchDirective.W_PASS) {
                Census.rec(getGame(), getPlayer(), "chooseSpellAbilityToPlay",
                        "by", "search", "pick", "pass");
                return null;
            }
            if (w.kind == SearchDirective.W_LEAF || w.kind == SearchDirective.W_VOID) {
                if (w.kind == SearchDirective.W_LEAF) {
                    sr.leafPeek = Obs.peekPriority(getGame(), player, options, true);
                }
                getGame().setAnvilCapReason(w.kind == SearchDirective.W_LEAF ? "leaf" : "search_void");
                getGame().setGameOver(forge.game.GameEndReason.Draw);
                return null;
            }
            if (w.kind == SearchDirective.W_FORCE) {
                if (sr.heuristicForce) {
                    // ADR-0114: the void-rescue copy — the forced option realized
                    // by the heuristic's planner on a bridged seat (the plan or the
                    // refusal recorded on the directive); the seat's later windows
                    // stay the network's, as on every copy. Recording only.
                    List<SpellAbility> hr = heuristicForce(w.ask.get(0), sr);
                    if (hr == null) {
                        searchVoid(sr, "heur_refuse");
                    }
                    return hr;
                }
                options = w.ask; // single-option forbid-decline ask
                schedForce = true;
            }
        }

        for (int attempt = 0;; attempt++) {
            // Index 0 = pass; one round-trip per attempt.
            List<String> labels = Lists.newArrayListWithCapacity(options.size() + 1);
            labels.add("pass");
            for (SpellAbility sa : options) {
                labels.add(Census.str(sa));
            }
            // Structured-opts dec (same basis as the corpus label path) so D8
            // eval games are analyzable trajectories and ret() can label "oi".
            // A re-ask mints a fresh seq; the vetoed dec's ret(null) has
            // already run, so the single-slot oi bookkeeping is clear.
            long obsSeq = Obs.decPriority(getGame(), getPlayer(), "bridge", options);

            // M1 one-shot: the composite CastPlan path; null = M0-shape bridge.
            CastPlanAnswer plan = bridge.priorityCastPlan(TAG_PRIORITY, labels,
                    Obs.lastDecForBridge(getGame()), attempt, forcedAct || seqAct || schedForce);
            if (plan != null) {
                OneShot r = oneShotCast(options, plan, obsSeq, attempt);
                if (r.sas != null || r.vetoedOption > 0) {
                    // Any cast ATTEMPT (realized or vetoed) ran canPlaySa on
                    // option SAs (targets/X mutation) — the seat's cached
                    // mask must not survive into the next window. A realized
                    // cast would invalidate via the timestamp anyway; the
                    // veto-then-pass case is the one that would not.
                    AnvilOptions.invalidate(getGame(), player);
                }
                if (r.vetoedOption <= 0) {
                    if (forcedAct) {
                        // sas null under the mask = server passed anyway
                        // (candidate-set mismatch or transport-failure PASS);
                        // the pair drops, loudly.
                        fd.result = r.sas != null ? ForcedResult.CAST
                                : ForcedResult.SKIP_PASS_RESPONSE;
                    } else if (seqAct) {
                        if (r.sas != null) {
                            sd.casts++;
                            if (sd.firstCastSa == null && !r.sas.isEmpty()) {
                                sd.firstCastSa = Census.str(r.sas.get(0));
                            }
                        } else {
                            sd.exhausts++; // server passed despite the mask
                        }
                    } else if (schedForce && sc != null) {
                        // realized cast advances the schedule (or settles the
                        // land question); a pass despite the mask degrades.
                        sc.onCast(r.sas != null && !r.sas.isEmpty() ? r.sas.get(0) : null);
                    } else if (schedForce && sr != null && (r.sas == null || r.sas.isEmpty())) {
                        // search copy: the server passed despite the single-
                        // option forbid-decline mask — the option is VOID here.
                        searchVoid(sr, "pass_masked");
                    } else if (sd != null && sd.mode == SeqMode.OBSERVE
                            && r.sas != null) {
                        // M8 D1: pure recording — the ask above ran exactly
                        // as unarmed natural (forced flag false, no re-ask
                        // semantics change), so counters stay untouched.
                        int t = getGame().getPhaseHandler().getTurn();
                        for (SpellAbility s : r.sas) {
                            if (s.isLandAbility()) {
                                if (sd.firstLandTurn < 0) {
                                    sd.firstLandTurn = t;
                                }
                            } else if (s.isSpell() && sd.firstSpellSa == null) {
                                sd.firstSpellSa = Census.str(s);
                                sd.firstSpellTurn = t;
                            }
                        }
                    }
                    return r.sas; // realized cast, model pass, or oor pass
                }
                if (vetoFallbackHeuristic && !forcedAct && !seqAct && !schedForce && r.vetoedOption > 0) {
                    SpellAbility pick = options.get(r.vetoedOption - 1);
                    List<SpellAbility> hr = heuristicRealize(pick);
                    Census.rec(getGame(), getPlayer(), "chooseSpellAbilityToPlay", "by", "bridge_hplan",
                            "pick", Census.str(pick), "realized", hr != null);
                    if (hr != null) {
                        Obs.ret(getGame(), obsSeq, hr);
                        return hr;
                    }
                }
                if (forcedAct || seqAct || schedForce) {
                    // Forced/seq/sched act re-asks on veto regardless of the
                    // global §6b flag: the arm is DEFINED as the best
                    // realizable cast under the mask. Exhaustion: one-shot =
                    // skip (pair drops); sequence = degrade to pass, counted;
                    // schedule = degrade-and-count (land asks just settle).
                    if (attempt + 1 >= REASK_CAP) {
                        if (forcedAct) {
                            fd.result = ForcedResult.SKIP_EXHAUSTED;
                        } else if (seqAct) {
                            sd.exhausts++;
                        } else if (sc != null) {
                            sc.onExhaust("veto_cap");
                        } else if (sr != null) {
                            searchVoid(sr, "veto_cap");
                        }
                        return null;
                    }
                } else if (!reaskOnVeto || attempt + 1 >= REASK_CAP) {
                    return null; // pre-amendment behavior: veto = pass
                }
                SpellAbility vetoed = options.get(r.vetoedOption - 1);
                if (plan.hostLevel && vetoed.getHostCard() != null) {
                    // Host-level plans exhausted the host's whole ladder.
                    final Card host = vetoed.getHostCard();
                    options.removeIf(sa -> sa.getHostCard() == host);
                } else {
                    options.remove(r.vetoedOption - 1);
                }
                if (options.isEmpty()) {
                    if (forcedAct) {
                        fd.result = ForcedResult.SKIP_EXHAUSTED;
                    } else if (seqAct) {
                        sd.exhausts++;
                    } else if (schedForce && sc != null) {
                        sc.onExhaust("veto");
                    } else if (schedForce && sr != null) {
                        searchVoid(sr, r.veto == null ? "veto" : r.veto); // the forced option vetoed at apply
                    }
                    return null; // only pass remains; nothing left to ask
                }
                continue;
            }
            if (forcedAct) {
                // M0-shape bridge can't honor the mask; skip, drop the pair.
                fd.result = ForcedResult.SKIP_NO_ONESHOT;
            } else if (seqAct) {
                sd.exhausts++; // M0-shape bridge can't honor the mask
                return null;
            } else if (schedForce && sc != null) {
                sc.onExhaust("no_oneshot"); // M0-shape bridge can't honor the mask
                return null;
            } else if (schedForce && sr != null) {
                searchVoid(sr, "no_oneshot");
                return null;
            }
            return selectOnePick(options, labels, obsSeq);
        }
    }

    /** M12 Build 2 control arm: the heuristic's own realization of one option
     *  (the M0 selectOne rule — canPlaySa sets targets / X, a veto = null). */
    private List<SpellAbility> heuristicRealize(SpellAbility sa) {
        AnvilOptions.invalidate(getGame(), player);
        if (!sa.isLandAbility() && getAi().canPlaySa(sa) != AiPlayDecision.WillPlay) {
            return null;
        }
        return Lists.newArrayList(sa);
    }

    /** ADR-0114 (the void-rescue instrument): the heuristic's realization of
     *  the forced option on a rescue copy — the realized plan (targets, X;
     *  Obs.planJson) or the AI's refusal recorded on the directive. */
    private List<SpellAbility> heuristicForce(SpellAbility sa, SearchDirective sr) {
        AnvilOptions.invalidate(getGame(), player);
        if (!sa.isLandAbility()) {
            AiPlayDecision d = getAi().canPlaySa(sa);
            if (d != AiPlayDecision.WillPlay) {
                sr.refuse = d.name();
                return null;
            }
        }
        sr.plan = Obs.planJson(sa);
        Census.rec(getGame(), getPlayer(), "chooseSpellAbilityToPlay",
                "by", "search_hplan", "pick", Census.str(sa));
        return Lists.newArrayList(sa);
    }

    /** Search copy (M12 Build 0): the forced option could not be applied
     *  (vetoed at apply / unhonored mask) — the candidate is VOID and the
     *  copy ends; a pass here would mislabel the pass leaf as this option. */
    private void searchVoid(SearchDirective sr, String reason) {
        sr.outcome = "void";
        sr.voidReason = reason;
        getGame().setAnvilCapReason("search_void");
        getGame().setGameOver(forge.game.GameEndReason.Draw);
    }

    /** M0 selectOne path (never re-asks; heuristic canPlaySa veto = pass). */
    private List<SpellAbility> selectOnePick(List<SpellAbility> options, List<String> labels,
            long obsSeq) {
        int pick = bridge.selectOne(TAG_PRIORITY, labels);
        if (pick == 0) {
            Census.rec(getGame(), getPlayer(), "chooseSpellAbilityToPlay",
                    "by", "bridge", "options", options.size(), "pick", "pass");
            Obs.ret(getGame(), obsSeq, null);
            return null;
        }
        SpellAbility chosen = options.get(pick - 1);
        // Non-pass pick: canPlaySa below mutates the chosen SA either way.
        AnvilOptions.invalidate(getGame(), player);
        if (!chosen.isLandAbility() && getAi().canPlaySa(chosen) != AiPlayDecision.WillPlay) {
            // Targets/X could not be set up; window passes. Counted so the
            // veto rate is visible (it biases the pick toward AI-playable).
            Census.rec(getGame(), getPlayer(), "chooseSpellAbilityToPlay",
                    "by", "bridge", "options", options.size(), "pick", Census.str(chosen), "veto", true);
            Obs.ret(getGame(), obsSeq, null);
            return null;
        }
        Census.rec(getGame(), getPlayer(), "chooseSpellAbilityToPlay",
                "by", "bridge", "options", options.size(), "pick", Census.str(chosen));
        Obs.ret(getGame(), obsSeq, chosen);
        return Lists.newArrayList(chosen);
    }

    /** One-shot attempt outcome: sas = the answer (null = window passes);
     *  vetoedOption = the 1-based option index the realizer vetoed (0 = no
     *  veto — the caller only re-asks on a veto). */
    private static final class OneShot {
        final List<SpellAbility> sas;
        final int vetoedOption;

        /** ADR-0114: the realizer's veto code on a veto (null otherwise). */
        final String veto;

        OneShot(List<SpellAbility> sas, int vetoedOption) {
            this(sas, vetoedOption, null);
        }

        OneShot(List<SpellAbility> sas, int vetoedOption, String veto) {
            this.sas = sas;
            this.vetoedOption = vetoedOption;
            this.veto = veto;
        }
    }

    /**
     * M1 D8: realize a composite CastPlan answer. The realizer adjudicates
     * legality only (never the heuristic's judgment — the M0 65% veto class);
     * a veto passes the window (or re-asks, D6 run-2), with the reason in the
     * census/provenance log. Census lines gain "reask"=attempt on re-asked
     * attempts (attempt > 0), so a success line with reask>0 = a rescue.
     */
    private OneShot oneShotCast(List<SpellAbility> options, CastPlanAnswer plan,
            long obsSeq, int attempt) {
        if (plan.optionIndex <= 0 || plan.optionIndex > options.size()) {
            boolean oor = plan.optionIndex != 0;
            if (attempt > 0) {
                Census.rec(getGame(), getPlayer(), "chooseSpellAbilityToPlay",
                        "by", "bridge", "options", options.size(), "pick", "pass",
                        "oneshot", true, "oor", oor, "reask", attempt);
            } else {
                Census.rec(getGame(), getPlayer(), "chooseSpellAbilityToPlay",
                        "by", "bridge", "options", options.size(), "pick", "pass",
                        "oneshot", true, "oor", oor);
            }
            Obs.ret(getGame(), obsSeq, null);
            return new OneShot(null, 0);
        }
        SpellAbility picked = options.get(plan.optionIndex - 1);
        List<SpellAbility> hostSas;
        if (plan.hostLevel && picked.getHostCard() != null) {
            hostSas = Lists.newArrayListWithCapacity(2);
            for (SpellAbility sa : options) {
                if (sa.getHostCard() == picked.getHostCard()) {
                    hostSas.add(sa);
                }
            }
        } else {
            hostSas = Lists.newArrayList(picked);
        }
        CastPlanRealizer.Result r = CastPlanRealizer.realize(getGame(), player, hostSas, plan);
        if (r.sa == null) {
            if (attempt > 0) {
                Census.rec(getGame(), getPlayer(), "chooseSpellAbilityToPlay",
                        "by", "bridge", "options", options.size(), "pick", Census.str(picked),
                        "oneshot", true, "veto", r.veto, "hostSas", r.hostSas, "fits", r.fitCount,
                        "ignoredTargets", r.ignoredTargets,
                        "reask", attempt);
            } else {
                Census.rec(getGame(), getPlayer(), "chooseSpellAbilityToPlay",
                        "by", "bridge", "options", options.size(), "pick", Census.str(picked),
                        "oneshot", true, "veto", r.veto, "hostSas", r.hostSas, "fits", r.fitCount,
                        "ignoredTargets", r.ignoredTargets);
            }
            Obs.ret(getGame(), obsSeq, null);
            return new OneShot(null, plan.optionIndex, r.veto);
        }
        if (attempt > 0) {
            Census.rec(getGame(), getPlayer(), "chooseSpellAbilityToPlay",
                    "by", "bridge", "options", options.size(), "pick", Census.str(r.sa),
                    "oneshot", true, "rung", r.rung, "hostSas", r.hostSas, "fits", r.fitCount,
                    "divided", r.divided, "ignoredTargets", r.ignoredTargets, "reask", attempt);
        } else {
            Census.rec(getGame(), getPlayer(), "chooseSpellAbilityToPlay",
                    "by", "bridge", "options", options.size(), "pick", Census.str(r.sa),
                    "oneshot", true, "rung", r.rung, "hostSas", r.hostSas, "fits", r.fitCount,
                    "divided", r.divided, "ignoredTargets", r.ignoredTargets);
        }
        Obs.ret(getGame(), obsSeq, Lists.newArrayList(r.sa));
        return new OneShot(Lists.newArrayList(r.sa), 0);
    }

    /** London mulligans are rules-unbounded (hand redraws to 7 every time), so
     *  a pathological bridge answer loops the game forever at turn 0 (D8
     *  smoke 1). Insurance cap, far beyond any sane line. */
    private static final int MULLIGAN_CAP = 12;
    private int mulligansAsked;

    @Override
    public boolean mulliganKeepHand(Player firstPlayer, int cardsToReturn) {
        if (!bridged(TAG_MULLIGAN)) {
            return super.mulliganKeepHand(firstPlayer, cardsToReturn);
        }
        long obsSeq = Obs.decBridged(getGame(), getPlayer(), "mulliganKeepHand", null);
        boolean keep = bridge.bool(TAG_MULLIGAN);
        if (!keep && ++mulligansAsked >= MULLIGAN_CAP) {
            keep = true;
            Census.rec(getGame(), getPlayer(), "mulliganKeepHand", "by", "bridge",
                    "keep", true, "mull_cap", true);
        } else {
            Census.rec(getGame(), getPlayer(), "mulliganKeepHand", "by", "bridge", "keep", keep);
        }
        Obs.ret(getGame(), obsSeq, keep);
        return keep;
    }

    @Override
    public CardCollectionView tuckCardsViaMulligan(CardCollectionView hand, int cardsToReturn) {
        if (!bridged(TAG_TUCK)) {
            return super.tuckCardsViaMulligan(hand, cardsToReturn);
        }
        List<String> handLabels = Lists.newArrayListWithCapacity(hand.size());
        for (Card c : hand) {
            handLabels.add(Census.str(c));
        }
        long obsSeq = Obs.decBridged(getGame(), getPlayer(), "tuckCardsViaMulligan", handLabels);
        int[] picks = bridge.selectK(TAG_TUCK, hand.size(), cardsToReturn);
        CardCollection tuck = new CardCollection();
        for (int i : picks) {
            tuck.add(hand.get(i));
        }
        Census.rec(getGame(), getPlayer(), "tuckCardsViaMulligan", "by", "bridge", "n", tuck.size());
        Obs.ret(getGame(), obsSeq, tuck);
        return tuck;
    }

    @Override
    public boolean confirmTrigger(WrappedAbility sa) {
        if (!bridged(TAG_TRIGGER)) {
            return super.confirmTrigger(sa);
        }
        long obsSeq = Obs.decBridged(getGame(), getPlayer(), "confirmTrigger", null, "sa", Census.str(sa));
        boolean yes = bridge.bool(TAG_TRIGGER);
        Census.rec(getGame(), getPlayer(), "confirmTrigger", "by", "bridge", "yes", yes);
        Obs.ret(getGame(), obsSeq, yes);
        return yes;
    }

    @Override
    public boolean playTrigger(Card host, WrappedAbility wrapperAbility, boolean isMandatory) {
        if (isMandatory || !bridged(TAG_TRIGGER)) {
            return super.playTrigger(host, wrapperAbility, isMandatory);
        }
        long obsSeq = Obs.decBridged(getGame(), getPlayer(), "playTrigger", null,
                "host", Census.str(host), "wrapperAbility", Census.str(wrapperAbility));
        boolean yes = bridge.bool(TAG_TRIGGER);
        Census.rec(getGame(), getPlayer(), "playTrigger", "by", "bridge", "yes", yes);
        Obs.ret(getGame(), obsSeq, yes);
        return yes && super.playTrigger(host, wrapperAbility, true);
    }

    @Override
    public boolean chooseBinary(SpellAbility sa, String question, BinaryChoiceType kindOfChoice, Boolean defaultVal) {
        if (!bridged(TAG_BINARY)) {
            return super.chooseBinary(sa, question, kindOfChoice, defaultVal);
        }
        long obsSeq = Obs.decBridged(getGame(), getPlayer(), "chooseBinary", null,
                "question", question, "kind", String.valueOf(kindOfChoice));
        boolean v = bridge.bool(TAG_BINARY);
        Census.rec(getGame(), getPlayer(), "chooseBinary", "by", "bridge", "v", v);
        Obs.ret(getGame(), obsSeq, v);
        return v;
    }

    @Override
    public int chooseNumber(SpellAbility sa, String title, int min, int max) {
        if (!bridged(TAG_NUMBER)) {
            return super.chooseNumber(sa, title, min, max);
        }
        long obsSeq = Obs.decBridged(getGame(), getPlayer(), "chooseNumber", null,
                "title", title, "min", min, "max", max);
        int v = bridge.intInRange(TAG_NUMBER, min, max);
        Census.rec(getGame(), getPlayer(), "chooseNumber", "by", "bridge", "v", v);
        Obs.ret(getGame(), obsSeq, v);
        return v;
    }

    @Override
    public int chooseNumber(SpellAbility sa, String title, List<Integer> values, Player relatedPlayer) {
        if (!bridged(TAG_NUMBER)) {
            return super.chooseNumber(sa, title, values, relatedPlayer);
        }
        List<String> valueLabels = Lists.newArrayListWithCapacity(values.size());
        for (Integer n : values) {
            valueLabels.add(String.valueOf(n));
        }
        long obsSeq = Obs.decBridged(getGame(), getPlayer(), "chooseNumber", valueLabels, "title", title);
        int v = values.get(bridge.selectOne(TAG_NUMBER, valueLabels));
        Census.rec(getGame(), getPlayer(), "chooseNumber", "by", "bridge", "v", v);
        Obs.ret(getGame(), obsSeq, v);
        return v;
    }

    /**
     * M9 D3 (§3c): conscious mana payment (m9-payment-surface-spec.md).
     * In-scope windows (effect=false, nonzero mana) run legality-derived
     * class enumeration; consequential windows (≥2 classes) bridge as
     * SELECT_ONE over {auto} ∪ classes on tag mtg.pay_mana_class — auto is
     * option 0, so a server that always answers 0 (or fallback/echo) is
     * bit-identical to today. A class answer executes float-then-apply
     * (PaymentEnumerator.executeDirected, the ADR-0065 primitive) and the
     * heuristic path completes the payment from the float (pool-first).
     * Failure semantics per spec §7: directed_salvage / directed_fail are
     * census reason codes, NEVER vetoes — D5's mechanism read must not
     * measure itself. Auto/heuristic payment goes through ComputerUtilMana
     * directly (calling super would double-log the census record).
     */
    /** Non-null while inside a bridged payCombatCost (the cousins touch,
     *  2026-08-28): the attacker/blocker whose combat cost is being paid.
     *  Marks the nested payManaCost window (which arrives with effect=true
     *  and a degenerate EmptySa) as in-scope combat — CostAdjustment is a
     *  no-op for effect costs, so these are the CLEAN enumeration case
     *  (no costmod branch). Game-thread-scoped, cleared in a finally. */
    private forge.game.card.Card combatPayCard;

    /**
     * M10 cousins touch: combat costs (CantAttackUnless/CantBlockUnless/
     * OptionalAttackCost statics) route through PlayerController.payManaCost
     * on the AI path — the effect=true gate was the single blocker. The
     * marker widens the nested window's scope; auto (option 0) remains
     * bit-identical to today's playNoStack behavior. Pay-vs-decline itself
     * stays heuristic (ComputerUtilCost.canPayCost upstream) — that is
     * ADR-0080's re-deferred genre; this is only HOW to pay.
     */
    @Override
    public boolean payCombatCost(forge.game.card.Card c, forge.game.cost.Cost cost,
            SpellAbility sa, String prompt) {
        if (!bridged(TAG_PAY_CLASS)) {
            return super.payCombatCost(c, cost, sa, prompt);
        }
        combatPayCard = c;
        try {
            return super.payCombatCost(c, cost, sa, prompt);
        } finally {
            combatPayCard = null;
        }
    }

    @Override
    public boolean payManaCost(forge.card.mana.ManaCost toPay, forge.game.cost.CostPartMana costPartMana,
            SpellAbility sa, String prompt, forge.game.mana.ManaConversionMatrix matrix, boolean effect) {
        // Cousins hygiene: any armed directive from a previous window is
        // stale by construction at the next window's entry (the certify
        // path's arm is consumed inside its own super auto-pay and has no
        // post-super hook to clear itself).
        CousinDirective.disarm(getPlayer());
        final boolean combat = combatPayCard != null;
        // M10 schedule directive (m10-ceiling-spec knob c): a JOINT arm owns
        // every in-scope payment window on its target turn regardless of the
        // serve config's bridged tags — schedule-consistent selection is
        // engine-side and deterministic given the schedule. A null return
        // falls through to the normal path (costmod / non-consequential /
        // enumeration error), reason-counted on the directive. Combat
        // windows are not schedule-owned (slots are casts).
        if (!effect && toPay != null && !toPay.isZero()) {
            final ScheduleDirective sdir = ScheduleDirective.paymentDirective(getGame(), player);
            if (sdir != null) {
                final Boolean paid = schedulePay(sdir, toPay, sa, effect);
                if (paid != null) {
                    return paid;
                }
            }
        }
        // M12 Build 3 evening 4 (ADR-0105): payment on a SEARCH COPY — the
        // copy-side gate (never bridged unless -searchpaybridge), the acting
        // seat's in-scope windows traced as a PAY surface / directed by a PAY
        // SurfaceDirective (copyPay); every other copy window pays auto.
        final boolean onCopy = SearchDirective.directive(getGame()) != null;
        if (onCopy && !copyPayBridge) {
            final SearchDirective sr = SearchDirective.active(getGame(), player);
            if (sr == null || !sr.applied || (effect && !combat) || toPay == null || toPay.isZero()) {
                return super.payManaCost(toPay, costPartMana, sa, prompt, matrix, effect);
            }
            return copyPay(sr, toPay, sa, effect, combat);
        }
        if (effect && !combat && !onCopy && PaymentTelemetry.enabled) {
            // Evening 4: the resolution-effect payment census (the ADR-0077
            // queue's item 3, deferred twice — this row is its measured
            // argument: how many of the ~51/game carry a real choice). Under
            // -paytelemetry only; scratch RNG (the auto-payer probe draws).
            resolutionEffectCensus(toPay, sa);
        }
        if (AnvilOptions.PAYRESCUE && !bridged(TAG_PAY_CLASS) && !(effect && !combat)
                && toPay != null && !toPay.isZero()) {
            // evening 4 (the rescue flag): an UNBRIDGED seat's forced window
            // (auto cannot pay, a plan exists) pays directed by the
            // enumerator's first plan — the natural line under the flag; a
            // null = not forced, the ordinary path
            final Boolean paid = rescuePay(toPay, sa, effect, combat);
            if (paid != null) {
                return paid;
            }
        }
        if (!bridged(TAG_PAY_CLASS) || (effect && !combat) || toPay == null || toPay.isZero()) {
            return super.payManaCost(toPay, costPartMana, sa, prompt, matrix, effect);
        }
        final PaymentEnumerator.Result r;
        final boolean conseq;
        final boolean forced;
        try {
            // cost-modified windows: out-of-scope v1 (spec §12b) — raw toPay
            // diverges from what auto pays; goal enumeration would target
            // the wrong cost. Auto + kv, never bridged. Combat windows skip
            // the detector: CostAdjustment never adjusts effect costs, and
            // the EmptySa's host is the attacker (its own casting keywords
            // must not trip the scan).
            if (!combat && PaymentEnumerator.costModified(sa)) {
                Census.rec(getGame(), getPlayer(), "payManaCost", "by", "auto",
                        "sa", Census.str(sa), "effect", false, "costmod", true);
                long s = Obs.dec(getGame(), getPlayer(), "payManaCost",
                        "sa", Census.str(sa), "effect", false);
                boolean paid = autoPay(toPay, sa, effect);
                Obs.ret(getGame(), s, paid);
                return paid;
            }
            r = quietProbe(() -> PaymentEnumerator.enumerate(getPlayer(), sa, toPay));
            final boolean auto = quietProbe(() -> PaymentEnumerator.autoPayable(getPlayer(), sa, toPay, effect));
            conseq = PaymentEnumerator.consequential(r, auto);
            forced = r.planCount >= 1 && !auto; // spec §12c: ¬costmod already holds here
        } catch (Exception e) {
            // enumeration must never kill the game thread: fall back to
            // today's behavior (auto), loudly reason-coded — never a veto.
            // Mirrors the non-consequential path (super would double-record).
            Census.rec(getGame(), getPlayer(), "payManaCost", "by", "auto",
                    "sa", Census.str(sa), "effect", effect,
                    "enumerr", e.getClass().getSimpleName());
            long s2 = Obs.dec(getGame(), getPlayer(), "payManaCost",
                    "sa", Census.str(sa), "effect", effect);
            boolean paid = autoPay(toPay, sa, effect);
            Obs.ret(getGame(), s2, paid);
            return paid;
        }
        if (!conseq) {
            // non-consequential windows never bridge (the sparsity contract);
            // the flag telemetry still lands on the census record. Recorded
            // AFTER autoPay so the §12b retrospective backstop (0 plans yet
            // auto pays = static-detector leak) is one kv, not a join.
            long s = Obs.dec(getGame(), getPlayer(), "payManaCost",
                    "sa", Census.str(sa), "effect", effect);
            boolean paid = autoPay(toPay, sa, effect);
            Census.rec(getGame(), getPlayer(), "payManaCost", "by", "auto",
                    "sa", Census.str(sa), "effect", effect,
                    "goals", r.options.size(), "plans", r.planCount, "conseq", false,
                    "trunc", r.goalCapHit, "nodecap", r.nodeCapHit,
                    "costmod_late", r.planCount == 0 && paid);
            Obs.ret(getGame(), s, paid);
            return paid;
        }
        final List<String> labels = paymentOptionLabels(r);
        // combat windows (cousins touch): extra kvs only there — non-combat
        // rows stay byte-shaped for the banked observe-frame joins
        final Object[] decKv = kvPlus(combat, new Object[] {
                "sa", Census.str(sa), "cost", String.valueOf(toPay), "effect", effect,
                "fpool", floatingPool(), "goals", r.options.size(), "plans", r.planCount,
                "trunc", r.goalCapHit, "forced", forced });
        long obsSeq = Obs.decBridged(getGame(), getPlayer(), "payManaCost", labels, decKv);
        int pick = bridge.selectOne(TAG_PAY_CLASS, labels);
        if (pick <= 0 || pick > r.options.size()) {
            if (AnvilOptions.PAYRESCUE && forced && !r.options.isEmpty()) {
                // evening 4 (the rescue flag): auto cannot pay here — the
                // enumerator's first plan is the natural line
                final boolean paid = directedPay(r.options.get(0).plan, toPay, sa, effect, "rescue");
                Census.rec(getGame(), getPlayer(), "payManaCost", kvPlus(combat, new Object[] {
                        "by", "bridge", "options", labels.size(), "pick", "auto", "rescue", true,
                        "paid", paid, "goals", r.options.size(), "plans", r.planCount, "conseq", true,
                        "trunc", r.goalCapHit, "forced", true }));
                Obs.ret(getGame(), obsSeq, "rescue:" + paid);
                return paid;
            }
            boolean paid = autoPay(toPay, sa, effect);
            Census.rec(getGame(), getPlayer(), "payManaCost", kvPlus(combat, new Object[] {
                    "by", "bridge",
                    "options", labels.size(), "pick", "auto", "paid", paid,
                    "goals", r.options.size(), "plans", r.planCount, "conseq", true,
                    "trunc", r.goalCapHit, "forced", forced }));
            Obs.ret(getGame(), obsSeq, "auto:" + paid);
            return paid;
        }
        final PaymentEnumerator.PaymentClass pc = r.options.get(pick - 1).plan;
        // Cousins: ALWAYS armed on a directed payment — an empty map is the
        // correct directive on a cousin-keyword spell whose chosen plan uses
        // no cousin resources (natural play would convoke/delve what the
        // plan deliberately spares). No-op for non-cousin spells.
        final CousinDirective.Armed cousins = CousinDirective.arm(getPlayer(), pc);
        final PaymentEnumerator.ExecOutcome out;
        final boolean paid;
        try {
            out = PaymentEnumerator.executeDirected(getPlayer(), pc);
            paid = autoPay(toPay, sa, effect); // completes from the float, pool-first
        } finally {
            CousinDirective.disarm(getPlayer());
        }
        int residue = getPlayer().getManaPool().totalMana();
        String exec = !paid ? "directed_fail"
                : out == PaymentEnumerator.ExecOutcome.DIRECTED_OK ? "directed_ok" : "directed_salvage";
        Object[] recKv = kvPlus(combat, new Object[] {
                "by", "bridge",
                "options", labels.size(), "pick", pick, "exec", exec, "paid", paid,
                "float_residue", residue,
                "goals", r.options.size(), "plans", r.planCount, "conseq", true,
                "trunc", r.goalCapHit, "forced", forced });
        if (pc.hasCousins()) {
            recKv = java.util.Arrays.copyOf(recKv, recKv.length + 2);
            recKv[recKv.length - 2] = "cousins";
            recKv[recKv.length - 1] = cousins.summary();
        }
        Census.rec(getGame(), getPlayer(), "payManaCost", recKv);
        Obs.ret(getGame(), obsSeq, exec);
        return paid;
    }

    /** Append the combat-marker kvs when marked (cousins touch): non-combat
     *  rows keep their pre-touch kv shape. */
    private Object[] kvPlus(boolean combat, Object[] kv) {
        if (!combat) {
            return kv;
        }
        final Object[] out = java.util.Arrays.copyOf(kv, kv.length + 4);
        out[kv.length] = "combat";
        out[kv.length + 1] = true;
        out[kv.length + 2] = "cmb";
        out[kv.length + 3] = combatPayCard.getName() + "#" + combatPayCard.getId();
        return out;
    }

    /** Current floating pool by mana type (spec §6 window context: WUBRGC). */
    private String floatingPool() {
        return floatingPool(getPlayer());
    }

    /** Static variant for the certify observe path (PaymentTelemetry). */
    static String floatingPool(forge.game.player.Player p) {
        StringBuilder sb = new StringBuilder(12);
        for (byte t : forge.card.mana.ManaAtom.MANATYPES) {
            if (sb.length() > 0) {
                sb.append(',');
            }
            sb.append(p.getManaPool().getAmountOfColor(t));
        }
        return sb.toString();
    }

    /**
     * Evening 4 (ADR-0105 addendum 09-09): the payment PROBE — the M9
     * enumeration and the auto-payability test — must not touch the game.
     * ComputerUtilMana's test-mode payment draws MyRandom.percentTrue per
     * candidate source (isManaSourceReserved) and clears / writes the AI's
     * mana-reservation memory sets, on every bridged in-scope window, BEFORE
     * the same auto payment the unbridged path makes: a bridged seat whose
     * head answered auto everywhere still played a different game (the
     * evening-4 reads: −2.9 / −3.4pp vs the tag withheld; the ADR-0102 scan
     * rule says probes are RNG-neutral). Scratch RNG around the body and the
     * memory sets the probe can change snapshotted and restored after.
     */
    private <T> T quietProbe(java.util.function.Supplier<T> body) {
        // ADR-0110 (the 09-16 upstream merge): the whole memory map, not a
        // named list — #11667 added MemorySetMana.UNPAID_COSTS, written by the
        // test-mode payment the probe runs and read by ChangeZoneAi; a named
        // list would have leaked it silently.
        final forge.game.player.Player p = getPlayer();
        java.util.Map<forge.ai.AiCardMemory.MemoryType, java.util.Set> saved = null;
        try {
            saved = forge.ai.AiCardMemory.snapshotAll(p);
        } catch (Exception ignored) {
        }
        try {
            return AnvilOptions.withScratchRng(body);
        } finally {
            try {
                forge.ai.AiCardMemory.restoreAll(p, saved);
            } catch (Exception ignored) {
            }
        }
    }

    /** The PlayerControllerAi payment body, called directly — super would
     *  re-record the census window. */
    private boolean autoPay(forge.card.mana.ManaCost toPay, SpellAbility sa, boolean effect) {
        return forge.ai.ComputerUtilMana.payManaCost(
                new forge.game.cost.Cost(toPay, effect), getPlayer(), sa, effect);
    }

    /**
     * Evening 4 (ADR-0105): an in-scope payment window of the ACTING seat on a
     * search copy (its SearchDirective applied). Enumerates the M9 goal
     * options exactly as the mainline bridge path does; a consequential window
     * is a PAY surface: the dec record (the wire shape, {auto} ∪ goals) is
     * emitted on the copy's session so a directed window's frame reaches the
     * sub row; a {@link SurfaceDirective} of kind PAY answers the ordinal-th
     * such window with a goal index (0 = auto) realized through the directed
     * executor; every window is TRACED (natural = the pick taken) so the
     * monitor's pay round can expand it. Never bridged (-searchpaybridge
     * off): the natural line is auto. Never throws into the game thread.
     */
    private boolean copyPay(SearchDirective sr, forge.card.mana.ManaCost toPay, SpellAbility sa,
            boolean effect, boolean combat) {
        final PaymentEnumerator.Result r;
        final boolean auto;
        try {
            if (!combat && PaymentEnumerator.costModified(sa)) {
                return autoPay(toPay, sa, effect);
            }
            r = quietProbe(() -> PaymentEnumerator.enumerate(getPlayer(), sa, toPay));
            auto = quietProbe(() -> PaymentEnumerator.autoPayable(getPlayer(), sa, toPay, effect));
        } catch (Exception e) {
            return autoPay(toPay, sa, effect);
        }
        if (!PaymentEnumerator.consequential(r, auto)) {
            return autoPay(toPay, sa, effect);
        }
        final List<String> labels = paymentOptionLabels(r);
        final int n = labels.size();
        final boolean forced = r.planCount >= 1 && !auto;
        final Object[] decKv = kvPlus(combat, new Object[] {
                "sa", Census.str(sa), "cost", String.valueOf(toPay), "effect", effect,
                "fpool", floatingPool(), "goals", r.options.size(), "plans", r.planCount,
                "trunc", r.goalCapHit, "forced", forced });
        final long obsSeq = Obs.decBridged(getGame(), getPlayer(), "payManaCost", labels, decKv);
        final SurfaceDirective d = SurfaceDirective.match(getGame(), player, Surfaces.PAY, n, 1, Census.str(sa));
        int pick = 0;
        String by = "natural";
        if (d != null) {
            d.frame = Obs.lastDecForBridge(getGame());
            final int a = d.answer == null || d.answer.length != 1 ? -1 : d.answer[0];
            if (a < 0 || a >= n) {
                d.miss("idx");
                by = "search_miss";
            } else {
                d.fired(n);
                pick = a;
                by = "search";
            }
        } else if (copyPayBridge && bridged(TAG_PAY_CLASS)) {
            final int p = bridge.selectOne(TAG_PAY_CLASS, labels);
            pick = p <= 0 || p > r.options.size() ? 0 : p;
            by = "bridge";
        }
        Surfaces.trace(getGame(), player, Surfaces.PAY, Census.str(sa), n, 1, 1, new int[] {pick}, null);
        final boolean paid;
        final String exec;
        String cousinsNote = null;
        if (pick == 0 && AnvilOptions.PAYRESCUE && forced && !r.options.isEmpty()) {
            // the rescue flag: auto cannot pay — the first plan is the natural line
            paid = directedPay(r.options.get(0).plan, toPay, sa, effect, "rescue");
            exec = "rescue:" + paid;
        } else if (pick == 0) {
            paid = autoPay(toPay, sa, effect);
            exec = "auto";
        } else {
            final PaymentEnumerator.PaymentClass pc = r.options.get(pick - 1).plan;
            final CousinDirective.Armed cousins = CousinDirective.arm(getPlayer(), pc);
            final PaymentEnumerator.ExecOutcome out;
            try {
                out = PaymentEnumerator.executeDirected(getPlayer(), pc);
                paid = autoPay(toPay, sa, effect);
            } finally {
                CousinDirective.disarm(getPlayer());
            }
            exec = !paid ? "directed_fail"
                    : out == PaymentEnumerator.ExecOutcome.DIRECTED_OK ? "directed_ok" : "directed_salvage";
            cousinsNote = pc.hasCousins() ? cousins.summary() : null;
        }
        if (d != null && d.fired && d.miss == null) {
            // the certifier merge (09-23): the certify row reads the answer's
            // execution off the directive (CensusRun.certRow's exec / goals)
            d.exec = exec;
            d.turn = getGame().getPhaseHandler().getTurn();
            if (pick > 0) {
                d.goals = new java.util.ArrayList<>(r.options.get(pick - 1).goals);
                d.kinds = new java.util.ArrayList<>(r.options.get(pick - 1).kinds);
            }
        }
        Object[] recKv = kvPlus(combat, new Object[] {
                "by", by, "copy", true, "options", n, "pick", pick == 0 ? "auto" : String.valueOf(pick),
                "exec", exec, "paid", paid, "goals", r.options.size(), "plans", r.planCount,
                "conseq", true, "forced", forced });
        if (cousinsNote != null) {
            recKv = java.util.Arrays.copyOf(recKv, recKv.length + 2);
            recKv[recKv.length - 2] = "cousins";
            recKv[recKv.length - 1] = cousinsNote;
        }
        Census.rec(getGame(), getPlayer(), "payManaCost", recKv);
        Obs.ret(getGame(), obsSeq, pick == 0 ? "auto:" + paid : exec);
        return paid;
    }

    /** Evening 4 (the rescue flag): a directed payment through the audited
     *  executor — the plan floated then auto completes pool-first, cousins
     *  armed; the failure semantics are the executor's (salvage, never a veto). */
    private boolean directedPay(PaymentEnumerator.PaymentClass pc, forge.card.mana.ManaCost toPay,
            SpellAbility sa, boolean effect, String why) {
        CousinDirective.arm(getPlayer(), pc);
        try {
            PaymentEnumerator.executeDirected(getPlayer(), pc);
            return autoPay(toPay, sa, effect);
        } finally {
            CousinDirective.disarm(getPlayer());
        }
    }

    /** Evening 4 (the rescue flag): an unbridged seat's in-scope window —
     *  forced (auto cannot pay, a plan exists) pays directed by the first
     *  plan and returns; null = not forced (or cost-modified, or an
     *  enumeration error): the ordinary path pays. */
    private Boolean rescuePay(forge.card.mana.ManaCost toPay, SpellAbility sa, boolean effect,
            boolean combat) {
        try {
            if (!combat && PaymentEnumerator.costModified(sa)) {
                return null;
            }
            final PaymentEnumerator.Result r = quietProbe(() -> PaymentEnumerator.enumerate(getPlayer(), sa, toPay));
            final boolean auto = quietProbe(() -> PaymentEnumerator.autoPayable(getPlayer(), sa, toPay, effect));
            if (auto || r.planCount < 1 || r.options.isEmpty()) {
                return null;
            }
            final long s = Obs.dec(getGame(), getPlayer(), "payManaCost",
                    "sa", Census.str(sa), "effect", effect, "rescue", true);
            final boolean paid = directedPay(r.options.get(0).plan, toPay, sa, effect, "rescue");
            Census.rec(getGame(), getPlayer(), "payManaCost", kvPlus(combat, new Object[] {
                    "by", "rescue", "sa", Census.str(sa), "effect", effect, "paid", paid,
                    "goals", r.options.size(), "plans", r.planCount, "conseq", true, "forced", true }));
            Obs.ret(getGame(), s, "rescue:" + paid);
            return paid;
        } catch (Exception e) {
            return null;
        }
    }

    /** Evening 4: one census row per resolution-effect payment window under
     *  -paytelemetry — goals / plans / consequential / auto-payable / costmod
     *  over the raw cost, auto pays as ever. Enumeration errors are a row too. */
    private void resolutionEffectCensus(forge.card.mana.ManaCost toPay, SpellAbility sa) {
        try {
            if (toPay == null || toPay.isZero()) {
                // the zero-mana / nested class (the spec's ~73/g): counted, never enumerated
                Census.rec(getGame(), getPlayer(), "payManaCost", "by", "auto", "effect", true, "reff", true,
                        "sa", Census.str(sa), "zero", true);
                return;
            }
            final boolean costmod = PaymentEnumerator.costModified(sa);
            final PaymentEnumerator.Result r = quietProbe(() -> PaymentEnumerator.enumerate(getPlayer(), sa, toPay));
            final boolean auto = quietProbe(() -> PaymentEnumerator.autoPayable(getPlayer(), sa, toPay, true));
            Census.rec(getGame(), getPlayer(), "payManaCost", "by", "auto", "effect", true, "reff", true,
                    "sa", Census.str(sa), "cost", String.valueOf(toPay),
                    "goals", r.options.size(), "plans", r.planCount,
                    "conseq", PaymentEnumerator.consequential(r, auto), "autoable", auto,
                    "costmod", costmod, "trunc", r.goalCapHit, "nodecap", r.nodeCapHit);
        } catch (Exception e) {
            Census.rec(getGame(), getPlayer(), "payManaCost", "by", "auto", "effect", true, "reff", true,
                    "sa", Census.str(sa), "enumerr", e.getClass().getSimpleName());
        }
    }

    /** Schedule-consistent directed payment (m10-ceiling-spec knob c): pick
     *  the enumerated plan that maximizes feasibility of the remaining
     *  scheduled items (ScheduleDirective.selectPlan), float it, and let
     *  auto complete from the float — the ADR-0065 float-then-apply
     *  primitive. Returns null to fall through to the normal payment path;
     *  failures are reason codes on the directive, NEVER vetoes, never
     *  exceptions into the game thread (the payManaCost bridged-path rule). */
    private Boolean schedulePay(ScheduleDirective sdir, forge.card.mana.ManaCost toPay,
            SpellAbility sa, boolean effect) {
        sdir.payWindows++;
        try {
            // cost-modified windows: out-of-scope v1 (spec §12b) — raw toPay
            // diverges from what auto pays; enumeration would target the
            // wrong cost.
            if (PaymentEnumerator.costModified(sa)) {
                sdir.payCostmod++;
                return null;
            }
            final PaymentEnumerator.Result r = PaymentEnumerator.enumerate(getPlayer(), sa, toPay);
            final boolean auto = PaymentEnumerator.autoPayable(getPlayer(), sa, toPay, effect);
            if (!PaymentEnumerator.consequential(r, auto) || r.options.isEmpty()) {
                sdir.payAuto++; // no real choice at this window
                return null;
            }
            final int pick = sdir.selectPlan(r, getPlayer());
            final PaymentEnumerator.PaymentClass pc = r.options.get(pick).plan;
            // cousins armed like the bridged path (empty maps = deliberate)
            CousinDirective.arm(getPlayer(), pc);
            final PaymentEnumerator.ExecOutcome out;
            final boolean paid;
            try {
                out = PaymentEnumerator.executeDirected(getPlayer(), pc);
                paid = autoPay(toPay, sa, effect); // completes from the float, pool-first
            } finally {
                CousinDirective.disarm(getPlayer());
            }
            if (!paid) {
                sdir.payFail++;
            } else if (out == PaymentEnumerator.ExecOutcome.DIRECTED_OK) {
                sdir.payDirected++;
            } else {
                sdir.paySalvage++;
            }
            Census.rec(getGame(), getPlayer(), "payManaCost", "by", "sched",
                    "sa", Census.str(sa), "effect", false,
                    "pick", pick + 1, "options", r.options.size(),
                    "exec", !paid ? "directed_fail"
                            : out == PaymentEnumerator.ExecOutcome.DIRECTED_OK
                                    ? "directed_ok" : "directed_salvage",
                    "paid", paid);
            return paid;
        } catch (Exception e) {
            sdir.payErr++;
            return null;
        }
    }

    /** Wire option labels (spec §5, goals per §12a): option 0 = auto; each
     *  option carries its goal names plus the representative plan's entity
     *  refs (the pointer-head substrate), pool spend by type, and phyrexian
     *  life count. Package-visible: the certify observe mode
     *  (PaymentTelemetry) emits the SAME labels the serve path presents —
     *  scorer/serve parity by construction. */
    static List<String> paymentOptionLabels(PaymentEnumerator.Result r) {
        List<String> labels = Lists.newArrayListWithCapacity(r.options.size() + 1);
        labels.add("{\"auto\":true}");
        for (PaymentEnumerator.GoalOption opt : r.options) {
            PaymentEnumerator.PaymentClass pc = opt.plan;
            StringBuilder sb = new StringBuilder(96);
            sb.append("{\"goals\":[");
            for (int i = 0; i < opt.goals.size(); i++) {
                if (i > 0) {
                    sb.append(',');
                }
                sb.append('"').append(opt.goals.get(i).replace("\"", "")).append('"');
            }
            // goal-kind codes (rung 3): the model-side featurizer keys the
            // pay_kind embedding on gk[0] — codes pinned in GoalOption's doc
            sb.append("],\"gk\":[");
            for (int i = 0; i < opt.kinds.size(); i++) {
                if (i > 0) {
                    sb.append(',');
                }
                sb.append(opt.kinds.get(i));
            }
            sb.append("],\"ents\":[");
            boolean first = true;
            for (PaymentEnumerator.Atom a : pc.atoms) {
                if (!first) {
                    sb.append(',');
                }
                sb.append(a.host.getId());
                first = false;
            }
            sb.append("],\"pool\":[");
            for (int i = 0; i < pc.poolSpend.length; i++) {
                if (i > 0) {
                    sb.append(',');
                }
                sb.append(pc.poolSpend[i]);
            }
            sb.append("],\"phy\":").append(pc.phyrexianLife).append('}');
            labels.add(sb.toString());
        }
        return labels;
    }

    /**
     * M2 D5 combat declarations. Labels are obs-join (post-declaration windows
     * carry the atk/blk flags), so no Obs.ret — same record shape as the
     * heuristic corpus. The realizer is engine-legality-only with
     * requirements repair (CombatRealizer); every deviation from the model's
     * raw map is census-counted (applied/dropped/forced/fallback). If the
     * bridge lacks the shape (misconfigured non-model arm), the AI brains
     * answer directly — super would double-log the dec.
     */
    @Override
    public void declareAttackers(Player attacker, Combat combat) {
        if (!bridged(TAG_ATTACK)) {
            super.declareAttackers(attacker, combat);
            return;
        }
        Obs.decBridged(getGame(), getPlayer(), "declareAttackers", null,
                "attacker", Census.str(attacker));
        CombatMapAnswer ans = bridge.attackMap(TAG_ATTACK, Obs.lastDecForBridge(getGame()));
        if (ans == null) {
            Census.rec(getGame(), getPlayer(), "declareAttackers", "by", "bridge",
                    "noShape", true);
            getAi().declareAttackers(attacker, combat);
            return;
        }
        CombatRealizer.Result r = CombatRealizer.realizeAttack(
                getGame(), attacker, combat, getAi(), ans);
        Census.rec(getGame(), getPlayer(), "declareAttackers", "by", "bridge",
                "assign", ans.assignments.size(), "applied", r.applied,
                "dropped", r.dropped, "forced", r.forced, "fallback", r.fallback);
    }

    @Override
    public void declareBlockers(Player defender, Combat combat) {
        if (!bridged(TAG_BLOCK)) {
            super.declareBlockers(defender, combat);
            return;
        }
        Obs.decBridged(getGame(), getPlayer(), "declareBlockers", null,
                "defender", Census.str(defender));
        CombatMapAnswer ans = bridge.blockMap(TAG_BLOCK, Obs.lastDecForBridge(getGame()));
        if (ans == null) {
            Census.rec(getGame(), getPlayer(), "declareBlockers", "by", "bridge",
                    "noShape", true);
            getAi().declareBlockersFor(defender, combat);
            return;
        }
        CombatRealizer.Result r = CombatRealizer.realizeBlock(
                getGame(), defender, combat, ans);
        Census.rec(getGame(), getPlayer(), "declareBlockers", "by", "bridge",
                "assign", ans.assignments.size(), "applied", r.applied,
                "dropped", r.dropped, "forced", r.forced, "fallback", r.fallback);
    }
}
