package forge.ai.anvil;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import forge.game.Game;
import forge.game.GameObject;
import forge.game.card.Card;
import forge.game.card.CardCopyService;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.spellability.SpellAbilityStackInstance;
import forge.game.spellability.TargetRestrictions;
import forge.game.staticability.StaticAbilityMustTarget;

/**
 * Realizes a rung-1 CastPlanAnswer against the chosen host's candidate SAs
 * and adjudicates LEGALITY ONLY (M1 D8). The M0 65% veto rate was
 * canPlaySa's heuristic *judgment* gating random proposals; a learned policy
 * must not be gated by the heuristic's opinion, so this deliberately never
 * calls the per-API AI (which is also where the AI sets targets/X — the model
 * supplies both, making that path unnecessary as well as unwanted).
 *
 * Host-level disambiguation ladder (model candidates are dedup host rows, so
 * several SAs can share the chosen host — measured 23.6% of expert actions on
 * the pilot): shape-fit (targets/X compatibility) -> legality+payability ->
 * kind preference (land > spell > ability; measured expert prior on
 * cross-kind ties) -> scan order. Every realization reports its rung and the
 * surviving-fit count so D8 telemetry can size the true ambiguity residue.
 *
 * The fit pass is side-effect-free: targets/X are cleared after each trial
 * and re-applied only to the final pick.
 */
public final class CastPlanRealizer {

    public static final class Result {
        public final SpellAbility sa;  // null = veto -> the window passes
        public final String veto;      // reason code when sa == null
        public final String rung;      // single|shape|pay|kind|order|modal
        public final int hostSas;      // options sharing the chosen host
        public final int fitCount;     // SAs that passed shape-fit
        public final boolean divided;  // even-split divided allocation applied
        public final boolean ignoredTargets; // target refs ignored on targetless SA

        private Result(SpellAbility sa, String veto, String rung, int hostSas, int fitCount,
                boolean ignoredTargets) {
            this.sa = sa;
            this.veto = veto;
            this.rung = rung;
            this.hostSas = hostSas;
            this.fitCount = fitCount;
            this.divided = sa != null && isDivided(sa);
            this.ignoredTargets = ignoredTargets;
        }

        private static Result veto(String reason, int hostSas, int fitCount) {
            return new Result(null, reason, null, hostSas, fitCount, false);
        }
    }

    private CastPlanRealizer() {
    }

    public static Result realize(Game game, Player player, List<SpellAbility> hostSas,
            CastPlanAnswer ans) {
        List<GameObject> refs = null;
        boolean refsResolved = false;
        boolean danglingRefs = false;
        List<SpellAbility> fits = new ArrayList<>(2);
        for (SpellAbility sa : hostSas) {
            List<GameObject> saRefs = Collections.emptyList();
            if (hasTargeting(sa)) {
                if (!refsResolved) {
                    refs = resolveRefs(game, ans.targets);
                    refsResolved = true;
                }
                if (refs == null) {
                    danglingRefs = true;
                    clear(sa);
                    continue;
                }
                saRefs = refs;
            }
            boolean ok = tryApply(sa, saRefs, ans);
            clear(sa);
            if (ok) {
                fits.add(sa);
            }
        }
        if (fits.isEmpty()) {
            if (danglingRefs) {
                return Result.veto("dangling_ref", hostSas.size(), 0);
            }
            // Modal spells (Charm api): the targeting lives inside modes that
            // bind at cast time, so the model's refs describe mode targets the
            // bare chain can't hold (smoke 3: 226/227 no-fit vetoes were
            // charms/commands). Cast mode-less — refs discarded, modes and
            // their targets fall to the heuristic interception points (M1
            // scope: modes are cut from rung 1; census rung "modal" sizes the
            // capability gap for the D8 report).
            List<GameObject> none = java.util.Collections.emptyList();
            for (SpellAbility sa : hostSas) {
                if (!isModal(sa) || !tryApply(sa, none, ans)) {
                    clear(sa);
                    continue;
                }
                String why = legality(game, player, sa);
                if (why == null && !sa.isLandAbility()
                        && !AnvilOptions.payableOrRescue(game, player, sa)) {
                    why = "unpayable";
                }
                if (why != null) {
                    clear(sa);
                    continue;
                }
                return new Result(sa, null, "modal", hostSas.size(), 0, false);
            }
            return Result.veto("no_shape_fit", hostSas.size(), 0);
        }
        List<SpellAbility> playable = new ArrayList<>(fits.size());
        String lastReason = null;
        for (SpellAbility sa : fits) {
            List<GameObject> saRefs = hasTargeting(sa) ? refs : Collections.emptyList();
            tryApply(sa, saRefs, ans); // legality/payability judged with targets+X set
            String why = legality(game, player, sa);
            if (why == null && !sa.isLandAbility()
                    && !AnvilOptions.payableOrRescue(game, player, sa)) {
                why = "unpayable";
            }
            clear(sa);
            if (why == null) {
                playable.add(sa);
            } else {
                lastReason = why;
            }
        }
        if (playable.isEmpty()) {
            return Result.veto(lastReason == null ? "illegal" : lastReason,
                    hostSas.size(), fits.size());
        }
        String rung;
        SpellAbility pick;
        if (hostSas.size() == 1) {
            rung = "single";
            pick = playable.get(0);
        } else if (fits.size() == 1) {
            rung = "shape";
            pick = playable.get(0);
        } else if (playable.size() == 1) {
            rung = "pay";
            pick = playable.get(0);
        } else {
            pick = preferByKind(playable);
            rung = sameKind(playable) ? "order" : "kind";
        }
        List<GameObject> pickRefs = hasTargeting(pick) ? refs : Collections.emptyList();
        tryApply(pick, pickRefs, ans); // leave the pick armed for the engine
        boolean ignoredTargets = !ans.targets.isEmpty() && !hasTargeting(pick);
        return new Result(pick, null, rung, hostSas.size(), fits.size(), ignoredTargets);
    }

    /** Model refs -> engine objects; null on any dangling ref. */
    private static List<GameObject> resolveRefs(Game game, List<CastPlanAnswer.Ref> refs) {
        List<GameObject> out = new ArrayList<>(refs.size());
        for (CastPlanAnswer.Ref r : refs) {
            if (r.isPlayer) {
                List<Player> ps = game.getRegisteredPlayers();
                if (r.player < 0 || r.player >= ps.size()) {
                    return null;
                }
                out.add(ps.get(r.player));
            } else if (r.onStack) {
                SpellAbility hit = null;
                for (SpellAbilityStackInstance si : game.getStack()) {
                    Card h = si.getSourceCard();
                    if (h != null && h.getId() == r.entity) {
                        hit = si.getSpellAbility();
                        break;
                    }
                }
                if (hit == null) {
                    return null;
                }
                out.add(hit);
            } else {
                Card c = game.findById(r.entity);
                if (c == null) {
                    return null;
                }
                out.add(c);
            }
        }
        return out;
    }

    /**
     * Distribute refs across the SA chain's targeting nodes in label-extractor
     * order (main SA first, then sub-abilities), greedily per node; set X.
     * True iff every node reaches a valid target count and every ref is
     * consumed. Mutates the SA — callers pair with clear().
     */
    static boolean tryApply(SpellAbility sa, List<GameObject> refs, CastPlanAnswer ans) {
        boolean saHasX = !sa.isLandAbility() && sa.getPayCosts() != null
                && sa.getPayCosts().getTotalMana() != null
                && sa.getPayCosts().getTotalMana().countX() > 0;
        // The server can't see which of a host's SAs is an X spell (rung-1
        // candidates are host-level), so it attaches its X reading to every
        // plan: an X-less plan can't power an X spell, but a plan's X is
        // simply ignored by non-X siblings rather than rejecting them.
        if (saHasX && !ans.hasX) {
            return false;
        }
        if (saHasX) {
            sa.setXManaCostPaid(ans.x);
        }
        int ri = 0;
        boolean hasTargeting = false;
        for (SpellAbility node = sa; node != null; node = node.getSubAbility()) {
            if (!node.usesTargeting()) {
                continue;
            }
            hasTargeting = true;
            node.clearTargets(); // also initializes dividedValue (X already set above)
            TargetRestrictions tr = node.getTargetRestrictions();
            int min = tr.getMinTargets(node.getHostCard(), node);
            int max = tr.getMaxTargets(node.getHostCard(), node);
            int added = 0;
            while (ri < refs.size() && added < max && node.canTarget(refs.get(ri))) {
                node.getTargets().add(refs.get(ri));
                ri++;
                added++;
            }
            if (added < min || !node.isTargetNumberValid()) {
                return false;
            }
            if (node.isDividedAsYouChoose() && added > 0) {
                // The model doesn't predict allocations at rung 1; resolution
                // NPEs without them (smoke 3). Deterministic legal default:
                // even split, remainder to the first target. Result.divided
                // marks these for the census.
                int total = node.getStillToDivide();
                if (total < added) {
                    return false; // can't give every target >= 1
                }
                int base = total / added;
                int rem = total % added;
                int i = 0;
                for (GameObject t : node.getTargets()) {
                    node.addDividedAllocation(t, base + (i++ == 0 ? rem : 0));
                }
            }
        }
        // Targetless spells have no shape slot for the shared decoder's
        // irrelevant refs. Target-bearing spells must consume every ref.
        return (!hasTargeting || ri == refs.size())
                && (!Obs.targetPlanMask || targetsMeetGlobalRestrictions(sa));
    }

    /** Whole-plan checks that only become meaningful after every target node
     * has been populated. Shared with the pre-inference enumerator. */
    static boolean targetsMeetGlobalRestrictions(SpellAbility sa) {
        if (!StaticAbilityMustTarget.meetsMustTargetRestriction(sa)) {
            return false;
        }
        for (SpellAbility node = sa; node != null; node = node.getSubAbility()) {
            if (node.isDividedAsYouChoose() && !node.getTargets().isEmpty()
                    && node.getStillToDivide() < node.getTargets().size()) {
                return false;
            }
        }
        return true;
    }

    private static boolean hasTargeting(SpellAbility sa) {
        for (SpellAbility node = sa; node != null; node = node.getSubAbility()) {
            if (node.usesTargeting()) {
                return true;
            }
        }
        return false;
    }

    static void clear(SpellAbility sa) {
        for (SpellAbility node = sa; node != null; node = node.getSubAbility()) {
            if (node.usesTargeting()) {
                node.clearTargets();
            }
        }
        sa.setXManaCostPaid(null);
    }

    private static boolean isDivided(SpellAbility sa) {
        for (SpellAbility node = sa; node != null; node = node.getSubAbility()) {
            if (node.usesTargeting() && node.isDividedAsYouChoose()) {
                return true;
            }
        }
        return false;
    }

    /**
     * The legality subset of AiController.canPlaySa — timing, casting
     * restrictions (with the same LKI stack-zone dance for spells), and
     * post-stack legality. Deliberately none of its judgment calls.
     */
    private static String legality(Game game, Player player, SpellAbility sa) {
        if (!sa.canPlay()) {
            return "timing";
        }
        if (!sa.canCastTiming(player)) {
            return "timing";
        }
        if (!sa.isLegalAfterStack()) {
            return "after_stack";
        }
        Card host = sa.getHostCard();
        Card spellHost = host;
        if (sa.isSpell() && host != null) {
            spellHost = CardCopyService.getLKICopy(host);
            spellHost.setLKICMC(-1);
            spellHost.setLastKnownZone(game.getStackZone());
            spellHost.setCastFrom(host.getZone());
        }
        if (!sa.checkRestrictions(spellHost, player)) {
            return "restrictions";
        }
        return null;
    }

    private static boolean isModal(SpellAbility sa) {
        for (SpellAbility node = sa; node != null; node = node.getSubAbility()) {
            if (node.getApi() == forge.game.ability.ApiType.Charm) {
                return true;
            }
        }
        return false;
    }

    // Measured expert prior on cross-kind host ties (pilot, 5K games):
    // land beats ability 3,175:<56, land beats spell 1,457:222, spell beats
    // ability 1,627:341. Within a kind, scan order (stable sort).
    private static int kindRank(SpellAbility sa) {
        if (sa.isLandAbility()) {
            return 0;
        }
        return sa.isSpell() ? 1 : 2;
    }

    private static SpellAbility preferByKind(List<SpellAbility> sas) {
        SpellAbility best = sas.get(0);
        for (SpellAbility sa : sas) {
            if (kindRank(sa) < kindRank(best)) {
                best = sa;
            }
        }
        return best;
    }

    private static boolean sameKind(List<SpellAbility> sas) {
        int k = kindRank(sas.get(0));
        for (SpellAbility sa : sas) {
            if (kindRank(sa) != k) {
                return false;
            }
        }
        return true;
    }
}
