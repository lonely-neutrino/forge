package forge.ai.anvil;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import forge.game.Game;
import forge.game.GameEntity;
import forge.game.GameObject;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.spellability.SpellAbilityStackInstance;
import forge.game.spellability.AbilitySub;
import forge.game.spellability.TargetChoices;
import forge.game.spellability.TargetRestrictions;

/**
 * Exact, bounded target-plan enumeration for the Anvil priority decoder.
 * The result is observation data, not an executor: CastPlanRealizer remains
 * the final authority and deliberately shares tryApply with this class.
 */
public final class CastTargetPlanEnumerator {
    public static final int MAX_TARGETS = 4;
    public static final int MAX_PLANS = 4096;
    public static final int MAX_STATES = 50000;
    public static final int X_CLASSES = 18;
    private static final int ALL_X = (1 << X_CLASSES) - 1;

    public static final class Plan {
        public final List<CastPlanAnswer.Ref> refs;
        public int xMask;

        private Plan(List<CastPlanAnswer.Ref> refs, int xMask) {
            this.refs = refs;
            this.xMask = xMask;
        }
    }

    public static final class Result {
        public final boolean complete;
        public final String reason;
        public final List<Plan> plans;

        private Result(boolean complete, String reason, List<Plan> plans) {
            this.complete = complete;
            this.reason = reason;
            this.plans = plans;
        }

        static Result fallback(String reason) {
            return new Result(false, reason, Collections.emptyList());
        }

        static Result complete(List<Plan> plans) {
            return new Result(true, null, plans);
        }

        /** Additive schema-v3 option fields. */
        public void appendJson(StringBuilder sb) {
            if (!complete) {
                sb.append(",\"tc\":0,\"tr\":").append(Obs.q(reason));
                return;
            }
            sb.append(",\"tc\":1,\"tp\":[");
            for (int i = 0; i < plans.size(); i++) {
                if (i > 0) sb.append(',');
                Plan p = plans.get(i);
                sb.append("{\"xm\":").append(p.xMask).append(",\"r\":[");
                for (int j = 0; j < p.refs.size(); j++) {
                    if (j > 0) sb.append(',');
                    CastPlanAnswer.Ref r = p.refs.get(j);
                    if (r.isPlayer) {
                        sb.append("{\"p\":").append(r.player).append('}');
                    } else {
                        sb.append("{\"e\":").append(r.entity);
                        if (r.onStack) sb.append(",\"ns\":1");
                        sb.append('}');
                    }
                }
                sb.append("]}");
            }
            sb.append(']');
        }
    }

    private final Game game;
    private final SpellAbility root;
    private final List<SpellAbility> nodes = new ArrayList<>();
    private final Map<String, Plan> plans = new LinkedHashMap<>();
    private int states;
    private String failure;

    private CastTargetPlanEnumerator(Game game, SpellAbility root) {
        this.game = game;
        this.root = root;
        SpellAbility previous = null;
        for (SpellAbility node = root; node != null; node = node.getSubAbility()) {
            if (previous != null && node instanceof AbilitySub) {
                ((AbilitySub) node).setParent(previous);
            }
            nodes.add(node);
            previous = node;
        }
    }

    public static Result enumerate(Game game, SpellAbility sa) {
        try {
            return AnvilOptions.withScratchRng(() -> new CastTargetPlanEnumerator(game, sa).run());
        } catch (RuntimeException e) {
            return Result.fallback("error:" + e.getClass().getSimpleName());
        }
    }

    private Result run() {
        for (SpellAbility node : nodes) {
            if (node.getApi() == forge.game.ability.ApiType.Charm) {
                return Result.fallback("modal");
            }
            if (node.usesTargeting() && node.hasParam("TargetingPlayer")) {
                return Result.fallback("targeting_player");
            }
        }

        List<TargetChoices> saved = new ArrayList<>(nodes.size());
        List<Integer> savedDivided = new ArrayList<>(nodes.size());
        for (SpellAbility node : nodes) {
            saved.add(node.getTargets().clone());
            savedDivided.add(node.getDividedValue());
        }
        Integer savedX = root.getXManaCostPaid();
        try {
            boolean hasX = hasX(root);
            if (!hasX) {
                enumerateX(0, ALL_X);
            } else {
                for (int x = 0; x <= 16 && failure == null; x++) {
                    int mask = 1 << x;
                    if (x == 16) mask |= 1 << 17; // overflow class clamps to 16 at serve
                    enumerateX(x, mask);
                }
            }
        } finally {
            for (int i = 0; i < nodes.size(); i++) {
                nodes.get(i).setTargets(saved.get(i));
                nodes.get(i).setDividedValue(savedDivided.get(i));
            }
            root.setXManaCostPaid(savedX);
        }
        if (failure != null) return Result.fallback(failure);
        return Result.complete(new ArrayList<>(plans.values()));
    }

    private void enumerateX(int x, int xMask) {
        clearTargets();
        root.setXManaCostPaid(x);
        List<GameObject> flat = new ArrayList<>(MAX_TARGETS);
        dfsNode(0, flat, x, xMask);
    }

    private void dfsNode(int ni, List<GameObject> flat, int x, int xMask) {
        if (!step()) return;
        if (ni >= nodes.size()) {
            addPlan(flat, x, xMask);
            return;
        }
        SpellAbility node = nodes.get(ni);
        if (!node.usesTargeting()) {
            dfsNode(ni + 1, flat, x, xMask);
            return;
        }
        node.clearTargets();
        dfsTargets(ni, node, flat, x, xMask);
        node.clearTargets();
    }

    private void dfsTargets(int ni, SpellAbility node, List<GameObject> flat, int x, int xMask) {
        if (!step()) return;
        TargetRestrictions tr = node.getTargetRestrictions();
        int count = node.getTargets().size();
        int min = tr.getMinTargets(node.getHostCard(), node);
        int max = tr.getMaxTargets(node.getHostCard(), node);
        if (count >= min && node.isTargetNumberValid()) {
            dfsNode(ni + 1, flat, x, xMask);
        }
        if (count >= max || flat.size() >= MAX_TARGETS || failure != null) return;

        // getAllCandidates updates changed target text after its player pass;
        // do it first so player and card candidates use the same restriction.
        tr.applyTargetTextChanges(node);
        List<GameObject> candidates = new ArrayList<>();
        for (GameEntity e : tr.getAllCandidates(node)) candidates.add(e);
        if (tr.getZone().contains(forge.game.zone.ZoneType.Stack)) {
            for (SpellAbilityStackInstance si : game.getStack()) {
                SpellAbility stackSa = si.getSpellAbility();
                if (node.canTargetSpellAbility(stackSa)) candidates.add(stackSa);
            }
        }
        for (GameObject candidate : candidates) {
            if (!canTarget(node, candidate)) continue;
            node.getTargets().add(candidate);
            flat.add(candidate);
            dfsTargets(ni, node, flat, x, xMask);
            flat.remove(flat.size() - 1);
            node.getTargets().remove(candidate);
            if (failure != null) return;
        }
    }

    private boolean canTarget(SpellAbility node, GameObject target) {
        return target instanceof SpellAbility
                ? node.canTargetSpellAbility((SpellAbility) target)
                : node.canTarget(target);
    }

    private void addPlan(List<GameObject> objects, int x, int xMask) {
        if (objects.size() > MAX_TARGETS) {
            failure = "target_cap";
            return;
        }
        if (!CastPlanRealizer.targetsMeetGlobalRestrictions(root)) return;

        List<CastPlanAnswer.Ref> refs = new ArrayList<>(objects.size());
        StringBuilder key = new StringBuilder();
        for (GameObject o : objects) {
            CastPlanAnswer.Ref ref;
            if (o instanceof Player) {
                int pi = game.getRegisteredPlayers().indexOf((Player) o);
                ref = new CastPlanAnswer.Ref(true, pi, -1, false);
                key.append('p').append(pi).append(';');
            } else if (o instanceof SpellAbility) {
                Card h = ((SpellAbility) o).getHostCard();
                if (h == null) return;
                ref = new CastPlanAnswer.Ref(false, -1, h.getId(), true);
                key.append('s').append(h.getId()).append(';');
            } else if (o instanceof Card) {
                ref = new CastPlanAnswer.Ref(false, -1, ((Card) o).getId(), false);
                key.append('e').append(((Card) o).getId()).append(';');
            } else {
                return;
            }
            refs.add(ref);
        }
        Plan existing = plans.get(key.toString());
        if (existing != null) {
            existing.xMask |= xMask;
            return;
        }
        if (plans.size() >= MAX_PLANS) {
            failure = "plan_cap";
            return;
        }
        plans.put(key.toString(), new Plan(refs, xMask));
    }

    private boolean step() {
        if (++states > MAX_STATES) {
            failure = "state_cap";
            return false;
        }
        return failure == null;
    }

    private void clearTargets() {
        for (SpellAbility node : nodes) node.clearTargets();
    }

    private static boolean hasX(SpellAbility sa) {
        return !sa.isLandAbility() && sa.getPayCosts() != null
                && sa.getPayCosts().getTotalMana() != null
                && sa.getPayCosts().getTotalMana().countX() > 0;
    }
}
