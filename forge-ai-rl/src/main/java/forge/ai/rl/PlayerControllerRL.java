package forge.ai.rl;

import forge.LobbyPlayer;
import forge.ai.rl.decisions.DecisionType;
import forge.ai.rl.features.ActionEncoder;
import forge.ai.rl.features.CardFeatures;
import forge.game.*;
import forge.game.card.*;
import forge.game.combat.Combat;
import forge.game.combat.CombatUtil;
import forge.game.player.*;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.WrappedAbility;
import forge.game.zone.ZoneType;
import forge.util.collect.FCollectionView;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.tinylog.Logger;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * PlayerController for the Reinforcement Learning AI.
 *
 * Extends PlayerControllerAi so ALL non-combat decisions use the proven
 * heuristic AI (ComputerUtil, AiController, etc.) without ClassCastException.
 *
 * declareAttackers and declareBlockers are overridden to either:
 * - Use the RL model (GRPC mode, server available)
 * - Or delegate to heuristic AND record the decision with pre-decision
 *   state for training data collection
 */
public class PlayerControllerRL extends forge.ai.PlayerControllerAi {

    private final RLController rl;

    // Diagnostic counters (per game)
    private int priorityModelAsked = 0;
    private int priorityModelPass = 0;
    private int priorityModelPlay = 0;
    private int priorityTargetingRejected = 0;
    private int priorityHeuristicBypass = 0;

    public PlayerControllerRL(Game game, Player p, LobbyPlayer lp, RLConfig config) {
        super(game, p, lp instanceof forge.ai.LobbyPlayerAi
                ? (forge.ai.LobbyPlayerAi) lp
                : createFallbackLobby(lp.getName()));
        this.rl = new RLController(config);
        this.rl.setPlayer(p);
    }

    public void logDiagnostics() {
        if (priorityModelAsked > 0 || priorityHeuristicBypass > 0) {
            Logger.info("RL_DIAG: priority: asked={} play={} pass={} targeting_rejected={} heuristic_bypass={}",
                    priorityModelAsked, priorityModelPlay, priorityModelPass,
                    priorityTargetingRejected, priorityHeuristicBypass);
        }
    }

    public void resetDiagnostics() {
        priorityModelAsked = 0;
        priorityModelPass = 0;
        priorityModelPlay = 0;
        priorityTargetingRejected = 0;
        priorityHeuristicBypass = 0;
    }

    private static forge.ai.LobbyPlayerAi createFallbackLobby(String name) {
        forge.ai.LobbyPlayerAi lp = new forge.ai.LobbyPlayerAi(name, null);
        lp.setAiProfile("Default");
        return lp;
    }

    public RLController getRLController() {
        return rl;
    }

    // ===== PRIORITY — which spell/ability to play =====

    @Override
    public List<SpellAbility> chooseSpellAbilityToPlay() {
        if (rl.isModelServerAvailable()) {
            // Let the heuristic build the candidate lists (lands, filtering, etc.)
            // then use the RL model to pick from the mechanically-legal set
            List<SpellAbility> heuristicResult = super.chooseSpellAbilityToPlay();

            // Get all mechanically legal spells (broad candidate set)
            List<SpellAbility> candidates = getAi().getLastPlayableSpellAbilities();
            if (candidates == null || candidates.isEmpty()) {
                priorityHeuristicBypass++;
                return heuristicResult; // land play or early-return — use heuristic
            }

            priorityModelAsked++;
            int idx = rl.decidePriorityAction(candidates);
            if (idx < 0 || idx >= candidates.size()) {
                priorityModelPass++;
                Logger.info("RL_PRIORITY: PASS ({} options available)", candidates.size());
                return null;
            }

            // RL picked a spell — set up targeting via RL model
            SpellAbility chosen = candidates.get(idx);
            chosen.setActivatingPlayer(player);
            String spellName = chosen.getHostCard() != null ? chosen.getHostCard().getName() : "?";
            String apiName = chosen.getApi() != null ? chosen.getApi().name() : "none";

            if (chosen.usesTargeting()) {
                if (chosen.getTargetRestrictions() == null) {
                    priorityTargetingRejected++;
                    Logger.info("RL_SPELL_REJECTED: {} ({}) reason=no_target_restrictions", spellName, apiName);
                    return null;
                }

                List<GameEntity> legalTargets = chosen.getTargetRestrictions()
                        .getAllCandidates(chosen, true);
                if (legalTargets.isEmpty()) {
                    priorityTargetingRejected++;
                    Logger.info("RL_SPELL_REJECTED: {} ({}) reason=no_legal_targets", spellName, apiName);
                    return null;
                }

                int minTgts = chosen.getTargetRestrictions().getMinTargets(chosen.getHostCard(), chosen);
                int maxTgts = chosen.getTargetRestrictions().getMaxTargets(chosen.getHostCard(), chosen);
                // Clamp to available targets
                maxTgts = Math.min(maxTgts, legalTargets.size());
                minTgts = Math.min(minTgts, maxTgts);
                float[] spellFeats = ActionEncoder.encode(chosen);
                List<Integer> targetIndices = rl.decideTargets(legalTargets, minTgts, maxTgts, spellFeats);
                if (targetIndices.isEmpty()) {
                    priorityTargetingRejected++;
                    Logger.info("RL_SPELL_REJECTED: {} ({}) reason=model_returned_no_targets", spellName, apiName);
                    return null;
                }

                chosen.resetTargets();
                for (int ti : targetIndices) {
                    if (ti >= 0 && ti < legalTargets.size()) {
                        GameEntity target = legalTargets.get(ti);
                        if (chosen.canTarget(target)) {
                            chosen.getTargets().add(target);
                        }
                    }
                }

                if (!chosen.isTargetNumberValid()) {
                    priorityTargetingRejected++;
                    Logger.info("RL_SPELL_REJECTED: {} ({}) reason=invalid_target_count", spellName, apiName);
                    return null;
                }

                String targetName = "?";
                if (!chosen.getTargets().isEmpty()) {
                    Object t = chosen.getTargets().get(0);
                    if (t instanceof Card) targetName = ((Card) t).getName();
                    else if (t instanceof Player) targetName = "Player:" + ((Player) t).getName();
                    else targetName = t.toString();
                }
                Logger.info("RL_PRIORITY_PLAY: {} ({}) -> target: {} (of {} legal)",
                        spellName, apiName, targetName, legalTargets.size());
            } else {
                Logger.info("RL_PRIORITY_PLAY: {} ({}) [no targeting needed]", spellName, apiName);
            }

            priorityModelPlay++;
            List<SpellAbility> rlResult = new ArrayList<>();
            rlResult.add(chosen);
            return rlResult;
        } else {
            // Heuristic decides — record the decision
            List<SpellAbility> result = super.chooseSpellAbilityToPlay();

            // Get all mechanically legal spells as candidates
            List<SpellAbility> candidates = getAi().getLastPlayableSpellAbilities();
            if (candidates == null || candidates.isEmpty()) {
                return result; // land play or early-return path — nothing to record
            }

            // Determine what the heuristic chose
            SpellAbility chosenSa = null;
            if (result != null && !result.isEmpty()) {
                chosenSa = result.get(0);
                if (chosenSa != null && chosenSa.isLandAbility()) {
                    return result; // don't record land plays
                }
            }

            rl.recordHeuristicPriority(candidates, chosenSa);
            return result;
        }
    }

    // ===== COMBAT — RL model or heuristic with recording =====

    @Override
    public void declareAttackers(Player attacker, Combat combat) {
        GameEntity defender = attacker.getWeakestOpponent();
        if (defender == null) return;

        // Build candidate list (creatures that can legally attack)
        CardCollection possibleAttackers = new CardCollection();
        for (Card c : attacker.getCreaturesInPlay()) {
            if (CombatUtil.canAttack(c, defender)) {
                possibleAttackers.add(c);
            }
        }
        if (possibleAttackers.isEmpty()) return;

        if (rl.isModelServerAvailable()) {
            // RL model makes the decision
            List<Integer> attackerIndices = rl.decideAttackers(possibleAttackers);
            for (int idx : attackerIndices) {
                if (idx >= 0 && idx < possibleAttackers.size()) {
                    Card c = possibleAttackers.get(idx);
                    if (CombatUtil.canAttack(c, defender)) {
                        combat.addAttacker(c, defender);
                    }
                }
            }
        } else {
            // Heuristic decides — but we record the decision
            // 1. Capture pre-decision state (creatures untapped, no combat)
            rl.capturePreDecisionState(possibleAttackers);

            // 2. Let heuristic make the decision (modifies combat object)
            super.declareAttackers(attacker, combat);

            // 3. Read back what the heuristic chose from the combat object
            CardCollection actualAttackers = combat.getAttackers();
            List<Integer> selectedIndices = new ArrayList<>();
            for (int i = 0; i < possibleAttackers.size(); i++) {
                if (actualAttackers.contains(possibleAttackers.get(i))) {
                    selectedIndices.add(i);
                }
            }

            // 4. Record: pre-decision state + heuristic's choice
            rl.recordHeuristicAttack(possibleAttackers, selectedIndices);
        }
    }

    @Override
    public void declareBlockers(Player defender, Combat combat) {
        // Build candidate list (untapped creatures that can block)
        List<Card> possibleBlockers = new ArrayList<>();
        for (Card c : defender.getCreaturesInPlay()) {
            if (!c.isTapped() && !c.hasKeyword("CARDNAME can't block.")) {
                possibleBlockers.add(c);
            }
        }
        CardCollection attackers = combat.getAttackers();
        if (possibleBlockers.isEmpty() || attackers.isEmpty()) return;

        if (rl.isModelServerAvailable()) {
            // RL model makes the decision
            List<int[]> assignments = rl.decideBlockers(possibleBlockers, attackers);
            for (int[] pair : assignments) {
                int blockerIdx = pair[0];
                int attackerIdx = pair[1];
                if (blockerIdx >= 0 && blockerIdx < possibleBlockers.size()
                        && attackerIdx >= 0 && attackerIdx < attackers.size()) {
                    Card blocker = possibleBlockers.get(blockerIdx);
                    Card att = attackers.get(attackerIdx);
                    if (CombatUtil.canBlock(att, blocker, combat)) {
                        combat.addBlocker(att, blocker);
                    }
                }
            }
        } else {
            // Heuristic decides — record with pre-decision state
            // Capture state BEFORE the heuristic modifies combat
            rl.capturePreDecisionState(possibleBlockers);

            super.declareBlockers(defender, combat);

            // Read back the full assignment: which blocker blocks which attacker
            // Record as (blocker, attacker) pairs matching inference format
            rl.recordHeuristicBlockAssignment(
                    possibleBlockers, attackers, combat);
        }
    }

    // ===== SPELL RESOLUTION — capture targeting at the moment of play =====

    @Override
    public boolean playChosenSpellAbility(SpellAbility sa) {
        // Capture targeting decisions BEFORE the spell goes on the stack.
        // Only record in heuristic mode — in GRPC mode, the RL path in
        // chooseSpellAbilityToPlay already recorded the targeting decision.
        if (!rl.isModelServerAvailable()
                && sa.usesTargeting() && sa.isTargetNumberValid()) {
            try {
                recordSpellTargeting(sa);
            } catch (Exception e) {
                // Never crash due to recording
            }
        }
        return super.playChosenSpellAbility(sa);
    }

    private void recordSpellTargeting(SpellAbility sa) {
        forge.game.spellability.TargetChoices chosen = sa.getTargets();
        if (chosen.isEmpty()) return;

        // Get the legal candidate list.
        // getAllCandidates excludes already-chosen targets, so we need to
        // include the chosen targets in the candidate list.
        List<GameEntity> candidates = new ArrayList<>();

        // Add all currently legal targets
        if (sa.getTargetRestrictions() != null) {
            candidates.addAll(sa.getTargetRestrictions().getAllCandidates(sa, true));
        }

        // Add the chosen targets back (they were excluded by getAllCandidates)
        for (Object obj : chosen) {
            if (obj instanceof GameEntity) {
                GameEntity ge = (GameEntity) obj;
                if (!candidates.contains(ge)) {
                    candidates.add(ge);
                }
            }
        }

        if (candidates.size() <= 1) return; // trivial choice, don't record

        // Encode candidates
        List<float[]> feats = new ArrayList<>();
        for (GameEntity entity : candidates) {
            if (entity instanceof Card) {
                feats.add(CardFeatures.encode((Card) entity, player));
            } else {
                feats.add(ActionEncoder.encodeTarget(entity));
            }
        }

        // Find selected indices
        List<Integer> selectedIndices = new ArrayList<>();
        for (Object obj : chosen) {
            if (obj instanceof GameEntity) {
                int idx = candidates.indexOf(obj);
                if (idx >= 0) selectedIndices.add(idx);
            }
        }

        if (selectedIndices.isEmpty()) return;

        // Include spell context info
        String spellName = sa.getHostCard() != null ? sa.getHostCard().getName() : "unknown";
        String apiName = sa.getApi() != null ? sa.getApi().name() : "unknown";

        float[] spellFeats = ActionEncoder.encode(sa);
        rl.recordDecisionDirect(DecisionType.TARGET_SELECTION,
                candidates.size(), selectedIndices, feats,
                "spell_target_" + spellName + "_" + apiName, spellFeats);
    }

    // ===== TARGET SELECTION =====

    @Override
    @SuppressWarnings("unchecked")
    public <T extends GameEntity> T chooseSingleEntityForEffect(
            FCollectionView<T> optionList,
            DelayedReveal delayedReveal,
            SpellAbility sa, String title,
            boolean isOptional, Player targetedPlayer,
            Map<String, Object> params) {

        if (rl.isModelServerAvailable()
                && optionList.size() > 1) {
            // RL model picks the target — pass spell features for context
            List<GameEntity> targets = new ArrayList<>(optionList);
            float[] spellFeats = sa != null ? ActionEncoder.encode(sa) : null;
            List<Integer> selected = rl.decideTargets(targets, 1, 1, spellFeats);
            int idx = 0; // default to first if model returns empty/invalid
            if (!selected.isEmpty()) {
                idx = Math.max(0, Math.min(selected.get(0), optionList.size() - 1));
            }
            return optionList.get(idx);
        }

        T result = super.chooseSingleEntityForEffect(
                optionList, delayedReveal, sa, title,
                isOptional, targetedPlayer, params);
        // Record heuristic's choice (RECORD_HEURISTIC mode only)
        if (result != null && optionList.size() > 1) {
            List<float[]> feats = new ArrayList<>();
            for (T entity : optionList) {
                if (entity instanceof Card) {
                    feats.add(CardFeatures.encode((Card) entity, player));
                } else {
                    feats.add(ActionEncoder.encodeTarget(entity));
                }
            }
            int idx = 0;
            for (int i = 0; i < optionList.size(); i++) {
                if (optionList.get(i) == result) {
                    idx = i;
                    break;
                }
            }
            float[] spellFeats = sa != null ? ActionEncoder.encode(sa) : null;
            rl.recordDecisionDirect(DecisionType.TARGET_SELECTION,
                    optionList.size(), List.of(idx), feats,
                    "target_" + title, spellFeats);
        }
        return result;
    }

    // ===== CARD SELECTION — discard, sacrifice, scry, etc. =====

    @Override
    public CardCollectionView chooseCardsForEffect(
            CardCollectionView sourceList, SpellAbility sa,
            String title, int min, int max,
            boolean isOptional,
            Map<String, Object> params) {
        if (rl.isModelServerAvailable() && sourceList.size() > 1) {
            List<Integer> selected = rl.decideCardSelection(sourceList, min, max);
            CardCollection rlResult = new CardCollection();
            for (int idx : selected) {
                if (idx >= 0 && idx < sourceList.size()) rlResult.add(sourceList.get(idx));
            }
            return rlResult;
        }
        CardCollectionView result = super.chooseCardsForEffect(
                sourceList, sa, title, min, max,
                isOptional, params);
        if (result != null && !result.isEmpty() && sourceList.size() > 1) {
            List<float[]> feats = new ArrayList<>();
            for (Card c : sourceList) {
                feats.add(CardFeatures.encode(c, player));
            }
            List<Integer> indices = new ArrayList<>();
            for (Card c : result) {
                int idx = sourceList.indexOf(c);
                if (idx >= 0) indices.add(idx);
            }
            rl.recordDecisionDirect(DecisionType.CARD_SELECTION,
                    sourceList.size(), indices, feats,
                    "cards_for_effect_" + title);
        }
        return result;
    }

    @Override
    public CardCollectionView choosePermanentsToSacrifice(
            SpellAbility sa, int min, int max,
            CardCollectionView validTargets, String msg) {
        if (rl.isModelServerAvailable() && validTargets.size() > 1) {
            List<Integer> selected = rl.decideCardSelection(validTargets, min, max);
            CardCollection rlResult = new CardCollection();
            for (int idx : selected) {
                if (idx >= 0 && idx < validTargets.size()) rlResult.add(validTargets.get(idx));
            }
            return rlResult;
        }
        CardCollectionView result = super.choosePermanentsToSacrifice(
                sa, min, max, validTargets, msg);
        if (result != null && validTargets.size() > 1) {
            List<float[]> feats = new ArrayList<>();
            for (Card c : validTargets) {
                feats.add(CardFeatures.encode(c, player));
            }
            List<Integer> indices = new ArrayList<>();
            for (Card c : result) {
                int idx = validTargets.indexOf(c);
                if (idx >= 0) indices.add(idx);
            }
            rl.recordDecisionDirect(DecisionType.CARD_SELECTION,
                    validTargets.size(), indices, feats,
                    "sacrifice");
        }
        return result;
    }

    @Override
    public CardCollection chooseCardsToDiscardFrom(
            Player p, SpellAbility sa,
            CardCollection validCards, int min, int max) {
        if (rl.isModelServerAvailable() && validCards.size() > 1) {
            List<Integer> selected = rl.decideCardSelection(validCards, min, max);
            CardCollection rlResult = new CardCollection();
            for (int idx : selected) {
                if (idx >= 0 && idx < validCards.size()) rlResult.add(validCards.get(idx));
            }
            return rlResult;
        }
        CardCollectionView result = super.chooseCardsToDiscardFrom(
                p, sa, validCards, min, max);
        if (result != null && validCards.size() > 1) {
            List<float[]> feats = new ArrayList<>();
            for (Card c : validCards) {
                feats.add(CardFeatures.encode(c, player));
            }
            List<Integer> indices = new ArrayList<>();
            for (Card c : result) {
                int idx = validCards.indexOf(c);
                if (idx >= 0) indices.add(idx);
            }
            rl.recordDecisionDirect(DecisionType.CARD_SELECTION,
                    validCards.size(), indices, feats,
                    "discard");
        }
        return new CardCollection(result);
    }

    // ===== SCRY =====

    @Override
    public ImmutablePair<CardCollection, CardCollection>
            arrangeForScry(CardCollection topN) {
        List<float[]> feats = new ArrayList<>();
        for (Card c : topN) {
            feats.add(CardFeatures.encode(c, player));
        }

        ImmutablePair<CardCollection, CardCollection> result =
                super.arrangeForScry(topN);

        // Record which cards stayed on top
        List<Integer> topIndices = new ArrayList<>();
        for (Card c : result.getLeft()) {
            int idx = topN.indexOf(c);
            if (idx >= 0) topIndices.add(idx);
        }
        rl.recordDecisionDirect(DecisionType.CARD_SELECTION,
                topN.size(), topIndices, feats,
                "scry_top_" + result.getLeft().size());
        return result;
    }

    // ===== MULLIGAN =====

    @Override
    public boolean mulliganKeepHand(
            Player firstPlayer, int cardsToReturn) {
        if (rl.isModelServerAvailable()) {
            CardCollectionView hand = player.getCardsIn(ZoneType.Hand);
            boolean keep = rl.decideMulligan(hand, cardsToReturn);
            Logger.info("RL_MULLIGAN: {} ({} cards, {} to return)",
                    keep ? "KEEP" : "MULLIGAN", hand.size(), cardsToReturn);
            return keep;
        }

        List<float[]> handFeats = new ArrayList<>();
        CardCollectionView hand = player.getCardsIn(ZoneType.Hand);
        for (Card c : hand) {
            handFeats.add(CardFeatures.encode(c, player));
        }

        boolean keep = super.mulliganKeepHand(firstPlayer, cardsToReturn);

        rl.recordDecisionDirect(DecisionType.MULLIGAN,
                2, List.of(keep ? 1 : 0), handFeats,
                "mulligan_" + cardsToReturn + (keep ? "_keep" : "_mull"));
        return keep;
    }

    // ===== BINARY DECISIONS =====

    @Override
    public boolean confirmAction(
            SpellAbility sa, PlayerActionConfirmMode mode,
            String message, List<String> options,
            Card cardToShow,
            Map<String, Object> params) {
        if (rl.isModelServerAvailable()) {
            boolean result = rl.decideBinary("confirm_" + mode);
            Logger.info("RL_BINARY: {} (confirm_{})", result ? "YES" : "NO", mode);
            return result;
        }

        boolean result = super.confirmAction(
                sa, mode, message, options, cardToShow, params);
        rl.recordDecisionDirect(DecisionType.BINARY_CHOICE,
                2, List.of(result ? 1 : 0), null,
                "confirm_" + mode);
        return result;
    }

    @Override
    public boolean confirmTrigger(WrappedAbility wrapper) {
        if (rl.isModelServerAvailable()) {
            boolean result = rl.decideBinary("trigger");
            Logger.info("RL_BINARY: {} (trigger)", result ? "YES" : "NO");
            return result;
        }

        boolean result = super.confirmTrigger(wrapper);
        rl.recordDecisionDirect(DecisionType.BINARY_CHOICE,
                2, List.of(result ? 1 : 0), null,
                "trigger");
        return result;
    }
}
