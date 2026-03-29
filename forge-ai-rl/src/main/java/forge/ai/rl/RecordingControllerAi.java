package forge.ai.rl;

import forge.LobbyPlayer;
import forge.ai.PlayerControllerAi;
import forge.ai.rl.decisions.DecisionContext;
import forge.ai.rl.decisions.DecisionResult;
import forge.ai.rl.decisions.DecisionType;
import forge.ai.rl.features.ActionEncoder;
import forge.ai.rl.features.CardFeatures;
import forge.ai.rl.features.GameStateEncoder;
import forge.ai.rl.features.GameStateFeatures;
import forge.ai.rl.training.TrajectoryRecorder;
import forge.game.Game;
import forge.game.GameEntity;
import forge.game.card.Card;
import forge.game.player.DelayedReveal;
import forge.game.card.CardCollection;
import forge.game.card.CardCollectionView;
import forge.game.combat.Combat;
import forge.game.player.Player;
import forge.game.player.PlayerActionConfirmMode;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.WrappedAbility;
import forge.game.zone.ZoneType;
import forge.util.collect.FCollectionView;
import org.apache.commons.lang3.tuple.ImmutablePair;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Subclass of PlayerControllerAi that records every major
 * decision as a trajectory step for imitation learning.
 *
 * This is the correct approach: by subclassing (not wrapping)
 * we avoid breaking the AiController's internal references.
 * AiController is created in the super constructor and holds
 * a reference to the Player, which references back to us.
 */
public class RecordingControllerAi extends PlayerControllerAi {

    private final GameStateEncoder encoder;
    private final TrajectoryRecorder recorder;
    private boolean recording = false;

    public RecordingControllerAi(
            Game game, Player p, LobbyPlayer lp,
            RLConfig config) {
        super(game, p, lp);
        this.encoder = new GameStateEncoder(config);
        this.recorder = new TrajectoryRecorder(
                config.getTrajectoryOutputDir(),
                config.isZeroIntermediateReward());
    }

    public TrajectoryRecorder getRecorder() {
        return recorder;
    }

    public void startRecording(String gameId) {
        recorder.startGame(gameId);
        recording = true;
    }

    public void stopRecording(boolean won) {
        if (recording) {
            recorder.endGame(won);
            recording = false;
        }
    }

    // --- Helper to record a decision ---

    private void record(DecisionType type,
                        int numCandidates,
                        List<Integer> selected,
                        List<float[]> candidateFeats,
                        String info) {
        if (!recording) {
            return;
        }
        try {
            GameStateFeatures gs = encoder.encode(
                    getGame(), player);
            DecisionContext ctx = new DecisionContext(
                    type, gs,
                    candidateFeats != null
                        ? candidateFeats : List.of(),
                    selected.size(), numCandidates, info);
            DecisionResult res = new DecisionResult(
                    selected, new float[0], 0f, true);
            Player opp = player.getWeakestOpponent();
            recorder.recordDecision(ctx, res,
                    player.getLife(),
                    opp != null ? opp.getLife() : 0,
                    player.getCardsIn(ZoneType.Hand).size(),
                    opp != null
                        ? opp.getCardsIn(ZoneType.Hand).size()
                        : 0,
                    countCreatures(player),
                    opp != null ? countCreatures(opp) : 0);
        } catch (Exception e) {
            // Never crash the game due to recording errors
        }
    }

    private int countCreatures(Player p) {
        int c = 0;
        for (Card card : p.getCardsIn(ZoneType.Battlefield)) {
            if (card.isCreature()) {
                c++;
            }
        }
        return c;
    }

    // === RECORDED DECISION OVERRIDES ===

    @Override
    public List<SpellAbility> chooseSpellAbilityToPlay() {
        // Let heuristic decide (also caches candidate list)
        List<SpellAbility> result =
                super.chooseSpellAbilityToPlay();

        // Get all mechanically legal spells as candidates
        List<SpellAbility> candidates =
                getAi().getLastPlayableSpellAbilities();
        if (candidates == null || candidates.isEmpty())
            return result;

        // Don't record land plays as priority decisions
        SpellAbility chosenSa = null;
        if (result != null && !result.isEmpty()) {
            chosenSa = result.get(0);
            if (chosenSa != null && chosenSa.isLandAbility())
                return result;
        }

        // Build candidate features (64-dim each) + pass
        List<float[]> feats = new ArrayList<>();
        for (SpellAbility sa : candidates) {
            feats.add(ActionEncoder.encode(sa));
        }
        feats.add(ActionEncoder.encodePassAction());

        // Find heuristic's choice index
        int selectedIdx = candidates.size(); // default=pass
        if (chosenSa != null) {
            for (int i = 0; i < candidates.size(); i++) {
                if (candidates.get(i) == chosenSa) {
                    selectedIdx = i;
                    break;
                }
            }
            if (selectedIdx == candidates.size()
                    && chosenSa.getHostCard() != null) {
                String name =
                    chosenSa.getHostCard().getName();
                Object api = chosenSa.getApi();
                for (int i = 0; i < candidates.size(); i++) {
                    SpellAbility sa = candidates.get(i);
                    if (sa.getHostCard() != null
                            && sa.getHostCard().getName()
                                .equals(name)
                            && sa.getApi() == api) {
                        selectedIdx = i;
                        break;
                    }
                }
            }
        }

        record(DecisionType.PRIORITY_ACTION,
                candidates.size() + 1,
                List.of(selectedIdx), feats,
                "priority_" + candidates.size()
                    + "_options");
        return result;
    }

    @Override
    public void declareAttackers(
            Player attacker, Combat combat) {
        // Record state BEFORE attack declaration
        int creaturesBeforeAttack =
                attacker.getCreaturesInPlay().size();
        List<float[]> creatureFeats = new ArrayList<>();
        List<Card> creatureList = new ArrayList<>(attacker.getCreaturesInPlay());
        for (Card c : creatureList) {
            creatureFeats.add(CardFeatures.encode(c, player));
        }
        super.declareAttackers(attacker, combat);

        // Record which creatures are now attacking
        int numAttacking = combat.getAttackers().size();
        List<Integer> attackerIndices = new ArrayList<>();
        List<Card> creatures =
                new ArrayList<>(attacker.getCreaturesInPlay());
        for (int i = 0; i < creatures.size(); i++) {
            if (combat.isAttacking(creatures.get(i))) {
                attackerIndices.add(i);
            }
        }
        record(DecisionType.DECLARE_ATTACKERS,
                creaturesBeforeAttack, attackerIndices,
                creatureFeats,
                "attackers_" + numAttacking
                    + "_of_" + creaturesBeforeAttack);
    }

    @Override
    public void declareBlockers(
            Player defender, Combat combat) {
        List<Card> defenderCreatures = new ArrayList<>(defender.getCreaturesInPlay());
        int creaturesAvail = defenderCreatures.size();
        List<float[]> creatureFeats = new ArrayList<>();
        for (Card c : defenderCreatures) {
            creatureFeats.add(CardFeatures.encode(c, player));
        }
        super.declareBlockers(defender, combat);

        // Record blocking assignments
        List<Integer> blockerIndices = new ArrayList<>();
        List<Card> creatures =
                new ArrayList<>(defender.getCreaturesInPlay());
        for (int i = 0; i < creatures.size(); i++) {
            if (combat.isBlocking(creatures.get(i))) {
                blockerIndices.add(i);
            }
        }
        record(DecisionType.DECLARE_BLOCKERS,
                creaturesAvail, blockerIndices,
                creatureFeats,
                "blockers_" + blockerIndices.size());
    }

    @Override
    public CardCollectionView chooseCardsForEffect(
            CardCollectionView sourceList, SpellAbility sa,
            String title, int min, int max,
            boolean isOptional,
            Map<String, Object> params) {
        CardCollectionView result =
                super.chooseCardsForEffect(
                        sourceList, sa, title, min, max,
                        isOptional, params);
        if (result != null && !result.isEmpty()
                && sourceList.size() > 1) {
            List<float[]> feats = new ArrayList<>();
            List<Card> cardList = new ArrayList<>();
            for (Card c : sourceList) {
                feats.add(CardFeatures.encode(c, player));
                cardList.add(c);
            }
            List<Integer> indices = new ArrayList<>();
            for (Card c : result) {
                int idx = sourceList.indexOf(c);
                if (idx >= 0) {
                    indices.add(idx);
                }
            }
            record(DecisionType.CARD_SELECTION,
                    sourceList.size(), indices, feats,
                    "cards_for_effect_" + title);
        }
        return result;
    }

    @Override
    public CardCollectionView choosePermanentsToSacrifice(
            SpellAbility sa, int min, int max,
            CardCollectionView validTargets, String msg) {
        CardCollectionView result =
                super.choosePermanentsToSacrifice(
                        sa, min, max, validTargets, msg);
        if (result != null && validTargets.size() > 1) {
            List<float[]> feats = new ArrayList<>();
            List<Card> cardList = new ArrayList<>();
            for (Card c : validTargets) {
                feats.add(CardFeatures.encode(c, player));
                cardList.add(c);
            }
            List<Integer> indices = new ArrayList<>();
            for (Card c : result) {
                int idx = validTargets.indexOf(c);
                if (idx >= 0) {
                    indices.add(idx);
                }
            }
            record(DecisionType.CARD_SELECTION,
                    validTargets.size(), indices, feats,
                    "sacrifice");
        }
        return result;
    }

    @Override
    public CardCollection chooseCardsToDiscardFrom(
            Player p, SpellAbility sa,
            CardCollection validCards, int min, int max) {
        // Cast to CardCollectionView for the super call
        CardCollectionView result =
                super.chooseCardsToDiscardFrom(
                        p, sa, validCards, min, max);
        if (result != null && validCards.size() > 1) {
            List<float[]> feats = new ArrayList<>();
            List<Card> cardList = new ArrayList<>(validCards);
            for (Card c : cardList) {
                feats.add(CardFeatures.encode(c, player));
            }
            List<Integer> indices = new ArrayList<>();
            for (Card c : result) {
                int idx = validCards.indexOf(c);
                if (idx >= 0) {
                    indices.add(idx);
                }
            }
            record(DecisionType.CARD_SELECTION,
                    validCards.size(), indices, feats,
                    "discard");
        }
        return new CardCollection(result);
    }

    @Override
    public boolean mulliganKeepHand(
            Player firstPlayer, int cardsToReturn) {
        // Record hand before decision
        List<float[]> handFeats = new ArrayList<>();
        CardCollectionView hand =
                player.getCardsIn(ZoneType.Hand);
        for (Card c : hand) {
            handFeats.add(CardFeatures.encode(c, player));
        }

        boolean keep = super.mulliganKeepHand(
                firstPlayer, cardsToReturn);

        record(DecisionType.MULLIGAN,
                2, List.of(keep ? 1 : 0), handFeats,
                "mulligan_" + cardsToReturn
                    + (keep ? "_keep" : "_mull"));
        return keep;
    }

    @Override
    public boolean confirmAction(
            SpellAbility sa, PlayerActionConfirmMode mode,
            String message, List<String> options,
            Card cardToShow,
            Map<String, Object> params) {
        boolean result = super.confirmAction(
                sa, mode, message, options,
                cardToShow, params);
        record(DecisionType.BINARY_CHOICE,
                2, List.of(result ? 1 : 0), null,
                "confirm_" + mode);
        return result;
    }

    @Override
    public boolean confirmTrigger(WrappedAbility wrapper) {
        boolean result = super.confirmTrigger(wrapper);
        record(DecisionType.BINARY_CHOICE,
                2, List.of(result ? 1 : 0), null,
                "trigger");
        return result;
    }

    @Override
    public ImmutablePair<CardCollection, CardCollection>
            arrangeForScry(CardCollection topN) {
        List<float[]> feats = new ArrayList<>();
        List<Card> cardList = new ArrayList<>(topN);
        for (Card c : cardList) {
            feats.add(CardFeatures.encode(c, player));
        }
        ImmutablePair<CardCollection, CardCollection> result =
                super.arrangeForScry(topN);

        // Record which cards stayed on top
        List<Integer> topIndices = new ArrayList<>();
        for (Card c : result.getLeft()) {
            int idx = topN.indexOf(c);
            if (idx >= 0) {
                topIndices.add(idx);
            }
        }
        record(DecisionType.CARD_SELECTION,
                topN.size(), topIndices, feats,
                "scry_top_" + result.getLeft().size());
        return result;
    }

    @Override
    @SuppressWarnings("unchecked")
    public <T extends GameEntity> T
            chooseSingleEntityForEffect(
            FCollectionView<T> optionList,
            DelayedReveal delayedReveal,
            SpellAbility sa, String title,
            boolean isOptional, Player targetedPlayer,
            Map<String, Object> params) {
        T result = super.chooseSingleEntityForEffect(
                optionList, delayedReveal, sa, title,
                isOptional, targetedPlayer, params);
        if (result != null && optionList.size() > 1) {
            List<float[]> feats = new ArrayList<>();
            for (T entity : optionList) {
                if (entity instanceof Card) {
                    feats.add(CardFeatures.encode(
                            (Card) entity, player));
                } else {
                    feats.add(ActionEncoder.encodeTarget(
                            entity));
                }
            }
            int idx = 0;
            for (int i = 0; i < optionList.size(); i++) {
                if (optionList.get(i) == result) {
                    idx = i;
                    break;
                }
            }
            record(DecisionType.TARGET_SELECTION,
                    optionList.size(), List.of(idx), feats,
                    "single_entity_" + title);
        }
        return result;
    }
}
