package forge.ai.rl;

import com.google.common.collect.ListMultimap;
import com.google.common.collect.Multimap;
import forge.LobbyPlayer;
import forge.card.ColorSet;
import forge.card.ICardFace;
import forge.card.mana.ManaCost;
import forge.card.mana.ManaCostShard;
import forge.deck.Deck;
import forge.deck.DeckSection;
import forge.game.*;
import forge.game.ability.effects.RollDiceEffect;
import forge.game.card.*;
import forge.game.combat.Combat;
import forge.game.cost.*;
import forge.game.keyword.KeywordInterface;
import forge.game.mana.Mana;
import forge.game.mana.ManaConversionMatrix;
import forge.game.mana.ManaCostBeingPaid;
import forge.game.player.*;
import forge.game.replacement.ReplacementEffect;
import forge.game.spellability.*;
import forge.game.staticability.StaticAbility;
import forge.game.trigger.WrappedAbility;
import forge.game.zone.PlayerZone;
import forge.game.zone.ZoneType;
import forge.item.PaperCard;
import forge.util.ITriggerEvent;
import forge.util.collect.FCollectionView;
import forge.ai.rl.decisions.*;
import forge.ai.rl.features.*;
import forge.ai.rl.training.TrajectoryRecorder;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;

import java.util.*;
import java.util.function.Predicate;

/**
 * Wraps a real PlayerController and records every decision as a
 * trajectory step. The wrapped controller makes all actual decisions.
 *
 * This is a delegation pattern — every method calls the delegate
 * and records the result. Only the most important decisions
 * (spell selection, combat, card choices) are recorded to avoid
 * massive trajectory files.
 */
public class RecordingPlayerController extends PlayerController {
    private final PlayerController delegate;
    private final GameStateEncoder encoder;
    private final TrajectoryRecorder recorder;
    private boolean gameStarted = false;

    public RecordingPlayerController(
            Game game, Player p, LobbyPlayer lp,
            PlayerController delegate, RLConfig config) {
        super(game, p, lp);
        this.delegate = delegate;
        this.encoder = new GameStateEncoder(config);
        this.recorder = new TrajectoryRecorder(
                config.getTrajectoryOutputDir(),
                config.isZeroIntermediateReward());
    }

    /** Call once at game start. */
    public void onGameStart(String gameId) {
        recorder.startGame(gameId);
        gameStarted = true;
    }

    /** Call once at game end. */
    public void onGameEnd(boolean won) {
        if (gameStarted) {
            recorder.endGame(won);
            gameStarted = false;
        }
    }

    // Helper to record a decision
    private void record(DecisionType type, int numCandidates,
                        List<Integer> selected, String info) {
        if (!gameStarted) {
            return;
        }
        try {
            GameStateFeatures gs = encoder.encode(getGame(), player);
            DecisionContext ctx = new DecisionContext(
                    type, gs, List.of(), 1,
                    numCandidates, info);
            DecisionResult res = new DecisionResult(
                    selected, new float[0], 0f, true);
            Player opp = player.getWeakestOpponent();
            recorder.recordDecision(ctx, res,
                    player.getLife(),
                    opp != null ? opp.getLife() : 0,
                    player.getCardsIn(ZoneType.Hand).size(),
                    opp != null
                        ? opp.getCardsIn(ZoneType.Hand).size() : 0,
                    countCreatures(player),
                    opp != null ? countCreatures(opp) : 0);
        } catch (Exception e) {
            // Don't let recording errors crash the game
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

    // ===== RECORDED DECISIONS =====

    @Override
    public boolean isAI() { return delegate.isAI(); }

    @Override
    public List<SpellAbility> chooseSpellAbilityToPlay() {
        List<SpellAbility> result =
                delegate.chooseSpellAbilityToPlay();
        if (result != null && !result.isEmpty()) {
            record(DecisionType.PRIORITY_ACTION, 1,
                    List.of(0), "spell_play");
        }
        return result;
    }

    @Override
    public boolean playChosenSpellAbility(SpellAbility sa) {
        return delegate.playChosenSpellAbility(sa);
    }

    @Override
    public SpellAbility getAbilityToPlay(Card hostCard,
            List<SpellAbility> abilities, ITriggerEvent e) {
        SpellAbility result =
                delegate.getAbilityToPlay(hostCard, abilities, e);
        if (result != null && abilities.size() > 1) {
            int idx = abilities.indexOf(result);
            record(DecisionType.PRIORITY_ACTION,
                    abilities.size(),
                    List.of(Math.max(0, idx)),
                    "ability_choice");
        }
        return result;
    }

    @Override
    public void declareAttackers(Player attacker, Combat combat) {
        int beforeCount = combat.getAttackers().size();
        delegate.declareAttackers(attacker, combat);
        int afterCount = combat.getAttackers().size();
        record(DecisionType.DECLARE_ATTACKERS,
                attacker.getCreaturesInPlay().size(),
                List.of(afterCount),
                "attackers_" + afterCount);
    }

    @Override
    public void declareBlockers(Player defender, Combat combat) {
        delegate.declareBlockers(defender, combat);
        record(DecisionType.DECLARE_BLOCKERS, 0,
                List.of(0), "blockers");
    }

    @Override
    public CardCollectionView chooseCardsForEffect(
            CardCollectionView sourceList, SpellAbility sa,
            String title, int min, int max,
            boolean isOptional, Map<String, Object> params) {
        CardCollectionView result = delegate.chooseCardsForEffect(
                sourceList, sa, title, min, max,
                isOptional, params);
        if (result != null && !result.isEmpty()) {
            record(DecisionType.CARD_SELECTION,
                    sourceList.size(),
                    List.of(result.size()),
                    "cards_for_effect");
        }
        return result;
    }

    @Override
    public CardCollectionView choosePermanentsToSacrifice(
            SpellAbility sa, int min, int max,
            CardCollectionView valid, String msg) {
        CardCollectionView result =
                delegate.choosePermanentsToSacrifice(
                        sa, min, max, valid, msg);
        record(DecisionType.CARD_SELECTION,
                valid.size(),
                List.of(result != null ? result.size() : 0),
                "sacrifice");
        return result;
    }

    @Override
    public CardCollectionView chooseCardsToDiscardFrom(
            Player p, SpellAbility sa,
            CardCollection valid, int min, int max) {
        CardCollectionView result =
                delegate.chooseCardsToDiscardFrom(
                        p, sa, valid, min, max);
        record(DecisionType.CARD_SELECTION,
                valid.size(),
                List.of(result != null ? result.size() : 0),
                "discard");
        return result;
    }

    @Override
    public boolean mulliganKeepHand(Player p, int cardsToReturn) {
        boolean result =
                delegate.mulliganKeepHand(p, cardsToReturn);
        record(DecisionType.MULLIGAN, 2,
                List.of(result ? 1 : 0),
                "mulligan_" + cardsToReturn);
        return result;
    }

    @Override
    public boolean confirmAction(SpellAbility sa,
            PlayerActionConfirmMode mode, String msg,
            List<String> options, Card cardToShow,
            Map<String, Object> params) {
        boolean result = delegate.confirmAction(
                sa, mode, msg, options, cardToShow, params);
        record(DecisionType.BINARY_CHOICE, 2,
                List.of(result ? 1 : 0), "confirm");
        return result;
    }

    @Override
    public boolean confirmTrigger(WrappedAbility sa) {
        boolean result = delegate.confirmTrigger(sa);
        record(DecisionType.BINARY_CHOICE, 2,
                List.of(result ? 1 : 0), "trigger");
        return result;
    }

    @Override
    public ImmutablePair<CardCollection, CardCollection>
            arrangeForScry(CardCollection topN) {
        ImmutablePair<CardCollection, CardCollection> result =
                delegate.arrangeForScry(topN);
        record(DecisionType.CARD_SELECTION,
                topN.size(),
                List.of(result.getLeft().size()),
                "scry");
        return result;
    }

    // ===== PURE DELEGATION (no recording) =====

    @Override
    public void playSpellAbilityNoStack(
            SpellAbility sa, boolean b) {
        delegate.playSpellAbilityNoStack(sa, b);
    }
    @Override
    public List<SpellAbility> orderSimultaneousSa(
            List<SpellAbility> l) {
        return delegate.orderSimultaneousSa(l);
    }
    @Override
    public void orderAndPlaySimultaneousSa(
            List<SpellAbility> l) {
        delegate.orderAndPlaySimultaneousSa(l);
    }
    @Override
    public boolean playTrigger(Card h, WrappedAbility w,
            boolean m) {
        return delegate.playTrigger(h, w, m);
    }
    @Override
    public boolean playSaFromPlayEffect(SpellAbility sa) {
        return delegate.playSaFromPlayEffect(sa);
    }
    @Override
    public List<PaperCard> sideboard(Deck d, GameType g,
            String m) {
        return delegate.sideboard(d, g, m);
    }
    @Override
    public List<PaperCard> chooseCardsYouWonToAddToDeck(
            List<PaperCard> l) {
        return delegate.chooseCardsYouWonToAddToDeck(l);
    }
    @Override
    public Map<Card, Integer> assignCombatDamage(
            Card a, CardCollectionView b, CardCollectionView r,
            int d, GameEntity def, boolean o) {
        return delegate.assignCombatDamage(a, b, r, d, def, o);
    }
    @Override
    public Map<GameEntity, Integer> divideShield(
            Card s, Map<GameEntity, Integer> a, int sh) {
        return delegate.divideShield(s, a, sh);
    }
    @Override
    public Map<Byte, Integer> specifyManaCombo(
            SpellAbility sa, ColorSet c, int m, boolean d) {
        return delegate.specifyManaCombo(sa, c, m, d);
    }
    @Override
    public CardCollectionView choosePermanentsToDestroy(
            SpellAbility sa, int min, int max,
            CardCollectionView v, String m) {
        return delegate.choosePermanentsToDestroy(
                sa, min, max, v, m);
    }
    @Override
    public Integer announceRequirements(
            SpellAbility a, int min, int max, String s) {
        return delegate.announceRequirements(a, min, max, s);
    }
    @Override
    public TargetChoices chooseNewTargetsFor(
            SpellAbility a, Predicate<GameObject> f, boolean o) {
        return delegate.chooseNewTargetsFor(a, f, o);
    }
    @Override
    public boolean chooseTargetsFor(SpellAbility sa) {
        return delegate.chooseTargetsFor(sa);
    }
    @Override
    public Pair<SpellAbilityStackInstance, GameObject>
            chooseTarget(SpellAbility sa,
            List<Pair<SpellAbilityStackInstance,
                    GameObject>> all) {
        return delegate.chooseTarget(sa, all);
    }
    @Override
    public boolean helpPayForAssistSpell(
            ManaCostBeingPaid c, SpellAbility sa,
            int max, int req) {
        return delegate.helpPayForAssistSpell(c, sa, max, req);
    }
    @Override
    public Player choosePlayerToAssistPayment(
            FCollectionView<Player> o, SpellAbility sa,
            String t, int max) {
        return delegate.choosePlayerToAssistPayment(
                o, sa, t, max);
    }
    @Override
    public CardCollection chooseCardsForEffectMultiple(
            Map<String, CardCollection> v, SpellAbility sa,
            String t, boolean o) {
        return delegate.chooseCardsForEffectMultiple(
                v, sa, t, o);
    }
    @Override
    public <T extends GameEntity> T
            chooseSingleEntityForEffect(
            FCollectionView<T> opt, DelayedReveal dr,
            SpellAbility sa, String t, boolean o,
            Player rp, Map<String, Object> p) {
        return delegate.chooseSingleEntityForEffect(
                opt, dr, sa, t, o, rp, p);
    }
    @Override
    public <T extends GameEntity> List<T>
            chooseEntitiesForEffect(
            FCollectionView<T> opt, int min, int max,
            DelayedReveal dr, SpellAbility sa, String t,
            Player rp, Map<String, Object> p) {
        return delegate.chooseEntitiesForEffect(
                opt, min, max, dr, sa, t, rp, p);
    }
    @Override
    public List<SpellAbility> chooseSpellAbilitiesForEffect(
            List<SpellAbility> s, SpellAbility sa,
            String t, int n, Map<String, Object> p) {
        return delegate.chooseSpellAbilitiesForEffect(
                s, sa, t, n, p);
    }
    @Override
    public SpellAbility chooseSingleSpellForEffect(
            List<SpellAbility> s, SpellAbility sa,
            String t, Map<String, Object> p) {
        return delegate.chooseSingleSpellForEffect(s, sa, t, p);
    }
    @Override
    public boolean confirmBidAction(SpellAbility sa,
            PlayerActionConfirmMode m, String s,
            int b, Player w) {
        return delegate.confirmBidAction(sa, m, s, b, w);
    }
    @Override
    public boolean confirmReplacementEffect(
            ReplacementEffect r, SpellAbility sa,
            GameEntity a, String q) {
        return delegate.confirmReplacementEffect(r, sa, a, q);
    }
    @Override
    public boolean confirmStaticApplication(
            Card h, PlayerActionConfirmMode m,
            String msg, String l) {
        return delegate.confirmStaticApplication(h, m, msg, l);
    }
    @Override
    public List<Card> exertAttackers(List<Card> a) {
        return delegate.exertAttackers(a);
    }
    @Override
    public List<Card> enlistAttackers(List<Card> a) {
        return delegate.enlistAttackers(a);
    }
    @Override
    public CardCollection orderBlockers(
            Card a, CardCollection b) {
        return delegate.orderBlockers(a, b);
    }
    @Override
    public CardCollection orderBlocker(
            Card a, Card b, CardCollection old) {
        return delegate.orderBlocker(a, b, old);
    }
    @Override
    public CardCollection orderAttackers(
            Card b, CardCollection a) {
        return delegate.orderAttackers(b, a);
    }
    @Override
    public void reveal(CardCollectionView c, ZoneType z,
            Player o, String m, boolean a) {
        delegate.reveal(c, z, o, m, a);
    }
    @Override
    public void reveal(List<CardView> c, ZoneType z,
            PlayerView o, String m, boolean a) {
        delegate.reveal(c, z, o, m, a);
    }
    @Override
    public void notifyOfValue(SpellAbility sa,
            GameObject t, String v) {
        delegate.notifyOfValue(sa, t, v);
    }
    @Override
    public ImmutablePair<CardCollection, CardCollection>
            arrangeForSurveil(CardCollection t) {
        return delegate.arrangeForSurveil(t);
    }
    @Override
    public boolean willPutCardOnTop(Card c) {
        return delegate.willPutCardOnTop(c);
    }
    @Override
    public CardCollectionView orderMoveToZoneList(
            CardCollectionView c, ZoneType z, SpellAbility s) {
        return delegate.orderMoveToZoneList(c, z, s);
    }
    @Override
    public CardCollectionView chooseCardsToDiscardUnlessType(
            int min, CardCollectionView h, String[] u,
            SpellAbility sa) {
        return delegate.chooseCardsToDiscardUnlessType(
                min, h, u, sa);
    }
    @Override
    public CardCollection chooseCardsToDiscardToMaximumHandSize(
            int n) {
        return delegate.chooseCardsToDiscardToMaximumHandSize(n);
    }
    @Override
    public CardCollectionView chooseCardsToDelve(
            int g, CardCollection gr) {
        return delegate.chooseCardsToDelve(g, gr);
    }
    @Override
    public Map<Card, ManaCostShard>
            chooseCardsForConvokeOrImprovise(
            SpellAbility sa, ManaCost mc,
            CardCollectionView u, boolean a, boolean c,
            Integer max) {
        return delegate.chooseCardsForConvokeOrImprovise(
                sa, mc, u, a, c, max);
    }
    @Override
    public List<Card> chooseCardsForSplice(
            SpellAbility sa, List<Card> c) {
        return delegate.chooseCardsForSplice(sa, c);
    }
    @Override
    public CardCollectionView chooseCardsToRevealFromHand(
            int min, int max, CardCollectionView v) {
        return delegate.chooseCardsToRevealFromHand(
                min, max, v);
    }
    @Override
    public List<SpellAbility>
            chooseSaToActivateFromOpeningHand(
            List<SpellAbility> u) {
        return delegate.chooseSaToActivateFromOpeningHand(u);
    }
    @Override
    public Player chooseStartingPlayer(boolean f) {
        return delegate.chooseStartingPlayer(f);
    }
    @Override
    public PlayerZone chooseStartingHand(
            List<PlayerZone> z) {
        return delegate.chooseStartingHand(z);
    }
    @Override
    public Mana chooseManaFromPool(List<Mana> m) {
        return delegate.chooseManaFromPool(m);
    }
    @Override
    public String chooseSomeType(String k, SpellAbility sa,
            Collection<String> v, boolean o) {
        return delegate.chooseSomeType(k, sa, v, o);
    }
    @Override
    public String chooseSector(Card a, String ai,
            List<String> s) {
        return delegate.chooseSector(a, ai, s);
    }
    @Override
    public List<Card> chooseContraptionsToCrank(
            List<Card> c) {
        return delegate.chooseContraptionsToCrank(c);
    }
    @Override
    public int chooseSprocket(Card a, List<Integer> s) {
        return delegate.chooseSprocket(a, s);
    }
    @Override
    public PlanarDice choosePDRollToIgnore(
            List<PlanarDice> r) {
        return delegate.choosePDRollToIgnore(r);
    }
    @Override
    public Integer chooseRollToIgnore(List<Integer> r) {
        return delegate.chooseRollToIgnore(r);
    }
    @Override
    public List<Integer> chooseDiceToReroll(
            List<Integer> r) {
        return delegate.chooseDiceToReroll(r);
    }
    @Override
    public Integer chooseRollToModify(List<Integer> r) {
        return delegate.chooseRollToModify(r);
    }
    @Override
    public RollDiceEffect.DieRollResult chooseRollToSwap(
            List<RollDiceEffect.DieRollResult> r) {
        return delegate.chooseRollToSwap(r);
    }
    @Override
    public String chooseRollSwapValue(List<String> s,
            Integer c, int p, int t) {
        return delegate.chooseRollSwapValue(s, c, p, t);
    }
    @Override
    public Object vote(SpellAbility sa, String p,
            List<Object> o, ListMultimap<Object, Player> v,
            Player f, boolean opt) {
        return delegate.vote(sa, p, o, v, f, opt);
    }
    @Override
    public CardCollectionView tuckCardsViaMulligan(
            CardCollectionView h, int c) {
        return delegate.tuckCardsViaMulligan(h, c);
    }
    @Override
    public List<AbilitySub> chooseModeForAbility(
            SpellAbility sa, List<AbilitySub> p,
            int min, int num, boolean a) {
        return delegate.chooseModeForAbility(
                sa, p, min, num, a);
    }
    @Override
    public int chooseNumberForCostReduction(
            SpellAbility sa, int min, int max) {
        return delegate.chooseNumberForCostReduction(
                sa, min, max);
    }
    @Override
    public int chooseNumberForKeywordCost(
            SpellAbility sa, Cost c, KeywordInterface k,
            String p, int max) {
        return delegate.chooseNumberForKeywordCost(
                sa, c, k, p, max);
    }
    @Override
    public int chooseNumber(SpellAbility sa, String t,
            int min, int max) {
        return delegate.chooseNumber(sa, t, min, max);
    }
    @Override
    public int chooseNumber(SpellAbility sa, String t,
            List<Integer> v, Player rp) {
        return delegate.chooseNumber(sa, t, v, rp);
    }
    @Override
    public boolean chooseBinary(SpellAbility sa, String q,
            BinaryChoiceType k, Boolean d) {
        return delegate.chooseBinary(sa, q, k, d);
    }
    @Override
    public boolean chooseFlipResult(SpellAbility sa,
            Player f, boolean c) {
        return delegate.chooseFlipResult(sa, f, c);
    }
    @Override
    public byte chooseColor(String m, SpellAbility sa,
            ColorSet c) {
        return delegate.chooseColor(m, sa, c);
    }
    @Override
    public byte chooseColorAllowColorless(String m,
            Card c, ColorSet cs) {
        return delegate.chooseColorAllowColorless(m, c, cs);
    }
    @Override
    public ColorSet chooseColors(String m, SpellAbility sa,
            int min, int max, ColorSet o) {
        return delegate.chooseColors(m, sa, min, max, o);
    }
    @Override
    public ICardFace chooseSingleCardFace(SpellAbility sa,
            String m, Predicate<ICardFace> p, String n) {
        return delegate.chooseSingleCardFace(sa, m, p, n);
    }
    @Override
    public ICardFace chooseSingleCardFace(SpellAbility sa,
            List<ICardFace> f, String m) {
        return delegate.chooseSingleCardFace(sa, f, m);
    }
    @Override
    public CardState chooseSingleCardState(SpellAbility sa,
            List<CardState> s, String m,
            Map<String, Object> p) {
        return delegate.chooseSingleCardState(sa, s, m, p);
    }
    @Override
    public boolean chooseCardsPile(SpellAbility sa,
            CardCollectionView p1, CardCollectionView p2,
            String f) {
        return delegate.chooseCardsPile(sa, p1, p2, f);
    }
    @Override
    public CounterType chooseCounterType(
            List<CounterType> o, SpellAbility sa,
            String p, Map<String, Object> params) {
        return delegate.chooseCounterType(o, sa, p, params);
    }
    @Override
    public String chooseKeywordForPump(List<String> o,
            SpellAbility sa, String p, Card t) {
        return delegate.chooseKeywordForPump(o, sa, p, t);
    }
    @Override
    public boolean confirmPayment(CostPart c,
            String s, SpellAbility sa) {
        return delegate.confirmPayment(c, s, sa);
    }
    @Override
    public ReplacementEffect chooseSingleReplacementEffect(
            List<ReplacementEffect> p) {
        return delegate.chooseSingleReplacementEffect(p);
    }
    @Override
    public StaticAbility chooseSingleStaticAbility(
            List<StaticAbility> p) {
        return delegate.chooseSingleStaticAbility(p);
    }
    @Override
    public String chooseProtectionType(SpellAbility sa,
            List<String> c) {
        return delegate.chooseProtectionType(sa, c);
    }
    @Override
    public void revealAnte(String m,
            Multimap<Player, PaperCard> r) {
        delegate.revealAnte(m, r);
    }
    @Override
    public void revealAISkipCards(String m,
            Map<Player, Map<DeckSection,
                    List<? extends PaperCard>>> d) {
        delegate.revealAISkipCards(m, d);
    }
    @Override
    public void revealUnsupported(
            Map<Player, List<PaperCard>> u) {
        delegate.revealUnsupported(u);
    }
    @Override
    public void resetAtEndOfTurn() {
        delegate.resetAtEndOfTurn();
    }
    @Override
    public List<OptionalCostValue> chooseOptionalCosts(
            SpellAbility s, List<OptionalCostValue> o) {
        return delegate.chooseOptionalCosts(s, o);
    }
    @Override
    public List<CostPart> orderCosts(List<CostPart> c) {
        return delegate.orderCosts(c);
    }
    @Override
    public boolean payCostToPreventEffect(Cost c,
            SpellAbility sa, boolean a,
            FCollectionView<Player> p) {
        return delegate.payCostToPreventEffect(c, sa, a, p);
    }
    @Override
    public boolean payCostDuringRoll(Cost c,
            SpellAbility sa) {
        return delegate.payCostDuringRoll(c, sa);
    }
    @Override
    public boolean payCombatCost(Card c, Cost cost,
            SpellAbility sa, String p) {
        return delegate.payCombatCost(c, cost, sa, p);
    }
    @Override
    public boolean payManaCost(ManaCost t, CostPartMana cpm,
            SpellAbility sa, String p,
            ManaConversionMatrix m, boolean e) {
        return delegate.payManaCost(t, cpm, sa, p, m, e);
    }
    @Override
    public boolean applyManaToCost(ManaCostBeingPaid t,
            SpellAbility a, String p,
            ManaConversionMatrix m, boolean e) {
        return delegate.applyManaToCost(t, a, p, m, e);
    }
    @Override
    public CardCollectionView chooseCardsForCost(
            CardCollectionView o, SpellAbility sa,
            CostPartWithList c, int a, boolean opt,
            String p) {
        return delegate.chooseCardsForCost(
                o, sa, c, a, opt, p);
    }
    @Override
    public CostDecisionMakerBase getCostDecisionMaker(
            Player p, SpellAbility a, boolean e, String pr) {
        return delegate.getCostDecisionMaker(p, a, e, pr);
    }
    @Override
    public String chooseCardName(SpellAbility sa,
            Predicate<ICardFace> p, String v, String m) {
        return delegate.chooseCardName(sa, p, v, m);
    }
    @Override
    public String chooseCardName(SpellAbility sa,
            List<ICardFace> f, String m) {
        return delegate.chooseCardName(sa, f, m);
    }
    @Override
    public Card chooseSingleCardForZoneChange(
            ZoneType d, List<ZoneType> o, SpellAbility sa,
            CardCollection f, DelayedReveal dr, String s,
            boolean opt, Player dec) {
        return delegate.chooseSingleCardForZoneChange(
                d, o, sa, f, dr, s, opt, dec);
    }
    @Override
    public List<Card> chooseCardsForZoneChange(
            ZoneType d, List<ZoneType> o, SpellAbility sa,
            CardCollection f, int min, int max,
            DelayedReveal dr, String s, Player dec) {
        return delegate.chooseCardsForZoneChange(
                d, o, sa, f, min, max, dr, s, dec);
    }
    @Override
    public void autoPassCancel() {
        delegate.autoPassCancel();
    }
    @Override
    public void awaitNextInput() {
        delegate.awaitNextInput();
    }
    @Override
    public void cancelAwaitNextInput() {
        delegate.cancelAwaitNextInput();
    }
}
