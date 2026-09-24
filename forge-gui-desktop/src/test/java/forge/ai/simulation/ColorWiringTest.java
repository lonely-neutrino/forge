package forge.ai.simulation;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import org.testng.AssertJUnit;
import org.testng.annotations.Test;

import forge.ai.anvil.AnvilBridge;
import forge.ai.anvil.CastPlanAnswer;
import forge.ai.anvil.PlayerControllerAnvil;
import forge.card.ColorSet;
import forge.card.MagicColor;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

/** M11 single-color bridge and targetless cast realization coverage. */
public class ColorWiringTest extends SimulationTest {

    private static final class StubBridge implements AnvilBridge {
        int answer;
        String tag;
        List<String> labels;

        @Override
        public int selectOne(String tag, List<String> optionLabels) {
            this.tag = tag;
            this.labels = new ArrayList<>(optionLabels);
            return answer;
        }

        @Override
        public int[] selectK(String tag, int n, int k) {
            throw new UnsupportedOperationException();
        }

        @Override
        public boolean bool(String tag) {
            throw new UnsupportedOperationException();
        }

        @Override
        public int intInRange(String tag, int min, int max) {
            throw new UnsupportedOperationException();
        }
    }

    @Test
    public void testChooseColorUsesCanonicalWubrgOptions() {
        Game game = initAndCreateGame();
        Player p = game.getPlayers().get(1);
        StubBridge bridge = new StubBridge();
        bridge.answer = 3;
        PlayerControllerAnvil controller = new PlayerControllerAnvil(
                game, p, p.getLobbyPlayer(), bridge, Set.of(PlayerControllerAnvil.TAG_COLOR));

        byte chosen = controller.chooseColor("Choose a color", null, ColorSet.WUBRG);

        AssertJUnit.assertEquals(PlayerControllerAnvil.TAG_COLOR, bridge.tag);
        AssertJUnit.assertEquals(List.of("white", "blue", "black", "red", "green"),
                bridge.labels);
        AssertJUnit.assertEquals(MagicColor.RED, chosen);
    }

    @Test
    public void testChooseColorsSingleUsesColorBridge() {
        Game game = initAndCreateGame();
        Player p = game.getPlayers().get(1);
        StubBridge bridge = new StubBridge();
        bridge.answer = 4;
        PlayerControllerAnvil controller = new PlayerControllerAnvil(
                game, p, p.getLobbyPlayer(), bridge, Set.of(PlayerControllerAnvil.TAG_COLOR));

        ColorSet chosen = controller.chooseColors(
                "Choose a color", null, 1, 1, ColorSet.WUBRG);

        AssertJUnit.assertEquals(PlayerControllerAnvil.TAG_COLOR, bridge.tag);
        AssertJUnit.assertEquals(List.of("white", "blue", "black", "red", "green"),
                bridge.labels);
        AssertJUnit.assertEquals(MagicColor.GREEN, chosen.getColor());
    }

    @Test
    public void testTargetlessBraveSurvivesIrrelevantTargetRef() {
        Game game = initAndCreateGame();
        Player p = game.getPlayers().get(1);
        Player opponent = game.getPlayers().get(0);
        addCard("Plains", p);
        Card brave = addCardToZone("Brave the Elements", p, ZoneType.Hand);
        Card decoy = addCard("Grizzly Bears", opponent);
        game.getPhaseHandler().devModeSet(PhaseType.MAIN1, p);
        game.getAction().checkStateEffects(true);

        SpellAbility sa = brave.getFirstSpellAbility();
        sa.setActivatingPlayer(p);
        CastPlanAnswer answer = new CastPlanAnswer(
                1, false,
                List.of(new CastPlanAnswer.Ref(false, -1, decoy.getId(), false)),
                false, 0);

        forge.ai.anvil.CastPlanRealizer.Result result =
                forge.ai.anvil.CastPlanRealizer.realize(game, p, List.of(sa), answer);

        AssertJUnit.assertNotNull("Brave should be shape-realizable", result.sa);
        AssertJUnit.assertTrue("the targetless recovery must be observable",
                result.ignoredTargets);
    }

    @Test
    public void testTargetBearingSpellRejectsDanglingTargetRef() {
        Game game = initAndCreateGame();
        Player p = game.getPlayers().get(1);
        addCard("Mountain", p);
        Card bolt = addCardToZone("Lightning Bolt", p, ZoneType.Hand);
        game.getPhaseHandler().devModeSet(PhaseType.MAIN1, p);
        game.getAction().checkStateEffects(true);

        SpellAbility sa = bolt.getFirstSpellAbility();
        sa.setActivatingPlayer(p);
        CastPlanAnswer answer = new CastPlanAnswer(
                1, false,
                List.of(new CastPlanAnswer.Ref(false, -1, Integer.MAX_VALUE, false)),
                false, 0);

        forge.ai.anvil.CastPlanRealizer.Result result =
                forge.ai.anvil.CastPlanRealizer.realize(game, p, List.of(sa), answer);

        AssertJUnit.assertNull("target-bearing spells remain strict", result.sa);
        AssertJUnit.assertEquals("dangling_ref", result.veto);
    }
}
