package forge.ai.anvil;

import org.testng.AssertJUnit;
import org.testng.annotations.Test;

import forge.ai.simulation.SimulationTest;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

/** Boundary tests for the additive schema-v3 legal target plans. */
public class CastTargetPlanEnumeratorTest extends SimulationTest {
    @Test
    public void exactlyOneCreatureTargetAndStateRestoration() {
        Game game = initAndCreateGame();
        Player caster = game.getPlayers().get(1);
        Player opponent = game.getPlayers().get(0);
        caster.setTeam(0);
        opponent.setTeam(1);

        Card ownBear = addCard("Runeclaw Bear", caster);
        Card opposingBear = addCard("Runeclaw Bear", opponent);
        Card growth = addCardToZone("Giant Growth", caster, ZoneType.Hand);
        SpellAbility sa = growth.getSpellAbilities().get(0);
        sa.setActivatingPlayer(caster);
        sa.getTargets().add(ownBear);

        CastTargetPlanEnumerator.Result result = CastTargetPlanEnumerator.enumerate(game, sa);

        AssertJUnit.assertTrue(result.reason, result.complete);
        AssertJUnit.assertEquals(2, result.plans.size());
        for (CastTargetPlanEnumerator.Plan plan : result.plans) {
            AssertJUnit.assertEquals(1, plan.refs.size());
            AssertJUnit.assertEquals((1 << CastTargetPlanEnumerator.X_CLASSES) - 1, plan.xMask);
        }
        AssertJUnit.assertTrue(result.plans.stream()
                .anyMatch(p -> p.refs.get(0).entity == ownBear.getId()));
        AssertJUnit.assertTrue(result.plans.stream()
                .anyMatch(p -> p.refs.get(0).entity == opposingBear.getId()));
        AssertJUnit.assertEquals(1, sa.getTargets().size());
        AssertJUnit.assertEquals(ownBear, sa.getTargets().getFirstTargetedCard());
    }

    @Test
    public void targetlessSpellHasOneEmptyPlan() {
        Game game = initAndCreateGame();
        Player caster = game.getPlayers().get(1);
        Card wrath = addCardToZone("Wrath of God", caster, ZoneType.Hand);
        SpellAbility sa = wrath.getSpellAbilities().get(0);
        sa.setActivatingPlayer(caster);

        CastTargetPlanEnumerator.Result result = CastTargetPlanEnumerator.enumerate(game, sa);

        AssertJUnit.assertTrue(result.reason, result.complete);
        AssertJUnit.assertEquals(1, result.plans.size());
        AssertJUnit.assertTrue(result.plans.get(0).refs.isEmpty());
    }
}
