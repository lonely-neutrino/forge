package forge.ai.rl.mcts;

import forge.ai.LobbyPlayerAi;
import forge.ai.PlayerControllerAi;
import forge.game.Game;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.List;

/**
 * A PlayerControllerAi that forces the first spell choice to be a specific
 * SpellAbility, then reverts to normal heuristic AI for all subsequent decisions.
 * The heuristic AI handles targeting, cost payment, and all other choices.
 *
 * Used by MCTS rollouts to test "what happens if I play this spell?"
 */
public class ForcedSpellController extends PlayerControllerAi {

    private SpellAbility forcedSpell;
    private boolean spellForced = false;

    public ForcedSpellController(Game game, Player player, SpellAbility forcedSa) {
        super(game, player, new LobbyPlayerAi(player.getName(), null));
        this.forcedSpell = forcedSa;
    }

    @Override
    public List<SpellAbility> chooseSpellAbilityToPlay() {
        if (!spellForced && forcedSpell != null) {
            spellForced = true;

            // Find the matching SA on this player's cards
            SpellAbility match = findMatchingSa(forcedSpell);
            if (match != null) {
                match.setActivatingPlayer(getPlayer());
                List<SpellAbility> result = new ArrayList<>();
                result.add(match);
                return result;
            }
            // Couldn't find match — fall through to heuristic
        }

        // After forced spell (or if forced was null/pass), use normal AI
        return super.chooseSpellAbilityToPlay();
    }

    /**
     * Find a SpellAbility on this player's cards that matches the forced one
     * by card name and API type.
     */
    private SpellAbility findMatchingSa(SpellAbility origSa) {
        if (origSa == null || origSa.getHostCard() == null) return null;

        String cardName = origSa.getHostCard().getName();
        Object apiType = origSa.getApi();

        // Search hand, battlefield, and other zones
        for (forge.game.card.Card card : getPlayer().getAllCards()) {
            if (card.getName().equals(cardName)) {
                for (SpellAbility sa : card.getAllSpellAbilities()) {
                    if (sa.getApi() == apiType) {
                        return sa;
                    }
                }
                // Fallback: any SA on matching card
                for (SpellAbility sa : card.getSpellAbilities()) {
                    if (sa.isSpell()) return sa;
                }
            }
        }
        return null;
    }
}
