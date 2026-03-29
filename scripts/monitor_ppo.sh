#!/bin/bash
# Monitor PPO training rounds and log stats to a file
# Usage: monitor_ppo.sh [log_file]
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/rl_common.sh"
LOG=${1:-"$FORGE_RL_LOG_DIR/ppo_monitor.log"}
STATE="$FORGE_RL_CHECKPOINT_DIR/ppo_training_state.json"
TRAJ_DIR="$FORGE_RL_PPO_TRAJ_DIR"
EVAL_DIR="$FORGE_RL_EVAL_DIR"
LAST_ROUND=-1

echo "PPO Monitor started — logging to $LOG" | tee "$LOG"
echo "Watching for new rounds..." | tee -a "$LOG"

analyze_trajectories() {
    local DIR="$1"
    local LABEL="$2"
    python3 << PYEOF
import json, os
from collections import defaultdict

DIR = "$DIR"
LABEL = "$LABEL"
TYPE_NAMES = ['Creature','Instant','Sorcery','Enchantment','Artifact','Planeswalker','Land']
COLOR_NAMES = {0:'White',1:'Blue',2:'Black',3:'Red',4:'Green'}

files = [f for f in os.listdir(DIR) if f.endswith('.jsonl')]
if not files:
    print(f"  [{LABEL}] No files")
    exit()

total_decisions = 0
total_fallback = 0
deck_wins = defaultdict(int)
deck_games = defaultdict(int)
type_counts = defaultdict(int)
attack_all = attack_partial = attack_none = 0
attackers_total = candidates_total = 0
game_spells = []
game_turns = []
wins = 0
spells_by_turn = defaultdict(list)

for fname in files:
    with open(os.path.join(DIR, fname)) as fh:
        lines = fh.readlines()
    if not lines: continue
    header = json.loads(lines[0])
    won = header.get('won', False)
    if won: wins += 1

    deck_color = None
    g_spells_by_turn = defaultdict(int)
    max_turn = 0

    for line in lines[1:]:
        try:
            rec = json.loads(line)
        except:
            continue
        total_decisions += 1
        if rec.get('usedFallback', False):
            total_fallback += 1

        gf = rec.get('globalFeatures', [])
        game_turn = int(round(gf[4] * 30)) if len(gf) > 4 else 0
        max_turn = max(max_turn, game_turn)
        dt = rec.get('decisionType', '')

        if dt == 'MULLIGAN' and deck_color is None:
            cands = rec.get('candidateFeatures', [])
            cc = defaultdict(int)
            for card in cands:
                if len(card) > 12:
                    for ci in range(5):
                        if card[7+ci] > 0.5:
                            cc[ci] += 1
            if cc:
                deck_color = COLOR_NAMES.get(max(cc, key=cc.get), '?')

        if dt == 'PRIORITY_ACTION':
            cands = rec.get('candidateFeatures', [])
            sel = rec.get('selectedIndices', [])
            if sel and sel[0] < len(cands):
                chosen = cands[sel[0]]
                card_type = None
                for ti in range(7):
                    if chosen[ti] > 0.5:
                        card_type = TYPE_NAMES[ti]
                        break
                if card_type and card_type != 'Land':
                    g_spells_by_turn[game_turn] += 1
                    type_counts[card_type] += 1

        if dt == 'DECLARE_ATTACKERS':
            sel = rec.get('selectedIndices', [])
            cands = rec.get('candidateFeatures', [])
            n_sel, n_cand = len(sel), len(cands)
            attackers_total += n_sel
            candidates_total += n_cand
            if n_sel == 0: attack_none += 1
            elif n_sel == n_cand: attack_all += 1
            else: attack_partial += 1

    if deck_color is None: deck_color = '?'
    deck_games[deck_color] += 1
    if won: deck_wins[deck_color] += 1
    game_spells.append(sum(g_spells_by_turn.values()))
    game_turns.append(max_turn)
    for t in range(max_turn + 1):
        spells_by_turn[t].append(g_spells_by_turn.get(t, 0))

n = len(files)
avg_spells = sum(game_spells) / n
avg_turns = sum(game_turns) / n
total_atk = attack_all + attack_partial + attack_none

print(f"")
print(f"  --- {LABEL} ({n} games) ---")
print(f"  Win rate: {wins}/{n} = {wins/n*100:.1f}%")
print(f"  Fallbacks: {total_fallback}/{total_decisions}")
print(f"  Decks: ", end="")
for d in sorted(deck_games.keys()):
    g, w = deck_games[d], deck_wins[d]
    print(f"{d}={w}/{g}({w/g*100:.0f}%) ", end="")
print()
if avg_turns > 0:
    print(f"  Spells/game: {avg_spells:.1f}  Turns/game: {avg_turns:.1f}  Spells/turn: {avg_spells/avg_turns:.2f}")

early = []
for t in range(1, 8):
    if t in spells_by_turn and len(spells_by_turn[t]) >= 10:
        avg = sum(spells_by_turn[t]) / len(spells_by_turn[t])
        early.append(f"T{t}:{avg:.2f}")
if early:
    print(f"  Spells/turn curve: {' '.join(early)}")

types_str = ' '.join(f"{k}={v}" for k, v in sorted(type_counts.items(), key=lambda x: -x[1]))
print(f"  Types: {types_str}")
if total_atk > 0:
    print(f"  Attacks: rate={attackers_total/candidates_total*100:.0f}% all-in={attack_all/total_atk*100:.0f}% hold={attack_none/total_atk*100:.0f}% avg={attackers_total/total_atk:.1f}/phase")
PYEOF
}

while true; do
    if [ ! -f "$STATE" ]; then
        sleep 10
        continue
    fi

    CURRENT_ROUND=$(python3 -c "import json; print(json.load(open('$STATE'))['completed_rounds'])" 2>/dev/null)
    if [ -z "$CURRENT_ROUND" ] || [ "$CURRENT_ROUND" = "$LAST_ROUND" ]; then
        sleep 15
        continue
    fi

    # New round completed
    LAST_ROUND=$CURRENT_ROUND
    {
        echo ""
        echo "================================================================"
        echo "ROUND $CURRENT_ROUND COMPLETED — $(date '+%Y-%m-%d %H:%M:%S')"
        echo "================================================================"

        # Training state summary
        python3 << PYEOF
import json
state = json.load(open("$STATE"))
wr = state.get('win_rates', [])
print(f"  Win rates:  {[f'{w:.1%}' for w in wr[-5:]]}")
print(f"  Best:       {state.get('best_win_rate',0):.1%} (round {state.get('best_round','?')})")
print(f"  Elo:        {state.get('current_elo',0):.0f}")
pl = state.get('policy_losses', [])
vl = state.get('value_losses', [])
ent = state.get('entropies', [])
if pl: print(f"  Policy loss: {pl[-1]:.4f}")
if vl: print(f"  Value loss:  {vl[-1]:.4f}")
if ent: print(f"  Entropy:     {ent[-1]:.4f}")
PYEOF

        # Analyze collect trajectories
        TRAJ_COUNT=$(ls "$TRAJ_DIR"/*.jsonl 2>/dev/null | wc -l)
        if [ "$TRAJ_COUNT" -ge 5 ]; then
            analyze_trajectories "$TRAJ_DIR" "COLLECT"
        fi

        # Analyze eval trajectories
        EVAL_COUNT=$(ls "$EVAL_DIR"/*.jsonl 2>/dev/null | wc -l)
        if [ "$EVAL_COUNT" -ge 5 ]; then
            analyze_trajectories "$EVAL_DIR" "EVAL"
        fi
    } >> "$LOG" 2>&1

    echo "Round $CURRENT_ROUND logged at $(date '+%H:%M:%S')"

    sleep 15
done
