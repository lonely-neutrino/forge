#!/bin/bash
# Step 9: ExIt data collection via MCTS rollouts
# Usage: 09_exit_collect.sh [games] [rollouts]
# Each decision point is evaluated by rolling out candidate actions
# to completion and picking the one with highest win rate.
# Much slower than heuristic collection (~25s/game vs ~0.5s/game).
set -e

GAMES=${1:-500}
ROLLOUTS=${2:-30}
OUTPUT=/home/maustin/forge/rl_data/exit_trajectories

echo "ExIt MCTS Collection: $GAMES games, $ROLLOUTS rollouts/candidate"
echo "Output: $OUTPUT"
echo ""

mkdir -p "$OUTPUT"

cd /home/maustin/forge/forge-gui-desktop
java -Xmx12g \
    --add-opens java.base/java.lang=ALL-UNNAMED \
    --add-opens java.base/java.util=ALL-UNNAMED \
    --add-opens java.base/java.text=ALL-UNNAMED \
    --add-opens java.base/java.lang.reflect=ALL-UNNAMED \
    --add-opens java.desktop/javax.imageio.spi=ALL-UNNAMED \
    -jar target/forge-gui-desktop-2.0.12-SNAPSHOT-jar-with-dependencies.jar \
    rltrain mcts-collect \
    -d "Green Stompy.dck" -d "White Weenie.dck" \
    -d "Blue Tempo.dck" -d "Red Aggro.dck" \
    -n "$GAMES" -t 4 -r "$ROLLOUTS" \
    -c 1800 \
    -o "$OUTPUT"
