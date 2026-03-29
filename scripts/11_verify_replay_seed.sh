#!/bin/bash
# Step 11: Verify deterministic replay by running the same seed twice
# Usage:
#   sh 11_verify_replay_seed.sh "<deck1>" ["<deck2>" [seed [prefix]]]
# Examples:
#   sh 11_verify_replay_seed.sh "C:\Users\tiger\AppData\Roaming\Forge\decks\Red Aggro.dck"
#   sh 11_verify_replay_seed.sh "C:\Decks\A.dck" "C:\Decks\B.dck" 12345 mycheck
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRACE_DIR="$PROJECT_ROOT/rl_data/replays"
PYTHON_DIR="$PROJECT_ROOT/forge-ai-rl/src/main/python"
VENV_PY="$PROJECT_ROOT/forge-ai-rl/venv/bin/python3"
VENV_PY_WIN="$PROJECT_ROOT/forge-ai-rl/venv/Scripts/python.exe"

if command -v cygpath >/dev/null 2>&1; then
    USERPROFILE_UNIX="$(cygpath "$USERPROFILE" 2>/dev/null || printf '%s' "$USERPROFILE")"
    CONDA_PREFIX_UNIX="$(cygpath "$CONDA_PREFIX" 2>/dev/null || printf '%s' "$CONDA_PREFIX")"
else
    USERPROFILE_UNIX="$USERPROFILE"
    CONDA_PREFIX_UNIX="$CONDA_PREFIX"
fi

CONDA_PY="$CONDA_PREFIX_UNIX/python"
CONDA_PY_WIN="$CONDA_PREFIX_UNIX/python.exe"
FORGE_RL_CONDA_WIN="$USERPROFILE_UNIX/anaconda3/envs/forge_rl/python.exe"

if [ -x "$VENV_PY" ]; then
    PYTHON="$VENV_PY"
elif [ -f "$VENV_PY_WIN" ]; then
    PYTHON="$VENV_PY_WIN"
elif [ -n "$CONDA_PREFIX" ] && [ -x "$CONDA_PY" ]; then
    PYTHON="$CONDA_PY"
elif [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PY_WIN" ]; then
    PYTHON="$CONDA_PY_WIN"
elif [ -n "$USERPROFILE" ] && [ -f "$FORGE_RL_CONDA_WIN" ]; then
    PYTHON="$FORGE_RL_CONDA_WIN"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
else
    echo "No Python interpreter found."
    exit 1
fi

if [ -z "$1" ]; then
    echo "Usage: sh 11_verify_replay_seed.sh \"<deck1>\" [\"<deck2>\" [seed [prefix]]]"
    exit 1
fi

DECK1="$1"
DECK2="${2:-$1}"
SEED="${3:-12345}"
PREFIX="${4:-verify_seed_${SEED}}"
RUN_A="${PREFIX}_a"
RUN_B="${PREFIX}_b"

echo "Running first replay..."
sh "$SCRIPT_DIR/09_replay_diff.sh" "$DECK1" "$DECK2" "$SEED" "$RUN_A"

echo
echo "Running second replay..."
sh "$SCRIPT_DIR/09_replay_diff.sh" "$DECK1" "$DECK2" "$SEED" "$RUN_B"

echo
echo "Comparing trace hashes..."

"$PYTHON" - "$TRACE_DIR" "$RUN_A" "$RUN_B" <<'PY'
import json
import sys
from pathlib import Path

trace_dir = Path(sys.argv[1])
run_a = sys.argv[2]
run_b = sys.argv[3]

def find_trace(run_id, label):
    matches = sorted(trace_dir.glob(f"replay_{run_id}_{label}_*.jsonl"))
    if not matches:
        raise SystemExit(f"Missing trace for {run_id} {label} in {trace_dir}")
    return matches[0]

def load_hashes(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            obj = json.loads(line)
            if idx == 0:
                continue
            rows.append((
                obj.get("stepIndex"),
                obj.get("stateHash"),
                obj.get("decisionHash"),
            ))
    return rows

def compare(label):
    a_path = find_trace(run_a, label)
    b_path = find_trace(run_b, label)
    a_rows = load_hashes(a_path)
    b_rows = load_hashes(b_path)
    if a_rows == b_rows:
        print(f"[OK] {label}: identical traces ({len(a_rows)} steps)")
        return True

    print(f"[FAIL] {label}: traces differ")
    max_len = min(len(a_rows), len(b_rows))
    for i in range(max_len):
        if a_rows[i] != b_rows[i]:
            print(f"  First mismatch at step {i + 1}")
            print(f"  A: {a_rows[i]}")
            print(f"  B: {b_rows[i]}")
            break
    else:
        print(f"  Different lengths: A={len(a_rows)} B={len(b_rows)}")
    return False

baseline_ok = compare("baseline")
compare_ok = compare("compare")

if baseline_ok and compare_ok:
    print()
    print("Determinism check passed: same seed produced identical baseline and compare traces.")
    sys.exit(0)

print()
print("Determinism check failed.")
sys.exit(1)
PY

echo
echo "Open one run in the viewer with:"
echo "  sh scripts/10_view_replay_diff.sh \"$RUN_A\""
