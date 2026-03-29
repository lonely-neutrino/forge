#!/bin/bash
# Step 12: Run many replaydiff seeds and analyze first divergences
# Usage:
#   sh 12_batch_replay_analysis.sh "<deck1>" ["<deck2>" [start_seed [count [prefix [seat [baseline [compare [model_dir [workers]]]]]]]]]
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRACE_DIR="$PROJECT_ROOT/rl_data/replays"
PYTHON_TOOL="$PROJECT_ROOT/forge-ai-rl/src/main/python/tools/analyze_replay_batch.py"
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
    echo "Usage: sh 12_batch_replay_analysis.sh \"<deck1>\" [\"<deck2>\" [start_seed [count [prefix [seat [baseline [compare [model_dir [workers]]]]]]]]]"
    exit 1
fi

DECK1="$1"
DECK2="${2:-$1}"
START_SEED="${3:-12345}"
COUNT="${4:-20}"
PREFIX="${5:-batch_replay}"
SEAT="${6:-1}"
BASELINE="${7:-heuristic}"
COMPARE="${8:-rl}"
MODEL_DIR="${9:-$PROJECT_ROOT/rl_data/models}"
WORKERS="${10:-2}"

case "$WORKERS" in
    ''|*[!0-9]*)
        echo "Workers must be a positive integer."
        exit 1
        ;;
esac
if [ "$WORKERS" -lt 1 ]; then
    echo "Workers must be at least 1."
    exit 1
fi

mkdir -p "$TRACE_DIR"
LOG_DIR="$TRACE_DIR/${PREFIX}_logs"
mkdir -p "$LOG_DIR"

echo "Batch replay analysis"
echo "  Deck 1: $DECK1"
echo "  Deck 2: $DECK2"
echo "  Start seed: $START_SEED"
echo "  Count: $COUNT"
echo "  Prefix: $PREFIX"
echo "  Seat: $SEAT"
echo "  Baseline: $BASELINE"
echo "  Compare: $COMPARE"
echo "  Workers: $WORKERS"
echo "  Logs: $LOG_DIR"
echo

ACTIVE_JOBS=""
FAILURES=0
SKIPPED=0

count_active_jobs() {
    if [ -z "$ACTIVE_JOBS" ]; then
        echo 0
        return
    fi
    printf '%s\n' "$ACTIVE_JOBS" | sed '/^$/d' | wc -l | tr -d ' '
}

pop_first_job() {
    FIRST_JOB="$(printf '%s\n' "$ACTIVE_JOBS" | sed '/^$/d' | head -n 1)"
    ACTIVE_JOBS="$(printf '%s\n' "$ACTIVE_JOBS" | sed '/^$/d' | tail -n +2)"
}

replay_exists() {
    replay_id="$1"
    baseline_matches=$(find "$TRACE_DIR" -maxdepth 1 -type f -name "replay_${replay_id}_baseline_*.jsonl" | wc -l | tr -d ' ')
    compare_matches=$(find "$TRACE_DIR" -maxdepth 1 -type f -name "replay_${replay_id}_compare_*.jsonl" | wc -l | tr -d ' ')
    if [ "$baseline_matches" -gt 0 ] && [ "$compare_matches" -gt 0 ]; then
        return 0
    fi
    return 1
}

wait_for_slot() {
    while :; do
        ACTIVE_COUNT="$(count_active_jobs)"
        if [ "$ACTIVE_COUNT" -lt "$WORKERS" ]; then
            return
        fi

        pop_first_job
        first_pid="${FIRST_JOB%%|*}"
        first_meta="${FIRST_JOB#*|}"

        if [ -n "$first_pid" ] && wait "$first_pid"; then
            echo "[done] $first_meta"
        else
            echo "[fail] $first_meta"
            FAILURES=$((FAILURES + 1))
        fi
    done
}

for ((offset=0; offset<COUNT; offset++)); do
    SEED=$((START_SEED + offset))
    REPLAY_ID="${PREFIX}_${SEED}"
    LOG_PATH="$LOG_DIR/${REPLAY_ID}.log"

    if replay_exists "$REPLAY_ID"; then
        echo "[$((offset + 1))/$COUNT] Skipping seed $SEED -> $REPLAY_ID (traces already exist)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    wait_for_slot

    echo "[$((offset + 1))/$COUNT] Launching seed $SEED -> $REPLAY_ID"

    (
        sh "$SCRIPT_DIR/09_replay_diff.sh" \
            "$DECK1" \
            "$DECK2" \
            "$SEED" \
            "$REPLAY_ID" \
            "$BASELINE" \
            "$COMPARE" \
            "$MODEL_DIR"
    ) >"$LOG_PATH" 2>&1 &

    pid=$!
    job_line="${pid}|${REPLAY_ID} (seed ${SEED})"
    if [ -z "$ACTIVE_JOBS" ]; then
        ACTIVE_JOBS="$job_line"
    else
        ACTIVE_JOBS="$ACTIVE_JOBS
$job_line"
    fi
done

while [ "$(count_active_jobs)" -gt 0 ]; do
    pop_first_job
    pid="${FIRST_JOB%%|*}"
    first_meta="${FIRST_JOB#*|}"
    if [ -n "$pid" ] && wait "$pid"; then
        echo "[done] $first_meta"
    else
        echo "[fail] $first_meta"
        FAILURES=$((FAILURES + 1))
    fi
done

CSV_OUT="$TRACE_DIR/replay_analysis_${PREFIX}_seat${SEAT}.csv"

echo
echo "Analyzing replay traces..."
"$PYTHON" "$PYTHON_TOOL" \
    --trace-dir "$TRACE_DIR" \
    --replay-prefix "$PREFIX" \
    --seat "$SEAT" \
    --output "$CSV_OUT"

echo
echo "Analysis complete."
if [ "$SKIPPED" -gt 0 ]; then
    echo "Replay runs skipped (already existed): $SKIPPED"
fi
if [ "$FAILURES" -gt 0 ]; then
    echo "Replay runs failed: $FAILURES"
    echo "Check logs under:"
    echo "  $LOG_DIR"
fi
echo "CSV:"
echo "  $CSV_OUT"
