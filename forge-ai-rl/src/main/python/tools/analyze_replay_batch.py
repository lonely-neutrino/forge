#!/usr/bin/env python3
"""
Analyze paired replaydiff traces and emit a CSV of first-divergence details.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def load_runs(trace_dir, replay_prefix, seat):
    grouped = {}
    for path in sorted(Path(trace_dir).glob("*.jsonl")):
        try:
            with path.open("r", encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
        except Exception:
            continue
        if not rows:
            continue
        header = rows[0]
        if header.get("recordType") != "header":
            continue
        replay_id = header.get("replayId")
        run_label = header.get("runLabel")
        seat_index = int(header.get("seatIndex", -1))
        if not replay_id or not replay_id.startswith(replay_prefix):
            continue
        if seat_index != seat:
            continue
        grouped.setdefault(replay_id, {})[run_label] = {
            "header": header,
            "steps": [r for r in rows[1:] if r.get("recordType") == "step"],
            "result": next((r for r in rows[1:] if r.get("recordType") == "result"), None),
            "path": str(path),
        }
    return grouped


def first_divergence(left_steps, right_steps):
    limit = min(len(left_steps), len(right_steps))
    for idx in range(limit):
        if left_steps[idx].get("decisionHash") != right_steps[idx].get("decisionHash"):
            return idx
    if len(left_steps) != len(right_steps):
        return limit
    return None


def snapshot_lives(record):
    snapshot = None
    if record:
        snapshot = record.get("snapshot")
    if snapshot:
        return snapshot.get("myLife"), snapshot.get("oppLife")
    return None, None


def outcome_label(result_record):
    if result_record is None:
        return "unknown"
    return "win" if result_record.get("won") else "loss"


def classify_divergence(left_step, right_step):
    if left_step is None or right_step is None:
        return "length_mismatch"

    left_type = left_step.get("decisionType")
    right_type = right_step.get("decisionType")
    if left_type != right_type:
        return "decision_type_mismatch"

    mapping = {
        "MULLIGAN": "mulligan_diff",
        "PRIORITY_ACTION": "priority_diff",
        "TARGET_SELECTION": "targeting_diff",
        "DECLARE_ATTACKERS": "attack_diff",
        "DECLARE_BLOCKERS": "block_diff",
        "CARD_SELECTION": "card_selection_diff",
        "BINARY_CHOICE": "binary_choice_diff",
    }
    return mapping.get(left_type, f"{str(left_type).lower()}_diff")


def selected_labels(step):
    if not step:
        return ""
    labels = step.get("selectedLabels") or []
    if labels:
        return " | ".join(str(x) for x in labels)
    return " | ".join(str(x) for x in (step.get("selectedIndices") or []))


def write_csv(rows, output_path):
    fieldnames = [
        "replay_id",
        "seed",
        "seat",
        "baseline_policy",
        "compare_policy",
        "baseline_outcome",
        "compare_outcome",
        "baseline_steps",
        "compare_steps",
        "first_divergence_step",
        "first_divergence_turn",
        "first_divergence_phase",
        "divergence_class",
        "baseline_decision_type",
        "compare_decision_type",
        "baseline_context",
        "compare_context",
        "baseline_selected",
        "compare_selected",
        "baseline_state_hash",
        "compare_state_hash",
        "baseline_decision_hash",
        "compare_decision_hash",
        "baseline_my_life",
        "baseline_opp_life",
        "compare_my_life",
        "compare_opp_life",
        "baseline_path",
        "compare_path",
    ]
    with Path(output_path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_rows(grouped, seat):
    rows = []
    for replay_id in sorted(grouped):
        pair = grouped[replay_id]
        baseline = pair.get("baseline")
        compare = pair.get("compare")
        if not baseline or not compare:
            continue

        baseline_steps = baseline["steps"]
        compare_steps = compare["steps"]
        idx = first_divergence(baseline_steps, compare_steps)
        left_step = baseline_steps[idx] if idx is not None and idx < len(baseline_steps) else None
        right_step = compare_steps[idx] if idx is not None and idx < len(compare_steps) else None
        divergence_class = "no_divergence" if idx is None else classify_divergence(left_step, right_step)

        base_life, base_opp_life = snapshot_lives(left_step or baseline.get("result"))
        cmp_life, cmp_opp_life = snapshot_lives(right_step or compare.get("result"))

        header = baseline["header"]
        rows.append({
            "replay_id": replay_id,
            "seed": header.get("seed"),
            "seat": seat,
            "baseline_policy": baseline["header"].get("policyMode"),
            "compare_policy": compare["header"].get("policyMode"),
            "baseline_outcome": outcome_label(baseline.get("result")),
            "compare_outcome": outcome_label(compare.get("result")),
            "baseline_steps": len(baseline_steps),
            "compare_steps": len(compare_steps),
            "first_divergence_step": "" if idx is None else idx + 1,
            "first_divergence_turn": "" if right_step is None else right_step.get("turn"),
            "first_divergence_phase": "" if right_step is None else right_step.get("phase"),
            "divergence_class": divergence_class,
            "baseline_decision_type": "" if left_step is None else left_step.get("decisionType"),
            "compare_decision_type": "" if right_step is None else right_step.get("decisionType"),
            "baseline_context": "" if left_step is None else left_step.get("contextInfo"),
            "compare_context": "" if right_step is None else right_step.get("contextInfo"),
            "baseline_selected": selected_labels(left_step),
            "compare_selected": selected_labels(right_step),
            "baseline_state_hash": "" if left_step is None else left_step.get("stateHash"),
            "compare_state_hash": "" if right_step is None else right_step.get("stateHash"),
            "baseline_decision_hash": "" if left_step is None else left_step.get("decisionHash"),
            "compare_decision_hash": "" if right_step is None else right_step.get("decisionHash"),
            "baseline_my_life": "" if base_life is None else base_life,
            "baseline_opp_life": "" if base_opp_life is None else base_opp_life,
            "compare_my_life": "" if cmp_life is None else cmp_life,
            "compare_opp_life": "" if cmp_opp_life is None else cmp_opp_life,
            "baseline_path": baseline["path"],
            "compare_path": compare["path"],
        })
    return rows


def print_summary(rows):
    print(f"Paired runs analyzed: {len(rows)}")
    if not rows:
        return

    compare_outcomes = Counter(row["compare_outcome"] for row in rows)
    print("Compare outcomes:")
    for key in ("win", "loss", "unknown"):
        if compare_outcomes.get(key):
            print(f"  {key}: {compare_outcomes[key]}")

    loss_rows = [row for row in rows if row["compare_outcome"] == "loss"]
    if loss_rows:
        counts = Counter(row["divergence_class"] for row in loss_rows)
        print("First-divergence classes in compare losses:")
        for label, count in counts.most_common():
            print(f"  {label}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Analyze replaydiff traces in batch")
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--replay-prefix", required=True)
    parser.add_argument("--seat", type=int, default=1)
    parser.add_argument("--output")
    args = parser.parse_args()

    output = args.output or str(
        Path(args.trace_dir) / f"replay_analysis_{args.replay_prefix}_seat{args.seat}.csv"
    )

    grouped = load_runs(args.trace_dir, args.replay_prefix, args.seat)
    rows = build_rows(grouped, args.seat)
    write_csv(rows, output)
    print_summary(rows)
    print(f"CSV written to: {output}")


if __name__ == "__main__":
    main()
