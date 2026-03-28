#!/usr/bin/env python3
"""
Analyze policy shift between heuristic imitation data and RL-reached states.

This script treats heuristic counterfactual labels as analysis-only metadata.
They are never fed into model inference or training code.
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.backend import resolve_backend
from model.mtg_model import MTGModel
from serving.model_server import ModelServer


SUPPORTED_DTYPES = {
    "DECLARE_ATTACKERS",
    "DECLARE_BLOCKERS",
    "PRIORITY_ACTION",
    "TARGET_SELECTION",
    "CARD_SELECTION",
    "MULLIGAN",
    "BINARY_CHOICE",
}


def parse_args():
    p = argparse.ArgumentParser(description="Analyze heuristic counterfactual agreement on RL trajectories.")
    p.add_argument("--heuristic-dir", type=Path, required=True,
                   help="Directory containing heuristic imitation trajectories.")
    p.add_argument("--rl-dir", type=Path, required=True,
                   help="Directory containing RL-vs-heuristic trajectories.")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Optional model checkpoint for current-model comparisons.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Directory for JSON/CSV outputs.")
    p.add_argument("--decision-types", nargs="*", default=None,
                   help="Optional subset of decision types to include.")
    p.add_argument("--max-files", type=int, default=None,
                   help="Optional max files per directory.")
    p.add_argument("--compute-embedding-distance", action="store_true",
                   help="Compute heuristic-state nearest-neighbor distance in encoder space.")
    p.add_argument("--won-only", action="store_true",
                   help="Only include winning games.")
    p.add_argument("--lost-only", action="store_true",
                   help="Only include losing games.")
    return p.parse_args()


def _normalize_selected(dt, selected):
    selected = selected or []
    if dt in ("DECLARE_ATTACKERS", "DECLARE_BLOCKERS"):
        return tuple(sorted(int(x) for x in selected))
    return tuple(int(x) for x in selected)


def _bucket_turn(turn_index):
    start = (int(turn_index) // 5) * 5
    return f"{start:02d}-{start + 4:02d}"


def _bucket_decision_depth(decision_depth):
    start = (int(decision_depth) // 10) * 10
    return f"{start:03d}-{start + 9:03d}"


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def _load_records(data_dir, max_files=None, decision_types=None,
                  won_only=False, lost_only=False):
    decision_types = set(decision_types) if decision_types else SUPPORTED_DTYPES
    files = sorted(Path(data_dir).glob("traj_*.jsonl"))
    if max_files:
        files = files[:max_files]

    records = []
    for filepath in files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) < 2:
                continue
            header = json.loads(lines[0])
            won = bool(header.get("won", False))
            if won_only and not won:
                continue
            if lost_only and won:
                continue

            decision_depth = 0
            source_depths = defaultdict(int)
            for line in lines[1:]:
                rec = json.loads(line)
                dt = rec.get("decisionType", "")
                if dt not in decision_types:
                    continue
                source = rec.get("source", "heuristic")
                records.append({
                    "filepath": str(filepath),
                    "game_id": header.get("gameId"),
                    "won": won,
                    "turnIndex": int(rec.get("turnIndex", 0)),
                    "decisionDepth": decision_depth,
                    "sourceDecisionDepth": source_depths[source],
                    "decisionType": dt,
                    "contextInfo": rec.get("contextInfo", ""),
                    "source": source,
                    "recorded": _normalize_selected(dt, rec.get("selectedIndices", [])),
                    "modelSelected": _normalize_selected(
                        dt, rec.get("modelSelectedIndices", rec.get("selectedIndices", []))),
                    "heuristic": _normalize_selected(
                        dt, rec.get("counterfactualHeuristicSelectedIndices", [])),
                    "heuristic_available": bool(rec.get("counterfactualHeuristicAvailable", False)),
                    "candidateFeatures": rec.get("candidateFeatures", []),
                    "globalFeatures": rec.get("globalFeatures", []),
                    "gameStateFlat": rec.get("gameStateFlat", []),
                    "spellFeatures": rec.get("spellFeatures"),
                    "actionProbabilities": rec.get("actionProbabilities", []),
                })
                decision_depth += 1
                source_depths[source] += 1
        except Exception as exc:
            print(f"warning: failed to load {filepath}: {exc}", flush=True)
    return records


class Predictor:
    def __init__(self, checkpoint, device):
        backend = resolve_backend(str(device))
        self.device = backend.torch_device
        self.model = MTGModel.load(str(checkpoint), device=self.device)
        self.model.eval()
        self.server = ModelServer(self.model, device=self.device, use_argmax=True)
        ModelServer._debug_count = 999

    def predict(self, rec):
        request = {
            "decisionType": rec["decisionType"],
            "globalFeatures": rec["globalFeatures"],
            "gameStateFlat": rec["gameStateFlat"],
            "candidateFeatures": rec["candidateFeatures"],
        }
        if rec.get("spellFeatures") is not None:
            request["spellFeatures"] = rec["spellFeatures"]

        # Analysis-only fields are deliberately omitted from the synthetic
        # request to mirror the live model path and prevent data leakage.
        result = self.server._process_request_impl(request)
        probs = result.get("actionProbabilities", []) or []
        selected = _normalize_selected(rec["decisionType"], result.get("selectedIndices", []))
        return selected, probs, _safe_float(result.get("valueEstimate", 0.0))


def _label_confidence(dt, probs, label, candidate_count):
    if not probs:
        return None
    if dt in ("PRIORITY_ACTION", "TARGET_SELECTION", "CARD_SELECTION",
              "MULLIGAN", "BINARY_CHOICE"):
        if not label:
            return None
        idx = label[0]
        if 0 <= idx < len(probs):
            return float(probs[idx])
        return None

    # Multi-select confidence: average per-candidate correctness confidence.
    total = 0.0
    count = 0
    selected = set(label)
    for i in range(min(candidate_count, len(probs))):
        p = float(probs[i])
        total += p if i in selected else (1.0 - p)
        count += 1
    return (total / count) if count else None


def _summarize_rows(rows):
    if not rows:
        return {"count": 0}

    summary = {
        "count": len(rows),
        "by_decision_type": {},
        "won": sum(1 for r in rows if r["won"]),
        "lost": sum(1 for r in rows if not r["won"]),
    }

    by_dt = defaultdict(list)
    for row in rows:
        by_dt[row["decisionType"]].append(row)

    for dt, items in sorted(by_dt.items()):
        summary["by_decision_type"][dt] = {
            "count": len(items),
            "won": sum(1 for r in items if r["won"]),
            "lost": sum(1 for r in items if not r["won"]),
        }
        for key in ("rl_vs_heuristic_match", "model_vs_label_match"):
            vals = [1.0 if r.get(key) else 0.0 for r in items if key in r]
            if vals:
                summary["by_decision_type"][dt][key] = sum(vals) / len(vals)

    return summary


def _compute_first_disagreement(rows):
    by_game = defaultdict(list)
    for row in rows:
        if "rl_vs_heuristic_match" not in row:
            continue
        by_game[row["filepath"]].append(row)

    result = Counter()
    for items in by_game.values():
        items.sort(key=lambda r: r["turnIndex"])
        first = None
        won = items[0]["won"] if items else False
        for row in items:
            if not row["rl_vs_heuristic_match"]:
                first = _bucket_turn(row["turnIndex"])
                break
        if first is not None:
            result[(first, "won" if won else "lost")] += 1
    return {
        f"{bucket}:{outcome}": count
        for (bucket, outcome), count in sorted(result.items())
    }


def _summarize_match_by_bucket(rows, match_key, bucket_fn, depth_key="turnIndex"):
    buckets = defaultdict(list)
    for row in rows:
        if match_key not in row:
            continue
        match = row.get(match_key)
        if match is None:
            continue
        buckets[bucket_fn(row.get(depth_key, 0))].append(1.0 if match else 0.0)

    summary = {}
    for bucket, values in sorted(buckets.items()):
        count = len(values)
        match_rate = sum(values) / count if count else None
        summary[bucket] = {
            "count": count,
            "match_rate": match_rate,
            "mismatch_rate": (1.0 - match_rate) if match_rate is not None else None,
        }
    return summary


def _depth_report(heldout_rows, rl_rows):
    rl_cf_rows = [r for r in rl_rows if r.get("heuristic_available")]
    report = {
        "by_turn_bucket": {
            "training_model_vs_heuristic": _summarize_match_by_bucket(
                heldout_rows, "model_vs_label_match", _bucket_turn),
            "rl_policy_rl_vs_heuristic": _summarize_match_by_bucket(
                rl_cf_rows, "rl_vs_heuristic_match", _bucket_turn),
            "rl_policy_model_vs_heuristic": _summarize_match_by_bucket(
                rl_cf_rows, "model_vs_label_match", _bucket_turn),
        },
        "by_decision_depth": {
            "training_model_vs_heuristic": _summarize_match_by_bucket(
                heldout_rows, "model_vs_label_match", _bucket_decision_depth, "sourceDecisionDepth"),
            "rl_policy_rl_vs_heuristic": _summarize_match_by_bucket(
                rl_cf_rows, "rl_vs_heuristic_match", _bucket_decision_depth, "sourceDecisionDepth"),
            "rl_policy_model_vs_heuristic": _summarize_match_by_bucket(
                rl_cf_rows, "model_vs_label_match", _bucket_decision_depth, "sourceDecisionDepth"),
        },
    }
    return report


def _write_depth_csv(output_path, depth_report):
    fieldnames = ["bucket_type", "bucket", "series", "count", "match_rate", "mismatch_rate"]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for bucket_type, series_map in depth_report.items():
            for series_name, bucket_map in series_map.items():
                for bucket, stats in bucket_map.items():
                    writer.writerow({
                        "bucket_type": bucket_type,
                        "bucket": bucket,
                        "series": series_name,
                        "count": stats.get("count"),
                        "match_rate": stats.get("match_rate"),
                        "mismatch_rate": stats.get("mismatch_rate"),
                    })


def _print_depth_table(depth_report, bucket_type):
    series_map = depth_report.get(bucket_type, {})
    all_buckets = sorted({
        bucket
        for bucket_map in series_map.values()
        for bucket in bucket_map.keys()
    })
    if not all_buckets:
        print(f"\n=== Depth Report: {bucket_type} ===", flush=True)
        print("No depth data available.", flush=True)
        return

    print(f"\n=== Depth Report: {bucket_type} ===", flush=True)
    print("bucket | train mismatch | rl-policy mismatch | model-on-rl mismatch | counts(train/rl/model)",
          flush=True)
    for bucket in all_buckets:
        train = series_map.get("training_model_vs_heuristic", {}).get(bucket, {})
        rlpol = series_map.get("rl_policy_rl_vs_heuristic", {}).get(bucket, {})
        model_rl = series_map.get("rl_policy_model_vs_heuristic", {}).get(bucket, {})

        def fmt(stats):
            rate = stats.get("mismatch_rate")
            return f"{rate:.1%}" if rate is not None else "n/a"

        train_count = train.get("count", 0)
        rl_count = rlpol.get("count", 0)
        model_count = model_rl.get("count", 0)
        print(
            f"{bucket} | {fmt(train):>14} | {fmt(rlpol):>18} | {fmt(model_rl):>20} | "
            f"{train_count:>5}/{rl_count:>5}/{model_count:>5}",
            flush=True
        )


def _embedding_distance(predictor, heuristic_records, rl_rows, max_refs=5000):
    if predictor is None or not heuristic_records or not rl_rows:
        return

    refs = heuristic_records[:max_refs]
    ref_vecs = []
    for rec in refs:
        request = {
            "decisionType": rec["decisionType"],
            "globalFeatures": rec["globalFeatures"],
            "gameStateFlat": rec["gameStateFlat"],
            "candidateFeatures": rec["candidateFeatures"],
        }
        state = predictor.server._encode_game_state(request)
        ref_vecs.append(state.detach().cpu().numpy().reshape(-1))
    ref_mat = np.stack(ref_vecs, axis=0)

    for row in rl_rows:
        request = {
            "decisionType": row["decisionType"],
            "globalFeatures": row["globalFeatures"],
            "gameStateFlat": row["gameStateFlat"],
            "candidateFeatures": row["candidateFeatures"],
        }
        state = predictor.server._encode_game_state(request)
        vec = state.detach().cpu().numpy().reshape(-1)
        dists = np.linalg.norm(ref_mat - vec, axis=1)
        row["heuristic_embedding_nn_distance"] = float(np.min(dists))


def main():
    args = parse_args()
    if args.won_only and args.lost_only:
        raise SystemExit("Choose at most one of --won-only and --lost-only")

    decision_types = args.decision_types or sorted(SUPPORTED_DTYPES)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictor = None
    if args.checkpoint is not None:
        predictor = Predictor(args.checkpoint, "cuda" if torch.cuda.is_available() else "cpu")

    heuristic_records = _load_records(
        args.heuristic_dir,
        max_files=args.max_files,
        decision_types=decision_types,
        won_only=args.won_only,
        lost_only=args.lost_only,
    )
    rl_records = _load_records(
        args.rl_dir,
        max_files=args.max_files,
        decision_types=decision_types,
        won_only=args.won_only,
        lost_only=args.lost_only,
    )

    heldout_rows = []
    for rec in heuristic_records:
        row = dict(rec)
        if predictor is not None:
            pred, probs, value = predictor.predict(rec)
            row["modelPrediction"] = pred
            row["model_vs_label_match"] = (pred == rec["recorded"])
            row["model_value"] = value
            conf = _label_confidence(
                rec["decisionType"], probs, rec["recorded"], len(rec["candidateFeatures"]))
            if conf is not None:
                row["label_confidence"] = conf
        heldout_rows.append(row)

    rl_rows = []
    for rec in rl_records:
        row = dict(rec)
        if row["heuristic_available"]:
            row["rl_vs_heuristic_match"] = (row["recorded"] == row["heuristic"])
        if predictor is not None:
            pred, probs, value = predictor.predict(rec)
            row["modelPrediction"] = pred
            row["model_value"] = value
            label = row["heuristic"] if row["heuristic_available"] else row["recorded"]
            row["model_vs_label_match"] = (pred == label)
            conf = _label_confidence(
                rec["decisionType"], probs, label, len(rec["candidateFeatures"]))
            if conf is not None:
                row["label_confidence"] = conf
        rl_rows.append(row)

    if args.compute_embedding_distance and predictor is not None:
        _embedding_distance(predictor, heuristic_records, rl_rows)

    depth_report = _depth_report(heldout_rows, rl_rows)

    summary = {
        "decision_types": decision_types,
        "heuristic_heldout": _summarize_rows(heldout_rows),
        "rl_reached": _summarize_rows(rl_rows),
        "rl_with_counterfactual": _summarize_rows(
            [r for r in rl_rows if r.get("heuristic_available")]),
        "first_rl_vs_heuristic_disagreement": _compute_first_disagreement(
            [r for r in rl_rows if r.get("heuristic_available")]),
        "depth_report": depth_report,
    }

    heldout_acc = summary["heuristic_heldout"].get("by_decision_type", {})
    rl_cf_acc = summary["rl_with_counterfactual"].get("by_decision_type", {})
    absolute_drops = {}
    for dt in sorted(set(heldout_acc) | set(rl_cf_acc)):
        h = heldout_acc.get(dt, {}).get("model_vs_label_match")
        r = rl_cf_acc.get(dt, {}).get("model_vs_label_match")
        if h is not None and r is not None:
            absolute_drops[dt] = h - r
    if absolute_drops:
        summary["model_agreement_drop_by_decision_type"] = absolute_drops

    with open(args.output_dir / "policy_shift_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    _write_depth_csv(args.output_dir / "policy_shift_depth.csv", depth_report)

    csv_rows = heldout_rows + rl_rows
    csv_fields = [
        "filepath", "game_id", "won", "turnIndex", "decisionDepth", "sourceDecisionDepth",
        "decisionType", "contextInfo",
        "source", "recorded", "modelSelected", "heuristic", "heuristic_available",
        "rl_vs_heuristic_match", "modelPrediction", "model_vs_label_match",
        "model_value", "label_confidence", "heuristic_embedding_nn_distance",
    ]
    with open(args.output_dir / "policy_shift_rows.csv", "w",
              encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in csv_rows:
            out = {k: row.get(k) for k in csv_fields}
            for key in ("recorded", "modelSelected", "heuristic", "modelPrediction"):
                if out.get(key) is not None:
                    out[key] = json.dumps(list(out[key]))
            writer.writerow(out)

    print("=== Policy Shift Summary ===", flush=True)
    print(f"Heuristic held-out rows: {summary['heuristic_heldout']['count']}", flush=True)
    print(f"RL rows: {summary['rl_reached']['count']}", flush=True)
    print(f"RL rows with heuristic counterfactual: {summary['rl_with_counterfactual']['count']}", flush=True)
    for dt, drop in summary.get("model_agreement_drop_by_decision_type", {}).items():
        print(f"  {dt}: held-out minus RL agreement drop = {drop:.3f}", flush=True)
    _print_depth_table(depth_report, "by_turn_bucket")
    _print_depth_table(depth_report, "by_decision_depth")


if __name__ == "__main__":
    main()
