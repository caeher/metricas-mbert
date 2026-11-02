#!/usr/bin/env python
"""Build a retrain-priority Argilla manifest (P0 discrepancies + P1 Facebook/classes 2/3)."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BATCH = ROOT / "reports" / "annotation" / "legacy_train_real_batch_v1.csv"
DEFAULT_DISCREPANCY = ROOT / "reports" / "annotation" / "legacy_train_real_heuristic_vs_mbert.csv"
DEFAULT_OUTPUT = ROOT / "reports" / "annotation" / "legacy_train_real_retrain_priority.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-manifest", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--discrepancy-csv", type=Path, default=DEFAULT_DISCREPANCY)
    parser.add_argument("--target-total", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_batch_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return {row["canonical_id"].strip(): row for row in reader if row.get("canonical_id")}


def discrepancy_ids(path: Path, batch_ids: set[str]) -> list[str]:
    if not path.is_file():
        return []
    ids: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            cid = (row.get("canonical_id") or "").strip()
            if cid and cid in batch_ids:
                ids.append(cid)
    return sorted(set(ids))


def p1_score(row: dict[str, str]) -> tuple[int, int, str]:
    platform = (row.get("platform") or "").strip()
    label = (row.get("provisional_label") or "").strip()
    fb = 0 if platform == "Facebook" else 1
    weak_class = 0 if label in {"2", "3", "0"} else 1
    return (fb, weak_class, row["canonical_id"])


def main() -> int:
    args = parse_args()
    batch = load_batch_rows(args.batch_manifest)
    if not batch:
        print(f"error: empty batch {args.batch_manifest}", file=sys.stderr)
        return 1

    batch_ids = set(batch.keys())
    p0_ids = discrepancy_ids(args.discrepancy_csv, batch_ids)

    selected: dict[str, dict[str, str]] = {}
    for cid in p0_ids:
        row = dict(batch[cid])
        row["selection_reason"] = "p0_heuristic_vs_mbert_discrepancy"
        row["stratum"] = "retrain_priority_p0"
        selected[cid] = row

    remaining_pool = [batch[cid] for cid in batch_ids if cid not in selected]
    rng = random.Random(args.seed)
    remaining_pool.sort(key=p1_score)
    need = max(0, args.target_total - len(selected))
    if need > 0 and remaining_pool:
        take = min(need, len(remaining_pool))
        for row in remaining_pool[:take]:
            cid = row["canonical_id"]
            enriched = dict(row)
            enriched["selection_reason"] = "p1_facebook_or_weak_class"
            enriched["stratum"] = "retrain_priority_p1"
            selected[cid] = enriched

    ordered = sorted(selected.values(), key=lambda r: r["canonical_id"])
    fieldnames = [
        "canonical_id",
        "stratum",
        "selection_reason",
        "provisional_label",
        "platform",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)

    platform_counts = Counter(r.get("platform", "") for r in ordered)
    label_counts = Counter(r.get("provisional_label", "") for r in ordered)
    reason_counts = Counter(r.get("selection_reason", "") for r in ordered)

    summary = {
        "batch_id": args.output.stem,
        "record_count": len(ordered),
        "target_total": args.target_total,
        "seed": args.seed,
        "source_batch": str(args.batch_manifest.relative_to(ROOT)).replace("\\", "/"),
        "source_discrepancy": str(args.discrepancy_csv.relative_to(ROOT)).replace("\\", "/"),
        "p0_discrepancy_count": len(p0_ids),
        "selection_reason_counts": dict(sorted(reason_counts.items())),
        "platform_counts": dict(sorted(platform_counts.items())),
        "provisional_label_counts": dict(sorted(label_counts.items())),
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(args.output), "summary": str(summary_path), **summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
