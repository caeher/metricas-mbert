#!/usr/bin/env python
"""Stratified Argilla batch from legacy_train_real campaign pool (ISSUE-06)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "argilla" / "scripts"))

from argilla_workflow import DEFAULT_RECORDS, WorkflowError, campaign_rows, load_canonical_rows

DEFAULT_MANIFEST = ROOT / "reports" / "annotation" / "legacy_train_real_batch_v1.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--total", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--facebook-share", type=float, default=0.5, help="Target fraction Facebook rows")
    parser.add_argument("--class-23-share", type=float, default=0.35, help="Target fraction labels 2 or 3")
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def provisional_label(row: dict[str, str]) -> str:
    raw = row.get("label_provisional_ids_json", "[]")
    try:
        labels = json.loads(raw)
    except json.JSONDecodeError:
        labels = []
    if not labels:
        return ""
    value = str(labels[0]).strip()
    return value if value in {"0", "1", "2", "3"} else ""


def main() -> int:
    args = parse_args()
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    records = load_canonical_rows(args.records)
    pool = campaign_rows(records, "legacy_train_real")
    if not pool:
        raise WorkflowError("Pool legacy_train_real vacío en canonical_records.")

    rng = random.Random(args.seed)
    facebook = [r for r in pool if r.get("platform_canonical") == "Facebook"]
    other = [r for r in pool if r.get("platform_canonical") != "Facebook"]
    n_fb = min(len(facebook), max(1, int(args.total * args.facebook_share)))
    n_other = min(len(other), args.total - n_fb)

    chosen: list[dict[str, str]] = []
    if facebook:
        chosen.extend(rng.sample(facebook, n_fb) if len(facebook) >= n_fb else facebook)
    if other and n_other > 0:
        chosen.extend(rng.sample(other, n_other) if len(other) >= n_other else other)

    if len(chosen) < args.total:
        already = {r["canonical_id"] for r in chosen}
        rest = [r for r in pool if r["canonical_id"] not in already]
        need = args.total - len(chosen)
        if rest and need > 0:
            chosen.extend(rng.sample(rest, min(need, len(rest))))

    class_23 = [r for r in chosen if provisional_label(r) in {"2", "3"}]
    target_23 = int(args.total * args.class_23_share)
    if len(class_23) < target_23:
        already = {r["canonical_id"] for r in chosen}
        extra_pool = [
            r
            for r in pool
            if r["canonical_id"] not in already and provisional_label(r) in {"2", "3"}
        ]
        need = min(target_23 - len(class_23), len(extra_pool), args.total - len(chosen))
        if need > 0:
            extras = rng.sample(extra_pool, need)
            chosen.extend(extras)
            if len(chosen) > args.total:
                chosen = rng.sample(chosen, args.total)

    manifest_rows = [
        {
            "canonical_id": row["canonical_id"],
            "stratum": "legacy_train_real",
            "selection_reason": "stratified_platform_class",
            "provisional_label": provisional_label(row),
            "platform": row.get("platform_canonical", ""),
        }
        for row in sorted(chosen, key=lambda r: r["canonical_id"])
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["canonical_id", "stratum", "selection_reason", "provisional_label", "platform"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "batch_id": args.output.stem,
        "record_count": len(manifest_rows),
        "seed": args.seed,
        "pool_size": len(pool),
        "canonical_records_sha256": hashlib.sha256(args.records.read_bytes()).hexdigest(),
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(args.output), **summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WorkflowError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
