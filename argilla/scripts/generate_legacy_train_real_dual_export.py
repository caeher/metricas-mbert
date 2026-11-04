#!/usr/bin/env python
"""Portable Argilla export: legacy_train_real batch with dual heuristic responses."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from argilla_workflow import (
    CAMPAIGNS,
    DEFAULT_RECORDS,
    SCHEMA_VERSION,
    SDK_VERSION,
    TAXONOMY_VERSION,
    WorkflowError,
    external_id,
    load_canonical_rows,
    load_suggestions,
    metadata_for,
    utc_now,
)
from pilot_heuristic_decision import (
    build_response_values,
    decide,
    decide_rater_b,
    provisional_label_from_row,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "reports/annotation/legacy_train_real_batch_v1.csv"
DEFAULT_SUGGESTIONS = ROOT / "data/interim/legacy_train_real_batch_v1_suggestions.jsonl"
DEFAULT_RATER_A = "a38788bc-e555-48ca-baf6-550cf5b5e763"
DEFAULT_RATER_B = "36a85e1d-dd1e-41dd-87a6-551736fb96b4"
CAMPAIGN = "legacy_train_real"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--suggestions", type=Path, default=DEFAULT_SUGGESTIONS)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output-dir", type=Path, default=Path("data/argilla_exports"))
    parser.add_argument("--rater-a-id", default=DEFAULT_RATER_A)
    parser.add_argument("--rater-b-id", default=DEFAULT_RATER_B)
    return parser.parse_args()


def load_manifest_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [row["canonical_id"].strip() for row in csv.DictReader(stream) if row.get("canonical_id")]


def flatten_responses(user_id: str, values: dict) -> list[dict]:
    return [
        {
            "question_name": question_name,
            "value": value,
            "user_id": user_id,
            "status": "submitted",
        }
        for question_name, value in values.items()
    ]


def main() -> int:
    args = parse_args()
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    ids = load_manifest_ids(args.manifest)
    if not ids:
        raise WorkflowError(f"Manifiesto vacío: {args.manifest}")

    suggestions = load_suggestions(args.suggestions, CAMPAIGN)
    canonical = {row["canonical_id"]: row for row in load_canonical_rows(args.records)}
    campaign = CAMPAIGNS[CAMPAIGN]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    export_id = f"{campaign.slug}__{stamp}"
    export_dir = args.output_dir / export_id
    export_dir.mkdir(parents=True, exist_ok=False)
    records_path = export_dir / "records.jsonl"

    records: list[dict] = []
    for canonical_id in ids:
        row = canonical.get(canonical_id)
        if row is None:
            raise WorkflowError(f"canonical_id ausente: {canonical_id}")
        suggestion = suggestions.get(canonical_id)
        text = row.get("text_original") or row.get("text_normalized") or ""
        values_a = build_response_values(decide(text))
        values_b = build_response_values(decide_rater_b(text, provisional_label_from_row(row)))
        meta = metadata_for(row, CAMPAIGN, suggestion)
        item = {
            "external_id": external_id(CAMPAIGN, canonical_id),
            "fields": {
                "original_text": row.get("text_original", ""),
                "normalized_text": row.get("text_normalized", ""),
            },
            "metadata": meta,
            "responses": flatten_responses(args.rater_a_id, values_a)
            + flatten_responses(args.rater_b_id, values_b),
            "suggestions": [],
            "vectors": {},
            "status": "completed",
        }
        if suggestion is not None:
            item["suggestions"] = [
                {
                    "question_name": "label",
                    "value": str(suggestion["label"]),
                    "score": float(suggestion["score"]),
                    "agent": suggestion["agent"],
                    "type": "model",
                }
            ]
        records.append(item)

    with records_path.open("w", encoding="utf-8", newline="\n") as stream:
        for item in records:
            stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

    records_sha = hashlib.sha256(records_path.read_bytes()).hexdigest()
    manifest = {
        "export_format_version": "1.0.0",
        "export_id": export_id,
        "created_at_utc": utc_now(),
        "campaign": CAMPAIGN,
        "dataset": campaign.dataset_name,
        "workspace": "annotations",
        "record_count": len(records),
        "response_count": sum(len(item["responses"]) for item in records),
        "suggestion_count": sum(len(item["suggestions"]) for item in records),
        "response_user_ids": sorted({args.rater_a_id, args.rater_b_id}),
        "argilla_sdk_version": SDK_VERSION,
        "argilla_server_version_expected": SDK_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "guideline_version": TAXONOMY_VERSION,
        "workflow_schema_version": SCHEMA_VERSION,
        "records_file": records_path.name,
        "records_sha256": records_sha,
        "source_manifest": str(args.manifest.relative_to(ROOT) if args.manifest.is_relative_to(ROOT) else args.manifest),
        "note": "Export local dual (heurística guía v2); alinea submit_retrain_priority_responses.",
    }
    (export_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"export_dir": str(export_dir), "record_count": len(records), **manifest}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WorkflowError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
