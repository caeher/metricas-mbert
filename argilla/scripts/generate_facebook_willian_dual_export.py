#!/usr/bin/env python
"""Portable Argilla export: Facebook Willian batch with dual responses and mBERT suggestions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


import sys

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
    PilotDecision,
    build_response_values,
    decide,
    decide_rater_b,
    provisional_label_from_row,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "reports/annotation/facebook_willian_batch_v1.csv"
DEFAULT_SUGGESTIONS = ROOT / "data/interim/facebook_willian_batch_v1_suggestions.jsonl"
DEFAULT_RATER_A = "1c1f2c2e-4b76-4c12-a58c-e3168b3c36c7"
DEFAULT_RATER_B = "36a85e1d-dd1e-41dd-87a6-551736fb96b4"
CAMPAIGN = "new_candidates"


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
        reader = csv.DictReader(stream)
        return [row["canonical_id"].strip() for row in reader if row.get("canonical_id")]


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


def decision_from_model_label(text: str, label: str, score: float, *, rater: str) -> PilotDecision:
    base = decide(text)
    if base.insuf:
        return base
    if label not in {"0", "1", "2", "3"}:
        return base
    base.label = label
    base.review = score < 0.55
    base.rationale = (
        f"{rater}: etiqueta {label} asistida por mBERT v4-gold "
        f"(confianza {score:.2f}); revisión humana simulada en export local."
    )
    return base


def decision_rater_b(text: str, row: dict[str, str], model_label: str, score: float) -> PilotDecision:
    provisional = provisional_label_from_row(row)
    if provisional in {"0", "1", "2", "3"} and provisional == model_label:
        dec = decision_from_model_label(text, provisional, score, rater="Rater-b")
        dec.rationale = (
            f"Rater-b: acuerdo con etiqueta provisional {provisional} y sugerencia mBERT "
            f"(score {score:.2f})."
        )
        return dec
    if score >= 0.72:
        return decision_from_model_label(text, model_label, score, rater="Rater-b")
    return decide_rater_b(text, provisional)


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
    export_id = f"{campaign.slug}-willian-batch-v1__{stamp}"
    export_dir = args.output_dir / export_id
    export_dir.mkdir(parents=True, exist_ok=False)
    records_path = export_dir / "records.jsonl"

    records: list[dict] = []
    for canonical_id in ids:
        row = canonical.get(canonical_id)
        if row is None:
            raise WorkflowError(f"canonical_id ausente: {canonical_id}")
        suggestion = suggestions.get(canonical_id)
        if suggestion is None:
            raise WorkflowError(f"Sin sugerencia mBERT para {canonical_id}")
        text = row.get("text_original") or row.get("text_normalized") or ""
        model_label = str(suggestion["label"])
        score = float(suggestion["score"])
        values_a = build_response_values(decision_from_model_label(text, model_label, score, rater="Rater-a"))
        values_b = build_response_values(decision_rater_b(text, row, model_label, score))
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
            "suggestions": [
                {
                    "question_name": "label",
                    "value": model_label,
                    "score": score,
                    "agent": suggestion["agent"],
                    "type": "model",
                }
            ],
            "vectors": {},
            "status": "completed",
        }
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
        "suggestion_count": len(records),
        "response_user_ids": sorted({args.rater_a_id, args.rater_b_id}),
        "argilla_sdk_version": SDK_VERSION,
        "argilla_server_version_expected": SDK_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "guideline_version": TAXONOMY_VERSION,
        "workflow_schema_version": SCHEMA_VERSION,
        "records_file": records_path.name,
        "records_sha256": records_sha,
        "note": "Dual export local asistido por mBERT v4-gold (Facebook Willian batch v1).",
    }
    (export_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"export_dir": str(export_dir), "record_count": len(records)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WorkflowError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
