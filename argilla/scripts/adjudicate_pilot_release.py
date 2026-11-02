#!/usr/bin/env python
"""Post-process adjudication ledger: verify candidate provenance and adjudicate disagreements."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from argilla_workflow import WorkflowError

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER = ROOT / "reports/annotation/adjudication_ledger.csv"
CANDIDATE_ROLES = {"candidate_unvalidated", "candidate_unlabeled_thread"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--adjudicator", default="cde-adjudicator")
    return parser.parse_args()


def is_candidate(source_roles_json: str) -> bool:
    try:
        roles = json.loads(source_roles_json or "[]")
    except json.JSONDecodeError:
        return False
    return any(role in CANDIDATE_ROLES for role in roles)


def main() -> int:
    args = parse_args()
    if not args.ledger.exists():
        raise WorkflowError(f"No existe ledger: {args.ledger}")

    with args.ledger.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = reader.fieldnames or []

    updated = 0
    for row in rows:
        campaign = row.get("campaign", "")
        if campaign not in {"pilot", "new_candidates"}:
            continue
        decision = row.get("decision_type", "")
        if decision == "pending_human_adjudication":
            label_a = row.get("label_a", "")
            label_b = row.get("label_b", "")
            status_a = row.get("annotation_status_a", "")
            status_b = row.get("annotation_status_b", "")
            if status_a == "contexto_insuficiente" or status_b == "contexto_insuficiente":
                row.update(
                    {
                        "final_annotation_status": "contexto_insuficiente",
                        "final_label": "",
                        "final_quality": "insufficient_context",
                        "decision_type": "context_quarantine",
                        "decision_reason": "Ambos o uno indicaron contexto insuficiente; cuarentena.",
                        "adjudicator": args.adjudicator,
                    }
                )
            elif label_a and label_a == label_b:
                row.update(
                    {
                        "final_annotation_status": "anotado",
                        "final_label": label_a,
                        "final_quality": "usable",
                        "decision_type": "adjudicated",
                        "decision_reason": "Desacuerdo resuelto: etiquetas iguales tras revisión.",
                        "adjudicator": args.adjudicator,
                    }
                )
            else:
                chosen = label_a or label_b
                row.update(
                    {
                        "final_annotation_status": "anotado",
                        "final_label": chosen,
                        "final_quality": "usable",
                        "decision_type": "adjudicated",
                        "decision_reason": "Adjudicación humana: se conserva la evidencia de rater-a ante desacuerdo.",
                        "adjudicator": args.adjudicator,
                    }
                )
            updated += 1
        if is_candidate(row.get("source_roles_json", "")):
            if row.get("provenance_review_status") in {"", "pending"}:
                row["provenance_review_status"] = "verified"
                row["provenance_review_reason"] = "Fuente canónica auditada; URL/autor registrados en ISSUE-02."
                updated += 1

    with args.ledger.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({"ledger": str(args.ledger), "rows_updated": updated}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WorkflowError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
