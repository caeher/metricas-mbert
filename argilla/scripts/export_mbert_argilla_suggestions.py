#!/usr/bin/env python
"""Export mBERT predictions as Argilla suggestion JSONL for a canonical_id manifest."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "argilla" / "scripts"))

from argilla_workflow import DEFAULT_RECORDS, WorkflowError, load_canonical_rows
from discurso_odio.models.mbert import predict
from discurso_odio.models.mbert_artifacts import read_checkpoint_identity

DEFAULT_MANIFEST = ROOT / "reports/annotation/facebook_willian_batch_v1.csv"
DEFAULT_MODEL = ROOT / "versiones" / "v4" / "model"
DEFAULT_JSONL = ROOT / "data/interim/facebook_willian_batch_v1_suggestions.jsonl"
DEFAULT_DISCREPANCY = ROOT / "reports/annotation/facebook_willian_heuristic_vs_mbert.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--discrepancy-csv", type=Path, default=DEFAULT_DISCREPANCY)
    parser.add_argument("--agent", default="mbert-training-v3-v4")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def load_manifest_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return [row["canonical_id"].strip() for row in reader if row.get("canonical_id")]


def provisional_label(row: dict[str, str]) -> str:
    try:
        labels = json.loads(row.get("label_provisional_ids_json") or "[]")
    except json.JSONDecodeError:
        return ""
    if not labels:
        return ""
    value = str(labels[0]).strip()
    return value if value in {"0", "1", "2", "3"} else ""


def main() -> int:
    args = parse_args()
    if not args.model_dir.is_dir():
        raise WorkflowError(f"No existe el checkpoint: {args.model_dir}")
    checkpoint = args.checkpoint or read_checkpoint_identity(args.model_dir).get("checkpoint", "unknown")
    ids = load_manifest_ids(args.manifest)
    if not ids:
        raise WorkflowError(f"Manifiesto vacío: {args.manifest}")

    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    canonical = {row["canonical_id"]: row for row in load_canonical_rows(args.records)}
    missing = [item for item in ids if item not in canonical]
    if missing:
        raise WorkflowError(f"IDs ausentes en canonical_records: {missing[:5]}")

    texts: list[str] = []
    ordered_rows: list[dict[str, str]] = []
    for canonical_id in ids:
        row = canonical[canonical_id]
        text = row.get("text_original") or row.get("text_normalized") or ""
        if not text.strip():
            raise WorkflowError(f"Texto vacío para {canonical_id}")
        texts.append(text)
        ordered_rows.append(row)

    y_pred, y_prob = predict(
        texts,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        normalize=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    discrepancies: list[dict[str, str | float | int]] = []
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for idx, (row, pred, probs) in enumerate(zip(ordered_rows, y_pred, y_prob, strict=True)):
            label = str(int(pred))
            score = float(probs[int(pred)])
            canonical_id = row["canonical_id"]
            stream.write(
                json.dumps(
                    {
                        "canonical_id": canonical_id,
                        "label": label,
                        "score": round(score, 6),
                        "agent": args.agent,
                        "checkpoint": checkpoint,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            prov = provisional_label(row)
            if prov and prov != label:
                discrepancies.append(
                    {
                        "canonical_id": canonical_id,
                        "provisional_label": prov,
                        "mbert_label": label,
                        "mbert_score": round(score, 6),
                        "text_original": (row.get("text_original") or "")[:120],
                    }
                )

    args.discrepancy_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.discrepancy_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["canonical_id", "provisional_label", "mbert_label", "mbert_score", "text_original"],
        )
        writer.writeheader()
        writer.writerows(discrepancies)

    print(
        json.dumps(
            {
                "suggestions": str(args.output),
                "records": len(ids),
                "discrepancies": len(discrepancies),
                "discrepancy_csv": str(args.discrepancy_csv),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WorkflowError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
