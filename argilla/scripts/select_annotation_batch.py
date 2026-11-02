#!/usr/bin/env python
"""Select a stratified Argilla annotation batch from canonical candidates (ISSUE-06)."""

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

from argilla_workflow import DEFAULT_RECORDS, WorkflowError, load_canonical_rows

DATA_PROCESSED = ROOT / "data" / "processed"
WILLIAN_SOURCE = "versiones/v3/sources/dataset_willian_facebook.csv"
HOLDOUT_BLOCKLIST = ROOT / "reports" / "annotation" / "willian_holdout_blocklist.csv"
DEFAULT_MANIFEST = ROOT / "reports" / "annotation" / "facebook_willian_batch_v1.csv"
DEFAULT_SUMMARY = DEFAULT_MANIFEST.with_suffix(".json")

LABEL_NAMES = {
    "0": "No Tóxico",
    "1": "Lenguaje Ofensivo",
    "2": "Discurso de Odio",
    "3": "Amenazas/Violencia",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--source-file", default=WILLIAN_SOURCE, help="Restrict to this candidate source path.")
    parser.add_argument("--total", type=int, default=500)
    parser.add_argument("--target-toxic-23", type=int, default=350, dest="target_toxic_23")
    parser.add_argument("--target-class-1", type=int, default=100, dest="target_class_1")
    parser.add_argument("--target-class-0", type=int, default=50, dest="target_class_0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--write-holdout-blocklist", action="store_true")
    return parser.parse_args()


def normalize_text(value: str) -> str:
    import unicodedata

    return " ".join(unicodedata.normalize("NFC", value or "").casefold().split())


def holdout_text_keys() -> set[str]:
    keys: set[str] = set()
    for name in ("train.csv", "val.csv", "test.csv"):
        path = DATA_PROCESSED / name
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            for row in reader:
                text = row.get("texto_modelo") or row.get("texto_original") or ""
                if text.strip():
                    keys.add(normalize_text(text))
    return keys


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


def source_files(row: dict[str, str]) -> list[str]:
    try:
        files = json.loads(row.get("source_files_json") or "[]")
    except json.JSONDecodeError:
        return []
    return [str(item) for item in files] if isinstance(files, list) else []


def is_willian_row(row: dict[str, str], source_file: str) -> bool:
    return source_file in source_files(row)


def confidence_from_row(row: dict[str, str]) -> float:
    meta_raw = row.get("source_metadata_json") or "[]"
    try:
        metas = json.loads(meta_raw)
    except json.JSONDecodeError:
        metas = []
    if isinstance(metas, list):
        for item in metas:
            if not isinstance(item, str):
                continue
            try:
                payload = json.loads(item)
            except json.JSONDecodeError:
                continue
            if "Confianza_Clasificador" in payload:
                try:
                    return float(payload["Confianza_Clasificador"])
                except (TypeError, ValueError):
                    pass
    return 0.7


def eligible_willian_rows(records: list[dict[str, str]], source_file: str, holdout: set[str]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in records:
        if not is_willian_row(row, source_file):
            continue
        if row.get("platform_canonical") != "Facebook":
            continue
        files = source_files(row)
        if any(path.endswith("/val.csv") or path.endswith("/test.csv") for path in files):
            continue
        disposition = row.get("record_disposition", "")
        if disposition in {"quarantine_exact_duplicate", "quarantine_label_conflict"}:
            continue
        text_key = normalize_text(row.get("text_normalized") or row.get("text_original") or "")
        if text_key in holdout:
            continue
        out.append(row)
    return out


def write_holdout_blocklist(rows: list[dict[str, str]], holdout: set[str], path: Path) -> int:
    blocked: list[dict[str, str]] = []
    for row in rows:
        if not is_willian_row(row, WILLIAN_SOURCE):
            continue
        text_key = normalize_text(row.get("text_normalized") or row.get("text_original") or "")
        if text_key in holdout:
            blocked.append(
                {
                    "canonical_id": row["canonical_id"],
                    "text_original": row.get("text_original", ""),
                    "reason": "normalized_text_overlap_val_or_test_holdout",
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["canonical_id", "text_original", "reason"])
        writer.writeheader()
        writer.writerows(blocked)
    return len(blocked)


def pick_stratum(
    pool: list[dict[str, str]],
    count: int,
    rng: random.Random,
    *,
    stratum: str,
    reason: str,
    sort_key,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if count <= 0 or not pool:
        return [], pool
    ordered = sorted(pool, key=sort_key)
    if len(ordered) <= count:
        chosen = ordered
        remaining = []
    else:
        chosen = rng.sample(ordered, count)
        chosen_ids = {row["canonical_id"] for row in chosen}
        remaining = [row for row in pool if row["canonical_id"] not in chosen_ids]
    manifest_rows = [
        {
            "canonical_id": row["canonical_id"],
            "stratum": stratum,
            "selection_reason": reason,
            "provisional_label": provisional_label(row),
            "platform": row.get("platform_canonical", ""),
        }
        for row in chosen
    ]
    return manifest_rows, remaining


def main() -> int:
    args = parse_args()
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    records = load_canonical_rows(args.records)
    holdout = holdout_text_keys()
    if args.write_holdout_blocklist or not HOLDOUT_BLOCKLIST.exists():
        n_blocked = write_holdout_blocklist(records, holdout, HOLDOUT_BLOCKLIST)
        print(json.dumps({"holdout_blocklist": str(HOLDOUT_BLOCKLIST), "rows": n_blocked}, ensure_ascii=False))

    eligible = eligible_willian_rows(records, args.source_file, holdout)
    if not eligible:
        raise WorkflowError("No hay filas Willian elegibles tras filtros de holdout/campaña.")

    rng = random.Random(args.seed)
    by_label: dict[str, list[dict[str, str]]] = {"0": [], "1": [], "2": [], "3": []}
    for row in eligible:
        label = provisional_label(row)
        if label in by_label:
            by_label[label].append(row)

    selected: list[dict[str, str]] = []

    for label, stratum in (("2", "class_2"), ("3", "class_3")):
        chunk, _ = pick_stratum(
            by_label[label],
            len(by_label[label]),
            rng,
            stratum=stratum,
            reason=f"all_eligible_label_{label}",
            sort_key=lambda r, lab=label: (confidence_from_row(r), r["canonical_id"]),
        )
        selected.extend(chunk)

    if len(selected) > args.target_toxic_23:
        selected = rng.sample(selected, args.target_toxic_23)

    chunk, _ = pick_stratum(
        by_label["1"],
        args.target_class_1,
        rng,
        stratum="class_1",
        reason="low_confidence_or_short_text",
        sort_key=lambda r: (confidence_from_row(r), -len(r.get("text_original") or ""), r["canonical_id"]),
    )
    selected.extend(chunk)

    chunk, _ = pick_stratum(
        by_label["0"],
        args.target_class_0,
        rng,
        stratum="class_0_hard",
        reason="low_confidence_neutral_facebook",
        sort_key=lambda r: (confidence_from_row(r), r["canonical_id"]),
    )
    selected.extend(chunk)

    if len(selected) > args.total:
        selected = rng.sample(selected, args.total)
    elif len(selected) < args.total:
        already = {row["canonical_id"] for row in selected}
        filler_pool = [row for row in eligible if row["canonical_id"] not in already]
        need = args.total - len(selected)
        extra, _ = pick_stratum(
            filler_pool,
            need,
            rng,
            stratum="filler",
            reason="fill_to_target_total",
            sort_key=lambda r: r["canonical_id"],
        )
        selected.extend(extra)

    selected.sort(key=lambda item: item["canonical_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["canonical_id", "stratum", "selection_reason", "provisional_label", "platform"],
        )
        writer.writeheader()
        writer.writerows(selected)

    records_sha = hashlib.sha256(args.records.read_bytes()).hexdigest()
    summary = {
        "batch_id": args.output.stem,
        "record_count": len(selected),
        "seed": args.seed,
        "source_file": args.source_file,
        "canonical_records_path": str(args.records.relative_to(ROOT)).replace("\\", "/"),
        "canonical_records_sha256": records_sha,
        "holdout_blocked_count": write_holdout_blocklist(records, holdout, HOLDOUT_BLOCKLIST),
        "stratum_counts": dict(
            sorted(
                (stratum, sum(1 for row in selected if row["stratum"] == stratum))
                for stratum in sorted({row["stratum"] for row in selected})
            )
        ),
        "provisional_label_counts": dict(
            sorted(
                (label, sum(1 for row in selected if row["provisional_label"] == label))
                for label in sorted({row["provisional_label"] for row in selected})
            )
        ),
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(args.output), "summary": str(summary_path), **summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WorkflowError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
