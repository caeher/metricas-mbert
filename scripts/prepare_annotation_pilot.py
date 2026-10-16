#!/usr/bin/env python
"""Prepare a deterministic, blind 200-record human annotation pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from argilla_workflow import DEFAULT_RECORDS, WorkflowError, load_canonical_rows


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports/annotation/pilot_sample_v1.csv"
LEGACY_TRAIN = "data/raw/train.csv"
NEW_SOURCES = ("data/raw/lenguaje_ofensivo.csv", "data/raw/thenewdata_X.csv")
GUIDELINE_VERSION = "2.0.0"


def parse_json_list(value: str) -> list[Any]:
    try:
        parsed = json.loads(value or "[]")
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def row_is_real(row: dict[str, str]) -> bool:
    return not any(str(origin).casefold().startswith(("synthetic", "sintetico")) for origin in parse_json_list(row.get("origin_types_json", "")))


def old_label(row: dict[str, str]) -> str | None:
    labels = {str(value) for value in parse_json_list(row.get("label_provisional_ids_json", ""))}
    if len(labels) == 1 and next(iter(labels)) in {"0", "1", "2", "3"}:
        return next(iter(labels))
    return None


def metadata_objects(row: dict[str, str]) -> list[dict[str, Any]]:
    values = parse_json_list(row.get("source_metadata_json", ""))
    result = []
    for value in values:
        if isinstance(value, dict):
            result.append(value)
        elif isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                result.append(parsed)
    return result


def subclasses(row: dict[str, str]) -> str:
    return " ".join(str(item.get("Subclase_Toxicidad", "")) for item in metadata_objects(row)).casefold()


def normalized_text(row: dict[str, str]) -> str:
    return unicodedata.normalize("NFKC", row.get("text_original", "")).casefold()


def stable_key(seed: str, canonical_id: str) -> str:
    return hashlib.sha256(f"issue05-pilot-v1\0{seed}\0{canonical_id}".encode("utf-8")).hexdigest()


def matches(row: dict[str, str], category: str) -> bool:
    text = normalized_text(row)
    subclass = subclasses(row)
    label = old_label(row)
    if category == "offense_hate_boundary":
        identity = re.search(
            r"\b(hondur|guatemal|nicarag|mexican|salvadoreñ|jud[ií]o|musulm|mujer|mujeres|negro|ind[ií]gen|gay|lesbian|pobre|partid|fmln|arena|nuevas ideas)\w*\b",
            text,
        )
        return bool(identity and label in {"1", "2"})
    if category == "hate_threat_boundary":
        harm = re.search(r"\b(matar|maten|matá|golpe|quem|dispar|muerte|muera|apuñal|amenaz|dar piso|dar cuello)\w*\b", text)
        return bool(harm and label in {"2", "3"})
    if category == "neutral_dialect":
        slang = re.search(r"\b(maje|majes|cerote|cerotes|chero|chera|cipote|cipotes|bicho|bichos|chivo|chiva|púchica|bayunco)\b", text)
        return bool(slang)
    if category == "politics":
        return "polit" in subclass or bool(
            re.search(r"\b(fmln|arena|nuevas ideas|bukele|president|diputad|alcald|partido|oposici[oó]n|gobierno)\w*\b", text)
        )
    if category == "sexism":
        return "sexis" in subclass or bool(
            re.search(r"\b(feminazi|mis[oó]gin|machist|zorra|perra|puta|puto|maric[oó]n|marica|mujer|mujeres)\w*\b", text)
        )
    return False


def source_group(row: dict[str, str]) -> str:
    files = set(parse_json_list(row.get("source_files_json", "")))
    if "data/raw/lenguaje_ofensivo.csv" in files and "data/raw/thenewdata_X.csv" in files:
        return "new_shared_conflict"
    if "data/raw/lenguaje_ofensivo.csv" in files:
        return "new_lenguaje_ofensivo"
    if "data/raw/thenewdata_X.csv" in files:
        return "new_thenewdata_X"
    return "legacy_train_real"


def prepare(rows: list[dict[str, str]], seed: str) -> tuple[list[dict[str, str]], dict[str, Any]]:
    eligible: dict[str, dict[str, str]] = {}
    for row in rows:
        files = set(parse_json_list(row.get("source_files_json", "")))
        roles = set(parse_json_list(row.get("source_roles_json", "")))
        if not row_is_real(row) or files.intersection({"data/raw/val.csv", "data/raw/test.csv"}):
            continue
        if LEGACY_TRAIN in files and row.get("record_disposition") == "legacy_split_label_unreviewed":
            eligible[row["canonical_id"]] = row
        elif "candidate_unvalidated" in roles and files.intersection(NEW_SOURCES):
            eligible[row["canonical_id"]] = row

    selected: dict[str, str] = {}

    def add_from(pool: list[dict[str, str]], count: int, stratum: str) -> int:
        available = [row for row in pool if row["canonical_id"] not in selected]
        available.sort(key=lambda row: stable_key(seed, row["canonical_id"]))
        chosen = available[:count]
        for row in chosen:
            selected[row["canonical_id"]] = stratum
        return len(chosen)

    legacy = [row for row in eligible.values() if LEGACY_TRAIN in parse_json_list(row.get("source_files_json", ""))]
    for platform in ("Facebook", "X"):
        for label in ("0", "1", "2", "3"):
            cell = [row for row in legacy if row.get("platform_canonical") == platform and old_label(row) == label]
            count = add_from(cell, 15, f"legacy_balance_{platform}_{label}")
            if count != 15:
                raise WorkflowError(f"Estrato histórico insuficiente: plataforma={platform}, etiqueta={label}, n={count}")

    candidate_rows = [row for row in eligible.values() if "candidate_unvalidated" in parse_json_list(row.get("source_roles_json", ""))]
    conflicts = [row for row in candidate_rows if row.get("record_disposition") == "human_adjudication_required"]
    for row in conflicts:
        selected[row["canonical_id"]] = "known_source_label_conflict"
    for source, stratum in ((NEW_SOURCES[0], "new_source_lenguaje_ofensivo"), (NEW_SOURCES[1], "new_source_thenewdata_X")):
        exclusive = [
            row for row in candidate_rows
            if source in parse_json_list(row.get("source_files_json", ""))
            and not set(parse_json_list(row.get("source_files_json", ""))).intersection(set(NEW_SOURCES).difference({source}))
        ]
        count = add_from(exclusive, 18, stratum)
        if count != 18:
            raise WorkflowError(f"La fuente nueva {source} no tiene 18 textos únicos elegibles")

    remaining = [row for row in eligible.values() if row["canonical_id"] not in selected]
    categories = (
        ("offense_hate_boundary", 8),
        ("hate_threat_boundary", 8),
        ("neutral_dialect", 8),
        ("politics", 8),
        ("sexism", 8),
    )
    for category, quota in categories:
        pool = [row for row in remaining if matches(row, category)]
        count = add_from(pool, quota, f"borderline_{category}")
        if count < quota:
            # Keep the sample at 200 while retaining a clear audit trail for sparse strata.
            backfill = [row for row in remaining if row["canonical_id"] not in selected]
            filled = add_from(backfill, quota - count, f"borderline_backfill_for_{category}")
            if filled < quota - count:
                raise WorkflowError("No hay suficientes registros reales para completar el piloto de 200")

    if len(selected) != 200:
        raise WorkflowError(f"El diseño produjo {len(selected)} IDs; se esperaban exactamente 200")

    by_id = {row["canonical_id"]: row for row in eligible.values()}
    manifest: list[dict[str, str]] = []
    for canonical_id, stratum in sorted(selected.items(), key=lambda item: (item[1], stable_key(seed, item[0]))):
        row = by_id[canonical_id]
        manifest.append(
            {
                "canonical_id": canonical_id,
                "pilot_stratum": stratum,
                "source_group": source_group(row),
                "platform": row.get("platform_canonical", ""),
            }
        )

    label_platform_counts = Counter(
        f"{row.get('platform_canonical', '')}:{old_label(row)}"
        for row in by_id.values()
        if row["canonical_id"] in selected and LEGACY_TRAIN in parse_json_list(row.get("source_files_json", ""))
    )
    source_counts = Counter(row["source_group"] for row in manifest)
    strata_counts = Counter(row["pilot_stratum"] for row in manifest)
    balanced_block_counts = Counter(
        f"{row['platform']}:{row['pilot_stratum'].rsplit('_', 1)[-1]}"
        for row in manifest
        if row["pilot_stratum"].startswith("legacy_balance_")
    )
    summary = {
        "status": "pilot_sample_prepared_pending_human_annotation",
        "sample_version": "annotation-pilot-v1",
        "seed": seed,
        "record_count": len(manifest),
        "eligible_historical_train_real_records": len(legacy),
        "eligible_new_candidate_records": len(candidate_rows),
        "guideline_version": GUIDELINE_VERSION,
        "canonical_records_sha256": hashlib.sha256(
            (ROOT / "data/interim/canonical_records.csv").read_bytes()
        ).hexdigest(),
        "historical_train_class_platform_cells_in_full_pilot": dict(sorted(label_platform_counts.items())),
        "balanced_block_class_platform_cells": dict(sorted(balanced_block_counts.items())),
        "source_group_counts": dict(sorted(source_counts.items())),
        "stratum_counts": dict(sorted(strata_counts.items())),
        "known_conflict_records_included": len(conflicts),
        "blindness": {
            "old_labels_in_argilla_fields": False,
            "model_suggestions": False,
            "sample_strata_sent_to_argilla": False,
        },
        "sampling_note": (
            "Las cuotas usan etiquetas históricas solo para muestreo estratificado; no se envían a Argilla. "
            "Esta muestra piloto no estima prevalencia ni reemplaza la revisión del corpus completo."
        ),
    }
    return manifest, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, help="Resumen JSON; por defecto comparte nombre con --output.")
    parser.add_argument("--seed", default="20261016", help="Semilla de selección estable.")
    args = parser.parse_args()
    try:
        manifest, summary = prepare(load_canonical_rows(args.records), args.seed)
        summary["canonical_records_sha256"] = hashlib.sha256(args.records.read_bytes()).hexdigest()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        summary_path = args.summary or args.output.with_suffix(".json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["canonical_id", "pilot_stratum", "source_group", "platform"])
            writer.writeheader()
            writer.writerows(manifest)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"manifest": str(args.output), "summary": str(summary_path), **summary}, ensure_ascii=False, indent=2))
        return 0
    except (WorkflowError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
