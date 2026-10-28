#!/usr/bin/env python3
"""Build deterministic historical split manifests without exporting text."""

from __future__ import annotations

import csv
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "splits"
CORE_FILES = {
    "train_legacy": "data/raw/train.csv",
    "development_legacy": "data/raw/val.csv",
    "historical_test": "data/raw/test.csv",
}
DIAGNOSTIC_FILES = {
    "diagnostic_adversarial_v1": "data/raw/test_adversarial.csv",
    "diagnostic_adversarial_v2": "data/processed/test_adversarial_144.csv",
}
LABEL_NAMES = {
    "0": "No Tóxico",
    "1": "Lenguaje Ofensivo",
    "2": "Discurso de Odio",
    "3": "Amenazas/Violencia",
}
CSV_FIELDS = [
    "release_id",
    "role",
    "source_file",
    "source_file_sha256",
    "record_id",
    "origin",
    "label_id",
    "label_name",
    "label_version",
    "platform_original",
    "canonical_platform",
    "raw_text_sha256",
    "normalized_text_sha256",
    "author_group_id_hash",
    "thread_group_id_hash",
    "event_group_id_hash",
    "duplicate_component_id_hash",
    "synthetic_template_group_id_hash",
    "source_row_number",
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_key(namespace: str, value: str) -> str:
    return sha256(f"{namespace}\0{value}".encode("utf-8"))


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).casefold()
    return " ".join(text.split())


def canonical_platform(value: str) -> str:
    value = value.strip()
    return "X" if value.casefold() == "twitter" else value


def load_rows(relative_path: str) -> tuple[bytes, list[dict[str, str]]]:
    path = ROOT / relative_path
    raw = path.read_bytes()
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"No records found in {relative_path}")
    return raw, rows


def build_records(role: str, relative_path: str) -> tuple[list[dict[str, str]], dict[str, Any]]:
    raw, rows = load_rows(relative_path)
    source_hash = sha256(raw)
    manifest_rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    label_counts: Counter[str] = Counter()
    platform_counts: Counter[str] = Counter()
    origin_counts: Counter[str] = Counter()
    authors: set[str] = set()
    missing_authors = 0
    label_pairs: list[tuple[str, str]] = []

    for line_no, row in enumerate(rows, start=2):
        record_id = (row.get("ID") or row.get("id") or f"row-{line_no - 1}").strip()
        if record_id in seen_ids:
            raise ValueError(f"Duplicate record ID {record_id!r} in {relative_path}")
        seen_ids.add(record_id)

        text = row.get("texto_original") or row.get("Texto_Original") or row.get("text") or row.get("texto") or ""
        label = (
            row.get("label")
            or row.get("clase")
            or row.get("Clase_Toxicidad")
            or row.get("label_esperado")
            or ""
        ).strip()
        if label in LABEL_NAMES:
            label_id = label
        elif label in LABEL_NAMES.values():
            label_id = next(key for key, name in LABEL_NAMES.items() if name == label)
        else:
            raise ValueError(f"Unexpected/missing label {label!r} in {relative_path}, record {record_id}")
        platform = (row.get("Plataforma") or row.get("plataforma") or "").strip()
        origin = (row.get("origen") or "real").strip()
        if role.startswith("diagnostic_") and "origen" not in row:
            origin = "diagnostic_curated_or_generated"
        author = (row.get("Usuario_ID_Anonimizado") or row.get("Usuario_ID_Anonimizado ") or "").strip()
        template = (row.get("plantilla_id") or "").strip()
        author_hash = hash_key("author-group-v1", author) if author else ""
        if author_hash:
            authors.add(author_hash)
        else:
            missing_authors += 1

        manifest_rows.append(
            {
                "release_id": "historical_snapshot_2026-10-16_v1",
                "role": role,
                "source_file": relative_path.replace("\\", "/"),
                "source_file_sha256": source_hash,
                "record_id": record_id,
                "origin": origin,
                "label_id": label_id,
                "label_name": LABEL_NAMES[label_id],
                "label_version": (
                    "diagnostic_curated_not_independent"
                    if role.startswith("diagnostic_")
                    else "legacy_unreviewed"
                ),
                "platform_original": platform,
                "canonical_platform": canonical_platform(platform),
                "raw_text_sha256": sha256(text.encode("utf-8")),
                "normalized_text_sha256": sha256(normalize_text(text).encode("utf-8")),
                "author_group_id_hash": author_hash,
                "thread_group_id_hash": "",
                "event_group_id_hash": "",
                "duplicate_component_id_hash": "",
                "synthetic_template_group_id_hash": hash_key("template-group-v1", template) if template else "",
                "source_row_number": str(line_no - 1),
            }
        )
        label_counts[label_id] += 1
        label_pairs.append((record_id, label_id))
        platform_counts[canonical_platform(platform) or "unknown"] += 1
        origin_counts[origin or "unknown"] += 1

    summary = {
        "path": relative_path.replace("\\", "/"),
        "role": role,
        "sha256": source_hash,
        "row_count": len(rows),
        "label_counts": dict(sorted(label_counts.items())),
        "platform_counts": dict(sorted(platform_counts.items())),
        "origin_counts": dict(sorted(origin_counts.items())),
        "author_group_count": len(authors),
        "records_missing_author_group": missing_authors,
        "thread_group_count": 0,
        "event_group_count": 0,
        "label_assignment_sha256": sha256(
            "".join(f"{record_id}\t{label}\n" for record_id, label in sorted(label_pairs)).encode("utf-8")
        ),
    }
    return manifest_rows, summary


def intersection_count(left: set[str], right: set[str]) -> int:
    return len(left.intersection(right))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, str]] = []
    summaries: dict[str, dict[str, Any]] = {}
    role_texts: dict[str, set[str]] = defaultdict(set)
    role_authors: dict[str, set[str]] = defaultdict(set)

    for role, relative_path in {**CORE_FILES, **DIAGNOSTIC_FILES}.items():
        records, summary = build_records(role, relative_path)
        all_records.extend(records)
        summaries[role] = summary
        for record in records:
            role_texts[role].add(record["normalized_text_sha256"])
            if record["author_group_id_hash"]:
                role_authors[role].add(record["author_group_id_hash"])

    manifest_path = OUT_DIR / "historical_split_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_records)
    manifest_bytes = manifest_path.read_bytes()

    core_roles = list(CORE_FILES)
    author_overlap: dict[str, int] = {}
    exact_overlap: dict[str, int] = {}
    for index, left in enumerate(core_roles):
        for right in core_roles[index + 1 :]:
            pair = f"{left}::{right}"
            author_overlap[pair] = intersection_count(role_authors[left], role_authors[right])
            exact_overlap[pair] = intersection_count(role_texts[left], role_texts[right])

    processed_test_path = ROOT / "data/processed/test.csv"
    processed_test_hash = sha256(processed_test_path.read_bytes())
    historical_test_hash = summaries["historical_test"]["sha256"]
    if processed_test_hash != historical_test_hash:
        raise ValueError("data/processed/test.csv is no longer byte-identical to data/raw/test.csv")

    report = {
        "release_id": "historical_snapshot_2026-10-16_v1",
        "manifest_version": "historical-snapshot-v1",
        "policy_version": "2.0.0",
        "taxonomy_version": "2.0.0",
        "label_version": "legacy_unreviewed",
        "generated_by": "scripts/build_split_manifests.py",
        "manifest_csv": manifest_path.relative_to(ROOT).as_posix(),
        "manifest_csv_sha256": sha256(manifest_bytes),
        "row_count": len(all_records),
        "splits": summaries,
        "model_test_input": {
            "path": "data/processed/test.csv",
            "sha256": processed_test_hash,
            "byte_identical_to": "data/raw/test.csv",
        },
        "overlap_audit": {
            "scope": "legacy train/development/historical test only",
            "author_group_intersections": author_overlap,
            "exact_normalized_text_intersections": exact_overlap,
            "thread_groups": "not_available_in_legacy_split_csvs",
            "event_groups": "not_available_in_legacy_split_csvs",
            "adjudicated_near_duplicate_components": "not_available_for_legacy_split_csvs",
            "interpretation": "Zero observed author/exact-text overlap does not establish event/thread/near-duplicate independence or untouched test status.",
        },
        "diagnostics": {
            "roles": list(DIAGNOSTIC_FILES),
            "selection": "curated/phenomenon-focused; report separately, not as prevalence estimates",
        },
        "limitations": [
            "Legacy labels have not been adjudicated under taxonomy v2.0.0.",
            "The historical test has prior analysis and is not an untouched final test.",
            "Author groups are hashed from legacy anonymized IDs; thread/event identifiers are absent.",
            "Candidate sources remain quarantined and are not included in any split.",
            "The new sealed test has not yet been collected or reserved.",
        ],
    }
    json_path = OUT_DIR / "historical_manifest.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote {manifest_path.relative_to(ROOT)} ({len(all_records)} records; sha256={sha256(manifest_bytes)})")
    print(f"Wrote {json_path.relative_to(ROOT)}")
    print("Observed author overlaps:", json.dumps(author_overlap, sort_keys=True))
    print("Observed exact-text overlaps:", json.dumps(exact_overlap, sort_keys=True))


if __name__ == "__main__":
    main()
