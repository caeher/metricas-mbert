#!/usr/bin/env python
"""Validate ml-metricas integration of cde-project dataset (e.g. v1.9.0)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CDE_ROOT = ROOT.parent / "cde-project-ml-2026"
HISTORICAL_VAL_TEST_REFERENCE = DEFAULT_CDE_ROOT / "data" / "history" / "v1.3.1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def read_split_keys(path: Path) -> dict[str, tuple[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows: dict[str, tuple[str, str]] = {}
        for row in reader:
            record_id = (row.get("ID") or "").strip()
            if not record_id:
                continue
            text = (row.get("texto_original") or "").strip()
            label = str(row.get("label", "")).strip()
            rows[record_id] = (text, label)
    return rows


def read_train_text_sets(train_path: Path) -> tuple[set[str], set[str]]:
    normalized: set[str] = set()
    raw: set[str] = set()
    with train_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            text = (row.get("texto_original") or "").strip()
            if text:
                raw.add(text)
                normalized.add(normalize_text(text))
    return raw, normalized


def read_holdout_texts(paths: list[Path]) -> tuple[set[str], set[str]]:
    normalized: set[str] = set()
    raw: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            text_columns = ("texto_original", "texto", "text")
            for row in reader:
                text = ""
                for column in text_columns:
                    if row.get(column, "").strip():
                        text = row[column].strip()
                        break
                if not text:
                    continue
                raw.add(text)
                normalized.add(normalize_text(text))
    return raw, normalized


def compare_historical_splits(local_val: Path, local_test: Path, reference_dir: Path) -> dict[str, object]:
    ref_val = reference_dir / "val.csv"
    ref_test = reference_dir / "test.csv"
    local_val_keys = read_split_keys(local_val)
    local_test_keys = read_split_keys(local_test)
    ref_val_keys = read_split_keys(ref_val)
    ref_test_keys = read_split_keys(ref_test)
    mismatches: list[str] = []
    for name, local_keys, ref_keys in (
        ("val", local_val_keys, ref_val_keys),
        ("test", local_test_keys, ref_test_keys),
    ):
        if set(local_keys) != set(ref_keys):
            mismatches.append(f"{name}: ID set mismatch")
        for record_id, ref_pair in ref_keys.items():
            local_pair = local_keys.get(record_id)
            if local_pair != ref_pair:
                mismatches.append(f"{name}:{record_id}")
    return {
        "reference_dir": str(reference_dir),
        "val_ids": len(local_val_keys),
        "test_ids": len(local_test_keys),
        "ok": not mismatches,
        "mismatches": mismatches[:20],
        "mismatch_count": len(mismatches),
    }


def verify_frozen_manifest(bank_dir: Path) -> dict[str, object]:
    manifest_path = bank_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks: list[dict[str, object]] = []
    ok = True
    for key, entry in manifest.items():
        if key == "test_oficial":
            continue
        if not isinstance(entry, dict) or "archivo" not in entry:
            continue
        rel = entry["archivo"].replace("data/evaluation/frozen_bank/", "")
        path = bank_dir / rel
        digest = sha256_file(path) if path.exists() else ""
        expected = entry.get("sha256", "")
        match = digest == expected
        ok = ok and match
        checks.append({"file": rel, "sha256_match": match, "expected": expected, "actual": digest})
    return {"ok": ok, "files": checks}


def verify_sync_manifest(sync_path: Path) -> dict[str, object]:
    if not sync_path.exists():
        return {"ok": False, "error": "missing last_sync.json"}
    manifest = json.loads(sync_path.read_text(encoding="utf-8"))
    mismatches: list[str] = []
    skipped_legacy = {
        "data/raw/test_adversarial.csv",
        "data/raw/corpus_completo.csv",
    }
    for entry in manifest.get("files", []):
        rel = entry.get("path", "")
        if rel in skipped_legacy:
            continue
        expected = entry.get("sha256", "")
        path = ROOT / rel
        if not path.exists():
            mismatches.append(f"missing:{rel}")
            continue
        if expected and sha256_file(path) != expected:
            mismatches.append(f"sha256:{rel}")
    corpus = manifest.get("corpus_completo", {})
    corpus_path_rel = str(corpus.get("path", ""))
    if corpus_path_rel.startswith("data/raw/"):
        corpus_path_rel = corpus_path_rel.replace("data/raw/", "data/interim/", 1)
    corpus_path = ROOT / corpus_path_rel
    if corpus_path.exists() and corpus.get("sha256") and sha256_file(corpus_path) != corpus["sha256"]:
        mismatches.append(f"sha256:{corpus_path.relative_to(ROOT).as_posix()}")
    return {"ok": not mismatches, "mismatches": mismatches, "dataset_version": manifest.get("dataset_version")}


def run_validation(*, reference_dir: Path) -> dict[str, object]:
    metadata_path = ROOT / "data" / "raw" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    train_path = ROOT / "data" / "raw" / "train.csv"
    val_path = ROOT / "data" / "raw" / "val.csv"
    test_path = ROOT / "data" / "raw" / "test.csv"
    adversarial_path = ROOT / "data" / "processed" / "test_adversarial.csv"
    bank_dir = ROOT / "data" / "evaluation" / "frozen_bank"

    row_counts_ok = (
        sum(1 for _ in open(train_path, encoding="utf-8-sig")) - 1 == metadata.get("train_total")
        and sum(1 for _ in open(val_path, encoding="utf-8-sig")) - 1 == 459
        and sum(1 for _ in open(test_path, encoding="utf-8-sig")) - 1 == 460
    )

    class_counts: dict[str, int] = {}
    with train_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            label_name = (row.get("clase") or "").strip()
            if label_name:
                class_counts[label_name] = class_counts.get(label_name, 0) + 1
    expected_classes = metadata.get("train_por_clase", {})
    class_ok = class_counts == expected_classes

    train_raw, train_norm = read_train_text_sets(train_path)
    holdout_paths = [val_path, test_path, adversarial_path]
    holdout_paths.extend(sorted(bank_dir.glob("*.csv")))
    holdout_raw, holdout_norm = read_holdout_texts(holdout_paths)
    leakage = {
        "raw_overlap": len(train_raw & holdout_raw),
        "normalized_overlap": len(train_norm & holdout_norm),
    }
    leakage_ok = leakage["raw_overlap"] == 0 and leakage["normalized_overlap"] == 0

    results = {
        "dataset_version": metadata.get("version"),
        "row_counts_ok": row_counts_ok,
        "train_por_clase_ok": class_ok,
        "train_por_clase_observed": class_counts,
        "historical_val_test": compare_historical_splits(val_path, test_path, reference_dir),
        "frozen_bank_manifest": verify_frozen_manifest(bank_dir),
        "train_holdout_leakage": leakage,
        "leakage_ok": leakage_ok,
        "sync_manifest": verify_sync_manifest(ROOT / "reports" / "data_sync" / "last_sync.json"),
    }
    supported_versions = {"1.8.0", "1.9.0"}
    results["ok"] = all(
        (
            metadata.get("version") in supported_versions,
            row_counts_ok,
            class_ok,
            results["historical_val_test"]["ok"],
            results["frozen_bank_manifest"]["ok"],
            leakage_ok,
            results["sync_manifest"]["ok"],
        )
    )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=HISTORICAL_VAL_TEST_REFERENCE,
        help="Historical val/test reference snapshot (default: cde-project history v1.3.1)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "data_sync" / "integration_validation.json",
    )
    args = parser.parse_args()
    report = run_validation(reference_dir=args.reference_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "output": str(args.output.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    if not report["ok"]:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
