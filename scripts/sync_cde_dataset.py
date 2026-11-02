#!/usr/bin/env python
"""Sync the active CDE dataset from cde-project-ml-2026 into ml-metricas (e.g. v1.9.0)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CDE_ROOT = ROOT.parent / "cde-project-ml-2026"

RAW_MIRROR = ("train.csv", "val.csv", "test.csv", "metadata.json")
PROCESSED_MIRROR = ("train.csv", "val.csv", "test.csv", "test_adversarial.csv", "metadata.json")
V4_DATASET_DIR = ROOT / "versiones" / "v4" / "dataset"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def atomic_copy(source: Path, destination: Path, *, dry_run: bool) -> dict[str, str]:
    rel = destination.relative_to(ROOT).as_posix()
    if dry_run:
        return {"path": rel, "sha256": sha256_file(source) if source.exists() else "", "action": "would_copy"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = source.read_bytes()
    digest = sha256_bytes(data)
    with tempfile.NamedTemporaryFile(delete=False, dir=destination.parent) as handle:
        handle.write(data)
        temp_path = Path(handle.name)
    temp_path.replace(destination)
    return {"path": rel, "sha256": digest, "action": "copied"}


def copy_tree_csvs(source_dir: Path, dest_dir: Path, *, dry_run: bool) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    if not source_dir.exists():
        return entries
    for source in sorted(source_dir.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(source_dir)
        destination = dest_dir / relative
        if source.suffix.lower() == ".csv" or source.name == "MANIFEST.json":
            entries.append(atomic_copy(source, destination, dry_run=dry_run))
    return entries


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def write_corpus_completo(train_path: Path, val_path: Path, test_path: Path, output_path: Path, *, dry_run: bool) -> dict[str, str | int]:
    splits = [("train", train_path), ("val", val_path), ("test", test_path)]
    all_fieldnames: list[str] = []
    by_id: dict[str, dict[str, str]] = {}
    for _role, path in splits:
        if not path.exists():
            raise FileNotFoundError(f"Missing split for corpus union: {path}")
        fieldnames, rows = read_csv_rows(path)
        for name in fieldnames:
            if name not in all_fieldnames:
                all_fieldnames.append(name)
        for row in rows:
            record_id = (row.get("ID") or "").strip()
            if not record_id:
                continue
            if record_id in by_id:
                continue
            by_id[record_id] = row
    ordered = sorted(by_id.values(), key=lambda row: str(row.get("ID", "")))
    if dry_run:
        return {"path": output_path.relative_to(ROOT).as_posix(), "row_count": len(ordered), "action": "would_write"}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=all_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)
    return {
        "path": output_path.relative_to(ROOT).as_posix(),
        "row_count": len(ordered),
        "sha256": sha256_file(output_path),
        "action": "written",
    }


def run_sync(cde_root: Path, *, dry_run: bool, update_v4_snapshot: bool = False) -> dict[str, object]:
    dataset_src = cde_root / "data" / "dataset"
    if not dataset_src.is_dir():
        raise FileNotFoundError(f"Expected dataset directory: {dataset_src}")

    copied: list[dict[str, str]] = []
    for name in RAW_MIRROR:
        source = dataset_src / name
        if not source.exists():
            raise FileNotFoundError(f"Missing dataset file: {source}")
        copied.append(atomic_copy(source, ROOT / "data" / "raw" / name, dry_run=dry_run))

    for name in PROCESSED_MIRROR:
        source = dataset_src / name
        if not source.exists():
            raise FileNotFoundError(f"Missing dataset file: {source}")
        copied.append(atomic_copy(source, ROOT / "data" / "processed" / name, dry_run=dry_run))

    if update_v4_snapshot:
        for name in RAW_MIRROR:
            source = dataset_src / name
            copied.append(atomic_copy(source, V4_DATASET_DIR / name, dry_run=dry_run))

    if not dry_run:
        sys.path.insert(0, str(ROOT / "src"))
        from discurso_odio.data.adversarial_battery import write_adversarial_144_csv

        path_144 = write_adversarial_144_csv()
        copied.append(
            {
                "path": path_144.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path_144),
                "action": "regenerated_adversarial_144",
            }
        )

    copied.extend(
        copy_tree_csvs(
            cde_root / "data" / "evaluation" / "frozen_bank",
            ROOT / "data" / "evaluation" / "frozen_bank",
            dry_run=dry_run,
        )
    )
    copied.extend(
        copy_tree_csvs(
            cde_root / "data" / "sources" / "x_threads",
            ROOT / "data" / "sources" / "x_threads",
            dry_run=dry_run,
        )
    )

    corpus_info = write_corpus_completo(
        dataset_src / "train.csv",
        dataset_src / "val.csv",
        dataset_src / "test.csv",
        ROOT / "data" / "interim" / "corpus_completo.csv",
        dry_run=dry_run,
    )

    metadata_path = dataset_src / "metadata.json"
    version = json.loads(metadata_path.read_text(encoding="utf-8")).get("version", "unknown")
    manifest = {
        "synced_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "cde_root": str(cde_root.resolve()),
        "dataset_version": version,
        "files": copied,
        "corpus_completo": corpus_info,
    }
    manifest_path = ROOT / "reports" / "data_sync" / "last_sync.json"
    if not dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["manifest_path"] = manifest_path.relative_to(ROOT).as_posix()
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cde-root", type=Path, default=DEFAULT_CDE_ROOT, help="Root of cde-project-ml-2026")
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without writing files")
    parser.add_argument(
        "--update-v4-snapshot",
        action="store_true",
        help="Refresh versiones/v4/dataset/ from CDE (same files as data/raw core).",
    )
    args = parser.parse_args()
    try:
        manifest = run_sync(
            args.cde_root.resolve(),
            dry_run=args.dry_run,
            update_v4_snapshot=args.update_v4_snapshot,
        )
    except (FileNotFoundError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
