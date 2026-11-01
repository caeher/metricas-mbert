#!/usr/bin/env python
"""Build an immutable dataset release from adjudicated Argilla exports (ISSUE-08)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from argilla_workflow import DEFAULT_RECORDS, TAXONOMY_VERSION, WorkflowError, load_canonical_rows

DATA_PROCESSED = ROOT / "data" / "processed"
RELEASES_ROOT = ROOT / "data" / "releases"

LABEL_NAMES = {
    0: "No Tóxico",
    1: "Lenguaje Ofensivo",
    2: "Discurso de Odio",
    3: "Amenazas/Violencia",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adjudicated",
        type=Path,
        default=ROOT / "reports/annotation/adjudicated_records.csv",
    )
    parser.add_argument("--release-id", default="v3-pilot-argilla-20261031")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_adjudicated(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise WorkflowError(f"No existe {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {row["canonical_id"]: row for row in rows if row.get("canonical_id")}


def apply_overrides(
    frame: pd.DataFrame,
    adjudicated: dict[str, dict[str, str]],
    canonical_by_id: dict[str, dict[str, str]],
    *,
    text_col: str = "texto_modelo",
) -> tuple[pd.DataFrame, int]:
    updated = 0
    out = frame.copy()
    norm_index: dict[str, list[int]] = {}
    for idx, value in out[text_col].astype(str).items():
        key = value.strip().casefold()
        norm_index.setdefault(key, []).append(idx)

    for canonical_id, decision in adjudicated.items():
        source = canonical_by_id.get(canonical_id)
        if source is None:
            continue
        key = (source.get("text_normalized") or source.get("text_original") or "").strip().casefold()
        if not key:
            continue
        label = int(decision["label_final"])
        for idx in norm_index.get(key, []):
            if int(out.at[idx, "label"]) != label:
                updated += 1
            out.at[idx, "label"] = label
            out.at[idx, "clase"] = LABEL_NAMES[label]
    return out, updated


def class_weights(train: pd.DataFrame, label_col: str = "label") -> dict[int, float]:
    counts = train[label_col].value_counts().sort_index()
    total = len(train)
    classes = len(counts)
    return {int(label): total / (classes * int(count)) for label, count in counts.items()}


def main() -> int:
    args = parse_args()
    adjudicated = load_adjudicated(args.adjudicated)
    if not adjudicated:
        raise WorkflowError("adjudicated_records.csv está vacío")

    canonical_by_id = {row["canonical_id"]: row for row in load_canonical_rows(DEFAULT_RECORDS)}
    release_dir = RELEASES_ROOT / args.release_id
    release_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA_PROCESSED / "train.csv")
    val = pd.read_csv(DATA_PROCESSED / "val.csv")
    test = pd.read_csv(DATA_PROCESSED / "test.csv")

    train, train_updates = apply_overrides(train, adjudicated, canonical_by_id)
    val, val_updates = apply_overrides(val, adjudicated, canonical_by_id)
    test, test_updates = apply_overrides(test, adjudicated, canonical_by_id)

    train_path = release_dir / "train.csv"
    val_path = release_dir / "val.csv"
    test_path = release_dir / "test.csv"
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)

    class_weights_map = class_weights(train)
    metadata = {
        "release_id": args.release_id,
        "taxonomy_version": TAXONOMY_VERSION,
        "seed": args.seed,
        "source_adjudicated": str(args.adjudicated.relative_to(ROOT)),
        "adjudicated_record_count": len(adjudicated),
        "label_overrides": {
            "train": train_updates,
            "val": val_updates,
            "test": test_updates,
        },
        "class_weights": {str(k): v for k, v in class_weights_map.items()},
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "test_rows": int(len(test)),
        "files": {
            "train": train_path.name,
            "val": val_path.name,
            "test": test_path.name,
        },
    }
    manifest_path = release_dir / "metadata.json"
    manifest_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"release_dir": str(release_dir), **metadata}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WorkflowError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
