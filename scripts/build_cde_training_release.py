#!/usr/bin/env python
"""Immutable release from synced CDE dataset splits (no Argilla adjudication)."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RAW = ROOT / "data" / "raw"
RELEASES_ROOT = ROOT / "data" / "releases"
SYNC_MANIFEST = ROOT / "reports" / "data_sync" / "last_sync.json"


def class_weights(train: pd.DataFrame, label_col: str = "label") -> dict[int, float]:
    counts = train[label_col].value_counts().sort_index()
    total = len(train)
    classes = len(counts)
    return {int(label): total / (classes * int(count)) for label, count in counts.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-id", default="training-v3-20261101")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata_path = DATA_RAW / "metadata.json"
    if not metadata_path.is_file():
        print(f"error: missing {metadata_path}", file=sys.stderr)
        return 1

    dataset_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    dataset_version = str(dataset_meta.get("version", "unknown"))
    cde_root = ""
    if SYNC_MANIFEST.is_file():
        sync = json.loads(SYNC_MANIFEST.read_text(encoding="utf-8"))
        cde_root = str(sync.get("cde_root", ""))

    release_dir = RELEASES_ROOT / args.release_id
    release_dir.mkdir(parents=True, exist_ok=True)

    for name in ("train.csv", "val.csv", "test.csv"):
        source = DATA_PROCESSED / name
        if not source.is_file():
            print(f"error: missing {source}", file=sys.stderr)
            return 1
        shutil.copy2(source, release_dir / name)

    train = pd.read_csv(release_dir / "train.csv")
    val = pd.read_csv(release_dir / "val.csv")
    test = pd.read_csv(release_dir / "test.csv")
    weights = class_weights(train)

    manifest = {
        "release_id": args.release_id,
        "dataset_version": dataset_version,
        "seed": args.seed,
        "source": "sync_cde_dataset",
        "cde_root": cde_root,
        "class_weights": {str(k): v for k, v in weights.items()},
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "test_rows": int(len(test)),
        "files": {
            "train": "train.csv",
            "val": "val.csv",
            "test": "test.csv",
        },
    }
    (release_dir / "metadata.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"release_dir": str(release_dir), **manifest}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
