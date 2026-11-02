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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "argilla" / "scripts"))

from argilla_workflow import DEFAULT_RECORDS, TAXONOMY_VERSION, WorkflowError, load_canonical_rows

DATA_PROCESSED = ROOT / "data" / "processed"
RELEASES_ROOT = ROOT / "data" / "releases"

LABEL_NAMES = {
    0: "No Tóxico",
    1: "Lenguaje Ofensivo",
    2: "Discurso de Odio",
    3: "Amenazas/Violencia",
}


WILLIAN_CANDIDATE_SOURCE = "versiones/v3/sources/dataset_willian_facebook.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adjudicated",
        type=Path,
        default=ROOT / "reports/annotation/adjudicated_records.csv",
    )
    parser.add_argument("--release-id", default="v3-pilot-argilla-20261031")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--append-adjudicated-candidates",
        action="store_true",
        help="Añade adjudicados nuevos (candidatos) solo a train.",
    )
    parser.add_argument(
        "--candidate-source",
        default=WILLIAN_CANDIDATE_SOURCE,
        help="source_files_json a incluir en append.",
    )
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


def _json_list(value: str) -> list:
    try:
        parsed = json.loads(value or "[]")
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _first_metadata_field(row: dict[str, str], field: str) -> str:
    for item in _json_list(row.get("source_metadata_json", "")):
        if not isinstance(item, str):
            continue
        try:
            payload = json.loads(item)
        except json.JSONDecodeError:
            continue
        if field in payload and payload[field] not in (None, ""):
            return str(payload[field])
    return ""


def append_adjudicated_candidates(
    train: pd.DataFrame,
    adjudicated: dict[str, dict[str, str]],
    canonical_by_id: dict[str, dict[str, str]],
    *,
    candidate_source: str,
) -> tuple[pd.DataFrame, int]:
    existing_norm = set(train["texto_modelo"].astype(str).str.strip().str.casefold())
    new_rows: list[dict[str, object]] = []
    appended = 0
    for canonical_id, decision in adjudicated.items():
        if decision.get("campaign") != "new_candidates":
            continue
        if decision.get("provenance_review_status") not in {"verified", "not_required"}:
            continue
        source = canonical_by_id.get(canonical_id)
        if source is None:
            continue
        if candidate_source not in _json_list(source.get("source_files_json", "")):
            continue
        text_orig = source.get("text_original") or ""
        text_norm = source.get("text_normalized") or text_orig
        key = str(text_norm).strip().casefold()
        if not key or key in existing_norm:
            continue
        label = int(decision["label_final"])
        record_ids = _json_list(source.get("source_record_ids_json", ""))
        row_id = str(record_ids[0]) if record_ids else f"WIL-{canonical_id[-8:]}"
        dates = _json_list(source.get("date_iso_values_json", ""))
        fecha = dates[0] if dates else _first_metadata_field(source, "Fecha_Recolección")
        authors = _json_list(source.get("author_group_ids_json", ""))
        user = authors[0] if authors else _first_metadata_field(source, "Usuario_ID_Anonimizado") or "USER_WIL"
        row = {col: "" for col in train.columns}
        row.update(
            {
                "ID": row_id,
                "Plataforma": source.get("platform_canonical") or "Facebook",
                "fecha": fecha,
                "Usuario_ID_Anonimizado": user,
                "texto_original": text_orig,
                "texto_modelo": text_norm,
                "texto_agresivo": text_norm,
                "clase": LABEL_NAMES[label],
                "label": label,
                "origen": "real_willian_argilla_gold",
                "validado": "SI",
                "recolector": "Willian",
            }
        )
        subclase = _first_metadata_field(source, "Subclase_Toxicidad")
        if subclase and subclase.upper() != "N/A":
            row["subclase"] = subclase
        grupos = _first_metadata_field(source, "Grupos_Afectados")
        if grupos and grupos.upper() != "N/A":
            row["Grupos_Afectados"] = grupos
        new_rows.append(row)
        existing_norm.add(key)
        appended += 1
    if not new_rows:
        return train, 0
    extra = pd.DataFrame(new_rows)
    for col in train.columns:
        if col not in extra.columns:
            extra[col] = ""
    extra = extra[train.columns]
    return pd.concat([train, extra], ignore_index=True), appended


def main() -> int:
    args = parse_args()
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    adjudicated_path = args.adjudicated.resolve()
    if not adjudicated_path.is_file():
        candidate = (ROOT / args.adjudicated).resolve()
        if candidate.is_file():
            adjudicated_path = candidate
    adjudicated = load_adjudicated(adjudicated_path)
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

    appended_train = 0
    if args.append_adjudicated_candidates:
        train, appended_train = append_adjudicated_candidates(
            train,
            adjudicated,
            canonical_by_id,
            candidate_source=args.candidate_source,
        )

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
        "source_adjudicated": str(
            adjudicated_path.relative_to(ROOT)
            if adjudicated_path.is_relative_to(ROOT)
            else adjudicated_path
        ),
        "adjudicated_record_count": len(adjudicated),
        "label_overrides": {
            "train": train_updates,
            "val": val_updates,
            "test": test_updates,
        },
        "appended_candidate_rows": appended_train,
        "candidate_source_appended": args.candidate_source if args.append_adjudicated_candidates else "",
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
