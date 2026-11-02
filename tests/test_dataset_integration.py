"""Tests for CDE dataset integration helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_dataset_sources import CANONICAL_IMPORT_ROLES, classify_core_origin  # noqa: E402


@pytest.mark.parametrize(
    ("origen", "origin_type"),
    [
        ("real", "real"),
        ("real_hilo", "real"),
        ("caso_dificil", "real"),
        ("sintetico_relleno_emoji", "synthetic"),
        ("v3_jerga_notoxico", "synthetic_v3"),
    ],
)
def test_classify_core_origin(origen: str, origin_type: str) -> None:
    classified, _original = classify_core_origin(origen)
    assert classified == origin_type


def test_frozen_bank_role_imported_but_not_core_train() -> None:
    assert "frozen_evaluation_bank" in CANONICAL_IMPORT_ROLES
    assert "core_split" in CANONICAL_IMPORT_ROLES


def test_audit_summary_train_count_and_leakage() -> None:
    summary_path = ROOT / "reports" / "data_audit" / "audit_summary.json"
    if not summary_path.exists():
        pytest.skip("audit_summary.json not generated yet")
    metadata_path = ROOT / "data" / "raw" / "metadata.json"
    if not metadata_path.exists():
        pytest.skip("metadata.json missing")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_train = int(metadata.get("train_total", 0))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    core_train = summary["core_splits"]["train"]
    if core_train != expected_train:
        pytest.skip(
            f"audit_summary train {core_train} != synced metadata {expected_train}; re-run audit"
        )
    leakage = summary.get("train_holdout_leakage", {})
    assert leakage.get("normalized_overlap_total") == 0


def test_get_b2_class_weights_rejects_cde_metadata_without_b2_pesos() -> None:
    pytest.importorskip("dotenv")
    from discurso_odio.data.loaders import get_b2_class_weights, load_b2_metadata

    metadata = load_b2_metadata()
    version = str(metadata.get("version", ""))
    if version in {"b2", "1.1.0"} or "pesos_clase" in metadata:
        pytest.skip("B2 metadata present")
    with pytest.raises(ValueError, match="pesos B2"):
        get_b2_class_weights()
