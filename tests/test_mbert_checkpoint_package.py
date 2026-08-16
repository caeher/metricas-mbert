"""Pruebas del paquete de auditoría mBERT (T-02)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.evaluation.predictions import MBERT_PRED_COL, validate_prediction_vector
from discurso_odio.models.inference_contract import get_inference_contract
from discurso_odio.models.mbert import default_mbert_model_dir
from discurso_odio.models.mbert_artifacts import (
    F1_TOLERANCE,
    GENERATED_FILES,
    MANIFEST_REQUIRED_FIELDS,
    REQUIRED_FILES,
    apply_id2label,
    list_package_files,
    package_readme,
    resolve_mbert_source,
    sha256_file,
)
from discurso_odio.paths import TEST_CSV

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = default_mbert_model_dir()
GITIGNORE = ROOT / ".gitignore"


def _has_weights() -> bool:
    try:
        source = resolve_mbert_source()
    except FileNotFoundError:
        return False
    return (source / "model.safetensors").exists()


def test_gitignore_keeps_weights_out_and_tracks_audit_artifacts():
    text = GITIGNORE.read_text(encoding="utf-8")
    assert "models/mbert-sv/" in text
    assert "artifacts/" in text
    assert "!reports/metrics/mbert_test_predictions.csv" in text
    assert "!reports/metrics/mbert_checkpoint_manifest.json" in text
    assert "!reports/metrics/mbert_checkpoint_verify.json" in text


def test_required_package_files_declared():
    assert REQUIRED_FILES == (
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    for name in GENERATED_FILES:
        assert name in {"inference_contract.json", "MANIFEST.json", "README.md"}


def test_manifest_required_fields():
    for field in (
        "git_commit",
        "seed",
        "base_model",
        "checkpoint",
        "sha256",
        "max_length",
        "normalizer",
        "test_sha256",
        "official_f1_macro",
    ):
        assert field in MANIFEST_REQUIRED_FIELDS


def test_sha256_file(tmp_path: Path):
    path = tmp_path / "sample.txt"
    path.write_text("mbert-audit", encoding="utf-8")
    digest = sha256_file(path)
    assert len(digest) == 64
    assert digest == sha256_file(path)


def test_apply_id2label(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"id2label": {"0": "LABEL_0"}, "label2id": {"LABEL_0": 0}}), encoding="utf-8")
    config = apply_id2label(config_path, ["No Tóxico", "Lenguaje Ofensivo", "Discurso de Odio", "Amenazas/Violencia"])
    assert config["id2label"]["0"] == "No Tóxico"
    assert config["label2id"]["Amenazas/Violencia"] == 3


def test_package_readme_mentions_contract():
    text = package_readme()
    assert "texto_modelo" in text
    assert "max_length=128" in text
    assert "normalize_for_model" in text


def test_prediction_vector_validation_460():
    test = pd.DataFrame({"ID": list(range(460)), "label": [0] * 460})
    preds = pd.DataFrame({"id": list(range(460)), MBERT_PRED_COL: ([0, 1, 2, 3] * 115)})
    y_pred = validate_prediction_vector(test, preds, MBERT_PRED_COL, "mBERT")
    assert len(y_pred) == 460
    assert y_pred.min() == 0
    assert y_pred.max() == 3


def test_inference_contract_max_length():
    contract = get_inference_contract()
    assert contract.max_length == 128
    with pytest.raises(ValueError, match="no coincide con el contrato"):
        contract.validate_max_length(256)


@pytest.mark.skipif(not _has_weights(), reason="Checkpoint mBERT no disponible")
def test_source_has_required_files():
    source = resolve_mbert_source()
    names = list_package_files(source)
    for required in REQUIRED_FILES:
        assert required in names


@pytest.mark.skipif(not _has_weights(), reason="Checkpoint mBERT no disponible")
def test_predict_rejects_max_length_override():
    from discurso_odio.models.mbert import predict

    with pytest.raises(ValueError, match="no coincide con el contrato"):
        predict(["hola"], MODEL_DIR if MODEL_DIR.exists() else resolve_mbert_source(), max_length=256)


def test_committed_manifest_identifies_checkpoint():
    path = ROOT / "reports" / "metrics" / "mbert_checkpoint_manifest.json"
    assert path.exists(), "Falta el manifiesto de auditoría"
    data = json.loads(path.read_text(encoding="utf-8"))
    for field in MANIFEST_REQUIRED_FIELDS:
        assert field in data, f"Manifiesto sin campo {field}"
    assert data["max_length"] == 128
    assert data["normalizer"] == "normalize_for_model"
    assert data["seed"] == 42
    assert data["official_f1_macro"] == 0.7777
    assert data["test_n"] == 460
    assert "model.safetensors" in data["sha256"]
    assert data["test_sha256"] == sha256_file(TEST_CSV)


def test_committed_manifest_test_sha256_matches_current_test_csv():
    path = ROOT / "reports" / "metrics" / "mbert_checkpoint_manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert TEST_CSV.exists(), "Falta data/processed/test.csv"
    assert data["test_sha256"] == sha256_file(TEST_CSV)


def test_committed_predictions_vector():
    path = ROOT / "reports" / "metrics" / "mbert_test_predictions.csv"
    assert path.exists(), "Falta el vector de predicciones mBERT"
    preds = pd.read_csv(path)
    test = pd.read_csv(ROOT / "data" / "processed" / "test.csv")
    y_pred = validate_prediction_vector(test, preds, MBERT_PRED_COL, "mBERT")
    assert len(y_pred) == 460
    assert int(y_pred.min()) >= 0
    assert int(y_pred.max()) <= 3


def test_committed_verify_report_passed():
    path = ROOT / "reports" / "metrics" / "mbert_checkpoint_verify.json"
    assert path.exists(), "Falta el log de verificación"
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["n"] == 460
    assert report["f1_match"] is True
    assert report["test_sha256_match"] is True
    assert report["used_repo_models_dir"] is False
    assert abs(report["f1_macro"] - 0.7777) <= F1_TOLERANCE
