"""Carga, validación y exportación de vectores de predicciones alineados a test."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from discurso_odio.config import load_settings
from discurso_odio.paths import REPORTS_METRICS, TEST_CSV

XLMR_PREDICTIONS_PATH = REPORTS_METRICS / "xlmr_test_predictions.csv"
XLMR_PREDICTIONS_JSON = REPORTS_METRICS / "xlmr_test_predictions.json"
MBERT_PREDICTIONS_PATH = REPORTS_METRICS / "mbert_test_predictions.csv"
MBERT_PREDICTIONS_JSON = REPORTS_METRICS / "mbert_test_predictions.json"

XLMR_PRED_COL = "y_pred_xlmr"
MBERT_PRED_COL = "y_pred_mbert"


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def expected_num_classes() -> int:
    settings = load_settings()
    return int(settings["labels"]["num_classes"])


def validate_prediction_vector(
    test: pd.DataFrame,
    preds: pd.DataFrame,
    pred_col: str,
    model_name: str,
    *,
    num_classes: int | None = None,
) -> np.ndarray:
    """Valida y alinea predicciones al orden de ``test``."""
    if pred_col not in preds.columns:
        raise ValueError(f"{model_name}: columna requerida ausente: {pred_col}")

    num_classes = num_classes or expected_num_classes()
    max_label = num_classes - 1

    if len(preds) != len(test):
        raise ValueError(
            f"{model_name}: se esperaban {len(test)} predicciones, se recibieron {len(preds)}"
        )

    if "id" in preds.columns and "ID" in test.columns:
        merged = test[["ID"]].merge(preds, left_on="ID", right_on="id", how="left")
        if merged[pred_col].isna().any():
            missing = int(merged[pred_col].isna().sum())
            raise ValueError(
                f"{model_name}: {missing} predicciones sin ID correspondiente en test"
            )
        y_pred = merged[pred_col].to_numpy()
    else:
        y_pred = preds[pred_col].to_numpy()

    try:
        raw = np.asarray(y_pred, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{model_name}: etiquetas deben ser enteras 0-{max_label}") from exc

    if not np.all(np.isfinite(raw)):
        raise ValueError(f"{model_name}: etiquetas deben ser enteras 0-{max_label}")
    if not np.all(raw == np.floor(raw)):
        raise ValueError(f"{model_name}: etiquetas deben ser enteras 0-{max_label}")

    y_pred = raw.astype(int)

    if y_pred.min() < 0 or y_pred.max() > max_label:
        raise ValueError(f"{model_name}: etiquetas fuera del rango 0-{max_label}")

    return y_pred


def load_aligned_predictions(
    test: pd.DataFrame,
    path: Path,
    pred_col: str,
    model_name: str,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"{model_name}: artefacto no encontrado: {path}")
    preds = pd.read_csv(path)
    return validate_prediction_vector(test, preds, pred_col, model_name)


def load_xlmr_predictions(test: pd.DataFrame) -> tuple[np.ndarray | None, str]:
    if not XLMR_PREDICTIONS_PATH.exists():
        return None, "predictions_not_found"
    y_pred = load_aligned_predictions(test, XLMR_PREDICTIONS_PATH, XLMR_PRED_COL, "XLM-R")
    return y_pred, "ok"


def export_predictions_csv(
    test: pd.DataFrame,
    y_pred: np.ndarray,
    *,
    csv_path: Path,
    json_path: Path,
    pred_col: str,
    model: str,
    checkpoint: str,
    max_length: int,
    split: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Exporta predicciones con IDs y metadatos de ejecución."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    split = split or TEST_CSV.name

    if len(y_pred) != len(test):
        raise ValueError(f"Longitud de predicciones ({len(y_pred)}) != test ({len(test)})")

    export_df = pd.DataFrame({"id": test["ID"].values, pred_col: np.asarray(y_pred, dtype=int)})
    export_df.to_csv(csv_path, index=False)

    metadata: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "model": model,
        "checkpoint": checkpoint,
        "split": split,
        "n": len(test),
        "max_length": max_length,
        "pred_column": pred_col,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return csv_path, json_path
