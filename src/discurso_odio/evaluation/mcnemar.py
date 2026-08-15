"""Ejecución y exportación de pruebas de McNemar."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from discurso_odio.config import load_settings
from discurso_odio.data.loaders import load_split
from discurso_odio.evaluation.metrics import mcnemar_test
from discurso_odio.evaluation.predictions import (
    MBERT_PRED_COL,
    MBERT_PREDICTIONS_PATH,
    XLMR_PRED_COL,
    XLMR_PREDICTIONS_PATH,
    load_aligned_predictions,
)
from discurso_odio.models.mbert import predict
from discurso_odio.paths import MODELS_DIR, REPORTS_METRICS

MCNEMAR_PATH = REPORTS_METRICS / "mcnemar_tests.csv"

BASELINE_NAMES = ["logisticregression", "linearsvc", "multinomialnb"]


def mcnemar_alpha() -> float:
    settings = load_settings()
    return float(settings.get("evaluation", {}).get("mcnemar_alpha", 0.05))


def mcnemar_conclusion(p_value: float, alpha: float | None = None) -> str:
    alpha = alpha if alpha is not None else mcnemar_alpha()
    if p_value < alpha:
        return f"diferencia significativa (alpha={alpha})"
    return f"no se rechaza H0 (alpha={alpha})"


def _load_mbert_predictions(test: pd.DataFrame, model_dir: Path) -> np.ndarray:
    if MBERT_PREDICTIONS_PATH.exists():
        return load_aligned_predictions(test, MBERT_PREDICTIONS_PATH, MBERT_PRED_COL, "mBERT")
    if not model_dir.exists():
        raise FileNotFoundError(
            f"No hay {MBERT_PREDICTIONS_PATH} ni checkpoint mBERT en {model_dir}"
        )
    y_pred, _ = predict(test["texto_modelo"].tolist(), model_dir)
    return np.asarray(y_pred, dtype=int)


def _baseline_row(
    y_true: np.ndarray,
    y_pred_mbert: np.ndarray,
    baseline_name: str,
    test: pd.DataFrame,
    alpha: float,
) -> dict[str, Any] | None:
    path = MODELS_DIR / f"baseline_{baseline_name}.joblib"
    if not path.exists():
        return None
    pipe = joblib.load(path)
    bl_pred = pipe.predict(test["texto_modelo"])
    mcn = mcnemar_test(y_true, y_pred_mbert, bl_pred)
    return _format_row("mbert", baseline_name, mcn, len(y_true), alpha)


def _format_row(
    model_a: str,
    model_b: str,
    mcn: dict[str, float],
    n: int,
    alpha: float,
) -> dict[str, Any]:
    p_value = float(mcn["p_value"])
    return {
        "model_a": model_a,
        "model_b": model_b,
        "statistic": float(mcn["statistic"]),
        "p_value": p_value,
        "b": int(mcn["b"]),
        "c": int(mcn["c"]),
        "n": n,
        "alpha": alpha,
        "significant": p_value < alpha,
        "conclusion": mcnemar_conclusion(p_value, alpha),
    }


def run_mcnemar_tests(
    *,
    model_dir: Path | None = None,
    require_xlmr: bool = True,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Calcula McNemar mBERT vs baselines y vs XLM-R."""
    model_dir = model_dir or MODELS_DIR / "mbert-sv"
    output_path = output_path or MCNEMAR_PATH
    alpha = mcnemar_alpha()

    test = load_split("test")
    y_true = test["label"].values
    y_pred_mbert = _load_mbert_predictions(test, model_dir)

    rows: list[dict[str, Any]] = []
    for name in BASELINE_NAMES:
        row = _baseline_row(y_true, y_pred_mbert, name, test, alpha)
        if row is not None:
            rows.append(row)

    if not XLMR_PREDICTIONS_PATH.exists():
        if require_xlmr:
            raise FileNotFoundError(
                f"Predicciones XLM-R requeridas para McNemar pareado: {XLMR_PREDICTIONS_PATH}. "
                "Ejecute scripts/export_xlmr_predictions.py primero."
            )
    else:
        y_pred_xlmr = load_aligned_predictions(
            test, XLMR_PREDICTIONS_PATH, XLMR_PRED_COL, "XLM-R"
        )
        mcn = mcnemar_test(y_true, y_pred_mbert, y_pred_xlmr)
        rows.append(_format_row("mbert", "xlm-roberta", mcn, len(y_true), alpha))

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
