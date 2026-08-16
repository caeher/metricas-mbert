"""Ejecución y exportación de pruebas de McNemar."""

from __future__ import annotations

import json
from datetime import datetime, timezone
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
    MBERT_PREDICTIONS_JSON,
    MBERT_PREDICTIONS_PATH,
    XLMR_PRED_COL,
    XLMR_PREDICTIONS_JSON,
    XLMR_PREDICTIONS_PATH,
    git_commit,
    load_aligned_predictions,
)
from discurso_odio.models.mbert import predict
from discurso_odio.models.xlmr import default_xlmr_model_dir
from discurso_odio.paths import MODELS_DIR, PROJECT_ROOT, REPORTS_METRICS, TEST_CSV

MCNEMAR_PATH = REPORTS_METRICS / "mcnemar_tests.csv"
MCNEMAR_JSON_PATH = REPORTS_METRICS / "mcnemar_tests.json"

BASELINE_NAMES = ["logisticregression", "linearsvc", "multinomialnb"]


def mcnemar_alpha() -> float:
    settings = load_settings()
    return float(settings.get("evaluation", {}).get("mcnemar_alpha", 0.05))


def mcnemar_conclusion(p_value: float, alpha: float | None = None) -> str:
    alpha = alpha if alpha is not None else mcnemar_alpha()
    if p_value < alpha:
        return f"diferencia significativa (alpha={alpha})"
    return f"no se rechaza H0 (alpha={alpha})"


def _rel_path(path: Path | str) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


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
    checkpoint_a: str,
) -> dict[str, Any] | None:
    path = MODELS_DIR / f"baseline_{baseline_name}.joblib"
    if not path.exists():
        return None
    pipe = joblib.load(path)
    bl_pred = pipe.predict(test["texto_modelo"])
    mcn = mcnemar_test(y_true, y_pred_mbert, bl_pred)
    return _format_row(
        "mbert",
        baseline_name,
        mcn,
        len(y_true),
        alpha,
        checkpoint_a=checkpoint_a,
        checkpoint_b=_rel_path(path),
    )


def _format_row(
    model_a: str,
    model_b: str,
    mcn: dict[str, float],
    n: int,
    alpha: float,
    *,
    checkpoint_a: str,
    checkpoint_b: str,
) -> dict[str, Any]:
    p_value = float(mcn["p_value"])
    return {
        "model_a": model_a,
        "model_b": model_b,
        "checkpoint_a": checkpoint_a,
        "checkpoint_b": checkpoint_b,
        "statistic": float(mcn["statistic"]),
        "p_value": p_value,
        "b": int(mcn["b"]),
        "c": int(mcn["c"]),
        "n": n,
        "alpha": alpha,
        "significant": p_value < alpha,
        "conclusion": mcnemar_conclusion(p_value, alpha),
    }


def export_mcnemar_json(
    df: pd.DataFrame,
    *,
    output_path: Path | None = None,
    checkpoint_a: str | None = None,
    checkpoint_b: str | None = None,
) -> Path | None:
    """Exporta metadatos de la comparación pareada mBERT vs XLM-R."""
    output_path = output_path or MCNEMAR_JSON_PATH
    xlmr_rows = df[df["model_b"] == "xlm-roberta"]
    if xlmr_rows.empty:
        return None

    row = xlmr_rows.iloc[0]
    payload: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "test_split": TEST_CSV.name,
        "alpha": float(row.get("alpha", mcnemar_alpha())),
        "n": int(row["n"]),
        "model_a": row["model_a"],
        "model_b": row["model_b"],
        "checkpoint_a": checkpoint_a or row.get("checkpoint_a"),
        "checkpoint_b": checkpoint_b or row.get("checkpoint_b"),
        "predictions_a": _rel_path(MBERT_PREDICTIONS_PATH),
        "predictions_b": _rel_path(XLMR_PREDICTIONS_PATH),
        "statistic": float(row["statistic"]),
        "p_value": float(row["p_value"]),
        "b": int(row["b"]),
        "c": int(row["c"]),
        "significant": bool(row.get("significant", False)),
        "conclusion": str(row["conclusion"]),
    }
    if MBERT_PREDICTIONS_JSON.exists():
        with open(MBERT_PREDICTIONS_JSON, encoding="utf-8") as f:
            payload["mbert_predictions_metadata"] = json.load(f)
    if XLMR_PREDICTIONS_JSON.exists():
        with open(XLMR_PREDICTIONS_JSON, encoding="utf-8") as f:
            payload["xlmr_predictions_metadata"] = json.load(f)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return output_path


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
    checkpoint_mbert = _rel_path(model_dir)
    checkpoint_xlmr = _rel_path(default_xlmr_model_dir())

    rows: list[dict[str, Any]] = []
    for name in BASELINE_NAMES:
        row = _baseline_row(y_true, y_pred_mbert, name, test, alpha, checkpoint_mbert)
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
        rows.append(
            _format_row(
                "mbert",
                "xlm-roberta",
                mcn,
                len(y_true),
                alpha,
                checkpoint_a=checkpoint_mbert,
                checkpoint_b=checkpoint_xlmr,
            )
        )

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    export_mcnemar_json(
        df,
        checkpoint_a=checkpoint_mbert,
        checkpoint_b=checkpoint_xlmr,
    )
    return df
