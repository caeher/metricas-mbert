#!/usr/bin/env python
"""Calcula y exporta métricas por cortes desde predicciones."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.data.loaders import load_split
from discurso_odio.evaluation.slices import (
    export_slice_metrics,
    slice_metrics,
    slice_metrics_to_rows,
)
from discurso_odio.models.mbert import predict
from discurso_odio.paths import MODELS_DIR, REPORTS_METRICS, TEST_CSV

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

XLMR_PREDICTIONS_PATH = REPORTS_METRICS / "xlmr_test_predictions.csv"


def _validate_predictions(
    test: pd.DataFrame,
    preds: pd.DataFrame,
    pred_col: str,
    model_name: str,
) -> np.ndarray:
    if len(preds) != len(test):
        raise ValueError(
            f"{model_name}: se esperaban {len(test)} predicciones, se recibieron {len(preds)}"
        )

    if "id" in preds.columns and "ID" in test.columns:
        merged = test[["ID"]].merge(preds, left_on="ID", right_on="id", how="left")
        if merged[pred_col].isna().any():
            raise ValueError(f"{model_name}: predicciones sin ID correspondiente en test")
        y_pred = merged[pred_col].to_numpy()
    else:
        y_pred = preds[pred_col].to_numpy()

    y_pred = np.asarray(y_pred, dtype=int)
    if y_pred.min() < 0 or y_pred.max() > 3:
        raise ValueError(f"{model_name}: etiquetas fuera del rango 0-3")
    return y_pred


def _load_mbert_predictions(test: pd.DataFrame, model_dir: Path) -> np.ndarray:
    if not model_dir.exists():
        raise FileNotFoundError(f"Checkpoint mBERT no encontrado: {model_dir}")
    y_pred, _ = predict(test["texto_modelo"].tolist(), model_dir)
    return np.asarray(y_pred, dtype=int)


def _load_xlmr_predictions(test: pd.DataFrame) -> tuple[np.ndarray | None, str]:
    if not XLMR_PREDICTIONS_PATH.exists():
        return None, "predictions_not_found"

    preds = pd.read_csv(XLMR_PREDICTIONS_PATH)
    pred_col = "y_pred_xlmr"
    if pred_col not in preds.columns:
        raise ValueError(f"Columna requerida ausente en {XLMR_PREDICTIONS_PATH}: {pred_col}")

    y_pred = _validate_predictions(test, preds, pred_col, "XLM-R")
    return y_pred, "ok"


def compute_all(
    *,
    model_dir: Path | None = None,
    include_xlmr: bool = True,
    run_id: str | None = None,
) -> pd.DataFrame:
    test = load_split("test")
    y_true = test["label"].values
    run_id = run_id or str(uuid4())
    model_dir = model_dir or MODELS_DIR / "mbert-sv"

    all_rows: list[dict] = []

    y_pred_mbert = _load_mbert_predictions(test, model_dir)
    mbert_slices = slice_metrics(test, y_true, y_pred_mbert)
    all_rows.extend(
        slice_metrics_to_rows(
            mbert_slices,
            model="mbert",
            checkpoint=str(model_dir),
            split=TEST_CSV.name,
            run_id=run_id,
            n_test=len(test),
        )
    )
    logger.info("mBERT: %d cortes calculados desde %s", len(mbert_slices), model_dir)

    xlmr_status = "skipped"
    if include_xlmr:
        y_pred_xlmr, xlmr_status = _load_xlmr_predictions(test)
        if y_pred_xlmr is not None:
            xlmr_slices = slice_metrics(test, y_true, y_pred_xlmr)
            all_rows.extend(
                slice_metrics_to_rows(
                    xlmr_slices,
                    model="xlm-roberta",
                    checkpoint=str(XLMR_PREDICTIONS_PATH),
                    split=TEST_CSV.name,
                    run_id=run_id,
                    n_test=len(test),
                )
            )
            logger.info(
                "XLM-R: %d cortes calculados desde %s",
                len(xlmr_slices),
                XLMR_PREDICTIONS_PATH,
            )

    result = pd.DataFrame(all_rows)
    result.attrs["xlmr_status"] = xlmr_status
    return result


def main():
    parser = argparse.ArgumentParser(description="Calcula métricas por cortes de evaluación.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODELS_DIR / "mbert-sv",
        help="Directorio del checkpoint mBERT",
    )
    parser.add_argument(
        "--skip-xlmr",
        action="store_true",
        help="No intentar cargar predicciones XLM-R",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Identificador de ejecución")
    args = parser.parse_args()

    slice_df = compute_all(
        model_dir=args.model_dir,
        include_xlmr=not args.skip_xlmr,
        run_id=args.run_id,
    )

    extra = {"xlmr_status": slice_df.attrs.get("xlmr_status", "unknown")}
    csv_path, json_path = export_slice_metrics(slice_df, REPORTS_METRICS, extra_metadata=extra)

    logger.info("Exportado: %s", csv_path)
    logger.info("Exportado: %s", json_path)
    for _, row in slice_df.iterrows():
        logger.info(
            "%s [%s] n=%d F1=%.4f",
            row["model"],
            row["slice"],
            row["n"],
            row["f1_macro"],
        )


if __name__ == "__main__":
    main()
