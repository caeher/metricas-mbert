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
from discurso_odio.evaluation.predictions import (
    XLMR_PREDICTIONS_PATH,
    load_xlmr_predictions,
)
from discurso_odio.evaluation.slices import (
    export_slice_metrics,
    slice_metrics,
    slice_metrics_to_rows,
)
from discurso_odio.models.inference_contract import get_inference_contract
from discurso_odio.models.mbert import predict
from discurso_odio.paths import MODELS_DIR, REPORTS_METRICS, TEST_CSV

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_mbert_predictions(test: pd.DataFrame, model_dir: Path) -> np.ndarray:
    if not model_dir.exists():
        raise FileNotFoundError(f"Checkpoint mBERT no encontrado: {model_dir}")
    y_pred, _ = predict(test["texto_modelo"].tolist(), model_dir)
    return np.asarray(y_pred, dtype=int)


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
        y_pred_xlmr, xlmr_status = load_xlmr_predictions(test)
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

    extra = {
        "xlmr_status": slice_df.attrs.get("xlmr_status", "unknown"),
        **get_inference_contract().metadata(),
    }
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
