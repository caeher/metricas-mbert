#!/usr/bin/env python
"""Calcula y exporta métricas por cortes desde predicciones oficiales."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.data.loaders import load_split
from discurso_odio.evaluation.official_mbert import (
    MBERT_PROBABILITIES_PATH,
    export_platform_traceability,
    load_official_run_metadata,
    official_checkpoint_label,
)
from discurso_odio.evaluation.predictions import (
    MBERT_PREDICTIONS_PATH,
    XLMR_PREDICTIONS_PATH,
    load_mbert_predictions,
    load_xlmr_predictions,
)
from discurso_odio.evaluation.slices import (
    export_slice_metrics,
    export_mbert_slice_artifacts,
    slice_metrics,
    slice_metrics_to_rows,
)
from discurso_odio.models.inference_contract import get_inference_contract
from discurso_odio.models.mbert import predict
from discurso_odio.paths import MODELS_DIR, REPORTS_METRICS, TEST_CSV

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_mbert_probabilities(test: pd.DataFrame) -> np.ndarray | None:
    if not MBERT_PROBABILITIES_PATH.exists():
        return None
    probs = pd.read_csv(MBERT_PROBABILITIES_PATH)
    if len(probs) != len(test):
        return None
    prob_cols = [c for c in probs.columns if c.startswith("p") and c[1:].isdigit()]
    if not prob_cols:
        return None
    return probs[prob_cols].to_numpy()


def _load_mbert_predictions(test: pd.DataFrame, model_dir: Path) -> np.ndarray:
    y_pred, status = load_mbert_predictions(test)
    if y_pred is not None:
        logger.info("mBERT: predicciones cargadas desde %s", MBERT_PREDICTIONS_PATH)
        return y_pred
    if not model_dir.exists():
        raise FileNotFoundError(
            f"No hay {MBERT_PREDICTIONS_PATH} ni checkpoint mBERT en {model_dir}"
        )
    logger.warning("Predicciones oficiales no encontradas; inferencia desde %s", model_dir)
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
    official = load_official_run_metadata()
    run_id = run_id or (official.get("run_id") if official else None)
    if not run_id:
        from uuid import uuid4

        run_id = str(uuid4())
    model_dir = model_dir or MODELS_DIR / "mbert-sv"
    checkpoint = official_checkpoint_label(official, model_dir)

    all_rows: list[dict] = []

    y_pred_mbert = _load_mbert_predictions(test, model_dir)
    y_prob = _load_mbert_probabilities(test)
    mbert_slices = slice_metrics(test, y_true, y_pred_mbert, y_prob)
    all_rows.extend(
        slice_metrics_to_rows(
            mbert_slices,
            model="mbert",
            checkpoint=checkpoint,
            split=TEST_CSV.name,
            run_id=run_id,
            n_test=len(test),
        )
    )
    logger.info(
        "mBERT: %d cortes calculados (checkpoint=%s, run_id=%s)",
        len(mbert_slices),
        checkpoint,
        run_id,
    )

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
    result.attrs["run_id"] = run_id
    result.attrs["checkpoint"] = checkpoint
    return result


def main():
    parser = argparse.ArgumentParser(description="Calcula métricas por cortes de evaluación.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODELS_DIR / "mbert-sv",
        help="Directorio del checkpoint mBERT (fallback si no hay predicciones oficiales)",
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

    official = load_official_run_metadata()
    run_id = slice_df.attrs.get("run_id")
    checkpoint = slice_df.attrs.get("checkpoint")

    extra = {
        "xlmr_status": slice_df.attrs.get("xlmr_status", "unknown"),
        "run_id": run_id,
        "checkpoint": checkpoint,
        "official_run_id": official.get("run_id") if official else None,
        **get_inference_contract().metadata(),
    }

    # Exportar mBERT con artefactos completos (per-class + dialect bias)
    test = load_split("test")
    y_true = test["label"].values
    y_pred, _ = load_mbert_predictions(test)
    if y_pred is not None:
        y_prob = _load_mbert_probabilities(test)
        export_mbert_slice_artifacts(
            test,
            y_true,
            y_pred,
            y_prob,
            REPORTS_METRICS,
            run_id=run_id,
            checkpoint=checkpoint,
            split=TEST_CSV.name,
            extra_metadata=extra,
            merge=True,
        )
        slice_df = pd.read_csv(REPORTS_METRICS / "slice_metrics.csv")
        mbert_only = slice_df[slice_df["model"] == "mbert"]
        export_platform_traceability(slice_df=mbert_only, official=official)
    else:
        csv_path, json_path = export_slice_metrics(slice_df, REPORTS_METRICS, extra_metadata=extra)
        logger.info("Exportado: %s", csv_path)
        logger.info("Exportado: %s", json_path)

    if slice_df.attrs.get("xlmr_status") == "ok":
        xlmr_rows = slice_df[slice_df["model"] == "xlm-roberta"]
        if not xlmr_rows.empty:
            export_slice_metrics(xlmr_rows, REPORTS_METRICS, extra_metadata=extra, merge=True)

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
