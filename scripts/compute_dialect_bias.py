#!/usr/bin/env python
"""Calcula y exporta la sonda de sesgo dialectal desde predicciones oficiales."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.data.loaders import load_split
from discurso_odio.evaluation.official_mbert import (
    load_official_run_metadata,
    official_checkpoint_label,
)
from discurso_odio.evaluation.predictions import (
    MBERT_PREDICTIONS_PATH,
    XLMR_PREDICTIONS_PATH,
    load_mbert_predictions,
    load_xlmr_predictions,
)
from discurso_odio.evaluation.slices import dialect_bias_probe, export_dialect_bias
from discurso_odio.models.inference_contract import get_inference_contract
from discurso_odio.models.mbert import predict
from discurso_odio.paths import MODELS_DIR, REPORTS_METRICS, TEST_CSV

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


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
    if not run_id:
        run_id = official.get("run_id") if official else None
    if not run_id:
        from uuid import uuid4

        run_id = str(uuid4())
    model_dir = model_dir or MODELS_DIR / "mbert-sv"
    checkpoint = official_checkpoint_label(official, model_dir)

    frames: list[pd.DataFrame] = []

    y_pred_mbert = _load_mbert_predictions(test, model_dir)
    frames.append(
        dialect_bias_probe(
            test,
            y_true,
            y_pred_mbert,
            model="mbert",
            checkpoint=checkpoint,
            run_id=run_id,
            split=TEST_CSV.name,
        )
    )
    logger.info("mBERT: sonda dialectal (checkpoint=%s, run_id=%s)", checkpoint, run_id)

    xlmr_status = "skipped"
    if include_xlmr:
        y_pred_xlmr, xlmr_status = load_xlmr_predictions(test)
        if y_pred_xlmr is not None:
            frames.append(
                dialect_bias_probe(
                    test,
                    y_true,
                    y_pred_xlmr,
                    model="xlm-roberta",
                    checkpoint=str(XLMR_PREDICTIONS_PATH),
                    run_id=run_id,
                    split=TEST_CSV.name,
                )
            )
            logger.info("XLM-R: sonda dialectal calculada desde %s", XLMR_PREDICTIONS_PATH)

    result = pd.concat(frames, ignore_index=True)
    result.attrs["xlmr_status"] = xlmr_status
    result.attrs["run_id"] = run_id
    return result


def main():
    parser = argparse.ArgumentParser(description="Calcula la sonda de sesgo dialectal.")
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

    probe_df = compute_all(
        model_dir=args.model_dir,
        include_xlmr=not args.skip_xlmr,
        run_id=args.run_id,
    )

    official = load_official_run_metadata()
    extra = {
        "xlmr_status": probe_df.attrs.get("xlmr_status", "unknown"),
        "run_id": probe_df.attrs.get("run_id"),
        "official_run_id": official.get("run_id") if official else None,
        **get_inference_contract().metadata(),
    }
    csv_path, json_path = export_dialect_bias(
        probe_df, REPORTS_METRICS, extra_metadata=extra, merge=True
    )

    logger.info("Exportado: %s", csv_path)
    logger.info("Exportado: %s", json_path)
    for _, row in probe_df.iterrows():
        logger.info(
            "%s [%s]: %d/%d FP (FPR=%.4f)",
            row["model"],
            row["group"],
            row["fp"],
            row["n"],
            row["fpr"] if pd.notna(row["fpr"]) else float("nan"),
        )


if __name__ == "__main__":
    main()
