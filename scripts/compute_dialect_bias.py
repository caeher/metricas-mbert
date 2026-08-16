#!/usr/bin/env python
"""Calcula y exporta la sonda de sesgo dialectal desde predicciones."""

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
from discurso_odio.evaluation.slices import dialect_bias_probe, export_dialect_bias
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

    frames: list[pd.DataFrame] = []

    y_pred_mbert = _load_mbert_predictions(test, model_dir)
    frames.append(
        dialect_bias_probe(
            test,
            y_true,
            y_pred_mbert,
            model="mbert",
            checkpoint=str(model_dir),
            run_id=run_id,
            split=TEST_CSV.name,
        )
    )
    logger.info("mBERT: sonda dialectal calculada desde %s", model_dir)

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
    return result


def main():
    parser = argparse.ArgumentParser(description="Calcula la sonda de sesgo dialectal.")
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

    probe_df = compute_all(
        model_dir=args.model_dir,
        include_xlmr=not args.skip_xlmr,
        run_id=args.run_id,
    )

    extra = {
        "xlmr_status": probe_df.attrs.get("xlmr_status", "unknown"),
        **get_inference_contract().metadata(),
    }
    csv_path, json_path = export_dialect_bias(probe_df, REPORTS_METRICS, extra_metadata=extra)

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
