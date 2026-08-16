#!/usr/bin/env python
"""Exporta predicciones mBERT alineadas con data/processed/test.csv."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.data.loaders import load_split
from discurso_odio.evaluation.predictions import (
    MBERT_PRED_COL,
    MBERT_PREDICTIONS_JSON,
    MBERT_PREDICTIONS_PATH,
    export_predictions_csv,
    validate_prediction_vector,
)
from discurso_odio.models.inference_contract import get_inference_contract
from discurso_odio.models.mbert import predict
from discurso_odio.paths import MODELS_DIR, REPORTS_METRICS, TEST_CSV

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def export_mbert_predictions(
    *,
    model_dir: Path | None = None,
    max_length: int | None = None,
    output_csv: Path | None = None,
    output_json: Path | None = None,
) -> tuple[Path, Path]:
    contract = get_inference_contract()
    model_dir = model_dir or MODELS_DIR / "mbert-sv"
    max_length = contract.validate_max_length(max_length)
    output_csv = output_csv or MBERT_PREDICTIONS_PATH
    output_json = output_json or MBERT_PREDICTIONS_JSON

    test = load_split("test")
    logger.info("Inferencia mBERT sobre %d instancias desde %s", len(test), model_dir)

    y_pred, _ = predict(test["texto_modelo"].tolist(), model_dir)
    import pandas as pd

    pred_df = pd.DataFrame({"id": test["ID"].values, MBERT_PRED_COL: y_pred})
    validate_prediction_vector(test, pred_df, MBERT_PRED_COL, "mBERT")

    csv_path, json_path = export_predictions_csv(
        test,
        y_pred,
        csv_path=output_csv,
        json_path=output_json,
        pred_col=MBERT_PRED_COL,
        model="mbert",
        checkpoint=str(model_dir.resolve()),
        max_length=max_length,
        split=TEST_CSV.name,
        extra_metadata=contract.metadata(),
    )
    logger.info("Exportado: %s", csv_path)
    logger.info("Exportado: %s", json_path)
    return csv_path, json_path


def main():
    parser = argparse.ArgumentParser(description="Exporta predicciones mBERT sobre test.csv")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODELS_DIR / "mbert-sv",
        help="Directorio del checkpoint mBERT",
    )
    args = parser.parse_args()

    REPORTS_METRICS.mkdir(parents=True, exist_ok=True)
    export_mbert_predictions(model_dir=args.model_dir)


if __name__ == "__main__":
    main()
