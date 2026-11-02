#!/usr/bin/env python
"""Exporta predicciones XLM-R alineadas con data/processed/test.csv."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from discurso_odio.data.loaders import load_split
from discurso_odio.evaluation.predictions import (
    XLMR_PRED_COL,
    XLMR_PREDICTIONS_JSON,
    XLMR_PREDICTIONS_PATH,
    export_predictions_csv,
    validate_prediction_vector,
)
from discurso_odio.models.inference_contract import resolve_inference_contract
from discurso_odio.models.xlmr import default_xlmr_model_dir, ensure_checkpoint_contract, predict
from discurso_odio.paths import REPORTS_METRICS, TEST_CSV

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def export_xlmr_predictions(
    *,
    model_dir: Path | None = None,
    max_length: int | None = None,
    output_csv: Path | None = None,
    output_json: Path | None = None,
) -> tuple[Path, Path]:
    model_dir = model_dir or default_xlmr_model_dir()
    contract = resolve_inference_contract(model_dir)
    max_length = contract.validate_max_length(max_length)
    output_csv = output_csv or XLMR_PREDICTIONS_PATH
    output_json = output_json or XLMR_PREDICTIONS_JSON

    ensure_checkpoint_contract(model_dir)
    test = load_split("test")
    logger.info("Inferencia XLM-R sobre %d instancias desde %s", len(test), model_dir)

    text_col = contract.text_column
    y_pred, _ = predict(
        test[text_col].tolist(),
        model_dir,
        max_length=max_length,
        normalize=True,
    )
    validate_prediction_vector(
        test,
        __build_pred_df(test, y_pred),
        XLMR_PRED_COL,
        "XLM-R",
    )

    csv_path, json_path = export_predictions_csv(
        test,
        y_pred,
        csv_path=output_csv,
        json_path=output_json,
        pred_col=XLMR_PRED_COL,
        model="xlm-roberta",
        checkpoint=str(model_dir.resolve()),
        max_length=max_length,
        split=TEST_CSV.name,
        extra_metadata=contract.metadata(),
    )
    logger.info("Exportado: %s", csv_path)
    logger.info("Exportado: %s", json_path)
    return csv_path, json_path


def __build_pred_df(test, y_pred):
    import pandas as pd

    return pd.DataFrame({"id": test["ID"].values, XLMR_PRED_COL: y_pred})


def main():
    parser = argparse.ArgumentParser(description="Exporta predicciones XLM-R sobre test.csv")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Directorio del checkpoint XLM-R (default: XLMR/02_modelo_prototipo/modelo_entrenado)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Longitud máxima de tokenización (debe coincidir con inference_contract)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=XLMR_PREDICTIONS_PATH,
        help="Ruta del CSV de salida",
    )
    args = parser.parse_args()

    REPORTS_METRICS.mkdir(parents=True, exist_ok=True)
    export_xlmr_predictions(
        model_dir=args.model_dir,
        max_length=args.max_length,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
