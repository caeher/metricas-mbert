#!/usr/bin/env python
"""Exporta predicciones mBERT alineadas con data/processed/test.csv."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.evaluation.official_mbert import evaluate_mbert_official
from discurso_odio.paths import REPORTS_METRICS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Exporta predicciones mBERT oficiales sobre test.csv"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Directorio del checkpoint mBERT (default: models/mbert-sv)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Longitud máxima de tokenización (debe coincidir con inference_contract)",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Alias de evaluate_mbert_official (misma corrida completa)",
    )
    args = parser.parse_args()

    REPORTS_METRICS.mkdir(parents=True, exist_ok=True)
    run = evaluate_mbert_official(
        model_dir=args.model_dir,
        max_length=args.max_length,
    )
    logger.info(
        "Exportación oficial: F1-macro=%.4f -> %s",
        run["official_f1_macro"],
        REPORTS_METRICS / "mbert_test_predictions.csv",
    )
    _ = args.metrics_only


if __name__ == "__main__":
    main()
