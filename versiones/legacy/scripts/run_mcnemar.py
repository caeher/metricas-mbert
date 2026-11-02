#!/usr/bin/env python
"""Ejecuta pruebas de McNemar pareadas mBERT vs baselines y XLM-R."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from discurso_odio.evaluation.mcnemar import run_mcnemar_tests
from discurso_odio.paths import MODELS_DIR, REPORTS_METRICS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ejecuta pruebas de McNemar pareadas.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODELS_DIR / "mbert-sv",
        help="Directorio del checkpoint mBERT",
    )
    parser.add_argument(
        "--allow-missing-xlmr",
        action="store_true",
        help="No fallar si faltan predicciones XLM-R (solo baselines)",
    )
    args = parser.parse_args()

    REPORTS_METRICS.mkdir(parents=True, exist_ok=True)
    df = run_mcnemar_tests(
        model_dir=args.model_dir,
        require_xlmr=not args.allow_missing_xlmr,
    )
    logger.info("Exportado: %s (%d filas)", REPORTS_METRICS / "mcnemar_tests.csv", len(df))
    for _, row in df.iterrows():
        logger.info(
            "%s vs %s: p=%.6f b=%d c=%d -> %s",
            row["model_a"],
            row["model_b"],
            row["p_value"],
            row["b"],
            row["c"],
            row["conclusion"],
        )


if __name__ == "__main__":
    main()
