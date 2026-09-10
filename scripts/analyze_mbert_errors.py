#!/usr/bin/env python
"""Análisis de errores mBERT B2 sobre test oficial (Fase 0)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.evaluation.error_analysis import export_error_analysis
from discurso_odio.paths import REPORTS_METRICS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analiza errores mBERT en test.csv")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=REPORTS_METRICS / "mbert_test_predictions.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPORTS_METRICS / "error_analysis_b2.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPORTS_METRICS / "error_analysis_b2_errors.csv",
    )
    parser.add_argument("--model", default="mbert-b2")
    parser.add_argument("--checkpoint", default="checkpoint-1644")
    args = parser.parse_args()

    report = export_error_analysis(
        output_json=args.output_json,
        output_csv=args.output_csv,
        predictions_path=args.predictions,
        model=args.model,
        checkpoint=args.checkpoint,
    )
    summary = {k: report[k] for k in ("n_test", "n_errors", "f1_macro", "adjacent_error_pct")}
    logger.info("Análisis guardado: %s", args.output_json)
    logger.info("Resumen: %s", json.dumps(summary, ensure_ascii=False))
    for rec in report.get("recommendations", []):
        logger.info("→ %s", rec)


if __name__ == "__main__":
    main()
