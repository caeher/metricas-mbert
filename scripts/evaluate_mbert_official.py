#!/usr/bin/env python
"""Evaluación oficial mBERT: una corrida, artefactos alineados (T-04)."""

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
    parser = argparse.ArgumentParser(description="Evalúa mBERT oficial sobre test.csv")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Directorio del checkpoint (default: models/mbert-sv)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Longitud máxima (default: inference_contract)",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Réplicas bootstrap para IC 95%%",
    )
    parser.add_argument(
        "--skip-official-f1-check",
        action="store_true",
        help="No exigir F1≈0.761 del checkpoint B2 empaquetado (baselines v3 u otros).",
    )
    args = parser.parse_args()

    REPORTS_METRICS.mkdir(parents=True, exist_ok=True)
    run = evaluate_mbert_official(
        model_dir=args.model_dir,
        max_length=args.max_length,
        n_bootstrap=args.n_bootstrap,
        skip_official_f1_check=args.skip_official_f1_check,
    )
    logger.info(
        "Evaluación oficial completada: F1-macro=%.4f [%.4f, %.4f] run_id=%s",
        run["official_f1_macro"],
        run["f1_bootstrap_lo"],
        run["f1_bootstrap_hi"],
        run["run_id"],
    )
    logger.info("Artefactos en %s", REPORTS_METRICS)


if __name__ == "__main__":
    main()
