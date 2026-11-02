#!/usr/bin/env python
"""Evalúa gates F1 0.80 (primario + robustez secundaria)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.evaluation.f1_gates import evaluate_gates
from discurso_odio.models.mbert import default_mbert_model_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gates oficiales F1-macro 0.80")
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument(
        "--use-existing-metrics",
        action="store_true",
        help="Usar métricas oficiales ya exportadas si no hay checkpoint",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Salir con código 2 si primary_pass es false.",
    )
    args = parser.parse_args()

    model_dir = args.model_dir or default_mbert_model_dir()
    exists = model_dir.exists()
    gates = evaluate_gates(
        model_dir if exists else None,
        run_official=exists,
        use_existing_metrics=args.use_existing_metrics or not exists,
    )
    logger.info("%s", gates["notes"])
    logger.info("Adoptado=%s test_f1=%.4f", gates["adopted"], gates["test_f1_macro"])
    if args.strict and not gates["primary_pass"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
