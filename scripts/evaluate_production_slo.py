#!/usr/bin/env python
"""Evalúa un checkpoint contra production_slo_v1 (score operativo ISSUE-14)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.evaluation.production_slo import (
    DEFAULT_SLO_PATH,
    RESULT_JSON,
    evaluate_production_slo,
)
from discurso_odio.paths import PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Score operativo production_slo_v1")
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument(
        "--slo",
        type=Path,
        default=DEFAULT_SLO_PATH,
        help="Ruta al JSON del contrato SLO",
    )
    parser.add_argument(
        "--skip-adversarial",
        action="store_true",
        help="No re-ejecutar batería adversarial (usa artefactos previos si existen)",
    )
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Exit 2 si no cumple slo_pass",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Ruta alternativa para JSON de resultado (default: production_slo_evaluation.json)",
    )
    args = parser.parse_args()

    model_dir = args.model_dir
    if model_dir is not None:
        model_dir = Path(model_dir)
        if not model_dir.is_absolute():
            model_dir = PROJECT_ROOT / model_dir

    result = evaluate_production_slo(
        model_dir=model_dir,
        slo_path=args.slo,
        run_adversarial=not args.skip_adversarial and model_dir is not None,
        use_existing_metrics=True,
        output_path=args.output,
    )
    logger.info("%s", result["notes"])
    logger.info(
        "operational_score=%s slo_pass=%s test_f1=%s",
        result["operational_score"],
        result["slo_pass"],
        result["test_f1_macro"],
    )
    if args.require_pass and not result["slo_pass"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
