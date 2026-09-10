#!/usr/bin/env python
"""Ensamble de checkpoints mBERT (Fase 3)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.evaluation.ensemble import evaluate_ensemble_on_split, export_ensemble_manifest
from discurso_odio.paths import MODELS_DIR, REPORTS_METRICS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensamble mBERT por promedio de probs")
    parser.add_argument(
        "--model-dirs",
        nargs="+",
        required=True,
        help="Checkpoints mBERT a combinar",
    )
    parser.add_argument(
        "--weights",
        nargs="*",
        type=float,
        default=None,
        help="Pesos opcionales (misma longitud que model-dirs)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_METRICS / "mbert_ensemble_manifest.json",
    )
    parser.add_argument("--eval-test", action="store_true", help="También evalúa test (una vez)")
    args = parser.parse_args()

    model_dirs = [Path(p) for p in args.model_dirs]
    manifest = export_ensemble_manifest(model_dirs, weights=args.weights, output_path=args.output)
    logger.info(
        "Ensamble val F1 argmax=%.4f tuned=%.4f",
        manifest["val_metrics"]["f1_macro_argmax"],
        manifest["val_metrics"]["f1_macro_with_thresholds"],
    )

    if args.eval_test:
        test_metrics = evaluate_ensemble_on_split(model_dirs, "test", weights=args.weights)
        manifest["test_metrics"] = test_metrics
        args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Test F1 argmax=%.4f", test_metrics["f1_macro_argmax"])


if __name__ == "__main__":
    main()
