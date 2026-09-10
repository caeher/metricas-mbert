#!/usr/bin/env python
"""Ajuste de umbrales por clase en val (Fase 3)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.evaluation.thresholds import tune_thresholds_on_val
from discurso_odio.models.mbert import default_mbert_model_dir
from discurso_odio.paths import REPORTS_METRICS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ajusta umbrales OvR en val")
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_METRICS / "mbert_thresholds.json",
    )
    parser.add_argument(
        "--grid",
        default="0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50",
        help="Valores de umbral por clase",
    )
    args = parser.parse_args()

    model_dir = args.model_dir or default_mbert_model_dir()
    grid = [float(x) for x in args.grid.split(",")]
    result = tune_thresholds_on_val(model_dir, grid=grid, output_path=args.output)
    logger.info(
        "Umbrales: %s | val F1 %.4f → %.4f (+%.2f pp)",
        result["thresholds"],
        result["val_f1_macro_baseline_argmax"],
        result["val_f1_macro_tuned"],
        result["val_gain_pp"] * 100,
    )


if __name__ == "__main__":
    main()
