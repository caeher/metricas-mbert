#!/usr/bin/env python
"""Entrena mBERT sobre el corpus salvadoreño."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse

from discurso_odio.config import load_settings
from discurso_odio.data.loaders import load_split, load_train_excluding_adversarial_overlap
from discurso_odio.evaluation.adversarial import find_train_overlaps
from discurso_odio.models.mbert import train_mbert
from discurso_odio.paths import MODELS_DIR
from discurso_odio.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Entrena mBERT sobre el corpus salvadoreño.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directorio de salida (default: models/mbert-sv en settings)",
    )
    parser.add_argument(
        "--exclude-adversarial-overlap",
        action="store_true",
        help="Excluye filas de train cuyo texto coincide con test_adversarial.csv (T-06)",
    )
    args = parser.parse_args()

    settings = load_settings()
    set_seed(settings["project"]["seed"])

    if args.exclude_adversarial_overlap:
        train_df = load_train_excluding_adversarial_overlap()
        overlaps = find_train_overlaps()
        logger.info(
            "Train sin solapamiento adversarial: %d filas (%d excluidas)",
            len(train_df),
            len(overlaps),
        )
    else:
        train_df = load_split("train")
    val_df = load_split("val")
    logger.info("Train: %d | Val: %d", len(train_df), len(val_df))

    trainer = train_mbert(train_df, val_df, output_dir=args.output_dir)
    logger.info("Entrenamiento completado. Métricas val: %s", trainer.evaluate())


if __name__ == "__main__":
    main()
