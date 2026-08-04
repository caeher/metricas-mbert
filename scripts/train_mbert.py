#!/usr/bin/env python
"""Entrena mBERT sobre el corpus salvadoreño."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.data.loaders import load_split
from discurso_odio.models.mbert import train_mbert
from discurso_odio.utils.seed import set_seed
from discurso_odio.config import load_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    settings = load_settings()
    set_seed(settings["project"]["seed"])

    train_df = load_split("train")
    val_df = load_split("val")
    logger.info("Train: %d | Val: %d", len(train_df), len(val_df))

    trainer = train_mbert(train_df, val_df)
    logger.info("Entrenamiento completado. Métricas val: %s", trainer.evaluate())


if __name__ == "__main__":
    main()
