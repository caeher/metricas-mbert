#!/usr/bin/env python
"""Entrena mBERT sobre el corpus salvadoreño."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse

from discurso_odio.config import load_settings
from discurso_odio.data.loaders import (
    compute_sample_loss_weights,
    get_b2_class_weights,
    load_split,
    load_train_excluding_adversarial_overlap,
)
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
        help="Excluye filas de train cuyo texto coincide con test_adversarial.csv (T-06/B2)",
    )
    parser.add_argument(
        "--b2",
        action="store_true",
        help="Modo B2: pesos de clase de metadata.json, rebalanceo de amenazas sintéticas",
    )
    parser.add_argument(
        "--threat-synthetic-effective-n",
        type=float,
        default=90.0,
        help="Masa efectiva total de amenazas sintéticas en la pérdida (default: 90)",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.05,
        help="Label smoothing para CrossEntropyLoss (default: 0.05; 0 desactiva)",
    )
    args = parser.parse_args()

    settings = load_settings()
    set_seed(settings["project"]["seed"])

    use_b2 = args.b2 or args.exclude_adversarial_overlap

    if args.exclude_adversarial_overlap or args.b2:
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

    sample_weights = None
    class_weights = None
    if use_b2:
        class_weights = get_b2_class_weights()
        sample_weights = compute_sample_loss_weights(
            train_df,
            threat_synthetic_effective_n=args.threat_synthetic_effective_n,
        )
        logger.info(
            "B2: pesos_clase=%s | amenazas sintéticas peso=%.4f (n=%d, efectivo=%.0f)",
            class_weights,
            float(sample_weights[sample_weights < 1.0].iloc[0]) if (sample_weights < 1.0).any() else 1.0,
            int((sample_weights < 1.0).sum()),
            args.threat_synthetic_effective_n,
        )

    trainer = train_mbert(
        train_df,
        val_df,
        output_dir=args.output_dir,
        class_weights=class_weights,
        sample_weights=sample_weights,
        label_smoothing=args.label_smoothing if use_b2 else 0.0,
        use_b2_class_weights=use_b2,
    )
    logger.info("Entrenamiento completado. Métricas val: %s", trainer.evaluate())


if __name__ == "__main__":
    main()
