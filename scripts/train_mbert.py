#!/usr/bin/env python
"""Entrena mBERT sobre el corpus salvadoreño (Fases 1–2 del plan F1 0.80)."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse

from discurso_odio.config import load_settings
from discurso_odio.data.loaders import (
    compute_sample_loss_weights,
    default_facebook_boost_weights,
    filter_train_by_synthetic_policy,
    get_b2_class_weights,
    get_class_weights,
    get_release_class_weights,
    load_release_split,
    load_split,
    load_train_excluding_adversarial_overlap,
)
from discurso_odio.evaluation.adversarial import find_train_overlaps
from discurso_odio.models.mbert import MbertTrainingConfig, train_mbert
from discurso_odio.paths import MODELS_DIR
from discurso_odio.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _parse_platform_boost(spec: str | None) -> dict[tuple[str, int], float] | None:
    if not spec or spec.lower() == "none":
        return None
    if spec.lower() == "default":
        return default_facebook_boost_weights()
    boosts: dict[tuple[str, int], float] = {}
    for part in spec.split(","):
        platform, label_s, factor_s = part.split(":")
        boosts[(platform.strip(), int(label_s))] = float(factor_s)
    return boosts


def main():
    parser = argparse.ArgumentParser(description="Entrena mBERT sobre el corpus salvadoreño.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--exclude-adversarial-overlap", action="store_true")
    parser.add_argument("--b2", action="store_true", help="Pesos B2 de metadata.json")
    parser.add_argument(
        "--synthetic-policy",
        choices=["b2_full", "real_only", "b2_no_template_heavy"],
        default="b2_full",
    )
    parser.add_argument("--threat-synthetic-effective-n", type=float, default=90.0)
    parser.add_argument(
        "--synthetic-effective-fraction",
        type=float,
        default=None,
        help="Downweight global de sintéticos (0.25–0.5 recomendado)",
    )
    parser.add_argument(
        "--platform-class-boost",
        default="none",
        help="none | default | Facebook:2:2.5,Facebook:3:1.8",
    )
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument(
        "--loss-type",
        choices=["ce", "focal", "ordinal"],
        default="ce",
    )
    parser.add_argument("--ordinal-weight", type=float, default=0.3)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--layerwise-lr-decay", type=float, default=None)
    parser.add_argument("--classifier-dropout", type=float, default=None)
    parser.add_argument("--use-v1-class-weights", action="store_true")
    parser.add_argument("--init-model-dir", type=Path, default=None, help="Checkpoint DAPT previo")
    parser.add_argument(
        "--release",
        type=str,
        default=None,
        help="Release ISSUE-08 en versiones/vN/dataset o data/releases/<id> (alias v1…v4).",
    )
    args = parser.parse_args()

    settings = load_settings()
    seed = args.seed or settings["project"]["seed"]
    set_seed(seed)

    use_b2 = args.b2 or args.exclude_adversarial_overlap

    if args.release:
        train_df = load_release_split(args.release, "train")
        val_df = load_release_split(args.release, "val")
        logger.info("Release %s: train=%d val=%d", args.release, len(train_df), len(val_df))
    elif args.exclude_adversarial_overlap or args.b2:
        train_df = load_train_excluding_adversarial_overlap()
        overlaps = find_train_overlaps()
        logger.info(
            "Train sin solapamiento adversarial: %d filas (%d excluidas)",
            len(train_df),
            len(overlaps),
        )
        val_df = load_split("val")
    else:
        train_df = load_split("train")
        val_df = load_split("val")

    train_df = filter_train_by_synthetic_policy(train_df, args.synthetic_policy)
    logger.info(
        "Train: %d (policy=%s) | Val: %d",
        len(train_df),
        args.synthetic_policy,
        len(val_df),
    )

    label_smoothing = args.label_smoothing
    if label_smoothing is None:
        label_smoothing = 0.05 if use_b2 else 0.0

    class_weights = None
    if args.use_v1_class_weights:
        class_weights = get_class_weights(train_df)
    elif args.release:
        class_weights = get_release_class_weights(args.release)
    elif use_b2:
        class_weights = get_b2_class_weights()

    platform_boost = _parse_platform_boost(args.platform_class_boost)
    sample_weights = compute_sample_loss_weights(
        train_df,
        threat_synthetic_effective_n=args.threat_synthetic_effective_n,
        synthetic_effective_fraction=args.synthetic_effective_fraction,
        platform_class_boost=platform_boost,
    )

    training_config = MbertTrainingConfig(
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        seed=seed,
        label_smoothing=label_smoothing,
        loss_type=args.loss_type,
        ordinal_weight=args.ordinal_weight,
        focal_gamma=args.focal_gamma,
        classifier_dropout=args.classifier_dropout,
        layerwise_lr_decay=args.layerwise_lr_decay,
    )

    output_dir = args.output_dir or (
        MODELS_DIR / "mbert-sv-f1" if use_b2 else MODELS_DIR / "mbert-sv"
    )

    training_recipe = {
        "release": args.release,
        "synthetic_policy": args.synthetic_policy,
        "synthetic_effective_fraction": args.synthetic_effective_fraction,
        "threat_synthetic_effective_n": args.threat_synthetic_effective_n,
        "platform_class_boost": args.platform_class_boost,
        "label_smoothing": label_smoothing,
        "loss_type": args.loss_type,
        "ordinal_weight": args.ordinal_weight,
        "focal_gamma": args.focal_gamma,
        "layerwise_lr_decay": args.layerwise_lr_decay,
        "classifier_dropout": args.classifier_dropout,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "seed": seed,
        "use_b2": use_b2,
        "exclude_adversarial_overlap": args.exclude_adversarial_overlap,
        "use_v1_class_weights": args.use_v1_class_weights,
        "init_model_dir": str(args.init_model_dir) if args.init_model_dir else None,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
    }

    trainer = train_mbert(
        train_df,
        val_df,
        output_dir=output_dir,
        class_weights=class_weights,
        sample_weights=sample_weights,
        label_smoothing=label_smoothing,
        use_b2_class_weights=use_b2 and not args.use_v1_class_weights,
        training_config=training_config,
        init_model_dir=args.init_model_dir,
        training_recipe=training_recipe,
    )
    logger.info("Entrenamiento completado. Métricas val: %s", trainer.evaluate())


if __name__ == "__main__":
    main()
