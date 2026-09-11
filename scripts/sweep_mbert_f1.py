#!/usr/bin/env python
"""Barrido de recetas mBERT para F1-macro (Fase 1–2). Selección solo en val."""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from discurso_odio.config import load_settings
from discurso_odio.data.loaders import (
    compute_sample_loss_weights,
    default_facebook_boost_weights,
    filter_train_by_synthetic_policy,
    get_b2_class_weights,
    get_class_weights,
    load_split,
    load_train_excluding_adversarial_overlap,
)
from discurso_odio.models.mbert import MbertTrainingConfig, train_mbert
from discurso_odio.paths import MODELS_DIR, REPORTS_METRICS
from discurso_odio.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SWEEP_CSV = REPORTS_METRICS / "mbert_sweep.csv"


def _load_existing_metrics(out_dir: Path) -> dict[str, float] | None:
    """Lee métricas de val de un entrenamiento ya completado."""
    metrics_path = out_dir / "train_metrics.json"
    if not metrics_path.exists():
        return None
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "val_f1_macro": float(data.get("eval_f1_macro", 0)),
        "val_accuracy": float(data.get("eval_accuracy", 0)),
    }


def _build_grid(args) -> list[dict]:
    lrs = [float(x) for x in args.learning_rates.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    policies = args.synthetic_policies.split(",")
    smoothings = [float(x) for x in args.label_smoothings.split(",")]
    loss_types = args.loss_types.split(",")
    synth_fracs = [None] if not args.synthetic_fractions else [
        float(x) for x in args.synthetic_fractions.split(",")
    ]

    combos = []
    for lr, seed, policy, smooth, loss, synth_frac in itertools.product(
        lrs, seeds, policies, smoothings, loss_types, synth_fracs
    ):
        combos.append(
            {
                "learning_rate": lr,
                "seed": seed,
                "synthetic_policy": policy,
                "label_smoothing": smooth,
                "loss_type": loss,
                "synthetic_effective_fraction": synth_frac,
            }
        )
    return combos[: args.max_runs]


def main() -> None:
    parser = argparse.ArgumentParser(description="Barrido mBERT (val F1-macro)")
    parser.add_argument("--learning-rates", default="1e-5,2e-5,3e-5")
    parser.add_argument("--seeds", default="42,7,13")
    parser.add_argument(
        "--synthetic-policies",
        default="b2_full,b2_no_template_heavy,real_only",
    )
    parser.add_argument("--label-smoothings", default="0.0,0.05")
    parser.add_argument("--loss-types", default="ce,ordinal,focal")
    parser.add_argument("--synthetic-fractions", default="0.35,0.5")
    parser.add_argument("--max-runs", type=int, default=12)
    parser.add_argument("--output-root", type=Path, default=MODELS_DIR / "sweep")
    parser.add_argument("--init-model-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Omitir corridas cuyo output_dir ya tenga train_metrics.json",
    )
    args = parser.parse_args()

    settings = load_settings()
    target = settings["project"]["target_f1_macro"]
    grid = _build_grid(args)
    logger.info("Barrido: %d configuraciones (max_runs=%d)", len(grid), args.max_runs)

    rows: list[dict] = []
    for idx, cfg in enumerate(grid):
        run_id = f"sweep_{idx:03d}_seed{cfg['seed']}"
        out_dir = args.output_root / run_id
        row = {**cfg, "run_id": run_id, "output_dir": str(out_dir)}

        if args.dry_run:
            rows.append(row)
            continue

        if args.skip_existing:
            existing = _load_existing_metrics(out_dir)
            if existing is not None:
                logger.info("Run %s: omitido (train_metrics.json existente)", run_id)
                row.update(existing)
                row["evaluated_at"] = datetime.now(timezone.utc).isoformat()
                row["status"] = "skipped"
                rows.append(row)
                continue

        set_seed(cfg["seed"])
        train_df = load_train_excluding_adversarial_overlap()
        train_df = filter_train_by_synthetic_policy(train_df, cfg["synthetic_policy"])
        val_df = load_split("val")

        sample_weights = compute_sample_loss_weights(
            train_df,
            threat_synthetic_effective_n=90.0,
            synthetic_effective_fraction=cfg["synthetic_effective_fraction"],
            platform_class_boost=default_facebook_boost_weights(),
        )

        training_config = MbertTrainingConfig(
            learning_rate=cfg["learning_rate"],
            num_train_epochs=6,
            early_stopping_patience=3,
            seed=cfg["seed"],
            label_smoothing=cfg["label_smoothing"],
            loss_type=cfg["loss_type"],
            ordinal_weight=0.3,
            focal_gamma=1.75,
            layerwise_lr_decay=0.95,
        )

        logger.info("Run %s: %s", run_id, cfg)
        trainer = train_mbert(
            train_df,
            val_df,
            output_dir=out_dir,
            class_weights=get_b2_class_weights(),
            sample_weights=sample_weights,
            use_b2_class_weights=True,
            training_config=training_config,
            init_model_dir=args.init_model_dir,
        )
        metrics = trainer.evaluate()
        row.update(
            {
                "val_f1_macro": float(metrics.get("eval_f1_macro", 0)),
                "val_accuracy": float(metrics.get("eval_accuracy", 0)),
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
                "status": "trained",
            }
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    SWEEP_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SWEEP_CSV, index=False)

    if not args.dry_run and len(df):
        best = df.sort_values("val_f1_macro", ascending=False).iloc[0]
        summary = {
            "best_run_id": best["run_id"],
            "best_val_f1_macro": float(best["val_f1_macro"]),
            "target_f1_macro": target,
            "passes_val_gate_0_77": bool(best["val_f1_macro"] >= 0.77),
            "sweep_csv": str(SWEEP_CSV),
        }
        (REPORTS_METRICS / "mbert_sweep_best.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Mejor val F1: %.4f (%s)", best["val_f1_macro"], best["run_id"])
    else:
        logger.info("Barrido registrado en %s", SWEEP_CSV)


if __name__ == "__main__":
    main()
