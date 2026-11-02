#!/usr/bin/env python
"""Backfill training_run.json for an existing checkpoint without re-training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.models.training_run import build_training_run_record, write_training_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--training-recipe", type=Path, help="Optional JSON with training_recipe fields")
    args = parser.parse_args()

    recipe: dict | None = None
    if args.training_recipe and args.training_recipe.is_file():
        recipe = json.loads(args.training_recipe.read_text(encoding="utf-8"))

    eval_metrics: dict | None = None
    train_metrics_path = args.model_dir / "train_metrics.json"
    if train_metrics_path.is_file():
        eval_metrics = json.loads(train_metrics_path.read_text(encoding="utf-8"))

    if recipe is None:
        recipe = {
            "release": "v4",
            "note": "backfilled from existing checkpoint",
            "platform_class_boost": "none",
            "synthetic_policy": "b2_full",
        }

    out = write_training_run(
        args.model_dir,
        trainer=None,
        training_recipe=recipe,
        eval_metrics=eval_metrics,
    )
    record = build_training_run_record(
        model_dir=args.model_dir,
        training_recipe=recipe,
        eval_metrics=eval_metrics,
    )
    if not record.get("parameters"):
        from transformers import AutoModelForSequenceClassification

        model = AutoModelForSequenceClassification.from_pretrained(str(args.model_dir))
        from discurso_odio.models.training_run import count_model_parameters

        record["parameters"] = count_model_parameters(model)
        out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({"ok": True, "path": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
