#!/usr/bin/env python
"""Orquesta evaluación adversarial, mitigación B2 y comparación (T-06)."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from discurso_odio.evaluation.adversarial import (
    build_battery_144_manifest,
    build_battery_manifest,
    decide_mitigation_adoption,
    evaluate_adversarial_144_battery,
    evaluate_adversarial_battery,
    evaluate_test_metrics_parallel,
    export_before_after,
    load_battery_144_manifest,
)
from discurso_odio.evaluation.official_mbert import (
    load_official_run_metadata,
    official_checkpoint_label,
)
from discurso_odio.paths import MODELS_DIR, REPORTS_METRICS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline T-06 robustez y equidad mBERT.")
    parser.add_argument("--skip-train", action="store_true", help="No reentrenar B2 aunque haya brecha")
    parser.add_argument("--train-max-steps", type=int, default=0, help="Límite de pasos (0 = épocas completas)")
    parser.add_argument(
        "--battery-144",
        action="store_true",
        help="También evalúa la batería extendida A–G (n=144)",
    )
    parser.add_argument(
        "--skip-battery-v1",
        action="store_true",
        help="Omite la batería v1 (n=75); implica evaluar solo --battery-144",
    )
    args = parser.parse_args()

    if not args.skip_battery_v1:
        load_battery_manifest()

        official = load_official_run_metadata()
        baseline_dir = MODELS_DIR / "mbert-sv"
        summary = evaluate_adversarial_battery(
            model_dir=baseline_dir,
            model_name="mbert",
            checkpoint=official_checkpoint_label(official, baseline_dir),
            run_id=official.get("run_id") if official else None,
            output_prefix="mbert",
        )
        evaluate_test_metrics_parallel(
            model_dir=baseline_dir,
            model_name="mbert",
            checkpoint=summary["checkpoint"],
            output_prefix="mbert",
        )
    else:
        official = load_official_run_metadata()
        baseline_dir = MODELS_DIR / "mbert-sv"
        summary = {"checkpoint": official_checkpoint_label(official, baseline_dir), "mitigation_required": False}

    if args.battery_144 or args.skip_battery_v1:
        load_battery_144_manifest()
        summary_144 = evaluate_adversarial_144_battery(
            model_dir=baseline_dir,
            model_name="mbert-v1",
            checkpoint=summary["checkpoint"],
            run_id=official.get("run_id") if official else None,
            output_prefix="mbert_v1",
        )
        logger.info(
            "Adversarial 144: %d/%d global (F1 diag=%.4f)",
            summary_144["global_correct"],
            summary_144["n_cases"],
            summary_144["f1_macro_diagnostic"],
        )
        if args.skip_battery_v1:
            summary = summary_144

    if not summary.get("mitigation_required") or args.skip_train:
        logger.info("Mitigación no requerida o omitida.")
        return

    b2_dir = MODELS_DIR / "mbert-sv-b2"
    if not b2_dir.exists() or not (b2_dir / "model.safetensors").exists():
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "train_mbert.py"),
            "--output-dir",
            str(b2_dir),
            "--exclude-adversarial-overlap",
        ]
        env = None
        if args.train_max_steps > 0:
            import os

            env = os.environ.copy()
            env["MAX_STEPS"] = str(args.train_max_steps)
        logger.info("Entrenando candidato B2: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, env=env)

    b2_summary = evaluate_adversarial_battery(
        model_dir=b2_dir,
        model_name="mbert-b2",
        checkpoint=official_checkpoint_label(None, b2_dir),
        run_id=None,
        output_prefix="mbert_b2",
    )
    evaluate_test_metrics_parallel(
        model_dir=b2_dir,
        model_name="mbert-b2",
        checkpoint=b2_summary["checkpoint"],
        output_prefix="mbert_b2",
    )
    export_before_after("mbert", "mbert_b2")
    before_after = __import__("pandas").read_csv(REPORTS_METRICS / "mbert_adversarial_before_after.csv")
    adopted, notes = decide_mitigation_adoption(before_after)
    meta_path = REPORTS_METRICS / "mbert_adversarial_before_after.json"
    with open(meta_path, encoding="utf-8") as handle:
        meta = json.load(handle)
    meta["mitigation_adopted"] = adopted
    meta["adoption_notes"] = notes
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)
    logger.info("Mitigación adoptada=%s: %s", adopted, notes)


if __name__ == "__main__":
    main()
