#!/usr/bin/env python
"""Evalúa mBERT sobre la batería adversarial versionada (T-06)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.evaluation.adversarial import (
    build_battery_144_manifest,
    build_battery_manifest,
    evaluate_adversarial_144_battery,
    evaluate_adversarial_battery,
    evaluate_test_metrics_parallel,
)
from discurso_odio.evaluation.official_mbert import (
    load_official_run_metadata,
    official_checkpoint_label,
)
from discurso_odio.paths import MODELS_DIR, REPORTS_METRICS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa robustez adversarial de mBERT (T-06).")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODELS_DIR / "mbert-sv",
        help="Directorio del checkpoint mBERT",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="mbert",
        help="Etiqueta del modelo en artefactos (p. ej. mbert, mbert-b2)",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Prefijo de salida (default: model-name con guiones)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="run_id a asociar (default: oficial T-04 si model-name=mbert)",
    )
    parser.add_argument(
        "--eval-test",
        action="store_true",
        help="También calcula métricas paralelas sobre test.csv (sin assert T-04)",
    )
    parser.add_argument(
        "--build-manifest",
        action="store_true",
        help="Regenera adversarial_battery_manifest.json",
    )
    parser.add_argument(
        "--battery-144",
        action="store_true",
        help="Evalúa la batería extendida A–G (n=144) en lugar de la v1 (n=75)",
    )
    parser.add_argument(
        "--build-manifest-144",
        action="store_true",
        help="Regenera adversarial_144_battery_manifest.json",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Tras --build-manifest/--build-manifest-144, no evaluar modelo",
    )
    args = parser.parse_args()

    if args.build_manifest:
        manifest = build_battery_manifest(write=True)
        logger.info("Manifiesto escrito: %s", manifest["sha256"])
    if args.build_manifest_144:
        manifest = build_battery_144_manifest(write=True)
        logger.info("Manifiesto 144 escrito: %s", manifest["sha256"])

    if args.manifest_only:
        return

    output_prefix = args.output_prefix or args.model_name.replace("-", "_")
    official = load_official_run_metadata() if args.model_name == "mbert" else None
    run_id = args.run_id or (official.get("run_id") if official else None)
    if args.model_name == "mbert":
        checkpoint = official_checkpoint_label(official, args.model_dir)
    else:
        from discurso_odio.models.mbert_artifacts import read_checkpoint_identity

        identity = read_checkpoint_identity(args.model_dir)
        checkpoint = identity.get("checkpoint") or args.model_dir.name

    if args.battery_144:
        if args.build_manifest_144 or not (REPORTS_METRICS / "adversarial_144_battery_manifest.json").exists():
            build_battery_144_manifest(write=True)
        summary = evaluate_adversarial_144_battery(
            model_dir=args.model_dir,
            model_name=args.model_name,
            checkpoint=checkpoint,
            run_id=run_id,
            output_prefix=output_prefix,
        )
    else:
        summary = evaluate_adversarial_battery(
            model_dir=args.model_dir,
            model_name=args.model_name,
            checkpoint=checkpoint,
            run_id=run_id,
            output_prefix=output_prefix,
        )

    logger.info(
        "Adversarial %s [%s]: %d/%d global, F1 diag=%.4f, mitigation=%s",
        args.model_name,
        checkpoint,
        summary["global_correct"],
        summary["n_cases"],
        summary["f1_macro_diagnostic"],
        summary["mitigation_required"],
    )
    if summary["mitigation_reasons"]:
        logger.info("Razones de mitigación: %s", summary["mitigation_reasons"])

    if args.eval_test:
        test_metrics = evaluate_test_metrics_parallel(
            model_dir=args.model_dir,
            model_name=args.model_name,
            checkpoint=checkpoint,
            output_prefix=output_prefix,
        )
        logger.info(
            "test.csv %s: F1-macro=%.4f, F1 amenazas=%.4f",
            args.model_name,
            test_metrics["f1_macro"],
            test_metrics["f1_Amenazas/Violencia"],
        )

    summary_path = REPORTS_METRICS / f"{output_prefix}_adversarial_summary.json"
    with open(summary_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
