#!/usr/bin/env python
"""ISSUE-09: train mBERT baseline on v3 pilot release and evaluate on historical test."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RELEASE = "v1"
MODEL_DIR = ROOT / "versiones" / "v1" / "model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=1, help="Epochs for smoke/full train.")
    parser.add_argument("--max-steps", type=int, default=0, help="If >0, sets MAX_STEPS for a smoke train.")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-slo", action="store_true", help="No adversarial/dialect/SLO tras eval oficial.")
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR)
    return parser.parse_args()


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True, env=env)


def main() -> int:
    args = parse_args()
    model_dir = args.output_dir
    if not args.skip_train:
        env = os.environ.copy()
        if args.max_steps > 0:
            env["MAX_STEPS"] = str(args.max_steps)
        run(
            [
                sys.executable,
                str(ROOT / "scripts" / "train_mbert.py"),
                "--release",
                RELEASE,
                "--epochs",
                str(args.epochs),
                "--platform-class-boost",
                "none",
                "--output-dir",
                str(model_dir),
            ],
            env=env,
        )
    if not args.skip_eval and model_dir.exists():
        run(
            [
                sys.executable,
                str(ROOT / "scripts" / "evaluate_mbert_official.py"),
                "--model-dir",
                str(model_dir),
                "--skip-official-f1-check",
            ]
        )
    if not args.skip_eval and not args.skip_slo and model_dir.exists():
        run(
            [
                sys.executable,
                str(ROOT / "scripts" / "compute_dialect_bias.py"),
                "--model-dir",
                str(model_dir),
                "--skip-xlmr",
            ]
        )
        run(
            [
                sys.executable,
                str(ROOT / "scripts" / "evaluate_production_slo.py"),
                "--model-dir",
                str(model_dir),
                "--output",
                "reports/metrics/production_slo_baseline_v3_pilot.json",
            ]
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
