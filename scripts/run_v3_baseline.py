#!/usr/bin/env python
"""ISSUE-09: train mBERT baseline on v3 pilot release and evaluate on historical test."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = "v3-pilot-argilla-20261031"
MODEL_DIR = ROOT / "models" / "mbert-sv-v3-pilot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=1, help="Epochs for smoke/full train.")
    parser.add_argument("--max-steps", type=int, default=0, help="If >0, sets MAX_STEPS for a smoke train.")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
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
            ]
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
