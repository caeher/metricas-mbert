#!/usr/bin/env python
"""Preentrenamiento MLM DAPT sobre train+val (Fase 4)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from discurso_odio.models.dapt import train_mbert_dapt
from discurso_odio.paths import MODELS_DIR
from discurso_odio.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="DAPT MLM mBERT (train+val, sin test)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODELS_DIR / "mbert-sv-dapt",
    )
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-val", action="store_true", help="Solo train (default incluye val)")
    args = parser.parse_args()

    set_seed(args.seed)
    out = train_mbert_dapt(
        args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        include_val=not args.no_val,
    )
    logger.info("DAPT completado: %s", out)


if __name__ == "__main__":
    main()
