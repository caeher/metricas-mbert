#!/usr/bin/env python
"""Import all ISSUE-05 annotation campaigns into Argilla (post-pilot)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CAMPAIGNS = (
    "legacy_train_real",
    "new_candidates",
    "legacy_train_synthetic",
    "calibration",
    "historical_val_audit",
    "historical_test_audit",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--env-file", type=Path, default=Path(".env.argilla"))
    parser.add_argument(
        "--only",
        nargs="+",
        choices=CAMPAIGNS,
        help="Importa solo las campañas indicadas (reanudar tras fallo parcial).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    importer = root / "scripts" / "import_argilla.py"
    campaigns = args.only or CAMPAIGNS
    for campaign in campaigns:
        command = [sys.executable, str(importer), "--campaign", campaign, "--env-file", str(args.env_file)]
        if args.dry_run:
            command.append("--dry-run")
        print("Running:", " ".join(command), flush=True)
        result = subprocess.run(command, cwd=root)
        if result.returncode != 0:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
