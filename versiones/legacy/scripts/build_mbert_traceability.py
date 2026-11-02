#!/usr/bin/env python
"""Genera la tabla de trazabilidad F1-macro mBERT (T-04)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from discurso_odio.evaluation.official_mbert import export_traceability


def main():
    parser = argparse.ArgumentParser(description="Inventario F1-macro mBERT")
    parser.add_argument(
        "--official-f1",
        type=float,
        default=None,
        help="F1 crudo de la corrida oficial (default: leer CSV si existe)",
    )
    args = parser.parse_args()
    csv_path, json_path = export_traceability(official_f1_raw=args.official_f1)
    print(f"Trazabilidad: {csv_path}")
    print(f"Trazabilidad: {json_path}")


if __name__ == "__main__":
    main()
