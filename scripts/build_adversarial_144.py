#!/usr/bin/env python
"""Genera data/processed/test_adversarial_144.csv (batería A–G, n=144)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.data.adversarial_battery import (
    BATTERY_144_N,
    FAMILY_SPECS_144,
    build_adversarial_144_dataframe,
    write_adversarial_144_csv,
)
from discurso_odio.evaluation.adversarial import build_battery_144_manifest


def main() -> None:
    path = write_adversarial_144_csv()
    df = build_adversarial_144_dataframe()
    assert len(df) == BATTERY_144_N
    for grupo, spec in FAMILY_SPECS_144.items():
        assert int((df["grupo"] == grupo).sum()) == spec["n"], grupo
    manifest = build_battery_144_manifest(write=True)
    print(f"Escrito {path} ({len(df)} casos)")
    print(f"Manifiesto SHA-256: {manifest['sha256']}")


if __name__ == "__main__":
    main()
