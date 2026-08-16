#!/usr/bin/env python
"""Exporta el manifiesto del contrato de inferencia a reports/metrics/."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.evaluation.predictions import git_commit
from discurso_odio.models.inference_contract import get_inference_contract, write_inference_contract
from discurso_odio.models.xlmr import default_xlmr_model_dir, ensure_checkpoint_contract
from discurso_odio.paths import REPORTS_METRICS

OUTPUT_PATH = REPORTS_METRICS / "inference_contract.json"


def export_inference_contract_manifest() -> Path:
    contract = get_inference_contract()
    model_dir = default_xlmr_model_dir()
    ensure_checkpoint_contract(model_dir)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "checkpoint": str(model_dir.resolve()),
        **contract.as_dict(),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    write_inference_contract(model_dir, contract)
    return OUTPUT_PATH


def main():
    path = export_inference_contract_manifest()
    print(f"Exportado: {path}")


if __name__ == "__main__":
    main()
