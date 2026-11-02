#!/usr/bin/env python
"""Fusiona métricas oficiales, training_run, fertility, gates y SLO en un informe único."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.evaluation.predictions import git_commit
from discurso_odio.paths import PROJECT_ROOT, REPORTS_METRICS

DEFAULT_OUT = REPORTS_METRICS / "mbert_training_v4_full_report.json"


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "versiones" / "v4" / "model")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    training_run = _load_json(model_dir / "training_run.json")
    train_metrics = _load_json(model_dir / "train_metrics.json")
    test_metrics = _load_json(model_dir / "test_metrics.json")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "model_dir": str(model_dir),
        "f1_macro": {
            "val": float(train_metrics.get("eval_f1_macro", 0)) if train_metrics else None,
            "test": float(test_metrics.get("f1_macro", 0)) if test_metrics else None,
            "test_bootstrap_lo": test_metrics.get("f1_bootstrap_lo") if test_metrics else None,
            "test_bootstrap_hi": test_metrics.get("f1_bootstrap_hi") if test_metrics else None,
            "val_meets_target_0_80": bool(
                train_metrics and float(train_metrics.get("eval_f1_macro", 0)) >= 0.80
            ),
        },
        "parameters": (training_run or {}).get("parameters"),
        "training_time": (training_run or {}).get("runtime"),
        "training_recipe": (training_run or {}).get("training_recipe"),
        "reproducibility": {
            "training_run": str((model_dir / "training_run.json").relative_to(PROJECT_ROOT)),
            "dataset": (training_run or {}).get("dataset"),
            "checkpoint_identity": (training_run or {}).get("checkpoint_identity"),
            "official_run": "reports/metrics/mbert_official_run.json",
            "checkpoint_manifest": "reports/metrics/mbert_checkpoint_manifest.json",
        },
        "tokenizer_fertility": _load_json(REPORTS_METRICS / "tokenizer_fertility.json"),
        "official_run": _load_json(REPORTS_METRICS / "mbert_official_run.json"),
        "f1_target_gates": _load_json(REPORTS_METRICS / "f1_target_gates.json"),
        "production_slo": _load_json(REPORTS_METRICS / "production_slo_baseline_training_v3.json"),
        "slice_metrics": _load_json(REPORTS_METRICS / "slice_metrics.json"),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    version_metrics = model_dir.parent / "metrics" / args.output.name
    version_metrics.parent.mkdir(parents=True, exist_ok=True)
    version_metrics.write_text(args.output.read_text(encoding="utf-8"), encoding="utf-8")

    print(json.dumps({"ok": True, "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
