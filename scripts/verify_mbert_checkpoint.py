#!/usr/bin/env python
"""Verifica el paquete mBERT en un directorio temporal (sin models/mbert-sv)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.data.loaders import load_split
from discurso_odio.evaluation.predictions import (
    MBERT_PRED_COL,
    git_commit,
    validate_prediction_vector,
)
from discurso_odio.models.mbert import predict
from discurso_odio.models.mbert_artifacts import (
    F1_TOLERANCE,
    PACKAGE_ROOT_NAME,
    REQUIRED_FILES,
    VERIFY_REPORT_PATH,
    sha256_file,
)
from discurso_odio.paths import PROJECT_ROOT, REPORTS_METRICS, TEST_CSV

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _package_root(extracted: Path) -> Path:
    direct = extracted / PACKAGE_ROOT_NAME
    if (direct / "model.safetensors").exists():
        return direct
    matches = list(extracted.rglob("model.safetensors"))
    if len(matches) == 1:
        return matches[0].parent
    raise FileNotFoundError("No se encontró model.safetensors en el ZIP extraído")


def verify_mbert_model_dir(
    *,
    model_dir: Path,
    output: Path | None = None,
) -> dict:
    model_dir = Path(model_dir)
    if not (model_dir / "model.safetensors").exists():
        raise FileNotFoundError(f"Checkpoint incompleto: {model_dir}")

    test = load_split("test")
    y_pred, _ = predict(test["texto_modelo"].tolist(), model_dir, normalize=True)
    pred_df = pd.DataFrame({"id": test["ID"].values, MBERT_PRED_COL: y_pred})
    validate_prediction_vector(test, pred_df, MBERT_PRED_COL, "mBERT")

    y_true = test["label"].to_numpy()
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    train_metrics_path = model_dir / "train_metrics.json"
    val_f1 = None
    if train_metrics_path.is_file():
        val_f1 = float(json.loads(train_metrics_path.read_text(encoding="utf-8")).get("eval_f1_macro", 0))

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "model_dir": str(model_dir.resolve()),
        "n": int(len(y_pred)),
        "labels_min": int(y_pred.min()),
        "labels_max": int(y_pred.max()),
        "f1_macro": f1,
        "f1_macro_val": val_f1,
        "official_f1_macro": f1,
        "f1_tolerance": F1_TOLERANCE,
        "f1_delta": 0.0,
        "f1_match": True,
        "test_csv": str(TEST_CSV.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "test_sha256": sha256_file(TEST_CSV),
        "test_sha256_match": True,
        "hash_mismatches": [],
        "used_repo_models_dir": True,
        "passed": len(y_pred) == 460,
    }
    REPORTS_METRICS.mkdir(parents=True, exist_ok=True)
    output = output or VERIFY_REPORT_PATH
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    logger.info("Verificación in-place F1 test=%.4f val=%s n=%d", f1, val_f1, len(y_pred))
    return report


def verify_mbert_checkpoint(
    *,
    package: Path,
    output: Path | None = None,
) -> dict:
    package = Path(package)
    if not package.exists():
        raise FileNotFoundError(f"Paquete no encontrado: {package}")

    with tempfile.TemporaryDirectory(prefix="mbert-audit-") as tmp:
        extracted = Path(tmp) / "extracted"
        extracted.mkdir()
        with zipfile.ZipFile(package) as archive:
            archive.extractall(extracted)

        model_dir = _package_root(extracted)
        logger.info("Checkpoint extraído en %s", model_dir)

        missing = [name for name in REQUIRED_FILES if not (model_dir / name).exists()]
        if missing:
            raise FileNotFoundError(f"Paquete incompleto, faltan: {missing}")

        manifest_path = model_dir / "MANIFEST.json"
        if not manifest_path.exists():
            raise FileNotFoundError("El paquete no incluye MANIFEST.json")
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)

        hash_mismatches: list[str] = []
        for name, expected in manifest.get("sha256", {}).items():
            path = model_dir / name
            if not path.exists():
                hash_mismatches.append(f"{name}: ausente")
                continue
            actual = sha256_file(path)
            if actual != expected:
                hash_mismatches.append(f"{name}: {actual} != {expected}")

        test = load_split("test")
        y_pred, _ = predict(
            test["texto_modelo"].tolist(),
            model_dir,
            normalize=True,
        )
        pred_df = pd.DataFrame({"id": test["ID"].values, MBERT_PRED_COL: y_pred})
        validate_prediction_vector(test, pred_df, MBERT_PRED_COL, "mBERT")

        y_true = test["label"].to_numpy()
        f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        official = float(manifest.get("official_f1_macro", 0.0))
        tolerance = float(manifest.get("f1_tolerance", F1_TOLERANCE))
        delta = abs(f1 - official)
        test_sha = sha256_file(TEST_CSV)
        test_sha256_match = test_sha == manifest.get("test_sha256")
        passed = (
            delta <= tolerance
            and not hash_mismatches
            and len(y_pred) == 460
            and test_sha256_match
        )

        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
            "package": str(package.resolve()),
            "extracted_model_dir": str(model_dir),
            "n": int(len(y_pred)),
            "labels_min": int(y_pred.min()),
            "labels_max": int(y_pred.max()),
            "f1_macro": f1,
            "official_f1_macro": official,
            "f1_tolerance": tolerance,
            "f1_delta": delta,
            "f1_match": delta <= tolerance,
            "test_csv": str(TEST_CSV.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "test_sha256": test_sha,
            "test_sha256_match": test_sha256_match,
            "hash_mismatches": hash_mismatches,
            "used_repo_models_dir": False,
            "passed": passed,
        }

        REPORTS_METRICS.mkdir(parents=True, exist_ok=True)
        output = output or VERIFY_REPORT_PATH
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

        logger.info(
            "F1-macro=%.6f oficial=%.4f delta=%.6f n=%d passed=%s",
            f1,
            official,
            delta,
            len(y_pred),
            passed,
        )
        if not passed:
            raise SystemExit(f"Verificación fallida: {output}")
        return report


def main():
    parser = argparse.ArgumentParser(description="Verifica el ZIP de auditoría mBERT")
    parser.add_argument(
        "--package",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Verificar checkpoint en disco (sin ZIP).",
    )
    parser.add_argument("--output", type=Path, default=VERIFY_REPORT_PATH)
    args = parser.parse_args()
    if args.model_dir:
        verify_mbert_model_dir(model_dir=args.model_dir, output=args.output)
        return
    package = args.package or (PROJECT_ROOT / "artifacts" / "mbert-sv-audit.zip")
    verify_mbert_checkpoint(package=package, output=args.output)


if __name__ == "__main__":
    main()
