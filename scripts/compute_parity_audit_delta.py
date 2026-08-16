#!/usr/bin/env python
"""Compara predicciones XLM-R bajo distintos contratos (diagnóstico T-06)."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.data.loaders import load_split
from discurso_odio.evaluation.predictions import git_commit
from discurso_odio.features.normalization import normalize_for_model
from discurso_odio.models.inference_contract import get_inference_contract
from discurso_odio.models.xlmr import default_xlmr_model_dir, predict
from discurso_odio.paths import REPORTS_METRICS, TEST_CSV

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_PATH = REPORTS_METRICS / "parity_audit_delta.json"
AUDIT_F1_MACRO_256 = 0.8058


def _predict_variant(
    test: pd.DataFrame,
    model_dir: Path,
    *,
    max_length: int,
    normalize: bool,
    text_source: str,
) -> np.ndarray:
    if text_source == "texto_modelo":
        texts = test["texto_modelo"].tolist()
    else:
        texts = test["texto_original"].tolist()

    if not normalize:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        model.to(device)
        model.eval()
        preds: list[int] = []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                logits = model(**enc).logits
                preds.extend(torch.argmax(logits, dim=-1).cpu().numpy().tolist())
        return np.asarray(preds, dtype=int)

    y_pred, _ = predict(texts, model_dir, max_length=max_length, normalize=normalize)
    return y_pred


def compute_parity_audit_delta() -> dict:
    contract = get_inference_contract()
    model_dir = default_xlmr_model_dir()
    if not model_dir.exists():
        raise FileNotFoundError(f"Checkpoint XLM-R no encontrado: {model_dir}")

    test = load_split("test")
    y_true = test["label"].values

    official = _predict_variant(
        test,
        model_dir,
        max_length=contract.max_length,
        normalize=True,
        text_source="texto_modelo",
    )
    audit_256 = _predict_variant(
        test,
        model_dir,
        max_length=256,
        normalize=False,
        text_source="texto_modelo",
    )
    raw_no_norm = _predict_variant(
        test,
        model_dir,
        max_length=contract.max_length,
        normalize=False,
        text_source="texto_original",
    )

    def _diff_stats(a: np.ndarray, b: np.ndarray) -> dict:
        changed = int(np.sum(a != b))
        return {
            "n_changed": changed,
            "pct_changed": round(changed / len(a) * 100, 2),
        }

    f1_official = float(f1_score(y_true, official, average="macro"))
    f1_audit_256 = float(f1_score(y_true, audit_256, average="macro"))

    norm_mismatch = 0
    for orig, modelo in zip(test["texto_original"], test["texto_modelo"], strict=True):
        if normalize_for_model(str(orig)) != str(modelo):
            norm_mismatch += 1

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "split": TEST_CSV.name,
        "checkpoint": str(model_dir.resolve()),
        "contract": contract.as_dict(),
        "audit_reference": {
            "f1_macro_reported": AUDIT_F1_MACRO_256,
            "max_length_reported": 256,
            "note": "Cifra del informe de auditoría con evaluación a max_length=256",
        },
        "official_contract": {
            "max_length": contract.max_length,
            "normalize": True,
            "text_column": contract.text_column,
            "f1_macro": round(f1_official, 4),
        },
        "diagnostic_audit_path": {
            "max_length": 256,
            "normalize": False,
            "text_column": "texto_modelo",
            "f1_macro": round(f1_audit_256, 4),
            "delta_f1_vs_official": round(f1_audit_256 - f1_official, 4),
        },
        "vs_audit_256": _diff_stats(official, audit_256),
        "vs_raw_no_normalization": _diff_stats(official, raw_no_norm),
        "normalization_pipeline": {
            "texto_original_vs_texto_modelo_mismatch": norm_mismatch,
            "pct_mismatch": round(norm_mismatch / len(test) * 100, 2),
        },
    }
    return report


def main():
    report = compute_parity_audit_delta()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Exportado: %s", OUTPUT_PATH)
    logger.info(
        "F1 oficial (contrato): %.4f | F1 diagnóstico 256: %.4f",
        report["official_contract"]["f1_macro"],
        report["diagnostic_audit_path"]["f1_macro"],
    )
    logger.info(
        "Predicciones distintas vs auditoría 256: %d (%.2f%%)",
        report["vs_audit_256"]["n_changed"],
        report["vs_audit_256"]["pct_changed"],
    )


if __name__ == "__main__":
    main()
