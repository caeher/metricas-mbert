#!/usr/bin/env python
"""Ejecuta análisis de interpretabilidad LIME + SHAP."""

import logging
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.config import load_settings
from discurso_odio.data.loaders import load_split
from discurso_odio.interpretability.lime_explainer import (
    aggregate_lime_weights,
    build_lime_explainer,
    explain_instance,
    jaccard_top_k,
    make_predict_proba_fn,
)
from discurso_odio.interpretability.shap_explainer import (
    explain_texts_shap,
    extract_top_tokens,
    make_shap_predict_fn,
)
from discurso_odio.models.mbert import predict
from discurso_odio.models.runtime import CachedPredictor
from discurso_odio.paths import MODELS_DIR, REPORTS_METRICS
from discurso_odio.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    settings = load_settings()
    set_seed(settings["project"]["seed"])
    model_dir = MODELS_DIR / "mbert-sv"
    if not model_dir.exists():
        logger.error("Modelo no encontrado. Ejecute train_mbert.py primero.")
        sys.exit(1)

    test = load_split("test")
    y_pred, _ = predict(test["texto_modelo"].tolist(), model_dir)
    y_true = test["label"].values

    samples = []
    for cls in range(4):
        idx_c = np.where((y_true == cls) & (y_pred == cls))[0][:3]
        idx_e = np.where((y_true == cls) & (y_pred != cls))[0][:2]
        samples.extend(idx_c.tolist() + idx_e.tolist())
    n_lime = min(settings["interpretability"]["lime_samples"], len(samples))
    texts = test.iloc[samples[:n_lime]]["texto_modelo"].tolist()

    predictor = CachedPredictor(model_dir)
    explainer = build_lime_explainer()
    predict_proba_fn = make_predict_proba_fn(predictor.predict_proba)

    lime_explanations = []
    for i, text in enumerate(texts[:12]):
        logger.info("LIME %d/%d", i + 1, min(12, len(texts)))
        lime_explanations.append(
            explain_instance(text, predict_proba_fn, explainer, num_samples=200, num_features=8)
        )

    global_lime = aggregate_lime_weights(lime_explanations)
    lime_rows = []
    for cls, tokens in global_lime.items():
        for token, weight in tokens:
            lime_rows.append({"class": cls, "token": token, "importance": weight})
    pd.DataFrame(lime_rows).to_csv(REPORTS_METRICS / "lime_global_weights.csv", index=False)

    shap_texts = texts[:4]
    jaccard_rows = []
    try:
        shap_predict_fn = make_shap_predict_fn(predictor.model, predictor.tokenizer, predictor.device)
        shap_values = explain_texts_shap(
            shap_texts, shap_predict_fn, tokenizer=predictor.tokenizer, max_evals=100
        )
        for i in range(min(len(lime_explanations), len(shap_texts))):
            lime_tokens = lime_explanations[i]["top_tokens"]
            shap_top = extract_top_tokens(shap_values, i, top_k=8)
            shap_tokens = [t for t, _ in shap_top]
            j = jaccard_top_k(lime_tokens, shap_tokens, k=8)
            jaccard_rows.append({"instance": i, "jaccard": j})
        jaccard_df = pd.DataFrame(jaccard_rows)
        jaccard_path = REPORTS_METRICS / "lime_shap_jaccard.csv"
        if jaccard_df["jaccard"].sum() == 0:
            logger.warning(
                "Jaccard LIME-SHAP=0 tras alineación; se omite tabla (ver interpretability_note.json)."
            )
            jaccard_path.unlink(missing_ok=True)
            pd.Series({"jaccard_available": False, "reason": "zero_overlap_after_alignment"}).to_json(
                REPORTS_METRICS / "interpretability_note.json"
            )
        else:
            jaccard_df.to_csv(jaccard_path, index=False)
            pd.Series({"jaccard_available": True}).to_json(
                REPORTS_METRICS / "interpretability_note.json"
            )
        logger.info("SHAP completado.")
    except Exception as e:
        logger.warning("SHAP falló: %s. LIME exportado.", e)

    fp_mask = (y_true == 0) & (y_pred >= 1)
    fp_jerga = int(((test["n_jerga"] > 0).values & fp_mask).sum())
    fp_no_jerga = int(((test["n_jerga"] == 0).values & fp_mask).sum())
    pd.Series({"fp_con_jerga": fp_jerga, "fp_sin_jerga": fp_no_jerga}).to_csv(
        REPORTS_METRICS / "interpretability_fp_jerga.csv"
    )
    logger.info("Interpretabilidad completada.")


if __name__ == "__main__":
    main()
