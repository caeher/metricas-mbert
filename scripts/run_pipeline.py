#!/usr/bin/env python
"""Ejecuta el pipeline completo: baselines, entrenamiento, métricas, benchmark."""

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.config import load_settings
from discurso_odio.data.loaders import load_split
from discurso_odio.evaluation.benchmark import run_benchmark
from discurso_odio.evaluation.metrics import bootstrap_ci, compute_all_metrics, mcnemar_test
from discurso_odio.evaluation.slices import (
    dialect_bias_probe,
    export_dialect_bias,
    export_slice_metrics,
    slice_metrics,
    slice_metrics_to_rows,
)
from discurso_odio.models.mbert import predict, train_mbert
from discurso_odio.paths import MODELS_DIR, REPORTS_FIGURES, REPORTS_METRICS, TEST_CSV
from discurso_odio.utils.plotting import setup_plot_style
from discurso_odio.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_baselines():
    logger.info("=== Baselines TF-IDF ===")
    train = load_split("train")
    test = load_split("test")
    X_train, y_train = train["texto_modelo"], train["label"]
    X_test, y_test = test["texto_modelo"], test["label"]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)
    models = {
        "LogisticRegression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "LinearSVC": LinearSVC(class_weight="balanced", random_state=42),
        "MultinomialNB": MultinomialNB(),
    }
    results = {}
    for name, clf in models.items():
        pipe = Pipeline([("tfidf", vectorizer), ("clf", clf)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        f1 = f1_score(y_test, preds, average="macro")
        results[name] = f1
        joblib.dump(pipe, MODELS_DIR / f"baseline_{name.lower()}.joblib")
        logger.info("%s F1-macro: %.4f", name, f1)

    pd.DataFrame({"model": list(results), "f1_macro": list(results.values())}).to_csv(
        REPORTS_METRICS / "baselines_tfidf.csv", index=False
    )
    return results


def run_training():
    logger.info("=== Fine-tuning mBERT ===")
    train = load_split("train")
    val = load_split("val")
    trainer = train_mbert(train, val)
    return trainer


def run_evaluation():
    logger.info("=== Evaluación test ===")
    test = load_split("test")
    model_dir = MODELS_DIR / "mbert-sv"
    y_pred, y_prob = predict(test["texto_modelo"].tolist(), model_dir)
    y_true = test["label"].values

    metrics = compute_all_metrics(y_true, y_pred, y_prob)
    f1_mean, f1_lo, f1_hi = bootstrap_ci(
        y_true, y_pred, lambda yt, yp: f1_score(yt, yp, average="macro")
    )
    metrics["f1_bootstrap_lo"] = f1_lo
    metrics["f1_bootstrap_hi"] = f1_hi

    slices = slice_metrics(test, y_true, y_pred, y_prob)
    slice_rows = slice_metrics_to_rows(
        slices,
        model="mbert",
        checkpoint=str(model_dir),
        split=TEST_CSV.name,
        n_test=len(test),
    )
    export_slice_metrics(
        pd.DataFrame(slice_rows),
        REPORTS_METRICS,
        extra_metadata={"xlmr_status": "not_requested"},
    )
    bias = dialect_bias_probe(
        test,
        y_true,
        y_pred,
        model="mbert",
        checkpoint=str(model_dir),
        split=TEST_CSV.name,
    )
    export_dialect_bias(bias, REPORTS_METRICS, extra_metadata={"xlmr_status": "not_requested"})

    export = {k: v for k, v in metrics.items() if not isinstance(v, (list, dict))}
    with open(model_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2)
    pd.DataFrame([export]).to_csv(REPORTS_METRICS / "mbert_test_metrics.csv", index=False)

    mcnemar_rows = []
    for name in ["logisticregression", "linearsvc", "multinomialnb"]:
        path = MODELS_DIR / f"baseline_{name}.joblib"
        if path.exists():
            pipe = joblib.load(path)
            bl_pred = pipe.predict(test["texto_modelo"])
            mcn = mcnemar_test(y_true, y_pred, bl_pred)
            mcnemar_rows.append({"baseline": name, **mcn})
    pd.DataFrame(mcnemar_rows).to_csv(REPORTS_METRICS / "mcnemar_tests.csv", index=False)

    logger.info("F1-macro: %.4f [%.4f, %.4f]", metrics["f1_macro"], f1_lo, f1_hi)
    return metrics


def run_benchmark_step():
    logger.info("=== Benchmark externo ===")
    results = run_benchmark(str(MODELS_DIR / "mbert-sv"))
    results.to_csv(REPORTS_METRICS / "benchmark_externo.csv", index=False)
    return results


def main():
    settings = load_settings()
    set_seed(settings["project"]["seed"])
    setup_plot_style()
    REPORTS_METRICS.mkdir(parents=True, exist_ok=True)
    REPORTS_FIGURES.mkdir(parents=True, exist_ok=True)

    run_baselines()
    run_training()
    run_evaluation()
    run_benchmark_step()
    logger.info("Pipeline completado.")


if __name__ == "__main__":
    main()
