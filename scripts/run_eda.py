#!/usr/bin/env python
"""Ejecuta notebooks 01-02 y genera figuras EDA."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu
from transformers import AutoTokenizer

from discurso_odio.config import load_settings
from discurso_odio.data.loaders import get_class_weights, load_all_splits
from discurso_odio.evaluation.metrics import trivial_baselines
from discurso_odio.features.text_features import build_text_features
from discurso_odio.paths import REPORTS_FIGURES, REPORTS_METRICS
from discurso_odio.utils.plotting import get_class_palette, setup_plot_style
from discurso_odio.utils.seed import set_seed

settings = load_settings()
set_seed(settings["project"]["seed"])
setup_plot_style()

splits = load_all_splits()
corpus, train, test = splits["corpus"], splits["train"], splits["test"]

# EDA figures
class_dist = corpus["clase"].value_counts()
fig, ax = plt.subplots(figsize=(10, 5))
class_dist.plot(kind="bar", color=get_class_palette(), ax=ax)
ax.set_title("Distribución de clases canónicas")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(REPORTS_FIGURES / "01_distribucion_clases.png")
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
pd.crosstab(corpus["Plataforma"], corpus["clase"], normalize="index").plot(
    kind="bar", stacked=True, ax=axes[0], color=get_class_palette()
)
corpus.boxplot(column="n_palabras", by="Plataforma", ax=axes[1])
plt.suptitle("")
plt.tight_layout()
plt.savefig(REPORTS_FIGURES / "02_plataforma_clases.png")
plt.close()

tokenizer = AutoTokenizer.from_pretrained(settings["model"]["name"])
lengths = [len(tokenizer.encode(t, add_special_tokens=True)) for t in corpus["texto_modelo"].head(500)]
plt.hist(lengths, bins=30, color="#3498db", edgecolor="white")
plt.axvline(settings["model"]["max_length"], color="red", linestyle="--", label="max_length=128")
plt.title("Distribución de longitud en tokens")
plt.legend()
plt.tight_layout()
plt.savefig(REPORTS_FIGURES / "03_token_lengths.png")
plt.close()

# Metrics exports (pisos triviales sobre test, n=460, alineado con evaluación)
counts = test["label"].value_counts().to_dict()
pd.Series(trivial_baselines(counts), name="F1-macro").to_csv(REPORTS_METRICS / "trivial_baselines.csv")
weights = get_class_weights(train)
pd.DataFrame({"label": sorted(weights), "weight": [weights[i] for i in sorted(weights)]}).to_csv(
    REPORTS_METRICS / "class_weights.csv", index=False
)
build_text_features(corpus).to_csv(REPORTS_METRICS / "features_sample.csv", index=False)
print("EDA artifacts generated.")
