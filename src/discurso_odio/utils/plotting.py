"""Estilo visual consistente para figuras."""

import matplotlib.pyplot as plt
import seaborn as sns

from discurso_odio.config import load_settings


def setup_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "figure.figsize": (10, 6),
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    })


def get_label_names() -> list[str]:
    return load_settings()["labels"]["names"]


def get_class_palette() -> list[str]:
    return ["#2ecc71", "#f39c12", "#e74c3c", "#8e44ad"]
