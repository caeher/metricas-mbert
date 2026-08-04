"""Carga de datasets locales."""

from __future__ import annotations

import pandas as pd

from discurso_odio.paths import CORPUS_CSV, TEST_CSV, TRAIN_CSV, VAL_CSV


def load_split(name: str) -> pd.DataFrame:
    paths = {"train": TRAIN_CSV, "val": VAL_CSV, "test": TEST_CSV, "corpus": CORPUS_CSV}
    if name not in paths:
        raise ValueError(f"Split desconocido: {name}. Use {list(paths)}")
    return pd.read_csv(paths[name])


def load_all_splits() -> dict[str, pd.DataFrame]:
    return {split: load_split(split) for split in ("train", "val", "test", "corpus")}


def get_class_weights(train_df: pd.DataFrame, label_col: str = "label") -> dict[int, float]:
    """Pesos inversamente proporcionales a la frecuencia (solo sobre train)."""
    counts = train_df[label_col].value_counts().sort_index()
    n = len(train_df)
    k = len(counts)
    return {int(cls): n / (k * cnt) for cls, cnt in counts.items()}
