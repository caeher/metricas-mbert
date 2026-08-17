"""Carga de datasets locales."""

from __future__ import annotations

import pandas as pd

from discurso_odio.paths import (
    CORPUS_CSV,
    TEST_ADVERSARIAL_CSV,
    TEST_CSV,
    TRAIN_CSV,
    VAL_CSV,
)


def load_split(name: str) -> pd.DataFrame:
    paths = {
        "train": TRAIN_CSV,
        "val": VAL_CSV,
        "test": TEST_CSV,
        "corpus": CORPUS_CSV,
        "adversarial": TEST_ADVERSARIAL_CSV,
    }
    if name not in paths:
        raise ValueError(f"Split desconocido: {name}. Use {list(paths)}")
    return pd.read_csv(paths[name])


def load_all_splits() -> dict[str, pd.DataFrame]:
    return {split: load_split(split) for split in ("train", "val", "test", "corpus")}


def load_train_excluding_adversarial_overlap() -> pd.DataFrame:
    """Train B2 sin filas cuyo texto coincide con la batería adversarial (T-06)."""
    from discurso_odio.evaluation.adversarial import adversarial_text_set, find_train_overlaps

    train_df = load_split("train")
    overlaps = find_train_overlaps(train_df)
    if overlaps.empty:
        return train_df
    adv_texts = adversarial_text_set()
    mask = ~train_df["texto_modelo"].astype(str).str.strip().str.lower().isin(adv_texts)
    return train_df.loc[mask].reset_index(drop=True)


def get_class_weights(train_df: pd.DataFrame, label_col: str = "label") -> dict[int, float]:
    """Pesos inversamente proporcionales a la frecuencia (solo sobre train)."""
    counts = train_df[label_col].value_counts().sort_index()
    n = len(train_df)
    k = len(counts)
    return {int(cls): n / (k * cnt) for cls, cnt in counts.items()}
