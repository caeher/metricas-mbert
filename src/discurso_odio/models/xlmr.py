"""Inferencia batch con checkpoint XLM-RoBERTa."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from discurso_odio.config import load_settings
from discurso_odio.paths import PROJECT_ROOT


def default_xlmr_model_dir() -> Path:
    return PROJECT_ROOT / "XLMR" / "02_modelo_prototipo" / "modelo_entrenado"


def predict(
    texts: list[str],
    model_dir: Path | str | None = None,
    *,
    max_length: int | None = None,
    batch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Predice etiquetas y probabilidades para una lista de textos."""
    settings = load_settings()
    model_dir = Path(model_dir) if model_dir else default_xlmr_model_dir()
    if not model_dir.exists():
        raise FileNotFoundError(f"Checkpoint XLM-R no encontrado: {model_dir}")

    max_length = max_length or settings["model"]["max_length"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    all_preds: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

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
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        all_preds.append(np.argmax(probs, axis=-1))

    return np.concatenate(all_preds), np.concatenate(all_probs)
