"""Predictor en memoria para serving (API, LIME, SHAP)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from discurso_odio.models.inference_contract import (
    InferenceContract,
    apply_normalization,
    resolve_inference_contract,
)
from discurso_odio.models.mbert import default_mbert_model_dir, get_device
from discurso_odio.models.xlmr import default_xlmr_model_dir


def resolve_model_dir(
    model_dir: Path | str | None = None,
    *,
    backend: str | None = None,
) -> Path:
    """Resuelve directorio del checkpoint desde env o backend."""
    if model_dir is not None:
        return Path(model_dir)

    env_dir = os.getenv("MODEL_DIR")
    if env_dir:
        return Path(env_dir)

    backend = (backend or os.getenv("MODEL_BACKEND", "mbert")).lower()
    if backend == "xlmr":
        return default_xlmr_model_dir()
    return default_mbert_model_dir()


class CachedPredictor:
    """Carga tokenizer + modelo una vez; aplica contrato de normalización."""

    def __init__(
        self,
        model_dir: Path | str | None = None,
        *,
        backend: str | None = None,
        batch_size: int = 16,
        normalize: bool = True,
    ):
        self.model_dir = resolve_model_dir(model_dir, backend=backend)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Checkpoint no encontrado: {self.model_dir}")

        self.contract: InferenceContract = resolve_inference_contract(self.model_dir)
        self.device = get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
        self.model.to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        self.normalize = normalize

    @property
    def max_length(self) -> int:
        return self.contract.max_length

    @torch.no_grad()
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Devuelve probabilidades (n_samples, n_classes)."""
        if self.normalize:
            texts = apply_normalization(texts, self.contract)
        else:
            texts = [str(t) if t else " " for t in texts]

        all_probs: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [str(t) if t else " " for t in texts[i : i + self.batch_size]]
            enc = self.tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=self.contract.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
        return np.concatenate(all_probs, axis=0)

    @torch.no_grad()
    def predict_single(self, text: str) -> tuple[int, float, np.ndarray]:
        """Predice una instancia: (label_index, confidence, probs)."""
        probs = self.predict_proba([text])[0]
        idx = int(np.argmax(probs))
        return idx, float(probs[idx]), probs
