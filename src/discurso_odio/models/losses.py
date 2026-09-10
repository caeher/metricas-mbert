"""Funciones de pérdida para fine-tuning mBERT (Fase 2)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def focal_loss(
    logits: Tensor,
    labels: Tensor,
    *,
    gamma: float = 2.0,
    class_weights: Tensor | None = None,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Focal loss multiclase (sin reducción)."""
    num_classes = logits.size(-1)
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()

    if label_smoothing > 0:
        smooth = label_smoothing / num_classes
        one_hot = torch.zeros_like(log_probs).scatter_(1, labels.unsqueeze(1), 1.0)
        targets = one_hot * (1.0 - label_smoothing) + smooth
        ce = -(targets * log_probs).sum(dim=-1)
        pt = (probs * targets).sum(dim=-1).clamp(min=1e-8)
    else:
        ce = F.cross_entropy(logits, labels, reduction="none")
        pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1).clamp(min=1e-8)

    focal = ((1.0 - pt) ** gamma) * ce
    if class_weights is not None:
        w = class_weights.gather(0, labels)
        focal = focal * w
    return focal


def ordinal_mae_penalty(logits: Tensor, labels: Tensor) -> Tensor:
    """Penalización |y - argmax| sobre predicciones duras (sin reducción)."""
    preds = logits.argmax(dim=-1)
    return (preds.float() - labels.float()).abs()


def combined_ordinal_loss(
    logits: Tensor,
    labels: Tensor,
    *,
    ordinal_weight: float = 0.3,
    gamma: float = 0.0,
    class_weights: Tensor | None = None,
    label_smoothing: float = 0.0,
) -> Tensor:
    """CE/focal + penalización ordinal |y-ŷ|."""
    if gamma > 0:
        base = focal_loss(
            logits,
            labels,
            gamma=gamma,
            class_weights=class_weights,
            label_smoothing=label_smoothing,
        )
    else:
        weight = class_weights
        base = F.cross_entropy(
            logits,
            labels,
            weight=weight,
            reduction="none",
            label_smoothing=label_smoothing,
        )
    if ordinal_weight <= 0:
        return base
    ordinal = ordinal_mae_penalty(logits, labels)
    return base + ordinal_weight * ordinal
