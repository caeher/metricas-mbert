"""Tests de utilidades CUDA para entrenamiento."""

import sys
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.models.mbert import cuda_training_kwargs, get_device


def test_cuda_training_kwargs_cpu():
    with patch.object(torch.cuda, "is_available", return_value=False):
        assert cuda_training_kwargs() == {"fp16": False, "dataloader_pin_memory": False}
        assert get_device().type == "cpu"


def test_cuda_training_kwargs_gpu():
    with patch.object(torch.cuda, "is_available", return_value=True):
        assert cuda_training_kwargs() == {"fp16": True, "dataloader_pin_memory": True}
        assert get_device().type == "cuda"
