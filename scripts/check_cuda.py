#!/usr/bin/env python
"""Diagnóstico de GPU/CUDA para entrenamiento mBERT en Windows o Linux."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from discurso_odio.config import get_run_profile
from discurso_odio.models.mbert import cuda_training_kwargs, get_device


def _run_nvidia_smi() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def main() -> int:
    print("=== Diagnóstico CUDA / mBERT ===\n")
    print(f"PyTorch:           {torch.__version__}")
    print(f"CUDA en PyTorch:   {torch.version.cuda or 'no compilado'}")
    print(f"torch.cuda:        {torch.cuda.is_available()}")
    print(f"RUN_PROFILE:       {get_run_profile()}")
    print(f"Dispositivo mBERT: {get_device()}")
    print(f"Trainer kwargs:    {cuda_training_kwargs()}")

    gpu_info = _run_nvidia_smi()
    if gpu_info:
        print(f"\nGPU (nvidia-smi):  {gpu_info}")
    else:
        print("\nGPU (nvidia-smi):  no detectada o comando no disponible")

    if torch.cuda.is_available():
        print(f"Nombre GPU:        {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"VRAM total:        {props.total_memory / (1024 ** 3):.1f} GB")
        print("\nListo para entrenar con GPU. Usa RUN_PROFILE=gpu en .env")
        return 0

    is_cpu_build = "+cpu" in torch.__version__
    print()
    if is_cpu_build and gpu_info:
        print("Hay GPU NVIDIA pero PyTorch está instalado solo para CPU (+cpu).")
        print("Reinstala con CUDA (cu128 recomendado para drivers recientes):")
        print()
        print("  pip uninstall -y torch torchvision")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")
        print()
        print("En Python 3.14 la wheel disponible suele ser torch 2.11.0+cu128 (no 2.13.0).")
        print()
        print("Luego vuelve a ejecutar: python scripts/check_cuda.py")
        return 1

    if not gpu_info:
        print("No se detectó GPU NVIDIA. El entrenamiento usará CPU.")
    else:
        print("Instala drivers/CUDA toolkit o la build correcta de PyTorch.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
