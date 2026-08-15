#!/usr/bin/env python
"""Instala dependencias en Colab sin romper el stack preinstalado."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], *, check: bool = True) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=check)


def install_project_deps() -> None:
    # No tocar pip/setuptools/torch/numpy: Colab ya trae un stack alineado (cu128).
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "-r",
            str(ROOT / "requirements-colab.txt"),
        ]
    )
    run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-e", str(ROOT)])


def verify_subprocess() -> None:
  code = """
import numpy as np
import pandas as pd
import torch
import torchvision
import torchaudio
from transformers import EarlyStoppingCallback, Trainer

print('numpy', np.__version__)
print('pandas', pd.__version__)
print('torch', torch.__version__)
print('torchvision', torchvision.__version__)
print('torchaudio', torchaudio.__version__)
print('CUDA', torch.cuda.is_available())
print('transformers OK', EarlyStoppingCallback, Trainer)
"""
  run([sys.executable, "-c", code])


def main() -> int:
    install_project_deps()
    verify_subprocess()
    print("Instalación Colab completada.")
    print("IMPORTANTE: Entorno de ejecución → Reiniciar sesión antes de continuar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
