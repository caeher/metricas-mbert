"""Ejecuta notebooks 01-07 en orden (pensado para Google Colab con GPU)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

NOTEBOOKS = [
    "01_eda.ipynb",
    "02_preprocesamiento_features.ipynb",
    "03_baselines_tfidf.ipynb",
    "04_finetuning_mbert.ipynb",
    "05_metricas_y_comparacion.ipynb",
    "06_benchmark_externo.ipynb",
    "07_interpretabilidad.ipynb",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Ejecuta notebooks del proyecto en orden.")
    parser.add_argument(
        "--from",
        dest="from_nb",
        type=int,
        default=1,
        help="Primer notebook (1-7, default: 1)",
    )
    parser.add_argument(
        "--to",
        dest="to_nb",
        type=int,
        default=7,
        help="Último notebook (1-7, default: 7)",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        type=int,
        default=[],
        help="Índices a omitir, ej: --skip 4 (saltar fine-tuning si ya tienes el modelo)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="Timeout por notebook en segundos (default: 7200)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[4]
    nb_dir = root / "notebooks"
    os.chdir(root)
    os.environ.setdefault("RUN_PROFILE", "gpu")

    # Verificar stack PyTorch antes de notebooks que usan transformers
    try:
        import torch
        import torchaudio
        import torchvision
        from transformers import EarlyStoppingCallback

        print(
            f"torch={torch.__version__} | torchvision={torchvision.__version__} | "
            f"torchaudio={torchaudio.__version__}"
        )
        print("CUDA:", torch.cuda.is_available())
    except Exception as exc:
        print(
            "ERROR: dependencias PyTorch/transformers no listas. "
            "Ejecuta: python scripts/colab_install_deps.py y reinicia el runtime.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    selected = [
        NOTEBOOKS[i - 1]
        for i in range(args.from_nb, args.to_nb + 1)
        if i not in args.skip and 1 <= i <= len(NOTEBOOKS)
    ]

    if not selected:
        print("No hay notebooks para ejecutar.", file=sys.stderr)
        return 1

    print("RUN_PROFILE =", os.environ.get("RUN_PROFILE"))
    print("Notebooks:", ", ".join(selected))

    for name in selected:
        path = nb_dir / name
        if not path.exists():
            print(f"Falta: {path}", file=sys.stderr)
            return 1
        print(f"\n=== Ejecutando {name} ===")
        cmd = [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            str(path),
            f"--output={name.replace('.ipynb', '_executed.ipynb')}",
            f"--ExecutePreprocessor.timeout={args.timeout}",
        ]
        result = subprocess.run(cmd, cwd=root)
        if result.returncode != 0:
            print(f"Falló: {name}", file=sys.stderr)
            return result.returncode

    print("\nTodos los notebooks completados.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
