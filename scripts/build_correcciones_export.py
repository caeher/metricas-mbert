#!/usr/bin/env python
"""Ensambla el paquete correcciones-export para Cristian (mBERT only)."""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

PROJECT_NAME = "metricas-proyecto-ml"
ZIP_NAME = "mbert-cristian-colab.zip"
AUDIT_ZIP_NAME = "mbert-sv-audit.zip"
SKIP_PATH_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache"}

# Tablas LaTeX a copiar al informe de correcciones
GENERATED_TABLES = (
    "tab_trivial_baselines.tex",
    "tab_mbert_global.tex",
    "tab_mbert_traceability.tex",
    "tab_mcnemar.tex",
    "tab_slices.tex",
    "tab_platform.tex",
    "tab_dialect_bias.tex",
    "tab_adversarial_families.tex",
    "tab_adversarial_144_families.tex",
    "tab_adversarial_dialect.tex",
    "tab_adversarial_before_after.tex",
)

# Figuras mBERT para el informe
FIGURE_NAMES = (
    "04_confusion_matrix.png",
    "07_slice_f1_heatmap.png",
    "08_platform_f1_grouped.png",
    "10_dialect_bias_fpr.png",
    "11_adversarial_families.png",
)

# Métricas versionadas para el ZIP Colab
METRICS_GLOBS = (
    "mbert_*.csv",
    "mbert_*.json",
    "slice_*.csv",
    "slice_*.json",
    "dialect_bias_probe.csv",
    "dialect_bias_probe.json",
    "adversarial_battery_manifest.json",
    "adversarial_144_battery_manifest.json",
    "mbert_v1_144_adversarial_summary.json",
    "mcnemar_tests.csv",
    "mcnemar_tests.json",
    "trivial_baselines.csv",
    "xlmr_test_predictions.csv",
)

# Scripts mBERT necesarios en Colab
MBERT_SCRIPTS = (
    "train_mbert.py",
    "evaluate_mbert_official.py",
    "export_mbert_predictions.py",
    "compute_slice_metrics.py",
    "compute_dialect_bias.py",
    "evaluate_adversarial.py",
    "build_adversarial_144.py",
    "run_interpretability.py",
    "run_mcnemar.py",
    "run_t06_pipeline.py",
    "package_mbert_checkpoint.py",
    "verify_mbert_checkpoint.py",
    "inferencia_mbert.py",
    "colab_install_deps.py",
    "colab_extract.py",
    "generate_figures.py",
    "generate_latex_report.py",
)


def _copy_tree(src: Path, dest: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dest, dirs_exist_ok=True)
    elif src.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _export_dir(repo_root: Path) -> Path:
    return repo_root / "correcciones-export"


def copy_latex_assets(repo_root: Path) -> None:
    export = _export_dir(repo_root)
    src_gen = repo_root / "reports" / "latex" / "generated"
    dest_gen = export / "latex" / "generated"
    dest_gen.mkdir(parents=True, exist_ok=True)

    for name in GENERATED_TABLES:
        src = src_gen / name
        if src.is_file():
            shutil.copy2(src, dest_gen / name)
            print(f"  Tabla: {name}")
        else:
            print(f"  AVISO: falta {name}")

    src_fig = repo_root / "reports" / "latex" / "figures"
    dest_fig = export / "latex" / "figures"
    dest_fig.mkdir(parents=True, exist_ok=True)
    for name in FIGURE_NAMES:
        src = src_fig / name
        if not src.is_file():
            src = repo_root / "reports" / "figures" / name
        if src.is_file():
            shutil.copy2(src, dest_fig / name)
            print(f"  Figura: {name}")
        else:
            print(f"  AVISO: falta figura {name}")


def compile_latex_pdf(repo_root: Path) -> bool:
    latex_dir = _export_dir(repo_root) / "latex"
    main_tex = latex_dir / "main.tex"
    if not main_tex.is_file():
        print("  AVISO: main.tex no encontrado")
        return False

    for cmd in (
        ["pdflatex", "-interaction=nonstopmode", "main.tex"],
        ["bibtex", "main"],
        ["pdflatex", "-interaction=nonstopmode", "main.tex"],
        ["pdflatex", "-interaction=nonstopmode", "main.tex"],
    ):
        try:
            result = subprocess.run(
                cmd,
                cwd=latex_dir,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            print("  AVISO: pdflatex/bibtex no disponible; intentando pandoc DOCX...")
            return compile_latex_docx(latex_dir)

        if result.returncode != 0 and cmd[0] == "pdflatex":
            print(f"  AVISO: {cmd[0]} terminó con código {result.returncode}")

    pdf = latex_dir / "main.pdf"
    if pdf.is_file():
        print(f"  PDF: {pdf}")
        return True
    print("  AVISO: no se generó main.pdf; intentando pandoc DOCX...")
    return compile_latex_docx(latex_dir)


def compile_latex_docx(latex_dir: Path) -> bool:
    docx = latex_dir / "main.docx"
    try:
        result = subprocess.run(
            [
                "pandoc",
                "main.tex",
                "-o",
                "main.docx",
                "--from=latex",
                "--resource-path=.;figures",
                "--citeproc",
            ],
            cwd=latex_dir,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        print("  AVISO: pandoc no disponible; compilar manualmente en latex/")
        return False
    if result.returncode == 0 and docx.is_file():
        print(f"  DOCX: {docx}")
        return True
    print(f"  AVISO: pandoc falló: {result.stderr[:200] if result.stderr else result.returncode}")
    return False


def _stage_colab(repo_root: Path, staging: Path) -> None:
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    _copy_tree(repo_root / "data" / "processed", staging / "data" / "processed")
    _copy_tree(repo_root / "config", staging / "config")

    src_pkg = repo_root / "src" / "discurso_odio"
    dest_pkg = staging / "src" / "discurso_odio"
    dest_pkg.mkdir(parents=True, exist_ok=True)
    for file in src_pkg.rglob("*"):
        if not file.is_file():
            continue
        if any(part in SKIP_PATH_PARTS for part in file.parts):
            continue
        if file.suffix == ".pyc":
            continue
        rel = file.relative_to(src_pkg)
        target = dest_pkg / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target)

    for name in (
        "requirements.txt",
        "requirements-colab.txt",
        "pyproject.toml",
        ".env.example",
    ):
        src = repo_root / name
        if src.is_file():
            _copy_tree(src, staging / name)

    docs_dest = staging / "docs"
    docs_dest.mkdir(parents=True, exist_ok=True)
    for doc in ("guia-inferencia-mbert.md", "auditoria-mbert-cristian.md"):
        src = repo_root / "docs" / doc
        if src.is_file():
            shutil.copy2(src, docs_dest / doc)

    nb_src = _export_dir(repo_root) / "notebooks" / "replicar_mbert_colab.ipynb"
    nb_dest = staging / "notebooks"
    nb_dest.mkdir(parents=True, exist_ok=True)
    if nb_src.is_file():
        shutil.copy2(nb_src, nb_dest / nb_src.name)

    scripts_dest = staging / "scripts"
    scripts_dest.mkdir(parents=True, exist_ok=True)
    for script in MBERT_SCRIPTS:
        src = repo_root / "scripts" / script
        if src.is_file():
            shutil.copy2(src, scripts_dest / script)

    metrics_dest = staging / "reports" / "metrics"
    metrics_dest.mkdir(parents=True, exist_ok=True)
    metrics_src = repo_root / "reports" / "metrics"
    for pattern in METRICS_GLOBS:
        for src in metrics_src.glob(pattern):
            if src.is_file():
                shutil.copy2(src, metrics_dest / src.name)

    (staging / "models").mkdir(parents=True, exist_ok=True)
    (staging / "reports" / "figures").mkdir(parents=True, exist_ok=True)

    readme = staging / "README_COLAB.md"
    readme.write_text(
        "# mBERT — paquete Colab (Cristian)\n\n"
        "1. Sube `mbert-cristian-colab.zip` a Colab.\n"
        "2. Abre `notebooks/replicar_mbert_colab.ipynb`.\n"
        "3. Opcional: sube `mbert-sv-audit.zip` para omitir reentrenamiento.\n",
        encoding="utf-8",
    )


def _write_zip(staging: Path, out_zip: Path) -> None:
    if out_zip.exists():
        out_zip.unlink()
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(staging.rglob("*")):
            if file.is_dir():
                continue
            rel = file.relative_to(staging).as_posix()
            arcname = f"{PROJECT_NAME}/{rel}"
            zf.write(file, arcname)


def build_colab_zip(repo_root: Path) -> Path:
    export = _export_dir(repo_root)
    out_zip = export / ZIP_NAME
    staging = repo_root / ".correcciones_staging" / PROJECT_NAME
    _stage_colab(repo_root, staging)
    _write_zip(staging, out_zip)
    shutil.rmtree(staging.parent, ignore_errors=True)
    size_mb = out_zip.stat().st_size / (1024 * 1024)
    print(f"  ZIP Colab: {out_zip} ({size_mb:.2f} MB)")
    return out_zip


def copy_audit_zip(repo_root: Path) -> Path | None:
    export = _export_dir(repo_root)
    dest = export / AUDIT_ZIP_NAME
    src = repo_root / "artifacts" / AUDIT_ZIP_NAME

    if not src.is_file():
        model_dir = repo_root / "models" / "mbert-sv" / "model.safetensors"
        if model_dir.is_file():
            print("  Empaquetando checkpoint mBERT...")
            subprocess.run(
                [sys.executable, str(repo_root / "scripts" / "package_mbert_checkpoint.py")],
                cwd=repo_root,
                check=True,
            )
        src = repo_root / "artifacts" / AUDIT_ZIP_NAME

    if src.is_file():
        shutil.copy2(src, dest)
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  Audit ZIP: {dest} ({size_mb:.1f} MB)")
        return dest

    print("  AVISO: no hay mbert-sv-audit.zip; ejecutar package_mbert_checkpoint.py")
    return None


def write_readme(repo_root: Path, audit_copied: bool, pdf_built: bool) -> None:
    export = _export_dir(repo_root)
    audit_row = (
        f"| `{AUDIT_ZIP_NAME}` | Checkpoint oficial (~630 MB, T-02) |"
        if audit_copied
        else f"| `{AUDIT_ZIP_NAME}` | Generar con `package_mbert_checkpoint.py` |"
    )
    if (export / "latex" / "main.pdf").is_file():
        report_row = "| `latex/main.pdf` | Informe compilado (PDF) |"
    elif (export / "latex" / "main.docx").is_file():
        report_row = "| `latex/main.docx` | Informe compilado (pandoc) |"
    else:
        report_row = "| `latex/main.tex` | Compilar con pdflatex + bibtex |"

    content = f"""# Paquete de entrega — Correcciones mBERT (Cristian)

Entrega consolidada de las tareas T-01 a T-07 derivadas de la auditoría y verificación de modelos.
**Alcance:** solo mBERT / Cristian Escalante. Sin XLMR ni correcciones de Julio.

## Contenido

| Archivo | Descripción |
|---------|-------------|
| `{ZIP_NAME}` | Código, datos, métricas y libreta Colab |
{audit_row}
{report_row}
| `notebooks/replicar_mbert_colab.ipynb` | Libreta única de replicación |
| `latex/` | Informe LaTeX de correcciones |

## Checklist T-01 … T-07

- [x] **T-01** Baseline clase mayoritaria: 0.128 (antes 0.087)
- [x] **T-02** Checkpoint reproducible + manifiesto SHA-256
- [x] **T-03** McNemar mBERT vs XLM-R: p=0.200 (no significativo)
- [x] **T-04** F1-macro oficial: **0.7777** (checkpoint-1072)
- [x] **T-05** Cortes por plataforma y contexto
- [x] **T-06** Batería adversarial v1 (n=75); mitigación B2 **no adoptada**
- [x] **T-07** Batería extendida 144 casos (A--G); tabla vs XLM-R verificación

## Cifras canónicas

| Métrica | Valor |
|---------|-------|
| F1-macro test | 0.7777 |
| Baseline mayoritaria | 0.128 |
| McNemar vs XLM-R | p = 0.200 |
| Adversarial global (144) | ver `mbert_v1_144_adversarial_summary.json` |
| Adversarial global (75, hist.) | 45/75 |
| Mitigación B2 | Rechazada (test 0.7161) |

## Uso en Google Colab

1. Runtime → GPU (T4).
2. Subir `{ZIP_NAME}`.
3. Abrir y ejecutar `notebooks/replicar_mbert_colab.ipynb`.
4. **Opción A:** subir `{AUDIT_ZIP_NAME}` para verificar F1 = 0.7777 sin reentrenar.
5. **Opción B:** reentrenar (2–4 h GPU); el F1 puede variar ligeramente.

## Verificación local

```powershell
python scripts/verify_mbert_checkpoint.py --package artifacts/mbert-sv-audit.zip
python scripts/evaluate_mbert_official.py --model-dir models/mbert-sv
pytest tests/test_adversarial_eval.py -q
```

## Regenerar este paquete

```powershell
python scripts/build_correcciones_export.py
```

## Fuera de alcance

- Carpeta `XLMR/`
- Hallazgos y correcciones exclusivas de Julio / XLM-RoBERTa
- Modelo candidato B2 (documentado pero no adoptado)
"""
    readme_path = export / "README.md"
    readme_path.write_text(content, encoding="utf-8")
    print(f"  README: {readme_path}")


def build_correcciones_export(repo_root: Path | None = None) -> None:
    repo_root = repo_root or _repo_root()
    export = _export_dir(repo_root)
    export.mkdir(parents=True, exist_ok=True)
    (export / "notebooks").mkdir(parents=True, exist_ok=True)

    print("1. Copiando tablas y figuras LaTeX...")
    copy_latex_assets(repo_root)

    print("2. Compilando PDF...")
    pdf_built = compile_latex_pdf(repo_root)

    print("3. Construyendo ZIP Colab...")
    build_colab_zip(repo_root)

    print("4. Copiando checkpoint de auditoría...")
    audit = copy_audit_zip(repo_root)

    print("5. Escribiendo README...")
    write_readme(repo_root, audit is not None, pdf_built)

    print("\nPaquete listo en:", export)


if __name__ == "__main__":
    build_correcciones_export()
