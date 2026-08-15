#!/usr/bin/env bash
# Genera tablas LaTeX / figuras y convierte main.tex → main.docx
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=python3
fi

echo "==> Generando tablas LaTeX..."
"$PYTHON" scripts/generate_latex_report.py

if [[ "${SKIP_FIGURES:-0}" != "1" ]]; then
  echo "==> Generando figuras..."
  "$PYTHON" scripts/generate_figures.py
  "$PYTHON" scripts/generate_latex_report.py
fi

echo "==> Verificando consistencia de métricas por plataforma..."
"$PYTHON" scripts/check_report_platform_metrics.py

LATEX_DIR="$ROOT/reports/latex"
RESOURCE_PATH="$LATEX_DIR:$LATEX_DIR/figures:$ROOT/reports/figures"

if [[ "${PDF:-0}" == "1" ]]; then
  echo "==> Compilando PDF..."
  (cd "$LATEX_DIR" && pdflatex -interaction=nonstopmode main.tex && bibtex main || true
   pdflatex -interaction=nonstopmode main.tex
   pdflatex -interaction=nonstopmode main.tex)
fi

echo "==> Convirtiendo a Word con pandoc..."
cd "$LATEX_DIR"
PANDOC_ARGS=(
  main.tex
  -o main.docx
  --from=latex
  --resource-path=".:figures:../figures"
  --citeproc
)
if [[ -f reference.docx ]]; then
  PANDOC_ARGS+=(--reference-doc=reference.docx)
fi
pandoc "${PANDOC_ARGS[@]}"
echo "Listo: $LATEX_DIR/main.docx"
