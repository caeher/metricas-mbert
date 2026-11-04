# Genera tablas/figuras LaTeX v6 y exporta informe_metricas_v6.docx / .pdf con pandoc.
# Uso (desde ml-metricas/):
#   powershell -ExecutionPolicy Bypass -File scripts/build_report.ps1

param(
    [switch]$SkipFigures,
    [switch]$SkipPdf
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
if ((Split-Path -Leaf $Root) -eq "scripts") {
    $Root = Split-Path -Parent $Root
}
Set-Location $Root

$Python = if (Test-Path ".venv\Scripts\python.exe") {
    (Resolve-Path ".venv\Scripts\python.exe").Path
} else {
    "python"
}

Write-Host "==> Generando tablas LaTeX..." -ForegroundColor Cyan
& $Python versiones\legacy\scripts\generate_latex_report.py
if ($LASTEXITCODE -ne 0) { throw "generate_latex_report.py falló" }

if (-not $SkipFigures) {
    Write-Host "==> Generando figuras..." -ForegroundColor Cyan
    & $Python versiones\legacy\scripts\generate_figures.py
    if ($LASTEXITCODE -ne 0) { throw "generate_figures.py falló" }
    & $Python versiones\legacy\scripts\generate_latex_report.py
    if ($LASTEXITCODE -ne 0) { throw "generate_latex_report.py falló (post-figuras)" }
}

Write-Host "==> Expandiendo \\input para pandoc..." -ForegroundColor Cyan
& $Python scripts\expand_latex_inputs.py
if ($LASTEXITCODE -ne 0) { throw "expand_latex_inputs.py falló" }

Write-Host "==> Placeholders EDA/benchmark (figuras faltantes)..." -ForegroundColor Cyan
& $Python -c @"
import matplotlib.pyplot as plt
import shutil
from pathlib import Path
root = Path('reports/figures')
root.mkdir(parents=True, exist_ok=True)
latex = Path('reports/latex/figures')
latex.mkdir(parents=True, exist_ok=True)
labels = {
    '01_distribucion_clases.png': 'EDA: distribucion de clases\n(no regenerado en corrida v6)',
    '02_plataforma_clases.png': 'EDA: plataforma x clases\n(no regenerado en corrida v6)',
    '03_token_lengths.png': 'EDA: longitudes de tokens\n(no regenerado en corrida v6)',
    '05_benchmark_externo.png': 'Benchmark externo (checkpoint v1)\n(ver tab_benchmark.tex)',
}
for name, text in labels.items():
    p = root / name
    if not p.exists():
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.axis('off')
        ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=12)
        fig.savefig(p, dpi=120, bbox_inches='tight')
        plt.close(fig)
    shutil.copy2(p, latex / name)
"@

$pandoc = Get-Command pandoc -ErrorAction SilentlyContinue
if (-not $pandoc) {
    throw "pandoc no está instalado o no está en el PATH."
}

$LatexDir = Join-Path $Root "reports\latex"
$BuildTex = Join-Path $LatexDir "build\main_expanded.tex"
$OutDocx = Join-Path $LatexDir "informe_metricas_v6.docx"
$OutPdf = Join-Path $LatexDir "informe_metricas_v6.pdf"
$FiguresDir = Join-Path $Root "reports\figures"
$LatexFigures = Join-Path $LatexDir "figures"

Write-Host "==> Convirtiendo a Word..." -ForegroundColor Cyan
Push-Location (Join-Path $LatexDir "build")
try {
    $resourcePath = "..;..\figures;..\..\figures"
    $pandocArgs = @(
        "main_expanded.tex",
        "-o", $OutDocx,
        "--from=latex",
        "--resource-path=$resourcePath",
        "--citeproc"
    )
    $refDoc = Join-Path $LatexDir "reference.docx"
    if (Test-Path $refDoc) {
        $pandocArgs += "--reference-doc=$refDoc"
    }
    & pandoc @pandocArgs
    if ($LASTEXITCODE -ne 0) { throw "pandoc docx falló" }
} finally {
    Pop-Location
}

Write-Host "Word: $OutDocx" -ForegroundColor Green

if (-not $SkipPdf) {
    $localTectonic = Join-Path $Root "tools\tectonic\tectonic.exe"
    if (-not (Test-Path $localTectonic)) {
        Write-Warning "tectonic no encontrado en tools/tectonic; omitiendo PDF."
    } else {
        Write-Host "==> Compilando PDF (tectonic + LaTeX expandido)..." -ForegroundColor Cyan
        $BuildDir = Join-Path $LatexDir "build"
        Copy-Item -Force (Join-Path $LatexDir "references.bib") (Join-Path $BuildDir "references.bib")
        $buildFigures = Join-Path $BuildDir "figures"
        if (Test-Path $buildFigures) { Remove-Item -Recurse -Force $buildFigures }
        Copy-Item -Recurse (Join-Path $LatexDir "figures") $buildFigures
        Push-Location $BuildDir
        try {
            & $localTectonic -X compile main_expanded.tex
            if ($LASTEXITCODE -ne 0) { throw "tectonic falló" }
            Move-Item -Force (Join-Path $BuildDir "main_expanded.pdf") $OutPdf
            Write-Host "PDF: $OutPdf" -ForegroundColor Green
        } finally {
            Pop-Location
        }
    }
}

Write-Host "Listo." -ForegroundColor Green
