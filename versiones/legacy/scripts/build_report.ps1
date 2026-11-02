# Genera tablas LaTeX, figuras del informe y convierte main.tex → main.docx con pandoc.
# Uso (desde la raíz del repo):
#   powershell -ExecutionPolicy Bypass -File scripts/build_report.ps1
# Flags:
#   -SkipFigures   no regenera figuras (solo tablas + pandoc)
#   -Pdf           intenta compilar PDF con pdflatex

param(
    [switch]$SkipFigures,
    [switch]$Pdf
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$Python = if (Test-Path ".venv\Scripts\python.exe") {
    (Resolve-Path ".venv\Scripts\python.exe").Path
} else {
    "python"
}

Write-Host "==> Generando tablas LaTeX..." -ForegroundColor Cyan
& $Python scripts/generate_latex_report.py
if ($LASTEXITCODE -ne 0) { throw "generate_latex_report.py falló" }
Write-Host "==> Verificando consistencia de métricas por plataforma..." -ForegroundColor Cyan
& $Python scripts/check_report_platform_metrics.py
if ($LASTEXITCODE -ne 0) { throw "check_report_platform_metrics.py falló" }

Write-Host "==> Verificando consistencia de McNemar..." -ForegroundColor Cyan
& $Python scripts/check_report_mcnemar.py
if ($LASTEXITCODE -ne 0) { throw "check_report_mcnemar.py falló" }

if (-not $SkipFigures) {
    Write-Host "==> Generando figuras..." -ForegroundColor Cyan
    $env:SKIP_CONFUSION_MATRIX = "1"
    & $Python scripts/generate_figures.py
    if ($LASTEXITCODE -ne 0) { throw "generate_figures.py falló" }
    & $Python scripts/generate_latex_report.py
    if ($LASTEXITCODE -ne 0) { throw "generate_latex_report.py falló (post-figuras)" }
}

Write-Host "==> Verificación final de consistencia..." -ForegroundColor Cyan
& $Python scripts/check_report_platform_metrics.py
if ($LASTEXITCODE -ne 0) { throw "check_report_platform_metrics.py falló" }
& $Python scripts/check_report_mcnemar.py
if ($LASTEXITCODE -ne 0) { throw "check_report_mcnemar.py falló" }

$LatexDir = Join-Path $Root "reports\latex"
$MainTex = Join-Path $LatexDir "main.tex"
$OutDocx = Join-Path $LatexDir "main.docx"

if ($Pdf) {
    Write-Host "==> Compilando PDF (pdflatex)..." -ForegroundColor Cyan
    Push-Location $LatexDir
    try {
        pdflatex -interaction=nonstopmode main.tex | Out-Null
        if (Test-Path "references.bib") {
            bibtex main 2>$null | Out-Null
            pdflatex -interaction=nonstopmode main.tex | Out-Null
            pdflatex -interaction=nonstopmode main.tex | Out-Null
        }
        Write-Host "PDF: $(Join-Path $LatexDir 'main.pdf')" -ForegroundColor Green
    } catch {
        Write-Warning "pdflatex no disponible o falló: $_"
    } finally {
        Pop-Location
    }
}

Write-Host "==> Convirtiendo a Word con pandoc..." -ForegroundColor Cyan
$pandoc = Get-Command pandoc -ErrorAction SilentlyContinue
if (-not $pandoc) {
    throw "pandoc no está instalado o no está en el PATH. Instálelo desde https://pandoc.org"
}

$refDoc = Join-Path $LatexDir "reference.docx"
Push-Location $LatexDir
try {
    $pandocArgs = @(
        "main.tex",
        "-o", "main.docx",
        "--from=latex",
        "--resource-path=.;figures;..\figures",
        "--citeproc"
    )
    if (Test-Path $refDoc) {
        $pandocArgs += "--reference-doc=$refDoc"
    }
    & pandoc @pandocArgs
    if ($LASTEXITCODE -ne 0) { throw "pandoc falló" }
} finally {
    Pop-Location
}

Write-Host "Listo: $OutDocx" -ForegroundColor Green
