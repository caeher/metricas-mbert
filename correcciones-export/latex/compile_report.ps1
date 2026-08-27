# Compila el informe de correcciones a PDF (requiere MiKTeX/TeX Live).
# Uso: powershell -ExecutionPolicy Bypass -File compile_report.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command pdflatex -ErrorAction SilentlyContinue)) {
    Write-Warning "pdflatex no encontrado. Alternativa: pandoc main.tex -o main.docx --from=latex --resource-path=.;figures --citeproc"
    exit 1
}

pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
Write-Host "PDF generado: $(Join-Path $PSScriptRoot 'main.pdf')" -ForegroundColor Green
