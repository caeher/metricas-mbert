# Genera metricas-proyecto-ml-colab.zip con lo mínimo para reproducir en Google Colab.
# Uso: powershell -ExecutionPolicy Bypass -File scripts/build_colab_zip.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = if (Test-Path (Join-Path $Root ".venv\Scripts\python.exe")) {
    Join-Path $Root ".venv\Scripts\python.exe"
} else {
    "python"
}

& $Python (Join-Path $Root "scripts\build_colab_zip.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
