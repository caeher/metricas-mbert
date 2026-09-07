# Genera google-colab/metricas-proyecto-ml-colab.zip con código, datos, baselines y pesos mBERT.
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

$ZipPath = Join-Path $Root "google-colab\metricas-proyecto-ml-colab.zip"
if (Test-Path $ZipPath) {
    $SizeMB = [math]::Round((Get-Item $ZipPath).Length / 1MB, 2)
    Write-Host "Listo: $ZipPath ($SizeMB MB)"
}
