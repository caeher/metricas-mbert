# Genera metricas-proyecto-ml-colab.zip con lo mínimo para reproducir en Google Colab.
# Uso: powershell -ExecutionPolicy Bypass -File scripts/build_colab_zip.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$OutZip = Join-Path $Root "metricas-proyecto-ml-colab.zip"
$Staging = Join-Path $env:TEMP "metricas-proyecto-ml-colab"

if (Test-Path $Staging) { Remove-Item $Staging -Recurse -Force }
if (Test-Path $OutZip) { Remove-Item $OutZip -Force }

New-Item -ItemType Directory -Path $Staging -Force | Out-Null

function Copy-Tree {
    param([string]$Source, [string]$Dest)
    if (-not (Test-Path $Source)) {
        throw "No existe: $Source"
    }
    New-Item -ItemType Directory -Path (Split-Path $Dest -Parent) -Force | Out-Null
    Copy-Item -Path $Source -Destination $Dest -Recurse -Force
}

# Datos procesados
Copy-Tree (Join-Path $Root "data\processed") (Join-Path $Staging "data\processed")

# Código fuente (sin __pycache__ ni egg-info)
$srcDest = Join-Path $Staging "src\discurso_odio"
New-Item -ItemType Directory -Path $srcDest -Force | Out-Null
Get-ChildItem (Join-Path $Root "src\discurso_odio") -Recurse -File |
    Where-Object { $_.FullName -notmatch '__pycache__|\.pyc$' } |
    ForEach-Object {
        $rel = $_.FullName.Substring((Join-Path $Root "src\discurso_odio").Length + 1)
        $target = Join-Path $srcDest $rel
        New-Item -ItemType Directory -Path (Split-Path $target -Parent) -Force | Out-Null
        Copy-Item $_.FullName $target -Force
    }

# Config y empaquetado
Copy-Tree (Join-Path $Root "config") (Join-Path $Staging "config")
Copy-Item (Join-Path $Root "requirements.txt") (Join-Path $Staging "requirements.txt")
Copy-Item (Join-Path $Root "pyproject.toml") (Join-Path $Staging "pyproject.toml")
Copy-Item (Join-Path $Root ".env.example") (Join-Path $Staging ".env.example")
Copy-Item (Join-Path $Root "README.md") (Join-Path $Staging "README.md")

# Notebooks 00-07 (sin *_executed.ipynb)
$nbDest = Join-Path $Staging "notebooks"
New-Item -ItemType Directory -Path $nbDest -Force | Out-Null
Get-ChildItem (Join-Path $Root "notebooks") -Filter "*.ipynb" |
    Where-Object { $_.Name -notmatch '_executed\.ipynb$' } |
    ForEach-Object { Copy-Item $_.FullName (Join-Path $nbDest $_.Name) }

# Scripts CLI
Copy-Tree (Join-Path $Root "scripts") (Join-Path $Staging "scripts")

# Carpetas vacías que el pipeline espera
New-Item -ItemType Directory -Path (Join-Path $Staging "models") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $Staging "reports\figures") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $Staging "reports\metrics") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $Staging "reports\tables") -Force | Out-Null

Compress-Archive -Path (Join-Path $Staging "*") -DestinationPath $OutZip -Force
Remove-Item $Staging -Recurse -Force

$sizeMB = [math]::Round((Get-Item $OutZip).Length / 1MB, 2)
Write-Host "ZIP creado: $OutZip ($sizeMB MB)"
