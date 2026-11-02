# Análisis de Discurso de Odio en Redes Sociales — El Salvador

Proyecto de clasificación multiclase de toxicidad con **mBERT** (`bert-base-multilingual-cased`), métricas comparativas, benchmarking zero-shot y interpretabilidad (SHAP + LIME).

## Requisitos

- Python 3.10+
- ~4 GB RAM libre (CPU)
- Opcional: GPU NVIDIA + CUDA para entrenamiento acelerado

### GPU local (Windows / Linux)

```powershell
python scripts/check_cuda.py
```

Si tienes GPU pero PyTorch dice `+cpu`, reinstala con CUDA:

```powershell
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

En Python 3.14 suele instalarse `torch 2.11.0+cu128` (la 2.13.0+cu128 aún no está publicada para ese runtime).

En `.env` usa `RUN_PROFILE=gpu` para batch sizes mayores. Con 4 GB VRAM y fp16 automático suele caber batch 16; si hay OOM, baja `per_device_train_batch_size` en `config/settings.yaml`.

## Instalación

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e .
copy .env.example .env
```

## Estructura

| Ruta | Descripción |
|------|-------------|
| `data/processed/` | Dataset **CDE v1.9.0** activo (train 21 585 / val / test) |
| `data/raw/` | Espejo mínimo del mismo core |
| `scripts/` | Pipeline **21k** (sync, train, eval oficial/SLO) |
| `versiones/v4/` | Release y checkpoint congelados CDE |
| `versiones/v1`–`v3`, `versiones/legacy/` | Líneas Argilla / histórico |
| `argilla/` | Anotación humana (scripts + [README](argilla/README.md)) |
| `src/discurso_odio/` | Paquete Python |
| `config/settings.yaml` | Hiperparámetros |
| `reports/` | Métricas y anotación |

## Uso rápido (CDE 21k)

```powershell
python scripts/sync_cde_dataset.py
python scripts/train_mbert.py --release v4 --output-dir versiones/v4/model
python scripts/evaluate_mbert_official.py --model-dir versiones/v4/model --skip-official-f1-check
python scripts/evaluate_production_slo.py --model-dir versiones/v4/model
```

Anotación Argilla: [`argilla/README.md`](argilla/README.md) (`make argilla-install`, `make argilla-up`).

## Uso rápido (legacy)

Notebooks, Colab y pipelines antiguos están en [`versiones/legacy/`](versiones/legacy/README.md).

```powershell
pytest tests/ -v
```

## Clases (ordinales 0–3)

| Label | Clase |
|-------|-------|
| 0 | No Tóxico |
| 1 | Lenguaje Ofensivo |
| 2 | Discurso de Odio |
| 3 | Amenazas/Violencia |

## Informes (legacy)

Scripts de informe LaTeX/Word están en [`versiones/legacy/scripts/`](versiones/legacy/scripts/) (`build_report.ps1`). Colab: [`versiones/legacy/google-colab/`](versiones/legacy/google-colab/).

## Checkpoint mBERT (v4 CDE)

Checkpoint congelado: [`versiones/v4/model/`](versiones/v4/model/).

```powershell
python scripts/verify_mbert_checkpoint.py --model-dir versiones/v4/model
python scripts/evaluate_mbert_official.py --model-dir versiones/v4/model --skip-official-f1-check
python scripts/package_mbert_checkpoint.py --model-dir versiones/v4/model
```

Pipelines F1 0.80 y B2 históricos: scripts en [`versiones/legacy/scripts/`](versiones/legacy/scripts/).

## Licencia

Investigación académica — UES ESI 2026.
