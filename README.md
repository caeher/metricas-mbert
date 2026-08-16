# Análisis de Discurso de Odio en Redes Sociales — El Salvador

Proyecto de clasificación multiclase de toxicidad con **mBERT** (`bert-base-multilingual-cased`), métricas comparativas, benchmarking zero-shot y interpretabilidad (SHAP + LIME).

## Requisitos

- Python 3.10+
- ~4 GB RAM libre (CPU)
- Opcional: GPU NVIDIA + CUDA para entrenamiento acelerado

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
| `data/processed/` | train/val/test/corpus_limpio.csv |
| `src/discurso_odio/` | Código fuente del paquete |
| `notebooks/` | 7 notebooks reproducibles (01–07) |
| `scripts/` | Entrenamiento y benchmarks CLI |
| `config/settings.yaml` | Hiperparámetros y rutas |
| `reports/` | Figuras, tablas y métricas exportadas |

## Uso rápido

```powershell
# Entrenar mBERT
python scripts/train_mbert.py

# Benchmark externo
python scripts/run_benchmarks.py

# Tests
pytest tests/ -v

# Notebooks
jupyter notebook notebooks/
```

## Notebooks

0. `00_colab_setup.ipynb` — Setup en Google Colab (subir ZIP, instalar deps)
1. `01_eda.ipynb` — Análisis exploratorio
2. `02_preprocesamiento_features.ipynb` — Pipeline de normalización
3. `03_baselines_tfidf.ipynb` — Baselines TF-IDF
4. `04_finetuning_mbert.ipynb` — Fine-tuning mBERT
5. `05_metricas_y_comparacion.ipynb` — Métricas completas
6. `06_benchmark_externo.ipynb` — Evaluación zero-shot
7. `07_interpretabilidad.ipynb` — SHAP y LIME

## Clases (ordinales 0–3)

| Label | Clase |
|-------|-------|
| 0 | No Tóxico |
| 1 | Lenguaje Ofensivo |
| 2 | Discurso de Odio |
| 3 | Amenazas/Violencia |

## Informes

El informe LaTeX de métricas vive en `reports/latex/`. Para regenerar tablas y
exportar a Word:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_report.ps1
# o solo tablas + docx (sin regenerar figuras):
powershell -ExecutionPolicy Bypass -File scripts/build_report.ps1 -SkipFigures
```

Salida: `reports/latex/main.docx` (y opcionalmente PDF con `-Pdf`).

## Google Colab

### 1. Generar el ZIP localmente

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_colab_zip.ps1
```

Esto crea `metricas-proyecto-ml-colab.zip` con la carpeta raíz `metricas-proyecto-ml/` (requerida por el setup de Colab).

### 2. Subir y configurar en Colab

1. Abre [Google Colab](https://colab.research.google.com/) y sube `notebooks/00_colab_setup.ipynb` o el ZIP completo.
2. **Runtime → Cambiar tipo de entorno → GPU (T4)** antes del fine-tuning.
3. Ejecuta las celdas de `00_colab_setup.ipynb` (sube el ZIP, instala dependencias, verifica datos).
4. La celda de diagnóstico confirma `PROJECT_ROOT`, CUDA y si el modelo ya existe.

### 3. Flujo A — Ejecución completa (primera vez, ~2–4 h con GPU)

```
00_colab_setup → 01 → 02 → 03 → 04 (GPU) → 05 → 06 → 07
```

Atajo desde la raíz del proyecto en Colab:

```python
!python scripts/run_notebooks_colab.py
!python scripts/run_interpretability.py  # exporta CSVs LIME/SHAP
```

### 4. Flujo B — Solo métricas (modelo ya entrenado)

Si ya tienes `models/mbert-sv/` y `models/baseline_*.joblib` (p. ej. en Google Drive):

```python
!python scripts/run_notebooks_colab.py --from 5 --to 7 --skip 4
!python scripts/run_interpretability.py
!python scripts/generate_figures.py
```

### 5. Artefactos a descargar al terminar

| Carpeta / archivo | Contenido |
|-------------------|-----------|
| `reports/metrics/*.csv` | Métricas, slices, McNemar, benchmark |
| `reports/figures/*.png` | Gráficos del informe |
| `models/mbert-sv/` | Modelo fine-tuned (~700 MB) |

### 6. Validar el ZIP antes de subir (solo estructura, sin entrenar)

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_colab_zip.ps1
pytest tests/test_colab_zip.py -v
```

Ver también [`docs/colab-run-log.md`](docs/colab-run-log.md) para el checklist paso a paso en Colab.

## Checkpoint mBERT (auditoría)

Los pesos fine-tuned **no** se versionan en Git (`models/mbert-sv/` está en `.gitignore`).
El paquete de auditoría autocontenido es `artifacts/mbert-sv-audit.zip` (tokenizer,
`model.safetensors`, `inference_contract.json` y `MANIFEST.json` con SHA-256).

```powershell
python scripts/package_mbert_checkpoint.py
python scripts/export_mbert_predictions.py
python scripts/verify_mbert_checkpoint.py --package artifacts/mbert-sv-audit.zip
```

Guía: [`docs/guia-inferencia-mbert.md`](docs/guia-inferencia-mbert.md).
Distribución a revisores: Google Drive (mismo canal que Colab). Pegar la URL
en el manifiesto con `--artifact-url` al publicar. F1-macro oficial del
checkpoint: **0.7777** (tolerancia `1e-4`) sobre `data/processed/test.csv` (n = 460).
El test B2 añade `origen`/`plantilla_id` sin cambiar IDs ni etiquetas; el
checkpoint sigue siendo el de train v1 (`checkpoint-1072`).

## Licencia

Investigación académica — UES ESI 2026.
