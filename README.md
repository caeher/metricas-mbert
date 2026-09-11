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

Paquete listo en [`google-colab/`](google-colab/README.md): código, datos procesados, baselines TF-IDF y pesos finales de mBERT (~700 MB).

### 1. Generar el ZIP localmente

```bash
python scripts/build_colab_zip.py
```

En Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_colab_zip.ps1
```

Salida: `google-colab/metricas-proyecto-ml-colab.zip` con carpeta raíz `metricas-proyecto-ml/` (requerida por el setup de Colab). También se genera `google-colab/00_colab_setup.ipynb` y el staging en `google-colab/metricas-proyecto-ml/`.

### 2. Subir y configurar en Colab

1. Sube el ZIP a **Google Drive** (recomendado por tamaño ~700 MB) o usa upload directo.
2. Abre [Google Colab](https://colab.research.google.com/) con `google-colab/00_colab_setup.ipynb`.
3. **Runtime → Cambiar tipo de entorno → GPU (T4)** antes del fine-tuning.
4. Ejecuta las celdas de setup (extrae ZIP, instala deps con `colab_install_deps.py`, verifica datos y modelo).
5. La celda de diagnóstico confirma `PROJECT_ROOT`, CUDA y `models/mbert-sv/model.safetensors`.

### 3. Flujo A — Ejecución completa (primera vez, ~2–4 h con GPU)

```
00_colab_setup → 01 → 02 → 03 → 04 (GPU) → 05 → 06 → 07
```

Atajo desde la raíz del proyecto en Colab:

```python
!python scripts/run_notebooks_colab.py
!python scripts/run_interpretability.py  # exporta CSVs LIME/SHAP
```

### 4. Flujo B — Solo métricas (pesos ya incluidos en el ZIP)

El paquete ya incluye `models/mbert-sv/` y `models/baseline_*.joblib`:

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

```bash
python scripts/build_colab_zip.py
pytest tests/test_colab_zip.py -v
```

Ver también [`google-colab/README.md`](google-colab/README.md) y [`docs/colab-run-log.md`](docs/colab-run-log.md).

## Checkpoint mBERT (auditoría)

Los pesos fine-tuned **no** se versionan en Git (`models/mbert-sv/` está en `.gitignore`).
El paquete de auditoría autocontenido es `artifacts/mbert-sv-audit.zip` (tokenizer,
`model.safetensors`, `inference_contract.json` y `MANIFEST.json` con SHA-256).

```powershell
python scripts/package_mbert_checkpoint.py
python scripts/verify_mbert_checkpoint.py --package artifacts/mbert-sv-audit.zip
python scripts/evaluate_mbert_official.py
python scripts/generate_latex_report.py
python scripts/check_report_mbert_official.py
```

Guía: [`docs/guia-inferencia-mbert.md`](docs/guia-inferencia-mbert.md).
Distribución a revisores: Google Drive (mismo canal que Colab). Pegar la URL
en el manifiesto con `--artifact-url` al publicar.

**Métrica oficial (T-04/B2):** F1-macro **0.7610** (tolerancia `1e-4`) sobre
`data/processed/test.csv` (n = 460), checkpoint `checkpoint-1644`, semilla 42.
Trazabilidad: `reports/metrics/mbert_f1_traceability.csv`. Una sola corrida
genera predicciones, métricas, matriz de confusión y calibración
(`reports/metrics/mbert_official_run.json`).

**Meta del proyecto:** F1-macro ≥ **0.80** (`config/settings.yaml`). Pipeline integral:

```powershell
# Fase 0: análisis de errores (sin entrenar)
python scripts/analyze_mbert_errors.py

# Fases 1–4 + gates (requiere GPU para entrenamiento)
python scripts/run_f1_target_pipeline.py

# Solo evaluar gates con métricas existentes
python scripts/evaluate_f1_gates.py --use-existing-metrics

# Barrido HP (val only)
python scripts/sweep_mbert_f1.py --max-runs 6

# Receta F1 recomendada (manual)
python scripts/train_mbert.py --b2 --exclude-adversarial-overlap `
  --synthetic-policy b2_no_template_heavy --synthetic-effective-fraction 0.35 `
  --platform-class-boost default --label-smoothing 0.0 --loss-type ordinal `
  --layerwise-lr-decay 0.95 --epochs 6 --output-dir models/mbert-sv-f1
```

Receta y gates en `config/settings.yaml` → sección `f1_target`.

## Licencia

Investigación académica — UES ESI 2026.
