# Guía de preparación para Google Colab

Este documento describe cómo llevar el proyecto a Colab. **No ejecuta entrenamiento ni métricas** — solo el setup y la verificación de que el bundle está listo.

## Qué incluye el paquete Colab

1. **ZIP con carpeta raíz** — `google-colab/metricas-proyecto-ml-colab.zip` empaqueta `metricas-proyecto-ml/` con datos, código, notebooks, baselines y pesos mBERT finales.
2. **Notebook de setup** — `google-colab/00_colab_setup.ipynb` (copia de `notebooks/00_colab_setup.ipynb`) extrae, instala deps y verifica rutas.
3. **Rutas unificadas** — notebooks 01–07 usan `discurso_odio.paths.PROJECT_ROOT`.
4. **Modelos incluidos** — `models/baseline_*.joblib` y `models/mbert-sv/` (pesos finales, sin checkpoints intermedios).
5. **Directorio `google-colab/`** — staging descomprimible y documentación en `google-colab/README.md`.

## Paso 1 — Generar el ZIP (en tu PC)

```bash
python scripts/build_colab_zip.py
```

En Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_colab_zip.ps1
```

Salida:

| Archivo | Descripción |
|---------|-------------|
| `google-colab/metricas-proyecto-ml-colab.zip` | Bundle completo (~700 MB) |
| `google-colab/metricas-proyecto-ml/` | Staging descomprimible |
| `google-colab/00_colab_setup.ipynb` | Notebook de entrada |

Validar estructura (opcional, no entrena nada):

```bash
pytest tests/test_colab_zip.py -v
```

## Paso 2 — Subir a Google Colab

1. Ir a [Google Colab](https://colab.research.google.com/).
2. **Subir el ZIP a Google Drive** (recomendado; ~700 MB es frágil con upload directo).
3. **Archivo → Subir notebook** → `google-colab/00_colab_setup.ipynb`.
4. **Runtime → Cambiar tipo de entorno → GPU (T4)** (necesario solo para notebook 04).

## Paso 3 — Ejecutar setup en Colab

En `00_colab_setup.ipynb`, en orden:

| Celda | Acción |
|-------|--------|
| Opción A | Subir `metricas-proyecto-ml-colab.zip` desde PC |
| Opción B | Montar Drive y apuntar al ZIP (recomendado) |
| Instalar deps | `python scripts/colab_install_deps.py` + reiniciar runtime |
| Variables | `RUN_PROFILE=gpu`, `MAX_STEPS=0` |
| Verificación | `Train: N filas` con N > 0 |
| Diagnóstico | `PROJECT_ROOT: /content/metricas-proyecto-ml`, `models/mbert-sv/model.safetensors` |

Si la verificación falla, regenera el ZIP con `python scripts/build_colab_zip.py`.

## Paso 4 — Ejecutar notebooks (en Colab, no en local)

| Notebook | Qué hace | GPU |
|----------|----------|-----|
| 01_eda | EDA | No |
| 02_preprocesamiento_features | Features | No |
| 03_baselines_tfidf | Baselines TF-IDF (ya incluidos; puede re-ejecutar) | No |
| 04_finetuning_mbert | Entrena mBERT (opcional si usas Flujo B) | **Sí** |
| 05_metricas_y_comparacion | Métricas test | No |
| 06_benchmark_externo | Benchmark HF | No (internet) |
| 07_interpretabilidad | LIME + SHAP | No |

**Atajo en Colab** (desde la raíz del proyecto):

```python
!python scripts/run_notebooks_colab.py
!python scripts/run_interpretability.py
```

**Flujo B — pesos ya en el ZIP** (saltar entrenamiento):

```python
!python scripts/run_notebooks_colab.py --from 5 --to 7 --skip 4
!python scripts/run_interpretability.py
!python scripts/generate_figures.py
```

## Paso 5 — Descargar resultados

Al terminar en Colab, descarga:

- `reports/metrics/*.csv`
- `reports/figures/*.png`
- `models/mbert-sv/` (si re-entrenaste en Colab)

## Checklist de validación en Colab

| Paso | Verificación | ☐ |
|------|--------------|---|
| ZIP en Drive o subido | `Proyecto en: /content/metricas-proyecto-ml` | |
| pip install | Sin errores; runtime reiniciado | |
| Datos | `Train: N filas`, N > 0 | |
| Modelo incluido | `models/mbert-sv/model.safetensors` | |
| GPU (antes 04) | `CUDA disponible: True` | |
| Baselines | `models/baseline_*.joblib` × 3 | |
| Métricas (05) | `reports/metrics/mbert_test_metrics.csv` | |

## Problemas frecuentes

| Síntoma | Solución |
|---------|----------|
| `Faltan archivos` tras extraer | Regenerar ZIP con `python scripts/build_colab_zip.py` |
| Upload ZIP falla en Colab | Usar Google Drive (Opción B) |
| `pip install -e .` falla | Verificar que `%cd` apunta a `/content/metricas-proyecto-ml` |
| Notebook 05: modelo no encontrado | Verificar extracción; el ZIP debe incluir `models/mbert-sv/` |
| Timeout en notebook 04 | Usar Flujo B con `--skip 4` |
