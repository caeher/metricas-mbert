# Guía de preparación para Google Colab

Este documento describe cómo llevar el proyecto a Colab. **No ejecuta entrenamiento ni métricas** — solo el setup y la verificación de que el bundle está listo.

## Qué se corrigió para Colab

1. **ZIP con carpeta raíz** — `scripts/build_colab_zip.ps1` empaqueta `metricas-proyecto-ml/` (antes los archivos quedaban sueltos y el setup fallaba).
2. **Notebook de setup** — `notebooks/00_colab_setup.ipynb` extrae, instala deps y verifica rutas.
3. **Rutas unificadas** — notebooks 01–07 usan `discurso_odio.paths.PROJECT_ROOT`.
4. **Exportación de métricas** — notebooks 05 y 07 guardan los mismos CSVs que los scripts CLI.
5. **Un solo repo** — se eliminó la carpeta duplicada `metricas-proyecto-ml-colab/`; el ZIP se genera desde la raíz.

## Paso 1 — Generar el ZIP (en tu PC)

```powershell
cd c:\Users\echoe\Desktop\metricas-proyecto-ml
powershell -ExecutionPolicy Bypass -File scripts/build_colab_zip.ps1
```

Salida: `metricas-proyecto-ml-colab.zip` (~0.6 MB con datos procesados, sin modelo entrenado).

Validar estructura (opcional, no entrena nada):

```powershell
pytest tests/test_colab_zip.py -v
```

## Paso 2 — Subir a Google Colab

1. Ir a [Google Colab](https://colab.research.google.com/).
2. **Archivo → Subir notebook** → `notebooks/00_colab_setup.ipynb`
   - O subir el ZIP completo y abrir el notebook desde ahí.
3. **Runtime → Cambiar tipo de entorno → GPU (T4)** (necesario solo para notebook 04).

## Paso 3 — Ejecutar setup en Colab

En `00_colab_setup.ipynb`, en orden:

| Celda | Acción |
|-------|--------|
| Opción A | Subir `metricas-proyecto-ml-colab.zip` |
| Instalar deps | `pip install -r requirements.txt` + `pip install -e .` |
| Variables | `RUN_PROFILE=gpu`, `MAX_STEPS=0` |
| Verificación | Debe mostrar `Train: N filas` con N > 0 |
| Diagnóstico | `PROJECT_ROOT: /content/metricas-proyecto-ml`, CUDA si GPU activa |

Si la verificación falla, regenera el ZIP con el script actualizado.

## Paso 4 — Ejecutar notebooks (en Colab, no en local)

| Notebook | Qué hace | GPU |
|----------|----------|-----|
| 01_eda | EDA | No |
| 02_preprocesamiento_features | Features | No |
| 03_baselines_tfidf | Baselines TF-IDF | No |
| 04_finetuning_mbert | Entrena mBERT | **Sí** |
| 05_metricas_y_comparacion | Métricas test | No |
| 06_benchmark_externo | Benchmark HF | No (internet) |
| 07_interpretabilidad | LIME + SHAP | No |

**Atajo en Colab** (desde la raíz del proyecto):

```python
!python scripts/run_notebooks_colab.py
!python scripts/run_interpretability.py
```

**Si ya tienes modelo entrenado** (subir `models/mbert-sv/` a Drive):

```python
!python scripts/run_notebooks_colab.py --from 5 --to 7 --skip 4
!python scripts/run_interpretability.py
```

## Paso 5 — Descargar resultados

Al terminar en Colab, descarga:

- `reports/metrics/*.csv`
- `reports/figures/*.png`
- `models/mbert-sv/` (si entrenaste ahí)

## Checklist de validación en Colab

| Paso | Verificación | ☐ |
|------|--------------|---|
| ZIP subido | `Proyecto en: /content/metricas-proyecto-ml` | |
| pip install | Sin errores | |
| Datos | `Train: N filas`, N > 0 | |
| GPU (antes 04) | `CUDA disponible: True` | |
| Baselines (03) | `models/baseline_*.joblib` × 3 | |
| mBERT (04) | `models/mbert-sv/model.safetensors` | |
| Métricas (05) | `reports/metrics/mbert_test_metrics.csv` | |

## Problemas frecuentes

| Síntoma | Solución |
|---------|----------|
| `Faltan archivos` tras extraer | Regenerar ZIP con `build_colab_zip.ps1` actualizado |
| `pip install -e .` falla | Verificar que `%cd` apunta a `/content/metricas-proyecto-ml` |
| Notebook 05: modelo no encontrado | Ejecutar notebook 04 o subir `models/mbert-sv/` |
| Timeout en notebook 04 | Subir modelo pre-entrenado y usar `--skip 4` |
