# Google Colab — Paquete de replicación

Este directorio contiene el paquete listo para ejecutar el proyecto en [Google Colab](https://colab.research.google.com/), incluyendo datos procesados, baselines TF-IDF y pesos finales de mBERT.

## Contenido del paquete

| Componente | Descripción |
|------------|-------------|
| `metricas-proyecto-ml-colab.zip` | Bundle completo (~700 MB) |
| `metricas-proyecto-ml/` | Staging descomprimible (generado por el script) |
| `00_colab_setup.ipynb` | Notebook de entrada (copia de `notebooks/00_colab_setup.ipynb`) |

El ZIP incluye:

- Código fuente (`src/discurso_odio/`), scripts y config
- Datos procesados (`data/processed/`)
- Notebooks 00–07 (sin versiones `*_executed`)
- Baselines TF-IDF (`models/baseline_*.joblib`)
- Pesos finales mBERT en `models/mbert-sv/` (sin checkpoints intermedios)

## Generar o actualizar el paquete

Desde la raíz del repositorio:

```bash
python scripts/build_colab_zip.py
```

En Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_colab_zip.ps1
```

Validar estructura (sin entrenar):

```bash
pytest tests/test_colab_zip.py -v
```

## Subir a Colab

El ZIP pesa ~700 MB. **Recomendado: Google Drive** (el upload directo en Colab puede fallar o ser lento).

1. Sube `google-colab/metricas-proyecto-ml-colab.zip` a tu Drive.
2. Abre Colab y sube `google-colab/00_colab_setup.ipynb` (o ábrelo desde Drive).
3. **Runtime → Cambiar tipo de entorno → GPU (T4)** antes del fine-tuning (notebook 04).
4. En el setup, usa **Opción B (Drive)** si el ZIP está en Drive, o **Opción A** si lo subes desde tu PC.

Requisitos en Colab:

- ~2 GB libres (ZIP + extracción + modelo)
- GPU T4 solo para notebook 04 (entrenamiento)
- Internet para benchmarks externos (notebook 06)

## Flujos de ejecución

### Flujo A — Ejecución completa (primera vez, ~2–4 h con GPU)

```
00_colab_setup → 01 → 02 → 03 → 04 (GPU) → 05 → 06 → 07
```

Atajo desde la raíz del proyecto en Colab:

```python
!python scripts/run_notebooks_colab.py
!python scripts/run_interpretability.py
```

### Flujo B — Solo métricas (pesos ya incluidos en el ZIP)

El paquete ya trae `models/mbert-sv/` y baselines. Salta el entrenamiento:

```python
!python scripts/run_notebooks_colab.py --from 5 --to 7 --skip 4
!python scripts/run_interpretability.py
!python scripts/generate_figures.py
```

## Descargar resultados

Al terminar en Colab, descarga:

- `reports/metrics/*.csv`
- `reports/figures/*.png`
- `models/mbert-sv/` (si re-entrenaste en Colab)

## Checklist rápido

| Paso | Verificación |
|------|--------------|
| ZIP en Drive o subido | `Proyecto en: /content/metricas-proyecto-ml` |
| Dependencias | `python scripts/colab_install_deps.py` sin errores; reiniciar runtime |
| Datos | `Train: N filas`, N > 0 |
| Modelo incluido | `models/mbert-sv/model.safetensors` existe |
| GPU (antes de 04) | `CUDA disponible: True` |

Guía detallada: [`docs/colab-run-log.md`](../docs/colab-run-log.md).
