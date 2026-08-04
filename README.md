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

## Licencia

Investigación académica — UES ESI 2026.
