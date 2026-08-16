# Guía de inferencia mBERT

Checkpoint oficial: `models/mbert-sv/` (mejor validación: `checkpoint-1072`, época 4,
semilla 42). Los pesos **no** van en Git. El paquete de auditoría es
`artifacts/mbert-sv-audit.zip`.

## Contrato (idéntico a entrenamiento y evaluación)

| Campo | Valor |
|-------|-------|
| Columna de texto | `texto_modelo` |
| `max_length` | 128 |
| Normalizador | `normalize_for_model` v1.1 |
| Etiquetas | 0 No Tóxico, 1 Lenguaje Ofensivo, 2 Discurso de Odio, 3 Amenazas/Violencia |
| Test | `data/processed/test.csv` (n = 460) |

**Corpus B2:** el test añade las columnas `origen` y `plantilla_id` respecto al
v1; los IDs, textos y etiquetas son idénticos. El checkpoint empaquetado
(`checkpoint-1072`) se entrenó sobre train v1 (2 144 reales). La métrica oficial
de test es F1-macro **0.7777** (T-04).

`texto_modelo` ya está normalizado. Si el texto es crudo, aplicar
`normalize_for_model` (el script de inferencia lo hace).

## Comandos

```powershell
# Empaquetar (genera ZIP + manifiesto SHA-256)
python scripts/package_mbert_checkpoint.py

# Vector de 460 predicciones alineadas al test (corrida oficial T-04)
python scripts/evaluate_mbert_official.py --model-dir models/mbert-sv

# Equivalente: export_mbert_predictions.py invoca la misma corrida completa
python scripts/export_mbert_predictions.py --model-dir models/mbert-sv

# Verificar el ZIP en un directorio temporal (no usa models/mbert-sv)
python scripts/verify_mbert_checkpoint.py --package artifacts/mbert-sv-audit.zip

# Demo / modo interactivo
python scripts/inferencia_mbert.py --model-dir models/mbert-sv
```

Salidas versionables:

- `reports/metrics/mbert_test_predictions.csv` — columnas `id`, `y_pred_mbert`
- `reports/metrics/mbert_test_probabilities.csv` — columnas `id`, `p0`…`p3`
- `reports/metrics/mbert_test_metrics.csv` — métricas globales y bootstrap
- `reports/metrics/mbert_confusion_matrix.csv` — matriz 4×4
- `reports/metrics/mbert_calibration.csv` — bins de confiabilidad (ECE)
- `reports/metrics/mbert_official_run.json` — metadatos de la corrida única
- `reports/metrics/mbert_f1_traceability.csv` — inventario histórico vs oficial
- `reports/metrics/mbert_checkpoint_manifest.json`
- `reports/metrics/mbert_checkpoint_verify.json`

## Carga mínima del paquete descomprimido

El tokenizer y los pesos se cargan solo desde el directorio del paquete:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_dir = "mbert-sv"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
```

No hace falta `bert-base-multilingual-cased` en caché ni otros archivos locales
no listados en `MANIFEST.json`.

## Dónde obtener el ZIP

1. Local: `artifacts/mbert-sv-audit.zip` (gitignored, ~700 MB).
2. Canal de artefactos de la tesina: Google Drive. Pegar la URL pública en
   `artifact_url` del manifiesto al publicar:

```powershell
python scripts/package_mbert_checkpoint.py --artifact-url "https://drive.google.com/..."
```
