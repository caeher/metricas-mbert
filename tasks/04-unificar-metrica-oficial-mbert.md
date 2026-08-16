# T-04 — Unificar la métrica oficial y su trazabilidad

**Responsable:** Cristian Escalante  
**Modelo:** mBERT (`bert-base-multilingual-cased`)  
**Prioridad:** alta  
**Origen:** consistencia detectada en la auditoría de mBERT

## Por qué

La auditoría cita F1-macro `0.7862` para mBERT, mientras que el pipeline del repositorio reporta aproximadamente `0.7777`. La auditoría no pudo reproducir mBERT porque faltaban pesos, por lo que ambas cifras pueden corresponder a checkpoints, semillas o ejecuciones distintas. Publicar ambas sin identificar su procedencia debilita la defensa del análisis.

## Objetivo

Definir una única métrica oficial de mBERT y conservar la evidencia que permita rastrearla desde el informe hasta el checkpoint y las predicciones que la originan.

## Qué se debe hacer

1. Inventariar cada F1-macro de mBERT publicada, su archivo fuente, checkpoint, commit, semilla, split y fecha de ejecución.
2. Seleccionar el checkpoint oficial conforme a una regla documentada —por ejemplo, mejor checkpoint de validación antes de evaluar una sola vez en test— y no por la cifra más alta.
3. Reejecutar la evaluación del checkpoint elegido sobre el `test.csv` identificado y guardar predicciones, probabilidades, métricas globales, intervalo bootstrap, matriz de confusión y calibración.
4. Actualizar CSV, tablas LaTeX, notebooks, capítulos y README para que usen exclusivamente la métrica oficial y redondeen desde la misma salida.
5. Registrar las cifras históricas distintas como resultados de otra ejecución o retirarlas con una nota de trazabilidad; no mezclarlas en una misma comparación.

## Criterios de aceptación

- [x] Existe una tabla de trazabilidad que explica cada cifra histórica de F1-macro de mBERT o la marca como no reproducible.
- [x] La métrica oficial identifica checkpoint, commit, semilla, hash del test y fecha de evaluación.
- [x] El vector de predicciones, las métricas CSV y las tablas del informe provienen de la misma ejecución.
- [x] El informe muestra una sola cifra oficial de mBERT para cada métrica y todas las referencias coinciden al redondear.
- [x] Un revisor puede reproducir la cifra oficial siguiendo el manifiesto de T-02.

## Evidencia T-04

| Artefacto | Rol |
|-----------|-----|
| `reports/metrics/mbert_f1_traceability.csv` | Inventario histórico vs oficial (`0.7862` → no reproducible) |
| `reports/metrics/mbert_official_run.json` | `run_id`, commit, semilla 42, `checkpoint-1072`, hash test, fecha |
| `reports/metrics/mbert_test_metrics.csv` | F1-macro 0.7777, bootstrap, ECE, métricas por clase |
| `reports/metrics/mbert_test_predictions.csv` | 460 predicciones alineadas al test |
| `reports/metrics/mbert_test_probabilities.csv` | Probabilidades `p0`–`p3` de la misma corrida |
| `reports/metrics/mbert_confusion_matrix.csv` | Matriz 4×4 |
| `reports/metrics/mbert_calibration.csv` | Bins de calibración |
| `reports/latex/generated/tab_mbert_traceability.tex` | Tabla LaTeX de trazabilidad |
| `reports/latex/generated/mbert_*_prose.tex` | Prosa generada desde CSV (sin `0.778` hardcodeado) |
| `tests/test_mbert_official_metric.py` | Consistencia CSV ↔ LaTeX ↔ `official_run` |
| `scripts/check_report_mbert_official.py` | Checker del informe |

**Métrica oficial:** F1-macro **0.7777** (`checkpoint-1072`, mejor val F1=0.7715, test n=460).

```powershell
python scripts/verify_mbert_checkpoint.py --package artifacts/mbert-sv-audit.zip
python scripts/evaluate_mbert_official.py
python scripts/generate_latex_report.py
python scripts/check_report_mbert_official.py
python -m pytest tests/test_mbert_official_metric.py -q
```
