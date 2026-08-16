# T-03 — Ejecutar McNemar pareado mBERT vs XLM-R

**Responsable:** Cristian Escalante  
**Modelo:** mBERT (`bert-base-multilingual-cased`)  
**Prioridad:** crítica  
**Hallazgo de auditoría:** 4  
**Dependencia externa:** Julio debe entregar las predicciones XLM-R del mismo test.

## Por qué

El objetivo del proyecto es comparar mBERT y XLM-R, pero la comparación actual se basa en dos F1-macro. La auditoría observó que los intervalos de confianza se solapan y que, con 460 instancias, la diferencia puntual puede ser ruido de muestreo. La función `mcnemar_test()` ya está disponible para mBERT; falta ejecutar la prueba pareada con predicciones alineadas de XLM-R.

## Objetivo

Determinar si la diferencia de aciertos entre mBERT y XLM-R es estadísticamente significativa sobre las mismas 460 instancias del conjunto de prueba.

## Qué se debe hacer

1. Generar desde el checkpoint oficial las predicciones mBERT y vincularlas al identificador/orden de `data/processed/test.csv`.
2. Recibir el CSV de XLM-R y validar: 460 filas, mismas ids u orden, etiquetas enteras 0–3 y metadatos de checkpoint.
3. Ejecutar `mcnemar_test(y_true, y_pred_mbert, y_pred_xlmr)`.
4. Exportar estadístico, p-value, conteos discordantes `b` y `c`, tamaño de muestra, modelos/checkpoints y nivel de significancia.
5. Regenerar la tabla y el texto de comparación en el informe, interpretando un resultado no significativo como equivalencia estadística bajo este test, no como ausencia de trabajo.

## Criterios de aceptación

- [x] Las predicciones mBERT y XLM-R tienen 460 filas y se validan contra el mismo `test.csv` antes de calcular McNemar.
- [x] Existe un resultado exportado de McNemar que identifica ambos checkpoints y contiene estadístico, p-value, `b`, `c` y `n`.
- [x] El informe deja de concluir superioridad de un modelo basándose solo en la diferencia de F1-macro.
- [x] Una prueba o validación rechaza vectores desalineados, longitudes distintas o etiquetas fuera de 0–3.
- [x] Si Julio aún no entrega el vector, la tarea queda marcada como bloqueada por esa dependencia y no se fabrican resultados.

## Evidencia T-03

| Artefacto | Rol |
|-----------|-----|
| `reports/metrics/mbert_test_predictions.csv` | 460 predicciones `id,y_pred_mbert` desde `models/mbert-sv` |
| `reports/metrics/xlmr_test_predictions.csv` | 460 predicciones `id,y_pred_xlmr` desde `XLMR/02_modelo_prototipo/modelo_entrenado` |
| `reports/metrics/mcnemar_tests.csv` | McNemar pareado + baselines; fila `xlm-roberta` con checkpoints |
| `reports/metrics/mcnemar_tests.json` | Sidecar con metadatos, rutas de predicciones y resultado mBERT vs XLM-R |
| `reports/latex/generated/tab_mcnemar.tex` | Tabla LaTeX regenerada desde CSV |
| `reports/latex/generated/mcnemar_prose.tex` | Párrafo interpretativo (no rechazo de H0) |
| `tests/test_mcnemar.py` | Validación de vectores desalineados y esquema de exportación |

**Resultado mBERT vs XLM-RoBERTa** (`n=460`, `α=0.05`):

| Campo | Valor |
|-------|-------|
| `checkpoint_a` | `models/mbert-sv` |
| `checkpoint_b` | `XLMR/02_modelo_prototipo/modelo_entrenado` |
| `statistic` | 0.4098 |
| `p_value` | 0.200031 |
| `b` | 25 (mBERT acierto, XLM-R error) |
| `c` | 36 (mBERT error, XLM-R acierto) |
| `significant` | No |
| `conclusion` | No se rechaza H0: equivalencia estadística de aciertos bajo este test |

**Dependencia Julio:** checkpoint XLM-R entregado en `XLMR/02_modelo_prototipo/modelo_entrenado`. El vector se exportó con `scripts/export_xlmr_predictions.py` sobre `data/processed/test.csv`; no se fabricaron conteos.

```powershell
python scripts/export_mbert_predictions.py
python scripts/export_xlmr_predictions.py
python scripts/run_mcnemar.py
python scripts/check_report_mcnemar.py
python -m pytest tests/test_mcnemar.py -q
```
