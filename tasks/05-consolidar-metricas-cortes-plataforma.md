# T-05 — Consolidar métricas por cortes y plataforma

**Responsable:** Cristian Escalante  
**Modelo:** mBERT (`bert-base-multilingual-cased`)  
**Prioridad:** alta  
**Origen:** recomendación de auditoría para la siguiente iteración

## Por qué

La auditoría reconoce que el código de mBERT calcula correctamente cortes con `SI` y `SÍ`, incluida la sonda de sesgo dialectal. Sin embargo, el objetivo del proyecto requiere reportar siempre el desempeño por plataforma y existen diferencias entre cifras históricas: la auditoría menciona aproximadamente 0.8503 en X y 0.6896 en Facebook, mientras el repositorio actual muestra cerca de 0.839 y 0.673. Los cortes deben formar parte de la evaluación oficial, no ser salidas parciales difíciles de relacionar con el checkpoint.

## Objetivo

Publicar de forma estable, completa y reproducible las métricas de mBERT por plataforma, jerga, sarcasmo, ironía y contexto político.

## Qué se debe hacer

1. Usar `slice_metrics()` y `dialect_bias_probe()` sobre el mismo vector de predicciones definido como oficial en T-04.
2. Exportar para cada corte: definición, tamaño de muestra, distribución de clases, F1-macro, accuracy y métricas por clase cuando el tamaño sea suficiente.
3. Incluir siempre X y Facebook; para jerga, sarcasmo, ironía y contexto político, normalizar las variantes `SI`/`SÍ`/`NO` y reportar ambos grupos cuando existan.
4. Añadir intervalos de confianza o una advertencia de baja potencia para subconjuntos pequeños.
5. Generar tablas y texto del informe desde el CSV de cortes, citando la ejecución y el checkpoint oficial.

## Criterios de aceptación

- [x] Una ejecución oficial genera resultados para X y Facebook con `n`, distribución de clases, F1-macro y accuracy.
- [x] Los cortes de jerga, sarcasmo, ironía y contexto político se calculan con las columnas y codificaciones reales del test.
- [x] Los resultados por corte proceden del mismo vector de predicciones que la métrica global de T-04.
- [x] El informe no contiene métricas de plataforma/corte escritas a mano y sus cifras coinciden con el CSV generado.
- [x] Los cortes con muestra pequeña muestran `n` y una advertencia o intervalo de confianza apropiado.

## Evidencia T-05

| Artefacto | Rol |
|-----------|-----|
| `reports/metrics/slice_metrics.csv` | Cortes mBERT/XLM-R con definición, n0–n3, F1, IC o baja potencia |
| `reports/metrics/slice_perclass_metrics.csv` | Métricas por clase en cortes con n ≥ 30 |
| `reports/metrics/dialect_bias_probe.csv` | Sonda dialectal alineada al run_id oficial |
| `reports/metrics/mbert_platform_traceability.csv` | Oficial vs histórico (0.8503/0.6896 → not_reproducible) |
| `reports/latex/generated/tab_slices.tex` | Tabla de cortes generada desde CSV |
| `reports/latex/generated/tab_platform.tex` | Plataforma con checkpoint-1072 y distribución |
| `tests/test_mbert_slices_official.py` | run_id, plataforma, SI/SÍ, baja potencia |

```powershell
python scripts/compute_slice_metrics.py
python scripts/compute_dialect_bias.py
python scripts/generate_latex_report.py
python -m pytest tests/test_slices.py tests/test_platform_report.py tests/test_dialect_bias.py tests/test_mbert_official_metric.py tests/test_mbert_slices_official.py -q
```
