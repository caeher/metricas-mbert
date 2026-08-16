# T-01 — Corregir el baseline de clase mayoritaria

**Responsable:** Cristian Escalante  
**Modelo:** mBERT (`bert-base-multilingual-cased`)  
**Prioridad:** alta  
**Hallazgo de auditoría:** 5

## Por qué

La función `trivial_baselines()` calcula como si la precisión y el recall de la clase mayoritaria fueran ambos su proporción en el test. Si un clasificador predice siempre la clase mayoritaria, su recall para esa clase es `1.0`. El error produjo F1-macro `0.087` en lugar de `0.128` y contradice el valor aproximado `0.129` publicado en el informe de preparación de datos.

## Objetivo

Corregir el piso de referencia de clase mayoritaria para que la comparación de mBERT contra baselines sea matemáticamente correcta y consistente en todos los informes.

## Qué se debe hacer

1. Corregir `trivial_baselines()` en `src/discurso_odio/evaluation/metrics.py`: para la clase mayoritaria usar precisión `p` y recall `1.0`; las demás clases deben contribuir F1 cero.
2. Mantener explícitos el número de clases y la definición de F1-macro para que la fórmula sea revisable.
3. Añadir una prueba unitaria con la distribución real de `data/processed/test.csv`: 159 de 460 ejemplos en la clase mayoritaria.
4. Regenerar `trivial_baselines.csv`, la comparación de modelos, las tablas LaTeX, notebooks y capítulos que consumen ese resultado.
5. Revisar que el informe EDA v1.1 y el informe final indiquen la misma fuente y regla de redondeo.

## Criterios de aceptación

- [x] La prueba con 159/460 produce F1-macro `0.128` para el baseline de clase mayoritaria.
- [x] Hay pruebas para el caso real y para al menos otra distribución de cuatro clases.
- [x] Ningún artefacto generado ni capítulo publicado conserva `0.087` como valor del baseline mayoritario.
- [x] Las tablas de EDA y evaluación usan el mismo valor y documentan el redondeo aplicado.
- [x] La regeneración completa de métricas termina sin cambios manuales en los resultados publicados.

