# T-01 — Recalcular el sesgo dialectal desde predicciones

**Prioridad:** crítica  
**Responsable:** Julio (XLM-R), con revisión del equipo de métricas  
**Hallazgo de auditoría:** 1

## Por qué

El informe de XLM-R presentó como calculadas las tasas de falsos positivos con y sin jerga, pero el notebook contenía los valores fijos `0.142` y `0.385`. Esto compromete la trazabilidad de la evidencia, aunque el recálculo de la auditoría confirmó una brecha incluso mayor: 15/94 (15.96 %) sin jerga y 13/29 (44.83 %) con jerga entre los textos No Tóxico.

## Objetivo

Obtener y publicar las tasas de falsos positivos dialectales exclusivamente a partir de las predicciones del checkpoint evaluado y del mismo `test.csv` usado para las métricas globales.

## Qué se debe hacer

1. Eliminar las constantes manuales de notebooks, scripts y tablas de XLM-R.
2. Reutilizar o implementar una función equivalente a `dialect_bias_probe()` que filtre la clase real No Tóxico y calcule, para cada grupo de jerga, numerador, denominador y FPR.
3. Exportar un artefacto tabular con el identificador de ejecución, checkpoint, definición de jerga, `n`, falsos positivos, FPR y, si es viable, intervalos de confianza.
4. Regenerar tablas, figuras y texto del informe desde ese artefacto; no transcribir los porcentajes a mano.
5. Añadir una prueba con un conjunto sintético que cubra los grupos con y sin jerga y los casos de denominador cero.

## Criterios de aceptación

- [x] No quedan valores literales de FPR dialectal usados como resultado en el flujo de evaluación de XLM-R.
- [x] Una ejecución reproducible sobre las predicciones auditadas produce 15/94 = 15.96 % sin jerga y 13/29 = 44.83 % con jerga, o documenta explícitamente por qué se evalúa otro checkpoint o split.
- [x] El CSV/JSON generado incluye numeradores, denominadores y metadatos de ejecución, no solo porcentajes.
- [x] La prueba unitaria valida el cálculo de FPR y los grupos vacíos.
- [x] El informe cita el artefacto generado y sus cifras coinciden con él al redondear.

