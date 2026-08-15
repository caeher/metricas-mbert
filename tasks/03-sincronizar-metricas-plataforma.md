# T-03 — Sincronizar métricas por plataforma en el informe

**Prioridad:** crítica  
**Responsable:** Julio (XLM-R)  
**Hallazgo de auditoría:** 3

## Por qué

El texto del informe de XLM-R atribuyó a ese modelo los resultados por plataforma de mBERT. Para XLM-R los valores auditados son F1-macro 0.867 en X y 0.715 en Facebook; los valores copiados pertenecían al otro modelo. Esta discrepancia permite cuestionar la fiabilidad del informe aun cuando la gráfica fuese correcta.

## Objetivo

Garantizar que cada número por plataforma pertenezca al modelo, checkpoint y ejecución que declara el informe.

## Qué se debe hacer

1. Localizar todas las referencias narrativas, tablas y figuras que contengan métricas por plataforma de XLM-R.
2. Generar una única tabla de resultados por plataforma para cada modelo, con nombre de modelo, hash/identificador de checkpoint, `n`, F1-macro y accuracy.
3. Sustituir en el texto de XLM-R las cifras ajenas por sus valores propios o por referencias generadas a la tabla.
4. Añadir una revisión de consistencia que compare las cifras publicadas con el CSV de salida antes de compilar el informe.

## Criterios de aceptación

- [x] El informe de XLM-R reporta 0.867 para X y 0.715 para Facebook al usar el checkpoint auditado, o explica el cambio de ejecución con su evidencia.
- [x] Ningún texto, tabla o leyenda de XLM-R utiliza los valores de plataforma de mBERT.
- [x] Cada métrica por plataforma identifica modelo, checkpoint y tamaño del subconjunto.
- [x] Una verificación automatizada falla si una cifra del informe difiere del artefacto de métricas generado.

