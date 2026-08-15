# Plan de tareas de la auditoría de métricas

Estas tareas convierten los hallazgos de `Informe_Auditoria_de_Modelos.docx` en trabajo verificable. Cubren las correcciones obligatorias de la Tabla 8 y las mejoras analíticas recomendadas por la auditoría.

## Prioridad crítica — antes de entregar

1. [T-01 — Recalcular el sesgo dialectal desde predicciones](01-recalcular-sesgo-dialectal.md)
2. [T-02 — Corregir y ampliar los cortes de evaluación](02-corregir-cortes-de-evaluacion.md)
3. [T-03 — Sincronizar métricas por plataforma en el informe](03-sincronizar-metricas-plataforma.md)
4. [T-04 — Comparar mBERT y XLM-R con McNemar pareado](04-mcnemar-pareado-mbert-xlm-r.md)

## Prioridad alta — antes de entregar

5. [T-05 — Corregir el baseline de clase mayoritaria](05-corregir-baseline-clase-mayoritaria.md)
6. [T-06 — Garantizar paridad entre entrenamiento e inferencia](06-paridad-entrenamiento-inferencia.md)
7. [T-07 — Completar el informe estadístico de XLM-R](07-completar-informe-estadistico-xlm-r.md)
8. [T-08 — Empaquetar checkpoints reproducibles](08-empaquetar-checkpoints-reproducibles.md)

## Prioridad media — higiene y trazabilidad

9. [T-09 — Configurar nombres de clase en el modelo](09-configurar-id2label.md)
10. [T-10 — Retirar artefactos pesados del control de versiones](10-higiene-control-versiones.md)
11. [T-15 — Automatizar la trazabilidad de las métricas](15-trazabilidad-automatizada-metricas.md)

## Mejoras para la siguiente iteración

12. [T-11 — Robustecer la detección de amenazas ofuscadas](11-robustez-amenazas-ofuscadas.md)
13. [T-12 — Corregir el sesgo espurio de emojis](12-sesgo-emoji-toxicidad.md)
14. [T-13 — Mitigar el sesgo dialectal salvadoreño](13-mitigar-sesgo-dialectal.md)
15. [T-14 — Institucionalizar el reporte por plataforma](14-reporte-por-plataforma.md)

## Criterio global de cierre

Una tarea no está cerrada por editar una cifra manualmente. Debe dejar un artefacto regenerable, una prueba o verificación automatizada cuando corresponda, y el informe debe consumir el resultado generado. Las métricas deben indicar conjunto de datos, checkpoint, configuración y fecha/commit de ejecución.

