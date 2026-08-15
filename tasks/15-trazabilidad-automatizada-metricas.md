# T-15 — Automatizar la trazabilidad de las métricas

**Prioridad:** media  
**Responsable:** equipo de métricas  
**Origen:** mejora transversal derivada de los hallazgos 1–7

## Por qué

La causa común de las cifras escritas a mano, cortes que no se ejecutaron y valores copiados de otro modelo no es el algoritmo, sino una cadena débil entre predicciones, métricas, figuras y texto. El proyecto ya dispone de utilidades de evaluación; falta convertirlas en la única fuente de los números publicados.

## Objetivo

Crear un flujo de evaluación que haga reproducibles y auditables todas las métricas publicadas, desde el checkpoint y el dataset hasta el informe final.

## Qué se debe hacer

1. Generar por ejecución un manifiesto de métricas con modelo, checkpoint/hash, commit, versión y hash del split, semilla, configuración, fecha y comando.
2. Guardar predicciones, probabilidades cuando existan y métricas derivadas en un directorio de ejecución inmutable o identificado de forma única.
3. Hacer que tablas, figuras y capítulos del informe se generen desde esos artefactos, no desde valores escritos a mano.
4. Añadir validaciones de esquema, tamaños esperados, rango de etiquetas, correspondencia de ids y presencia de todos los cortes requeridos.
5. Incorporar pruebas de regresión para los cálculos sensibles de la auditoría: FPR dialectal, cortes `SI`/`SÍ`, baseline mayoritario, paridad de configuración y McNemar.

## Criterios de aceptación

- [ ] Una ejecución produce un manifiesto suficiente para identificar exactamente datos, modelo y configuración de cada métrica.
- [ ] Predicciones y resultados derivados se pueden recomputar a partir del manifiesto en un entorno limpio.
- [ ] Tablas y figuras del informe se regeneran automáticamente desde los artefactos de ejecución.
- [ ] Las validaciones fallan ante métricas hardcodeadas, predicciones desalineadas, cortes vacíos inesperados o metadatos faltantes.
- [ ] Las pruebas de regresión cubren los hallazgos críticos de la auditoría y se ejecutan antes de publicar el informe.

