# T-04 — Comparar mBERT y XLM-R con McNemar pareado

**Prioridad:** crítica  
**Responsables:** Julio y Cristian  
**Hallazgo de auditoría:** 4

## Por qué

La comparación central del proyecto se limita hoy a dos F1-macro. Los intervalos de confianza se solapan y el conjunto de prueba tiene 460 instancias; por ello la diferencia puntual no demuestra que un modelo sea superior. La auditoría indica que ya existe una implementación de McNemar para mBERT, pero falta el vector de predicciones de XLM-R.

## Objetivo

Establecer si la diferencia entre mBERT y XLM-R es estadísticamente significativa sobre exactamente las mismas 460 instancias de prueba.

## Qué se debe hacer

1. Exportar las predicciones de XLM-R en un CSV versionado como artefacto de evaluación, con `id` si está disponible, `y_pred_xlmr`, checkpoint y metadatos de ejecución.
2. Validar que tiene 460 filas, etiquetas enteras de 0 a 3 y el mismo orden o los mismos identificadores que `data/processed/test.csv`.
3. Generar o cargar las predicciones mBERT de la misma versión de `test.csv`.
4. Ejecutar `mcnemar_test(y_true, y_pred_mbert, y_pred_xlmr)` y guardar estadístico, p-value y los conteos discordantes `b` y `c`.
5. Incorporar el resultado y su interpretación en el informe, sin tratar un resultado no significativo como un fallo.

## Criterios de aceptación

- [x] Existe un vector de predicciones XLM-R validado contra las 460 filas del test.
- [x] El resultado McNemar se exporta a `mcnemar_tests.csv` o a un artefacto equivalente con ambos modelos claramente identificados.
- [x] Se reportan estadístico, p-value, `b`, `c`, tamaño de muestra y la conclusión con el umbral de significancia definido.
- [x] El informe compara los modelos usando la prueba pareada, no solo la diferencia de F1.
- [x] Hay una prueba que rechaza vectores de longitud, orden o espacio de etiquetas incompatibles.

