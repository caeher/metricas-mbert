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

- [ ] Existe una tabla de trazabilidad que explica cada cifra histórica de F1-macro de mBERT o la marca como no reproducible.
- [ ] La métrica oficial identifica checkpoint, commit, semilla, hash del test y fecha de evaluación.
- [ ] El vector de predicciones, las métricas CSV y las tablas del informe provienen de la misma ejecución.
- [ ] El informe muestra una sola cifra oficial de mBERT para cada métrica y todas las referencias coinciden al redondear.
- [ ] Un revisor puede reproducir la cifra oficial siguiendo el manifiesto de T-02.

