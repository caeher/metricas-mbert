# T-07 — Completar el informe estadístico de XLM-R

**Prioridad:** alta  
**Responsable:** Julio  
**Hallazgo de auditoría:** 7

## Por qué

El informe de XLM-R omitió intervalo bootstrap, matriz de confusión numérica y precisión suficiente en las métricas. En contraste, el informe mBERT ya ofrece estas piezas y la auditoría pudo calcular para XLM-R un IC bootstrap de F1-macro de [0.7665, 0.8420]. Sin incertidumbre ni errores por clase, una cifra global de 0.81 es insuficiente para interpretar el modelo.

## Objetivo

Publicar un reporte XLM-R reproducible que exponga desempeño global, incertidumbre y patrón de errores por clase.

## Qué se debe hacer

1. Calcular F1-macro, accuracy, precision macro y recall macro con cuatro decimales desde las predicciones exportadas.
2. Ejecutar bootstrap determinista (semilla, número de réplicas y método documentados) para F1-macro y registrar su IC 95 %.
3. Exportar la matriz de confusión con orden de clases explícito, reporte por clase y tamaños de soporte.
4. Generar tablas y texto del informe desde dichos artefactos; incorporar la comparación con baselines y McNemar de T-04 cuando esté disponible.
5. Añadir comprobaciones de esquema y rango para que una ejecución incompleta no se publique como resultado final.

## Criterios de aceptación

- [ ] El informe muestra métricas globales con cuatro decimales, IC 95 % bootstrap y matriz de confusión numérica etiquetada.
- [ ] Los parámetros del bootstrap y la semilla son reproducibles y están registrados.
- [ ] Cada clase tiene precision, recall, F1 y soporte reportados.
- [ ] Las tablas publicadas se regeneran desde archivos de métricas y coinciden con ellos.
- [ ] El pipeline rechaza o marca como incompleta una salida sin probabilidades o sin matriz de confusión cuando estas son requeridas.

