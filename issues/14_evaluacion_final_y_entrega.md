# ISSUE-14 — Evaluar receta congelada y entregar el modelo reproducible

**Estado:** pendiente · **Prioridad:** P1  
**Dependencias:** [ISSUE-03](03_splits_y_evaluacion_independiente.md), [ISSUE-06](06_recoleccion_y_aprendizaje_activo.md), [ISSUE-07](07_sinteticos_controlados.md), [ISSUE-09](09_baselines_y_ablaciones.md), [ISSUE-11](11_hiperparametros_y_perdidas.md)

## Problema y evidencia

El F1 macro oficial es 0.7610 y la meta configurada 0.80. Superar 0.90 es una aspiración a comprobar, no un resultado garantizado por datos sintéticos o búsqueda de hiperparámetros.

## Trabajo a realizar

1. Congelar receta, datos y checkpoint antes de abrir la prueba nueva. Si se ejecutaron DAPT/calibración/ensamble, resolver sus issues opcionales antes del cierre.

2. Reportar macro-F1 de las cuatro clases con IDs explícitos y manejo de clases sin predicciones, precision/recall/F1/support, confusión y cortes por plataforma, dialecto, identidad neutral, ironía y amenazas.

3. Calcular intervalos bootstrap pareados y por grupos donde corresponda para diferencias frente al baseline. Añadir prueba pareada como complemento; una cifra puntual de una muestra pequeña no acredita una mejora estable.

4. Reportar test histórico aparte, baterías 75/144 como diagnósticos conocidos y batería nueva protegida desde ISSUE-03. No sumar diagnósticos sintéticos al test principal.

5. Predefinir en desarrollo criterios de adopción: mejora macro, recall amenazas y FPR dialectal sin deterioro material, y costo asumible. Tolerancia absoluta inicial propuesta de 0.02 para recall/FPR debe revisarse con soporte/intervalos antes de abrir test.

6. Informar hitos 0.80,0.85 y >0.90 con estado real e incertidumbre. No retirar casos difíciles, modificar etiquetas por predicciones del candidato ni alterar muestreo para cruzar la meta.

7. Entregar checkpoint/tokenizer, dependencias verificadas, hashes de datos, taxonomía, normalizador, revisión del modelo base, semillas, métricas/predicciones y guía de reproducción. Actualizar gates y reportes existentes por versión.

## Entregables

reports/final_v3/, paquete autocontenido verificable y guía de inferencia/reproducción; rutas propuestas.

## Criterios de aceptación

- [ ] Baseline y candidato usan la misma evaluación compatible y se identifican las métricas históricas.
- [ ] Se informa el resultado aunque no alcance 0.90; la adopción considera rendimiento por clase y costo.
- [ ] Integridad del paquete y paridad de inferencia verificadas con scripts existentes; límites y decisiones documentados.

