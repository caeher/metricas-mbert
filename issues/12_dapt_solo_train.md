# ISSUE-12 — Evaluar DAPT con corpus real sin exposición a evaluación

**Estado:** pendiente · **Prioridad:** P2 opcional  
**Dependencias:** [ISSUE-03](03_splits_y_evaluacion_independiente.md), [ISSUE-08](08_builder_dataset_y_pesos.md), [ISSUE-09](09_baselines_y_ablaciones.md)

## Problema y evidencia

train_mbert_dapt.py y models/dapt.py usan include_val=True por defecto y todo train, incluidos sintéticos. Aunque MLM no usa etiquetas, exponer validación al preentrenamiento no cumple el protocolo inductivo independiente propuesto.

## Trabajo a realizar

1. Cambiar default a train real y corpus de dominio adicional elegible; excluir grupos/duplicados de val, calibración, test y baterías. Filtrar sintéticos de MLM.

2. Actualizar scripts/run_f1_target_pipeline.py para respetar manifiestos; no depender de recordar --no-val manualmente.

3. Comprobar diversidad/tamaño del corpus. Con pocos miles de frases el beneficio es incierto. Proponer piloto acotado 1-2 épocas, LR [2e-5,5e-5] y masking inicial 0.15; son parámetros experimentales.

4. Inicializar la clasificación desde el encoder MLM con cabeza nueva de cuatro clases y compararla con el mismo control supervisado sin DAPT. Conservar hashes de encoder/tokenizer.

5. Comparar por clase/plataforma, jerga y costo en desarrollo. Descartar si no mejora o degrada resultados; no convertir DAPT en requisito para completar el plan.

## Entregables

DAPT y orquestador actualizados, manifiesto MLM y reports/experiments/dapt_ablation.*.

## Criterios de aceptación

- [ ] Se demuestra exclusión de textos/grupos de evaluación incluso durante MLM.
- [ ] Comparación supervisada equivalente y confirmación en varias semillas si hay señal de mejora.
- [ ] Si no aporta, se conserva la receta sin DAPT y se documenta el resultado.

