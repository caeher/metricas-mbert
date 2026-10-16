# ISSUE-11 — Corregir cobertura del barrido y optimizar la receta

**Estado:** pendiente · **Prioridad:** P1  
**Dependencias:** [ISSUE-09](09_baselines_y_ablaciones.md), [ISSUE-10](10_normalizacion_y_contexto.md)

## Problema y evidencia

En sweep_mbert_f1.py, _build_grid construye itertools.product y toma combos[:max_runs]. Con defaults de 12, las 12 combinaciones interiores consumen el presupuesto del primer LR, primera semilla y primera política. No se exploran los valores exteriores. Ya existen CE/focal/ordinal y layerwise decay.

## Trabajo a realizar

1. Sustituir el corte por muestreo determinista con cobertura o búsqueda aleatoria/Optuna. Exportar plan antes de entrenar, evitar duplicados y comprobar que LR y política no quedan constantes accidentalmente.

2. Proponer 12-18 configuraciones en una semilla/dataset fijo: LR [1e-5,2e-5,3e-5,5e-5], batch efectivo [16,32], épocas máximas [3,5,8], warmup [0,0.06,0.1], weight decay [0.01,0.05,0.1]. Muestrear combinaciones, no ejecutar todo el producto. Acumular gradientes según VRAM.

3. Seleccionar checkpoint mediante F1 macro de desarrollo y early stopping. Reinicializar modelo/optimizador por corrida y reanudar solo si coinciden hashes de configuración/datos.

4. Comparar CE ponderada/no ponderada, focal, smoothing 0/0.05 y layerwise decay 1/0.95 como factores controlados. Tratar ordinal como hipótesis: las cuatro categorías no representan necesariamente una escala ordinal semántica válida.

5. Confirmar 2-3 finalistas con semillas 42,7,13 y desarrollo/folds por grupos si el tamaño lo requiere. No optimizar la semilla por test ni ampliar búsqueda sin hipótesis.

6. Registrar métricas, fallos OOM, costos y hashes completos. Cambiar resúmenes/gates hardcodeados como passes_val_gate_0_77 por criterios versionados, sin tratar el objetivo 0.90 como resultado.

## Entregables

Ampliación de sweep_mbert_f1.py, plan/config de búsqueda y comparación reproducible de finalistas.

## Criterios de aceptación

- [ ] El presupuesto limitado cubre LR y factores declarados; añadir control específico de cobertura.
- [ ] Objetivo de selección=f1_macro; configuraciones reinicializadas y seeds usadas para medir variabilidad.
- [ ] Se conserva toda corrida, incluso negativa; mejora estable y costo documentados frente al control.

