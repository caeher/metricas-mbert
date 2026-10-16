# ISSUE-13 — Comparar calibración y ensamble con validación independiente

**Estado:** pendiente · **Prioridad:** P2 opcional  
**Dependencias:** [ISSUE-11](11_hiperparametros_y_perdidas.md)

## Problema y evidencia

Existen evaluation/thresholds.py, evaluation/ensemble.py y sus scripts. Pueden mejorar decisiones/estabilidad, pero sobreajustar una validación pequeña o aumentar el costo.

## Trabajo a realizar

1. Auditar la implementación existente. Separar calibración de confianza de cambios de decisión: una temperatura escalar positiva preserva argmax y por sí sola no cambia F1 multiclase.

2. Si se prueban offsets/umbrales por clase, definir una regla que produzca exactamente una de cuatro clases. Ajustar en calibración o predicciones fuera de fold del desarrollo y controlar sobreajuste.

3. Comparar mejor mBERT individual con promedio de probabilidades de hasta tres finalistas. Alinear clases/normalizadores/manifiestos y reportar ensamble explícitamente como ensamble.

4. Medir F1 macro, recall amenazas, FPR neutral, Brier/ECE y latencia/VRAM. Elegir solo si la ganancia estable justifica costo.

5. Si se añade abstención para revisión humana, informar cobertura y errores de todos los registros; no presentar F1 de casos fáciles aceptados como F1 global.

## Entregables

Manifiestos, comparación de checkpoint individual/ensamble/calibración y contrato de inferencia reproducible.

## Criterios de aceptación

- [ ] Ningún umbral, temperatura o peso se elige con test final.
- [ ] Cada texto tiene salida válida y reproducible y se distingue calibración de mejora en F1.
- [ ] Si no hay beneficio estable se conserva el modelo simple.

