# T-05 — Corregir el baseline de clase mayoritaria

**Prioridad:** alta  
**Responsable:** Cristian  
**Hallazgo de auditoría:** 5

## Por qué

`trivial_baselines()` calcula la precisión y el recall de la clase mayoritaria con su proporción. Para un clasificador que siempre predice esa clase, su recall es 1.0. El error reduce artificialmente el F1-macro de 0.128 a 0.087 y entra en conflicto con el informe de preparación de datos, que reporta aproximadamente 0.129.

## Objetivo

Corregir la fórmula del baseline, regenerar todos los derivados y proteger el resultado con pruebas.

## Qué se debe hacer

1. Implementar el F1 de la clase mayoritaria usando precisión `p` y recall `1.0`; las demás clases deben aportar F1 cero.
2. Mantener explícita la definición de F1-macro y el número de clases para evitar fórmulas implícitas.
3. Añadir una prueba con la distribución real del test: 159 de 460 observaciones en la clase mayoritaria.
4. Regenerar `trivial_baselines.csv`, `model_comparison.csv`, tablas LaTeX, notebooks y cualquier texto que consuma el valor.
5. Revisar el informe EDA v1.1 y el informe final para que usen la misma fuente y el mismo redondeo documentado.

## Criterios de aceptación

- [x] `trivial_baselines({0: ..., 1: 159, ...})` produce F1-macro 0.128 para la distribución de prueba auditada.
- [x] La prueba unitaria cubre el caso real y al menos una distribución adicional.
- [x] Todos los artefactos regenerados muestran el valor correcto; no permanece 0.087 como resultado publicado.
- [x] Los informes de EDA y evaluación indican la misma definición, fuente y regla de redondeo.

