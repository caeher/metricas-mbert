# ISSUE-06 — Cubrir Facebook y clases ausentes con datos reales y aprendizaje activo

**Estado:** pendiente · **Prioridad:** P1  
**Dependencias:** [ISSUE-03](03_splits_y_evaluacion_independiente.md), [ISSUE-05](05_anotacion_y_adjudicacion.md), [ISSUE-09](09_baselines_y_ablaciones.md)

## Problema y evidencia

Las dos fuentes nuevas son X y priorizan ofensa/odio. El reporte histórico muestra Facebook F1 macro=0.664845 y X=0.818979; recall de odio/Facebook=0.4333 en 30 casos. Agregar solo X puede dejar intacto el problema principal.

## Trabajo a realizar

1. Actualizar cobertura real por plataforma/clase/fenómeno tras adjudicar. Medir jerga amistosa, identidad neutral, amenaza explícita/ofuscada, ironía, cita y política. Usar resultados históricos como pista, no como descripción del nuevo corpus.

2. Recolectar primero Facebook real y clases 0/3 ausentes de las fuentes nuevas. Añadir menciones neutrales de identidad y amenazas reales; evitar muestreo exclusivo por groserías que asocie palabras o plataformas a una etiqueta.

3. Construir pool elegible independiente de holdouts y seleccionar rondas propuestas de 300-500 registros para Argilla: 40% incertidumbre/desacuerdo, 30% diversidad/cortes deficitarios, 30% aleatorio. Registrar cada criterio.

4. Combinar margen/entropía con diversidad y calidad; no seleccionar solo textos incomprensibles. Incluir casos de alta confianza en que el modelo discrepa de personas.

5. Reentrenar con receta de control y desarrollo fijo por ronda; registrar curva de aprendizaje, número de reales aceptados y costo de anotación. Cambiar estrategia si la mejora se estabiliza.

6. Para datasets externos comprobar licencia, dominio y etiquetas. Una etiqueta binaria de toxicidad no se mapea directamente a cuatro clases sin revisión adicional.

## Entregables

scripts/select_annotation_batch.py, manifiestos de rondas, reporte de cobertura y curva de aprendizaje; rutas propuestas.

## Criterios de aceptación

- [ ] Las rondas documentan seleccionados/descartados, aceptación y métricas por plataforma/clase.
- [ ] Facebook y las clases ausentes reciben datos reales revisados; la localización inferida se mantiene explícita.
- [ ] Ninguna ronda usa errores del test sellado para seleccionar candidatos.

