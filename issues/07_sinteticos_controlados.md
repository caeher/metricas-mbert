# ISSUE-07 — Generar sintéticos contrastivos y medir su contribución

**Estado:** pendiente · **Prioridad:** P1  
**Dependencias:** [ISSUE-01](01_taxonomia_y_correcciones.md), [ISSUE-03](03_splits_y_evaluacion_independiente.md), [ISSUE-05](05_anotacion_y_adjudicacion.md), [ISSUE-09](09_baselines_y_ablaciones.md)

## Problema y evidencia

B2 ya tiene 1,149 sintéticos, 34.9% del train por conteo, muchos de plantillas. Más cantidad puede reforzar atajos; se debe medir la contribución frente a reales adjudicados.

## Trabajo a realizar

1. Preparar un piloto propuesto de 200-400 candidatos según errores de desarrollo y huecos: pares de jerga amistosa/insulto, identidad neutral/ataque, cita/amenaza y negación/afirmación. No construir etiquetas por mera inserción de una palabra o emoji.

2. Diversificar temas, longitudes, registros, posiciones de emojis y fuentes; revisión nativa del dialecto. Evitar asociación determinista grupo-emoji-clase o lenguaje que no parezca natural.

3. Guardar generador/modelo/revisión, prompt/versión, parámetros, semilla si está soportada, familia/plantilla, ejemplo padre de train, clase sugerida y revisores. La revisión humana no cambia su origen sintético.

4. Importar en Argilla y aplicar anotación/adjudicación de ISSUE-05. Rechazar clase incorrecta, baja plausibilidad, duplicados y variaciones cosméticas de ejemplos protegidos.

5. Comparar 0%,10%,25% de peso efectivo sintético con base real fija; el 35% anterior puede servir de referencia adicional. Distinguir fracción por conteo de peso efectivo y ablar plantillas antiguas/nuevos sintéticos/combinación por separado.

6. Deduplicar por texto, similitud, familia y pares. Ningún prompt puede usar val/test/baterías, y los pares/familias generados permanecen en train. Repetir auditoría tras generación.

## Entregables

Corpus sintético versionado, plantillas/prompts, ledger humano y reports/experiments/synthetic_ablation.*; rutas propuestas.

## Criterios de aceptación

- [ ] Cada sintético incorporado tiene aprobación humana, procedencia y control de solapamiento.
- [ ] La proporción elegida mejora desarrollo frente a reales con receta fija y estabilidad entre semillas, sin degradar amenazas/Facebook/jerga neutral de manera material.
- [ ] Si no aporta mejora se excluye de la receta; se conservan resultados negativos.

