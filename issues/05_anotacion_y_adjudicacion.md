# ISSUE-05 — Revisar datos antiguos y nuevos y adjudicar desacuerdos

**Estado:** piloto importado en Argilla local; anotación y adjudicación humanas pendientes · **Prioridad:** P0  
**Dependencias:** [ISSUE-04](04_integracion_argilla.md)

## Problema y evidencia

Las 2,328 filas nuevas equivalen exploratoriamente a 2,325 textos únicos entre fuentes, todos pendientes de validar. Sus rótulos solo cubren ofensa y odio. Las correcciones de taxonomía también pueden exigir revisar las etiquetas antiguas.

## Trabajo a realizar

1. Ejecutar un piloto estratificado propuesto de 200 textos reales: cuatro clases antiguas, ambas plataformas, nuevas fuentes y casos fronterizos. Dos anotadores trabajan de forma independiente antes de ver etiquetas previas o predicciones.

2. Medir acuerdo porcentual y Cohen kappa nominal con n, matriz de desacuerdos y cortes por fuente/clase; usar otro estimador apropiado si hay más anotadores. Un segundo LLM o el clasificador no sustituye un segundo anotador humano independiente.

3. Proponer kappa >=0.75 como objetivo operativo inicial, junto con resultados por clase y análisis cualitativo. Si falla, revisar instrucciones y recalibrar con otra muestra; no presentar el umbral como garantía de F1.

4. Revisar todos los candidatos nuevos y el corpus real antiguo. Priorizar los tres conflictos, fronteras ofensa/odio/amenaza, jerga neutral, política y sexismo. La auditoría de evaluación sigue una campaña separada y ciega.

5. Resolver desacuerdos por un tercer criterio humano, conservando respuestas originales, decisión final, motivo y versión de guía. Poner en cuarentena contexto insuficiente y procedencia dudosa.

6. Revisar sintéticos existentes en su campaña propia y exportar únicamente registros con clase final y controles de calidad completos. No convertir Validado=NO a SI por una simple coincidencia de lexicón.

## Entregables

Exportaciones versionadas de Argilla, reports/annotation/agreement.*, ledger de adjudicación y registros adjudicados; rutas propuestas.

Preparación disponible: [guía de ejecución del piloto](../docs/annotation_pilot_issue05.md), [selector estratificado reproducible](../scripts/prepare_annotation_pilot.py), [analizador de acuerdo, ledger y release QC](../scripts/analyze_annotation_exports.py) y manifiesto generado `reports/annotation/pilot_sample_v1.csv`. El manifiesto contiene solo IDs y estratos internos; no es una exportación de etiquetas.

## Criterios de aceptación

- [ ] Cada registro elegible tiene respuestas y decisión trazables; todos los candidatos siguen bloqueados hasta revisión.
- [ ] Se cuantifican cambios de clase antiguos/provisionales por fuente y subclase.
- [ ] Los casos insuficientes no pasan a clase 0 y hay revisión humana familiarizada con dialecto salvadoreño.

