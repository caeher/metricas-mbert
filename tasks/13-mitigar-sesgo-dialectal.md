# T-13 — Mitigar el sesgo dialectal salvadoreño

**Prioridad:** mejora de alta urgencia  
**Responsable:** equipo de datos y modelado  
**Hallazgo de auditoría:** 11

## Por qué

La auditoría confirmó falsos positivos sobre habla cotidiana salvadoreña: 6 de 14 frases inofensivas con jerga (43 %) fueron marcadas como tóxicas frente a 0 de 12 equivalentes en español estándar. El modelo penaliza una variante lingüística de la población que el proyecto busca proteger.

## Objetivo

Reducir la brecha de falsos positivos entre habla coloquial salvadoreña y español estándar, preservando la detección de toxicidad real.

## Qué se debe hacer

1. Ampliar y documentar ejemplos No Tóxico con jerga salvadoreña, con revisión lingüística y de anotación.
2. Mantener un conjunto de evaluación dialectal bloqueado, con pares semánticamente equivalentes con y sin jerga y sin solapamiento con entrenamiento.
3. Medir FPR, tamaño de muestra e intervalos de confianza para ambos grupos, además de F1-macro y métricas de las clases tóxicas.
4. Evaluar intervenciones de datos y, si se propone, de calibración; no adoptar una mitigación que solo oculte los falsos positivos a costa de falsos negativos graves.
5. Documentar casos residuales y las limitaciones de representatividad de la sonda dialectal.

## Criterios de aceptación

- [ ] La línea base auditada de 6/14 con jerga y 0/12 estándar queda registrada con su definición y tamaño de muestra.
- [ ] El conjunto dialectal de evaluación está separado del entrenamiento y versionado.
- [ ] Cada experimento informa FPR por grupo, brecha de FPR, intervalos de confianza, F1-macro y métricas de Amenazas/Violencia.
- [ ] El informe justifica la intervención elegida con evidencia de reducción de sesgo y análisis de compensaciones.
- [ ] Los ejemplos y la guía de anotación respetan la variación lingüística sin etiquetar la jerga como ofensiva por defecto.

