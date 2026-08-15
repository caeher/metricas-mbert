# T-11 — Robustecer la detección de amenazas ofuscadas

**Prioridad:** mejora de alta urgencia  
**Responsable:** equipo de datos y modelado  
**Hallazgo de auditoría:** 9

## Por qué

La batería adversarial de XLM-R acertó solo 5 de 11 amenazas ofuscadas. Variantes como `m4t4r`, `m*tar`, el eufemismo local «dar piso» y amenazas indirectas se degradan a Ofensivo o No Tóxico. Es el fallo de mayor riesgo porque afecta la clase Amenazas/Violencia y apunta a falta de cobertura en el corpus, no a una discrepancia cosmética de métricas.

## Objetivo

Mejorar y medir la detección de amenazas escritas con ofuscación, leetspeak, eufemismos locales y formulaciones indirectas sin contaminar el conjunto de prueba principal.

## Qué se debe hacer

1. Diseñar una guía de anotación y recolectar ejemplos de entrenamiento de amenazas ofuscadas, con revisión de calidad y procedencia.
2. Crear una batería adversarial separada, versionada y bloqueada para evaluación; conservar los 11 casos auditados como línea base cuando sea permitido.
3. Ajustar entrenamiento o estrategia de normalización solo con datos de entrenamiento/validación.
4. Comparar antes y después por tipo de ofuscación, clase real, matriz de confusión y confianza; acompañar el resultado con F1 global y por clase.
5. Analizar manualmente los falsos negativos residuales y documentar los riesgos que no se resuelvan.

## Criterios de aceptación

- [ ] Existe un conjunto de evaluación adversarial versionado, separado de train/val/test y con etiquetas revisadas.
- [ ] El reporte conserva la línea base de 5/11 aciertos para amenazas auditadas y presenta el resultado posterior desagregado.
- [ ] Las nuevas muestras de entrenamiento no aparecen en la batería de evaluación ni en el test original.
- [ ] Se reportan F1 de Amenazas/Violencia, tasa de falsos negativos y F1-macro junto con los cambios en robustez.
- [ ] Los falsos negativos restantes tienen análisis y plan de mitigación documentados.

