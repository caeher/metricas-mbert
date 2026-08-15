# T-12 — Corregir el sesgo espurio de emojis

**Prioridad:** mejora  
**Responsable:** equipo de datos y modelado  
**Hallazgo de auditoría:** 10

## Por qué

En la batería adversarial, XLM-R acertó 6 de 18 casos con emoji. Los emojis no modifican la predicción en la dirección esperada y un emoji positivo llevó un texto amable a Lenguaje Ofensivo. La evidencia sugiere una correlación espuria: los emojis del corpus aparecen sobre todo en textos tóxicos.

## Objetivo

Reducir la dependencia espuria de la presencia de emojis y demostrarlo mediante evaluaciones contrafactuales y por grupos de emoji.

## Qué se debe hacer

1. Auditar la distribución de emojis por clase, plataforma y split; identificar escasez de ejemplos No Tóxico.
2. Incorporar ejemplos no tóxicos con emojis y revisar el equilibrio semántico de los ejemplos de entrenamiento.
3. Definir pares contrafactuales que solo cambien el emoji y una batería separada de evaluación con emojis positivos, negativos y de amenaza.
4. Comparar, mediante experimentos versionados, la conversión de emoji a descripción textual frente a alternativas de normalización.
5. Reportar accuracy, F1 por clase, cambios de predicción en pares y falsos positivos en textos amables.

## Criterios de aceptación

- [ ] Hay un análisis reproducible de distribución de emojis por clase y split.
- [ ] Se añaden ejemplos no tóxicos con emojis sin introducirlos en el conjunto de evaluación bloqueado.
- [ ] La batería contrafactual contiene pares cuya única diferencia sustantiva es el emoji.
- [ ] El informe compara al menos dos decisiones de normalización/entrenamiento y justifica la seleccionada.
- [ ] Se publican el resultado de 6/18 como línea base y los cambios de error por tipo de emoji.

