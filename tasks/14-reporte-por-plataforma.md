# T-14 — Institucionalizar el reporte por plataforma

**Prioridad:** mejora  
**Responsable:** equipo de métricas  
**Recomendación de auditoría:** reportar siempre F1 desagregado por plataforma

## Por qué

La plataforma es un objetivo explícito del proyecto y el desempeño cambia de forma material entre X y Facebook. El informe actual lo cubre de forma parcial y una de las inconsistencias críticas surgió precisamente al copiar cifras por plataforma entre modelos.

## Objetivo

Hacer del reporte por plataforma un componente obligatorio, trazable y comparable de toda evaluación de modelo.

## Qué se debe hacer

1. Definir un esquema común de salida por plataforma: modelo, checkpoint, plataforma, `n`, distribución de clases, F1-macro, accuracy y métricas por clase.
2. Generar la tabla y figura desde el pipeline para cada ejecución de evaluación.
3. Añadir intervalos de confianza o una advertencia visible cuando un subconjunto sea pequeño.
4. Incorporar el resultado en la metodología, evaluación y conclusiones, distinguiendo hechos de hipótesis sobre la causa de las diferencias.
5. Configurar una comprobación que marque una ejecución como incompleta si falta una plataforma presente en el test.

## Criterios de aceptación

- [ ] Cada evaluación final incluye métricas para X y Facebook con tamaño de muestra y distribución de clases.
- [ ] Las métricas de plataforma se generan desde el mismo vector de predicciones que la métrica global.
- [ ] El informe no contiene valores manuales de plataforma; consume la tabla generada.
- [ ] Los subconjuntos pequeños se señalan con su limitación estadística.
- [ ] El pipeline falla o advierte de forma explícita si falta el reporte de una plataforma observada.

