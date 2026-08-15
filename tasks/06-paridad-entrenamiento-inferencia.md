# T-06 — Garantizar paridad entre entrenamiento e inferencia

**Prioridad:** alta  
**Responsable:** Julio (XLM-R); aplicar la misma regla a todo endpoint futuro  
**Hallazgo de auditoría:** 6

## Por qué

La auditoría detectó que XLM-R se entrenó con `max_length=128` y se evaluó/infería con 256. También observó que 11 % de las predicciones cambia según se aplique o no la normalización del proyecto. Sin paridad, las métricas publicadas no describen el comportamiento que verá una persona usuaria.

## Objetivo

Convertir la normalización y la longitud máxima de secuencia en un contrato único y verificable entre entrenamiento, evaluación, inferencia por lote y prototipo.

## Qué se debe hacer

1. Definir `max_length=128` y la función de normalización como parámetros de configuración versionados junto al modelo de XLM-R.
2. Hacer que entrenamiento, evaluación, scripts de inferencia y endpoint importen la misma configuración, sin valores locales duplicados.
3. Persistir esos parámetros en el manifiesto del checkpoint y en la salida de métricas.
4. Construir una prueba de paridad que compare el texto normalizado, los tokens y las predicciones de una muestra representativa entre evaluación e inferencia.
5. Volver a ejecutar las métricas afectadas una vez unificado el contrato y documentar cualquier variación respecto a la auditoría.

## Criterios de aceptación

- [ ] No existe un `max_length` de 256 en la ruta de evaluación o inferencia de XLM-R si el entrenamiento usa 128.
- [ ] Los cuatro contextos (entrenamiento, evaluación, lote y endpoint) consumen una única fuente de configuración.
- [ ] Cada reporte de métricas registra versión de normalizador y `max_length`.
- [ ] La prueba de paridad verifica resultados idénticos para entradas compartidas.
- [ ] Las métricas y el informe se regeneran después del cambio de contrato.

