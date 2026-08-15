# T-09 — Configurar nombres de clase en el modelo

**Prioridad:** media  
**Responsable:** Julio (XLM-R)  
**Hallazgo de auditoría:** 12

## Por qué

El modelo XLM-R guardado no expone los nombres reales de clase mediante `id2label`. Esto hace que salidas, depuración e integración muestren identificadores numéricos ambiguos y aumenta el riesgo de interpretar mal una predicción.

## Objetivo

Persistir un mapeo de etiquetas consistente con el conjunto de datos, los reportes y la interfaz de inferencia.

## Qué se debe hacer

1. Definir el mapeo bidireccional para las cuatro clases: No Tóxico, Lenguaje Ofensivo, Discurso de Odio y Amenazas/Violencia.
2. Guardarlo como `id2label` y `label2id` en la configuración del checkpoint XLM-R.
3. Asegurar que las salidas de inferencia, matriz de confusión y reportes consuman este mismo mapeo.
4. Añadir una prueba de serialización: guardar, volver a cargar y verificar el mapeo y el nombre producido para cada id.

## Criterios de aceptación

- [ ] El checkpoint recargado expone `id2label` y `label2id` completos para ids 0–3.
- [ ] Las etiquetas coinciden exactamente con la configuración y el orden de `test.csv`.
- [ ] Las salidas humanas de inferencia no dependen de índices numéricos sin nombre.
- [ ] Una prueba evita que un cambio de orden de etiquetas pase inadvertido.

