# ISSUE-10 — Auditar normalización, emojis y truncamiento de mBERT

**Estado:** pendiente · **Prioridad:** P1  
**Dependencias:** [ISSUE-08](08_builder_dataset_y_pesos.md), [ISSUE-09](09_baselines_y_ablaciones.md)

## Problema y evidencia

El contrato actual usa max_length=128. normalize_for_model preserva tildes/mayúsculas y convierte emojis; conviven texto_original y texto_modelo. Las nuevas fuentes incluyen cadenas con ? sustituyendo caracteres, cuya recuperación debe comprobarse.

## Trabajo a realizar

1. Auditar normalización nueva/antigua para evitar doble demojización y aliases inconsistentes. Versionar normalizador y contrato con el checkpoint.

2. Medir con el tokenizer real longitudes, fragmentación de jerga/ofuscación y truncamiento por clase/plataforma. Comparar 128/256 y 384 solo si hay truncamiento material y presupuesto.

3. Preservar mBERT cased, negación, cita, emojis y grafía dialectal. No traducir jerga a insultos ni eliminar vocabulario de daño. Reparar encoding cuando sea recuperable; marcar pérdida irreversible.

4. Comparar representación vigente de emojis con alternativas controladas, manteniendo consistencia entrenamiento/inferencia. Si se agregan tokens, documentar embeddings y efectos.

5. Usar contexto padre solo con procedencia y disponibilidad esperada en inferencia. Mantener hilos agrupados y no inventar el contexto ausente.

6. Preparar controles de regresión de las correcciones: vieja, fuera, gentilicios, ataques sexistas, jerga amistosa, noticia/cita y amenaza ofuscada.

## Entregables

Reporte de tokenización/truncamiento y comparación de normalización; contrato actualizado únicamente si se adopta la variante.

## Criterios de aceptación

- [ ] Alternativas comparadas en desarrollo con entrenamiento/datos fijos.
- [ ] Paridad comprobada entre entrenamiento e inferencia; negación/citas no se destruyen.
- [ ] Una limpieza se adopta por corregir inconsistencia demostrada o mejorar resultados, no por apariencia.

