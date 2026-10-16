# ISSUE-02 — Auditar y unificar fuentes antiguas y nuevas

**Estado:** en curso — auditor e inventarios generados; pendientes revisión humana y adjudicación · **Prioridad:** P0  
**Dependencias:** [ISSUE-01](01_taxonomia_y_correcciones.md)

## Problema y evidencia

Train contiene 3,293 filas (2,144 reales y 1,149 sintéticas), val=459 y test=460. lenguaje_ofensivo.csv tiene 1,328 filas Plataforma=Twitter y thenewdata_X.csv tiene 1,000 Plataforma=X: ambas fuentes nuevas son X. Todas sus filas tienen Validado=NO y Listo_Entrenamiento=NO. Una comparación exploratoria encontró tres textos compartidos entre ellas, con etiquetas de origen diferentes.

## Trabajo a realizar

1. Implementar scripts/audit_dataset_sources.py y un esquema canónico con ID global, ID/fuente original, texto original/normalizado, plataforma original/canónica, origen real/sintético, etiqueta original/final, estado, fecha, URL, grupo de autor/hilo cuando exista y hashes. Son entregables propuestos.

2. Inventariar train/val/test, corpus_completo y baterías; no duplicar el corpus importando corpus_completo además de sus splits. Canonizar Twitter como X preservando la procedencia.

3. Auditar duplicados de texto exacto, URL/publicación, similitud, autor/hilo y plantillas. Enviar a adjudicación los tres conflictos conocidos; no resolver por orden de archivo ni por mayoría automática.

4. Auditar codificación: diferenciar caracteres realmente perdidos de problemas de visualización. Detectar vacíos, fechas anómalas, versiones de texto y fuentes sin procedencia verificable; conservar originales y cuarentena.

5. Revisar origen geográfico y lenguaje: 497 notas de X indican ubicación inferida y no confirmada por perfil. Registrar país/dialecto como confirmado, inferido o desconocido, sin asumir salvadoreñidad por pertenecer al archivo.

6. Exportar inventario, registros canónicos y ledger de exclusiones/conflictos bajo reports/data_audit/ y data/interim/ (propuestos).

## Entregables

Auditor reproducible, inventario por clase/plataforma/origen, corpus canónico y ledger de deduplicación.

## Criterios de aceptación

- [ ] Se reproducen recuentos y se explica cualquier diferencia; los tres conflictos tienen resolución humana trazable.
- [ ] Ningún duplicado se cuenta dos veces y las etiquetas provisionales nunca se presentan como finales.
- [x] Se documenta que cero coincidencias exactas con splits antiguos no prueba ausencia de similitud o fuga.

## Ejecución de la auditoría

El auditor y el esquema se documentan en
[`docs/auditoria_fuentes_datos.md`](../docs/auditoria_fuentes_datos.md). Para
regenerar inventarios, registros y ledgers desde la raíz del proyecto:

```powershell
python scripts/audit_dataset_sources.py
```

En el corte del 2026-10-16, el script reproduce train=3,293 (2,144 reales,
1,149 sintéticas), val=459 y test=460. `corpus_completo.csv` tiene los mismos
IDs y campos de los splits y no se suma otra vez. Las dos fuentes candidatas
suman 2,328 filas X; todas permanecen sin validar y no listas para
entrenamiento. No se encontraron coincidencias exactas entre ellas y los splits,
lo cual no descarta similitud ni fuga.

La cola de adjudicación contiene los tres conflictos exactos conocidos y un
conflicto adicional de la misma publicación con versiones de texto diferentes.
Se preservan las cuatro parejas y sus etiquetas originales; ningún valor se
eligió como etiqueta final. También quedan 1,728 pares de similitud para
revisión y las anomalías de fecha/codificación registradas en sus ledgers. La
resolución humana sigue pendiente.

