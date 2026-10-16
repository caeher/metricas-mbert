# Plan de issues: mejorar mBERT con Argilla y datos revisados

**Fecha:** 2026-10-16 · **Estado:** plan y seguimiento de implementación por issue; ISSUE-04 tiene el flujo de código preparado y el piloto humano pendiente.

Este directorio organiza la integración del dataset antiguo, las dos fuentes nuevas y las correcciones del usuario mediante Argilla, seguida de experimentos de datos, normalización e hiperparámetros. Cada issue incluye evidencia, dependencias, acciones, entregables y criterios de aceptación. Algunas rutas ya se implementaron según el estado de cada issue; eso no acredita despliegue, anotación o entrenamiento realizado.

## Evidencia de partida

- Fuente oficial: [mbert_official_run.json](../reports/metrics/mbert_official_run.json). F1 macro **0.7610126**, checkpoint-1644, test n=460; intervalo bootstrap histórico aproximadamente **[0.7179,0.7981]**. El modelo base es bert-base-multilingual-cased y la longitud registrada es 128.
- [Análisis de errores](../reports/metrics/error_analysis_b2.json): **107 errores/460**; Facebook **0.664845** frente a X **0.818979**, recall de odio/Facebook **0.4333** en 30 casos e ironía **0.5781**. Son métricas históricas, no una evaluación nueva.
- Train B2: **3,293** filas, **2,144 reales + 1,149 sintéticas**; val=459 y test=460, ambos reales. Los datos antiguos reales suman 3,063.
- [lenguaje_ofensivo.csv](../data/raw/lenguaje_ofensivo.csv): **1,328** filas y Plataforma=Twitter. [thenewdata_X.csv](../data/raw/thenewdata_X.csv): **1,000** y Plataforma=X. **Las dos fuentes nuevas son X**, no Facebook/X. Todas tienen Validado=NO y Listo_Entrenamiento=NO.
- Inspección exploratoria de texto original con NFC, casefold y espacios colapsados: **tres textos compartidos** entre las fuentes nuevas con etiquetas de origen diferentes; **2,325 textos candidatos únicos** entre ambas. No hubo coincidencias exactas así normalizadas con train/val/test antiguos. Falta auditar similitud, URLs, autores y grupos; esos recuentos no son tamaños finales de entrenamiento.
- En 497 notas de la fuente nueva X, la ubicación se declara inferida y no confirmada por perfil. La etiqueta de dialecto/geografía requiere validación.
- [Correcciones del usuario](../docs/lexico_taxonomia/README.md): menciones identitarias neutrales, órdenes y referencias de edad requieren contexto; se separaron ataques sexistas de menciones a mujeres/feministas. Siguen abiertas reglas de política, clasismo y solapamientos.
- [Config actual](../config/settings.yaml): meta **0.80**. Superar **0.90** es un objetivo exploratorio; solo debe declararse cuando se observe en una evaluación independiente con incertidumbre reportada.

## Lista de issues

| Issue | Prioridad | Trabajo |
|---|---|---|
| [01](01_taxonomia_y_correcciones.md) | P0 | Consolidar la taxonomía y las correcciones del usuario |
| [02](02_auditoria_fuentes_y_duplicados.md) | P0 | Auditar y unificar fuentes antiguas y nuevas |
| [03](03_splits_y_evaluacion_independiente.md) | P0 | Versionar splits por grupos y reservar una prueba nueva |
| [04](04_integracion_argilla.md) | P0 | Integrar Argilla con campañas, preguntas y trazabilidad |
| [05](05_anotacion_y_adjudicacion.md) | P0 | Revisar datos antiguos y nuevos y adjudicar desacuerdos |
| [06](06_recoleccion_y_aprendizaje_activo.md) | P1 | Cubrir Facebook y clases ausentes con datos reales y aprendizaje activo |
| [07](07_sinteticos_controlados.md) | P1 | Generar sintéticos contrastivos y medir su contribución |
| [08](08_builder_dataset_y_pesos.md) | P0 | Construir dataset versionado y pesos dinámicos de entrenamiento |
| [09](09_baselines_y_ablaciones.md) | P1 | Reproducir baseline y aislar efectos de datos y etiquetas |
| [10](10_normalizacion_y_contexto.md) | P1 | Auditar normalización, emojis y truncamiento de mBERT |
| [11](11_hiperparametros_y_perdidas.md) | P1 | Corregir cobertura del barrido y optimizar la receta |
| [12](12_dapt_solo_train.md) | P2 opcional | Evaluar DAPT con corpus real sin exposición a evaluación |
| [13](13_calibracion_y_ensamble.md) | P2 opcional | Comparar calibración y ensamble con validación independiente |
| [14](14_evaluacion_final_y_entrega.md) | P1 | Evaluar receta congelada y entregar el modelo reproducible |

## Orden de ejecución

1. Resolver taxonomía (01), inventario/deduplicación (02), protocolo de splits/evaluación (03) e integración Argilla (04).
2. Revisar/adjudicar orígenes antiguos y nuevos (05) y construir releases/pesos reproducibles (08).
3. Establecer baseline/ablaciones (09). Abrir rondas de recolección real (06), aumentación sintética (07) y normalización/contexto (10), guiadas por desarrollo.
4. Cada ronda de datos regresa al builder (08) y a comparaciones equivalentes (09); no requiere abrir test. Después, ejecutar búsqueda corregida (11).
5. Resolver DAPT (12) y calibración/ensamble (13) como experimentos opcionales; ejecutar evaluación y entrega (14) con la receta congelada.

Las dependencias indican requisitos para iniciar cada trabajo. Las issues complementan [tasks existentes](../tasks/README.md) y reutilizan los scripts actuales. No hay que implementar desde cero las pérdidas, los evaluadores, umbrales o ensambles ya disponibles.

## Fallos concretos a resolver

- El barrido actual toma el prefijo del producto cartesiano: con defaults de 12 corridas puede dejar learning rate, semilla y política constantes. ISSUE-11 exige cobertura efectiva antes de consumir GPU.
- DAPT incluye validación por defecto; el protocolo nuevo requiere excluir evaluación incluso de MLM. ISSUE-12 cambia ese comportamiento y el orquestador.
- Los pesos de B2 son estáticos; ISSUE-08 recalcula por train/fold y controla la combinación de pesos y muestreo.
- Incorporar solo nuevas clases ofensa/odio de X dejaría sin cubrir no toxicidad, amenazas y el dominio Facebook. ISSUE-06 mide y atiende esos vacíos.
- Los comentarios sobre sexismo y las referencias neutrales deben convertirse en reglas contextualizadas y pruebas de regresión, sin reglas automáticas por palabras.

## Presupuesto inicial propuesto

Piloto humano: 200 textos; rondas de adquisición: 300-500 candidatos; piloto sintético: 200-400; búsqueda: 12-18 recetas en una semilla y confirmación de 2-3 finalistas en tres semillas. Son presupuestos ajustables a recursos y señal de mejora, no compromisos de rendimiento. Medir VRAM/tiempo por corrida y fijar criterio de parada antes de ampliar.

Para comparar, registrar hipótesis, dataset/taxonomía/splits, normalización, receta, semilla, costos, métricas de desarrollo y decisión. Informar resultados negativos. Los cambios de etiquetas o evaluación exigen versiones explícitas y reevaluación del baseline.

## Documentación oficial consultada

- [Argilla: configuración de datasets/preguntas](https://docs.argilla.io/latest/how_to_guides/dataset/).
- [Argilla: distribución y mínimo de respuestas](https://docs.argilla.io/v2.2/how_to_guides/distribution/).
- [Argilla: respuestas por usuario](https://docs.argilla.io/latest/reference/argilla/records/responses/).
- [Hugging Face: importación/exportación Argilla](https://huggingface.co/docs/hub/datasets-argilla).
- [Transformers: búsqueda de hiperparámetros](https://huggingface.co/docs/transformers/en/hpo_train).
- [Transformers: selección de checkpoint en Trainer](https://huggingface.co/docs/transformers/main_classes/trainer).

La integración comprobará APIs en las versiones elegidas del SDK y servidor. Una referencia documental con versión explícita explica la capacidad; no impone instalar esa versión antigua.

