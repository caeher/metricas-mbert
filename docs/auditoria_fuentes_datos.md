# Auditoría de fuentes del corpus

El auditor `scripts/audit_dataset_sources.py` crea un inventario reproducible y
una vista canónica de observaciones. No cambia los CSV originales, no adjudica
etiquetas y no habilita automáticamente las fuentes candidatas para entrenar.

## Ejecución

Desde la raíz de `ml-metricas`:

```powershell
python scripts/audit_dataset_sources.py
```

La fecha de referencia predeterminada para marcar fechas futuras es
`2026-10-16`. Se puede fijar otra fecha con `--as-of YYYY-MM-DD`. El umbral
predeterminado para candidatos de similitud es Jaccard de trigramas 0.82; se
puede cambiar con `--similarity-threshold`. El proceso usa solo la biblioteca
estándar de Python.

## Qué se considera una observación

- `data/processed/train.csv`, `val.csv` y `test.csv` (y espejo en `data/raw/`) forman el corpus activo
  (**dataset v1.9.0** sincronizado desde `cde-project-ml-2026/data/dataset/`).
- Releases y checkpoints congelados: [`versiones/README.md`](../versiones/README.md) (`v1`…`v4`).
- `data/evaluation/frozen_bank/*.csv` entra a la vista canónica solo para
  trazabilidad, detección de duplicados y **comprobación de fugas**; nunca es
  corpus entrenable (`frozen_evaluation_bank`).
- `data/sources/x_threads/**/*.csv` son hilos X **sin etiquetar**
  (`candidate_unlabeled_thread`); quedan en cuarentena hasta anotación ISSUE-05.
- `data/sources/candidates/lenguaje_ofensivo.csv` y `thenewdata_X.csv` se inventarían como
  fuentes candidatas en cuarentena. No se integran a un split.
- `versiones/v3/sources/dataset_willian_facebook.csv` (7 045 comentarios Facebook, octubre
  2026) entra como `candidate_unvalidated`; etiquetas auto-heurísticas sin
  validar. Tras auditoría, **29** textos normalizados coinciden con val/test y
  deben excluirse del lote de anotación (`willian_holdout_blocklist.csv`).
- Las baterías `test_adversarial.csv` (144 filas en v1.8) y
  `data/processed/test_adversarial_144.csv` (144 filas) se conservan como
  diagnóstico y nunca se tratan como corpus de entrenamiento.
- `data/interim/corpus_completo.csv` se compara con train/val/test y se registra en el
  inventario, pero no se vuelve a importar. Las copias de `data/processed/`
  también se inventarían sin sumar filas nuevas.

`data/interim/canonical_records.csv` agrupa por texto normalizado exacto o por
URL normalizada. Cada grupo recibe `canonical_id` global y mantiene las
referencias, IDs, textos, versiones, etiquetas originales y hashes de sus
observaciones. Las similitudes, autores, hilos y plantillas se informan como
señales de revisión: no fusionan ni eliminan registros automáticamente.

## Interpretación de etiquetas y geografía

`label_original` y `label_provisional_id` conservan lo que decía cada fuente;
`label_final` permanece vacío porque no hay aprobación humana trazable en los
archivos. `record_disposition` indica si la observación pertenece a un split
histórico sin revisar, sigue en cuarentena, es solo diagnóstica o requiere
adjudicación. No usar los CSV de esta auditoría como exportación de etiquetas
gold.

Twitter y X se normalizan a plataforma canónica `X`, conservando el valor
original. País y dialecto se registran con estado `confirmed`, `inferred` o
`unknown`. La nota geográfica no confirmada no especifica país; por eso sus
registros tienen país vacío y estado `inferred`, sin convertirlos en casos de
El Salvador. Las etiquetas dialectales de las fuentes candidatas también se
tratan como inferencias mientras no exista validación.

## Sincronización CDE (v1.9.0 vigente)

Desde la raíz de `ml-metricas`:

```powershell
python scripts/sync_cde_dataset.py
python scripts/validate_dataset_integration.py
```

El manifiesto `reports/data_sync/last_sync.json` guarda SHA-256 por archivo;
`integration_validation.json` comprueba conteos, MANIFEST del banco congelado,
val/test históricos y fugas train→holdouts.

## Hallazgos del corte actual (post v1.9.0)

Consulte `reports/data_audit/audit_summary.json` tras cada auditoría. Tras la
integración v1.9.0, los splits activos son train **21 585**, val **459** y test
**460**; el banco congelado y los hilos sin etiquetar suman observaciones
canónicas adicionales sin habilitar entrenamiento. El resumen incluye
`train_holdout_leakage` (objetivo: **0** fugas normalizadas train vs
val/test/batería/banco).

Los párrafos siguientes describen el corte B2 previo solo como referencia
histórica de comparación con métricas antiguas:

- Train/val/test B2: 3,293 / 459 / 460 filas; train conservaba 2,144 reales y 1,149
  sintéticas. El total de los splits era 4,212.
- `corpus_completo.csv` tiene los mismos 4,212 IDs y todos los campos de los
  splits; no agrega observaciones.
- Las fuentes candidatas tienen 1,328 y 1,000 filas, ambas X/Twitter; las 2,328
  filas tienen `Validado=NO` y `Listo_Entrenamiento=NO`.
- Se confirman tres conflictos de etiqueta con texto exacto y una discrepancia
  adicional para la misma URL con versiones de texto diferentes. Los cuatro
  casos aparecen en `reports/data_audit/adjudication_queue.csv`; sus campos de
  resolución humana siguen vacíos. No se escogió ninguna etiqueta.
- La vista canónica contiene 6,640 entidades, agrupando 6,759 observaciones
  cargadas de splits, candidatas y baterías. Se unifican 83 grupos de texto
  exacto y un grupo de URL/publicación; los candidatos similares siguen
  separados hasta revisión humana.
- No hay texto exacto compartido entre las fuentes candidatas y los splits
  históricos. Esto no demuestra que no existan duplicados aproximados, URLs o
  publicaciones compartidas, grupos de autor/hilo relacionados ni plantillas
  filtradas.
- 497 notas de `thenewdata_X.csv` dicen que la ubicación fue inferida y no
  confirmada por perfil. El país no se especifica de forma confiable.
- Los IDs de publicación de X permiten decodificar fechas de publicación. El
  auditor detectó 445 fechas de recolección anteriores a esa fecha en
  `thenewdata_X.csv`; ambas fechas se conservan en `date_audit.csv` para revisar
  la fuente, no para corregirlas automáticamente.
- Los archivos auditados se decodifican como UTF-8 y no contienen U+FFFD. Hay
  signos `?` literales entre letras en notas/campos de 505 filas de
  `thenewdata_X.csv`, 4 de train y 1 de `lenguaje_ofensivo.csv`; diez textos
  también tienen esa señal. Son posibles caracteres perdidos antes de guardar
  estos CSV, no una diferencia de fuente o de visor. Se conservan los textos y
  hashes originales y no se reconstruye el carácter que pudo haberse perdido.
- La búsqueda de similitud encontró 1,728 pares con Jaccard de trigramas ≥0.82.
  Son candidatos que requieren revisión; no se asumen como duplicados ni se
  fusionan automáticamente. Se reportan 223 grupos de relación por autor o
  plantilla; las fuentes no tienen identificadores estables de hilo.

## Archivos de salida

En `reports/data_audit/`:

- `audit_summary.json`: configuración, recuentos y limitaciones.
- `source_inventory.csv`: filas, hashes, clases, plataformas, validación,
  procedencia, campos ausentes y reconciliación de vistas.
- `adjudication_queue.csv`: conflictos con texto, URL, etiquetas originales y
  campos para persona adjudicadora, fecha, decisión y justificación.
- `similarity_candidates.csv`: similitudes de trigramas que requieren revisión;
  no se deduplican automáticamente.
- `author_thread_template_relations.csv`: grupos de autor y plantilla. No hay
  identificadores estables de hilo en las fuentes cargadas.
- `encoding_audit.csv` y `encoding_suspects.csv`: señales agregadas y filas
  sospechosas que se preservan para revisar.
- `date_audit.csv`: fechas no parseables, ausentes en registros reales, futuras
  o anteriores a la publicación codificada en una URL de X.
- `geography_language_audit.csv`: estados confirmados, inferidos y desconocidos.

En `data/interim/`:

- `canonical_records.csv`: registros agrupados, procedencia y etiquetas
  provisionales con `label_final` vacío.
- `deduplication_exclusion_ledger.csv`: duplicados agrupados, candidatos
  excluidos, vistas de archivo no importadas y conflictos con resolución humana
  pendiente.

La adjudicación de los cuatro grupos de conflicto requiere una decisión de una
persona anotadora con identificador, fecha, etiqueta y fundamento. Los pares de
similitud y las anomalías de fecha/codificación también quedan señalados para
revisión; el auditor no convierte esas señales en exclusiones ni etiquetas.
