# Flujo de anotación Argilla — ISSUE-04

**Versión del esquema:** `argilla-issue04-v1`  
**Argilla SDK y servidor:** `2.8.0`  
**Taxonomía / guía:** `2.0.0` (`config/taxonomy_v2.yaml` y `docs/annotation_guidelines_v2.md`)

Este flujo crea campañas separadas sobre la vista canónica de ISSUE-02. Las etiquetas antiguas, predicciones y notas de origen no se muestran como respuestas; solo una respuesta de persona cuenta como anotación. Argilla es opcional: los scripts de inferencia y entrenamiento no importan el SDK.

## Versiones y despliegue local

Se fijó `argilla==2.8.0` en el extra opcional de `pyproject.toml`. El servidor y el worker usan `argilla/argilla-server:v2.8.0`; Elasticsearch queda en `8.17.0`, la versión de la configuración oficial de Argilla 2.8.0. PostgreSQL y Redis también tienen etiquetas fijadas en [docker-compose.yml](../infra/argilla/docker-compose.yml). El SDK 2.8.0 se instaló aislado para esta integración y se inspeccionaron las firmas de `Record`, `Response`, `Suggestion`, `Settings`, preguntas, `TaskDistribution`, `Dataset.records.log` y la lectura de respuestas.

La configuración soportada aquí es Docker Compose local. La interfaz solo se publica en `127.0.0.1:6900`; Postgres, Elasticsearch y Redis no publican puertos al host. El servidor requiere Docker con Compose y recursos suficientes para Elasticsearch.

```powershell
Copy-Item .env.argilla.example .env.argilla
# Edita .env.argilla y reemplaza todos los secretos de ejemplo.
docker compose --env-file .env.argilla -f infra/argilla/docker-compose.yml up -d
```

El estado persiste en los volúmenes `cde-ml-argilla-home`, `cde-ml-argilla-postgres` y `cde-ml-argilla-elasticsearch`. `docker compose down` conserva esos volúmenes. **No uses `down -v`** salvo que quieras borrar la instancia completa. El archivo `.env.argilla` está ignorado por Git; no pongas claves en el Compose, notebooks, CSV de sugerencias ni argumentos de línea de comandos.

`ARGILLA_API_KEY` es la clave del propietario inicial configurada al arrancar. `ARGILLA_AUTH_SECRET_KEY` mantiene estable la firma de sesión entre reinicios. Los scripts leen la clave API de `.env.argilla` o del entorno; el secreto de firma solo lo usa el servidor. Antes de importar, crea en la interfaz dos workspaces distintos: uno para anotación/adquisición y otro restringido para auditoría. Añade a cada workspace únicamente sus anotadores. El propietario/API key de importación debe poder crear datasets en ambos; usa cuentas de anotador diferentes para las respuestas. No des acceso al workspace de auditoría a quien selecciona candidatos, entrena o ajusta modelos.

### Copia completa y restauración de Docker

Además de exportar los registros desde Argilla, conserva una copia de los tres volúmenes para restaurar usuarios, permisos, datasets y respuestas con sus UUID originales. En PowerShell, detén primero el stack y crea los archivos:

```powershell
$backup = Join-Path (Get-Location) ("backups/argilla/" + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Force -Path $backup | Out-Null
docker compose --env-file .env.argilla -f infra/argilla/docker-compose.yml stop
$backupPath = (Resolve-Path $backup).Path
docker run --rm --mount 'type=volume,source=cde-ml-argilla-home,target=/source,readonly' --mount "type=bind,source=$backupPath,target=/backup" alpine:3.21 tar -czf /backup/home.tar.gz -C /source .
docker run --rm --mount 'type=volume,source=cde-ml-argilla-postgres,target=/source,readonly' --mount "type=bind,source=$backupPath,target=/backup" alpine:3.21 tar -czf /backup/postgres.tar.gz -C /source .
docker run --rm --mount 'type=volume,source=cde-ml-argilla-elasticsearch,target=/source,readonly' --mount "type=bind,source=$backupPath,target=/backup" alpine:3.21 tar -czf /backup/elasticsearch.tar.gz -C /source .
docker compose --env-file .env.argilla -f infra/argilla/docker-compose.yml start
```

Guarda juntas las tres copias y el `.env.argilla` en almacenamiento privado. Para restaurar, establece `$backupPath` a esa carpeta, detén el stack y extrae cada archivo en su volumen correspondiente:

```powershell
$backupPath = (Resolve-Path 'backups/argilla/FECHA-DE-COPIA').Path
docker compose --env-file .env.argilla -f infra/argilla/docker-compose.yml stop
docker run --rm --mount 'type=volume,source=cde-ml-argilla-home,target=/target' --mount "type=bind,source=$backupPath,target=/backup,readonly" alpine:3.21 tar -xzf /backup/home.tar.gz -C /target
docker run --rm --mount 'type=volume,source=cde-ml-argilla-postgres,target=/target' --mount "type=bind,source=$backupPath,target=/backup,readonly" alpine:3.21 tar -xzf /backup/postgres.tar.gz -C /target
docker run --rm --mount 'type=volume,source=cde-ml-argilla-elasticsearch,target=/target' --mount "type=bind,source=$backupPath,target=/backup,readonly" alpine:3.21 tar -xzf /backup/elasticsearch.tar.gz -C /target
docker compose --env-file .env.argilla -f infra/argilla/docker-compose.yml start
```

Restaura la misma versión de las imágenes primero. Una exportación JSONL aislada también puede reimportar registros y respuestas, pero requiere que existan las cuentas con los mismos UUID; la copia de volúmenes preserva esas cuentas directamente. Las copias completas están ignoradas por Git en `backups/argilla/`.

## Campañas y límites entre cohorts

El importador acepta el CSV canónico de `data/interim/canonical_records.csv`, nunca `corpus_completo.csv` además de sus splits. Asigna cada entidad canónica a una campaña y usa IDs externos `cde-issue04-<campaña>-<canonical_id>` estables:

| `--campaign` | Cohorte | Workspace |
|---|---|---|
| `pilot` | 200 IDs del muestreo real ciego; requiere `reports/annotation/pilot_sample_v1.csv` | `ARGILLA_WORKSPACE` |
| `legacy_train_real` | Reales de `data/raw/train.csv` | `ARGILLA_WORKSPACE` |
| `legacy_train_synthetic` | Sintéticos históricos de train | `ARGILLA_WORKSPACE` |
| `new_candidates` | Fuentes candidatas nuevas, sin duplicar train/evaluación | `ARGILLA_WORKSPACE` |
| `new_synthetic` | Sintéticos nuevos; queda vacío hasta agregarlos al corpus canónico. Las baterías diagnósticas quedan excluidas | `ARGILLA_WORKSPACE` |
| `calibration` | 40 casos de formación de `docs/calibration_cases_v2.md`, sin clase esperada ni sugerencias visibles | `ARGILLA_WORKSPACE` |
| `historical_val_audit` | val histórico | `ARGILLA_AUDIT_WORKSPACE` |
| `historical_test_audit` | test histórico | `ARGILLA_AUDIT_WORKSPACE` |

Las entidades con referencia a val/test nunca entran a train ni a la cola de candidatos. Val y test se separan en dos datasets de auditoría. Para ambas campañas restringidas, `ARGILLA_AUDIT_WORKSPACE` es obligatorio y cualquier `--workspace` explícito debe coincidir; no se permite redirigirlas al workspace de anotación. Estas campañas y calibración rechazan sugerencias del modelo. No se incluye `label_original`, `label_provisional_id`, metadatos del anotador fuente ni notas que puedan revelar etiquetas previas. La procedencia se conserva como metadatos extra accesibles por SDK, sin mostrarse en la interfaz. La campaña de calibración lee las filas del Markdown, asigna IDs sin clase y omite la columna de etiquetas esperadas; estos ejemplos no se reportan como prueba independiente ni prevalencia.

La campaña `pilot` exige un manifiesto de exactamente 200 IDs generado por `scripts/prepare_annotation_pilot.py`; `pilot` rechaza sugerencias. Los estratos y las etiquetas usadas para balancear no se copian a los registros Argilla. El manifiesto local es información interna de muestreo: no se comparte con los anotadores. El piloto incluye reales históricos de train y candidatos nuevos; no incorpora val/test, sintéticos ni baterías diagnósticas.

## Crear las campañas

Instala la integración solo cuando vayas a usarla:

```powershell
python -m pip install -e ".[argilla]"
```

Simula antes cada importación; la simulación no necesita conexión ni credenciales de Argilla:

```powershell
python scripts/import_argilla.py --campaign legacy_train_real --limit 200 --dry-run
python scripts/import_argilla.py --campaign legacy_train_real --limit 200
python scripts/prepare_annotation_pilot.py
python scripts/import_argilla.py --campaign pilot --dry-run
python scripts/import_argilla.py --campaign calibration --dry-run
```

Repetir el segundo comando no duplica registros. El script agrega registros faltantes, actualiza campos/procedencia de los existentes sin respuestas, y se detiene si detecta un cambio en un registro que ya tiene respuestas humanas. Puedes omitir `--limit` para importar la cohorte completa. `--records` permite pasar una nueva exportación canónica; la selección por campaña aplica los mismos filtros de aislamiento.

Antes de iniciar la anotación, verifica que el workspace tenga al menos dos cuentas de anotación independientes. Cada dataset queda configurado con `TaskDistribution(min_submitted=2)`: una fila permanece pendiente hasta alcanzar dos respuestas enviadas. La propiedad se valida en las importaciones siguientes y el script se detiene si una campaña existente no la tiene. Argilla puede retirar una fila de la cola compartida al completar el mínimo, así que usa usuarios distintos y revisa autores por registro antes de adjudicar.

## Preguntas, guía y cegamiento

Todos los datasets llevan los campos visibles `original_text` y `normalized_text`. Las preguntas recogen estado (`anotado` / `contexto_insuficiente`), etiqueta primaria 0–3, subclase, grupos objetivo, dialecto, cita/reporte, negación, tipo de violencia, calidad, motivo de contexto insuficiente y justificación. La etiqueta primaria es opcional para permitir `contexto_insuficiente` sin crear una quinta clase. Argilla limita las instrucciones del dataset a 10,000 caracteres: se carga la [guía breve v2.0.0](annotation_guidelines_v2_argilla.md), mientras la [guía completa](annotation_guidelines_v2.md) y el [registro de decisiones](taxonomy_decisions_v2.md) permanecen versionados en el repositorio.

El importador no convierte etiquetas de fuente en respuestas. Las sugerencias son optativas y viajan por `rg.Suggestion`, separadas de `rg.Response`. Si se usa un modelo para priorizar revisión en una campaña de train/candidatos, prepara un JSONL como este y pasa `--suggestions`:

```json
{"canonical_id":"CAN-0123456789abcdef0123","label":"2","score":0.91,"agent":"mbert-v2","checkpoint":"checkpoint-1644"}
```

Se validan clase, agente, checkpoint y score [0,1]. La sugerencia guarda clase/score/agente en Argilla y checkpoint en metadatos de procedencia. No se acepta para auditoría. Las campañas de val/test están en otro workspace y no muestran etiquetas anteriores ni sugerencias, para reducir anclaje.

Las respuestas originales nunca se consolidan automáticamente: desacuerdos, contexto insuficiente y adjudicación quedan como eventos diferentes. El exportador conserva todas las respuestas con pregunta, valor, UUID de usuario y estado; una respuesta más reciente no se trata como decisión final.

## Exportación, reanudación y restauración portable

Exporta una campaña a un directorio inmutable con JSONL y manifiesto:

```powershell
python scripts/export_argilla.py --campaign legacy_train_real
```

`data/argilla_exports/<campaña>__<UTC>/records.jsonl` incluye todos los campos, metadatos de procedencia, vectores, sugerencias y respuestas individuales. `manifest.json` registra versiones SDK/servidor esperada, taxonomía/guía/esquema, conteos de respuestas y autores y SHA-256 del JSONL. Conserva estas carpetas como releases locales de solo adición en almacenamiento privado; están ignoradas por Git porque contienen textos de origen y UUID de anotadores. No se publica nada en Hub.

Restaura en una instancia compatible:

```powershell
python scripts/import_argilla.py --restore-export data/argilla_exports/<directorio>
```

La restauración conserva los IDs, la procedencia y los autores de respuestas si las cuentas con esos UUID ya existen en el workspace destino. Es reanudable e idempotente: conserva respuestas existentes, agrega las ausentes y se detiene ante conflictos de valor/estado para el mismo usuario y pregunta. No altera una sugerencia asociada a respuestas humanas. Para recuperar toda la instancia, usuarios y permisos, prefiere restaurar la copia completa de los volúmenes.

El exportador produce una exportación de campaña, no un dataset de entrenamiento. La construcción del dataset de entrenamiento y la adjudicación continúan bajo ISSUE-05/08: ninguna sugerencia cuenta como validación humana y ninguna campaña de evaluación se etiqueta como train.

## Referencias API de la versión fijada

- [Guía de datasets de Argilla 2.8](https://docs.argilla.io/v2.8/how_to_guides/dataset/) documenta preguntas, guía y `TaskDistribution(min_submitted=2)`.
- [Distribución de anotaciones de Argilla 2.8](https://docs.argilla.io/v2.8/how_to_guides/distribution/) explica las respuestas enviadas y el mínimo por registro.
- [Respuestas](https://docs.argilla.io/v2.8/reference/argilla/records/responses/) y [sugerencias](https://docs.argilla.io/v2.8/reference/argilla/records/suggestions/) distinguen autores humanos de agentes de modelo.
- [Compose oficial de Argilla v2.8.0](https://github.com/argilla-io/argilla/blob/v2.8.0/examples/deployments/docker/docker-compose.yaml) fija Elasticsearch 8.17.0 y configura persistencia para Argilla y su base de datos; [Docker Hub](https://hub.docker.com/r/argilla/argilla-server/tags) publica la imagen `v2.8.0`.
