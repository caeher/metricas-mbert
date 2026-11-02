# Argilla — anotación humana (ISSUE-04–08)

Stack local **Argilla 2.8.0** + scripts de import/export/adjudicación. No forma parte del pipeline de entrenamiento **CDE 21k** (`scripts/` en la raíz del repo).

## 1. Dependencias Python

Desde la raíz de `ml-metricas`:

```powershell
pip install -e .
pip install -e ".[argilla]"
python argilla/scripts/check_argilla_env.py
```

## 2. Servidor Docker

```powershell
Copy-Item .env.argilla.example .env.argilla
# Edita secretos en .env.argilla
make argilla-up
```

UI: http://localhost:6900 (solo `127.0.0.1`).

Detalle de volúmenes, backup y campañas: [`docs/argilla_workflow.md`](../docs/argilla_workflow.md).

## 3. Cuentas y campañas

```powershell
python argilla/scripts/setup_argilla_annotators.py
python argilla/scripts/import_argilla.py --campaign pilot --records data/interim/canonical_records.csv
```

Estado de releases y métricas: [`reports/annotation/STATUS.md`](../reports/annotation/STATUS.md).

## 4. Datos relacionados

| Ruta | Uso |
|------|-----|
| `data/interim/canonical_records.csv` | Vista canónica ISSUE-02 |
| `data/sources/candidates/` | Candidatos X sin validar |
| `versiones/v3/sources/dataset_willian_facebook.csv` | Candidatos Facebook Willian |
| `data/argilla_exports/` | Exports versionados (local, gitignored) |
| `reports/annotation/` | Manifiestos, adjudicación, acuerdo |

## 5. Scripts

Todos en [`argilla/scripts/`](scripts/): `build_dataset_v3.py`, `adjudicate_pilot_release.py`, `select_annotation_batch.py`, etc.

Compose: [`infra/argilla/docker-compose.yml`](../infra/argilla/docker-compose.yml) (también `make argilla-up`).
