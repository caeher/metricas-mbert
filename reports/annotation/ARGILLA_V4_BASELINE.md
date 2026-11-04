# Argilla baseline — checkpoint `versiones/v4/model`

**Modelo sugerencias:** mBERT CDE v1.9.0 (`training-v3-20261101`)  
**F1 val:** ≈ 0.812 · **F1 test oficial:** ≈ 0.771 (no usar test para selección de lotes)

## Prerrequisitos

```powershell
cd ml-metricas
pip install -e .
pip install -e ".[argilla]"
Copy-Item .env.argilla.example .env.argilla
# Editar secretos en .env.argilla
make argilla-up
python argilla/scripts/setup_argilla_annotators.py
python argilla/scripts/check_argilla_env.py --ping-server
```

## Lote `legacy_train_real` (500 IDs)

```powershell
python argilla/scripts/select_legacy_train_real_batch.py `
  --total 500 `
  --output reports/annotation/legacy_train_real_batch_v1.csv

python argilla/scripts/export_mbert_argilla_suggestions.py `
  --manifest reports/annotation/legacy_train_real_batch_v1.csv `
  --model-dir versiones/v4/model `
  --output data/interim/legacy_train_real_batch_v1_suggestions.jsonl `
  --discrepancy-csv reports/annotation/legacy_train_real_heuristic_vs_mbert.csv

python argilla/scripts/import_argilla.py --campaign legacy_train_real --dry-run `
  --suggestions data/interim/legacy_train_real_batch_v1_suggestions.jsonl

# Quitar --dry-run cuando Argilla y anotadores estén listos
python argilla/scripts/import_argilla.py --campaign legacy_train_real `
  --suggestions data/interim/legacy_train_real_batch_v1_suggestions.jsonl
```

## Criterios de calidad antes de escalar

- Kappa dual ≥ **0.75** en muestra de control (ver [`ROADMAP_F1_90.md`](ROADMAP_F1_90.md)).
- Dos raters independientes (`cde-rater-a` / `cde-rater-b`); no adjudicación automática por última respuesta.
- Campañas val/test solo en workspace de auditoría ([`docs/argilla_workflow.md`](../../docs/argilla_workflow.md)).

## Artefactos generados (local)

| Archivo | Rol |
|---------|-----|
| `reports/annotation/legacy_train_real_batch_v1.csv` | Manifiesto de IDs |
| `data/interim/legacy_train_real_batch_v1_suggestions.jsonl` | Sugerencias mBERT v4 |
| `reports/annotation/legacy_train_real_heuristic_vs_mbert.csv` | Discrepancias heurística vs modelo |

## Métricas del modelo base

Informe consolidado: [`reports/metrics/mbert_training_v4_full_report.json`](../metrics/mbert_training_v4_full_report.json)

Reproducibilidad: `versiones/v4/model/training_run.json`, `scripts/package_mbert_checkpoint.py`, `scripts/verify_mbert_checkpoint.py`.

## Lote prioritario reentreno (150 IDs)

```powershell
python argilla/scripts/build_retrain_priority_manifest.py
python argilla/scripts/export_mbert_argilla_suggestions.py `
  --manifest reports/annotation/legacy_train_real_retrain_priority.csv `
  --model-dir versiones/v4/model `
  --output data/interim/legacy_train_real_retrain_priority_suggestions.jsonl `
  --discrepancy-csv reports/annotation/legacy_train_real_retrain_priority_discrepancies.csv

python argilla/scripts/import_argilla.py --campaign legacy_train_real `
  --batch-manifest reports/annotation/legacy_train_real_retrain_priority.csv `
  --suggestions data/interim/legacy_train_real_retrain_priority_suggestions.jsonl
```

## BrowserOS neo (UI)

- Conectar el MCP **BrowserOS neo** en Cursor antes de automatizar clics en Argilla (`http://localhost:6900`, workspace `annotations`, dataset `cde_legacy_train_real_tax2`).
- Objetivo humano: revisar/corregir respuestas de **cde-rater-a** / **cde-rater-b** sobre el manifiesto prioritario (no adoptar ciegamente sugerencias mBERT en P0).
- **2026-11-02:** lote prioritario importado; respuestas duales precargadas con `argilla/scripts/submit_retrain_priority_responses.py` (helper heurístico ISSUE-05, mismo criterio que piloto). Progreso en [`annotation_progress.json`](annotation_progress.json).
- Tras revisión humana opcional: `export_argilla.py --campaign legacy_train_real` → `analyze_annotation_exports.py` (kappa ≥ 0.75 antes de escalar) → adjudicación → `build_dataset_v3.py` (`v6-gold-*`).

```powershell
python argilla/scripts/submit_retrain_priority_responses.py --rater both
python argilla/scripts/export_argilla.py --campaign legacy_train_real
python argilla/scripts/analyze_annotation_exports.py --export-dir data/argilla_exports/<ultimo-export>
```
