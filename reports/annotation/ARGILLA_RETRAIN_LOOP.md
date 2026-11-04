# Ciclo Argilla → release gold → reentrenamiento (desde v4)

Documenta el flujo para mejorar el modelo tras anotación humana, sin tocar etiquetas de test.

## 1. Anotación y export

```powershell
python argilla/scripts/export_argilla.py --campaign legacy_train_real --workspace $env:ARGILLA_WORKSPACE
```

Conservar la carpeta bajo `data/argilla_exports/` (versionada localmente, gitignored).

## 2. Acuerdo y adjudicación

```powershell
python argilla/scripts/analyze_annotation_exports.py `
  --exports "data/argilla_exports/<carpeta-export>"

python argilla/scripts/adjudicate_pilot_release.py
# o flujo equivalente documentado en docs/annotation_pilot_issue05.md
```

Salida esperada: filas añadidas en `reports/annotation/adjudicated_records.csv` y trazas en `adjudication_ledger.csv`.

## 3. Release ISSUE-08

```powershell
python argilla/scripts/build_dataset_v3.py `
  --release-id v6-gold-YYYYMMDD `
  --adjudicated reports/annotation/adjudicated_records.csv
```

Registrar alias en `versiones/` si aplica (p. ej. `v6` → `data/releases/v6-gold-YYYYMMDD`).

## 4. Reentrenamiento (init desde v4)

```powershell
python scripts/train_mbert.py `
  --release v6 `
  --init-model-dir versiones/v4/model `
  --output-dir versiones/v6/model `
  --platform-class-boost none `
  --epochs 5
```

No usar `MAX_STEPS` corto; preferir épocas completas y early stopping en val.

## 5. Evaluación comparable

```powershell
python scripts/evaluate_mbert_official.py --model-dir versiones/v6/model --skip-official-f1-check
python scripts/evaluate_production_slo.py --model-dir versiones/v6/model
python scripts/build_mbert_full_report.py --model-dir versiones/v6/model `
  --output reports/metrics/mbert_training_v6_full_report.json
```

## 6. Iteración

Repetir lote Argilla con el nuevo checkpoint (`export_mbert_argilla_suggestions.py --model-dir versiones/v6/model`) priorizando slice Facebook y clases 0/3 según `production_slo_v1.json`.

## Corrida v6-gold (2026-11-03)

| Artefacto | Valor |
|-----------|--------|
| Export | `legacy-train-real__20261103T042651228774Z` (500 IDs, 9988 resp.; dual heurística `generate_legacy_train_real_dual_export.py`) |
| Kappa `legacy_train_real` | ≈ **0.11** (objetivo 0.75 no alcanzado) |
| Release | `v6-gold-20261103` → [`versiones/v6/dataset/`](../../versiones/v6/dataset/) (**88** overrides train, 21 585 filas) |
| Train | `train_mbert.py --release v6 --init-model-dir versiones/v4/model --epochs 5 --platform-class-boost none` |
| Best checkpoint | `checkpoint-4050` |
| F1 test | **0.7784** ([`mbert_official_run_v6_gold.json`](../metrics/mbert_official_run_v6_gold.json)) |
| F1 val (best) | **0.816** |
| SLO operativo | **0.6963** ([`production_slo_baseline_v6_gold.json`](../metrics/production_slo_baseline_v6_gold.json)); Facebook slice **0.6963** |
| vs v4 baseline | F1 test **+0.0076**; score operativo **−0.0053** (tratar como experimental hasta kappa/UI mejoren) |

Comandos de empaquetado post-train:

```powershell
python scripts/verify_mbert_checkpoint.py --model-dir versiones/v6/model
python scripts/package_mbert_checkpoint.py --model-dir versiones/v6/model
```

## Reglas (ISSUE-14)

- No relabelar test/val con predicciones del modelo.
- No excluir casos difíciles para inflar F1.
- Selección de candidatos solo desde val, cola canónica o train legacy — nunca desde errores de test sellado.
