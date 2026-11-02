# Estado de ISSUE-05

**Piloto con dos anotadores independientes (cde-rater-a / cde-rater-b), exportación dual, acuerdo medido y release parcial QC.** Cuentas creadas en Argilla local (`http://localhost:6900`). Exportación versionada en `data/argilla_exports/pilot-v1__*` con 200 registros y dos respuestas humanas por fila. Kappa piloto ≈ **0.23** (objetivo operativo 0.75 no alcanzado); **180** registros en `adjudicated_records.csv` tras adjudicación. Resto en cuarentena (`contexto_insuficiente` o calidad).

- `pilot_sample_v1.csv` / `.json`: muestra estratificada (200 IDs).
- Scripts: [`setup_argilla_annotators.py`](../../argilla/scripts/setup_argilla_annotators.py), [`generate_pilot_dual_export.py`](../../argilla/scripts/generate_pilot_dual_export.py), [`adjudicate_pilot_release.py`](../../argilla/scripts/adjudicate_pilot_release.py), [`build_dataset_v3.py`](../../argilla/scripts/build_dataset_v3.py).
- Release de entrenamiento: [`versiones/v1/dataset/`](../../versiones/v1/dataset/) (`v3-pilot-argilla-20261031`, overlays B2).
- Release gold ISSUE-08: [`versiones/v2/dataset/`](../../versiones/v2/dataset/) (`v4-gold-20261101`). Checkpoint [`versiones/v2/model`](../../versiones/v2/model): **F1-macro test 0.7641**, score operativo SLO **0.6867** ([`production_slo_baseline_v4_gold.json`](../metrics/production_slo_baseline_v4_gold.json)).
- Campaña **Facebook Willian** (`new_candidates`, batch v1): manifiesto [`facebook_willian_batch_v1.csv`](facebook_willian_batch_v1.csv) (500 IDs), export dual [`data/argilla_exports/new-candidates-willian-batch-v1__20261101T101145980164Z/`](../../data/argilla_exports/new-candidates-willian-batch-v1__20261101T101145980164Z/), kappa batch ≈ **0.92** (export local asistido mBERT). **644** filas en `adjudicated_records.csv` (464 `new_candidates` elegibles para train).
- Release v5: [`versiones/v3/dataset/`](../../versiones/v3/dataset/) (`v5-facebook-willian-20261101`) — train **13105** (+464 gold FB append), val/test sin cambio. Checkpoint [`versiones/v3/model`](../../versiones/v3/model) (fine-tune desde v4-gold, **`MAX_STEPS=500`** ≈ 0.61 época, `platform-class-boost none`): **F1-macro test 0.7678**, slice Facebook **0.6829**, score operativo **0.6829** ([`production_slo_baseline_v5_facebook_willian.json`](../metrics/production_slo_baseline_v5_facebook_willian.json), [`mbert_official_run_v5_facebook_willian.json`](../metrics/mbert_official_run_v5_facebook_willian.json)). vs v4-gold: +0.0037 F1 global, Facebook **−0.0038** (meta SLO slice 0.75 no alcanzada).
- **training-v3 CDE v1.9.0** (sin Argilla): sync desde `cde-project-ml-2026` → [`versiones/v4/dataset/`](../../versiones/v4/dataset/) (`train` **21 585**). Vigente en `data/processed/`. Checkpoint [`versiones/v4/model`](../../versiones/v4/model) (init `bert-base-multilingual-cased`, 5 épocas, `platform-class-boost none`): **F1-macro test 0.7708**, slice Facebook **0.7016**, score operativo **0.7016** ([`production_slo_baseline_training_v3.json`](../metrics/production_slo_baseline_training_v3.json), [`mbert_official_run_training_v3.json`](../metrics/mbert_official_run_training_v3.json)). No confundir con `versiones/v1` (Argilla pilot).
- [`agreement.json`](agreement.json), [`adjudicated_records.csv`](adjudicated_records.csv), [`adjudication_ledger.csv`](adjudication_ledger.csv).
- Métrica operativa (SLO v1): [`production_slo_v1.json`](../metrics/production_slo_v1.json), baseline piloto [`production_slo_baseline_v3_pilot.json`](../metrics/production_slo_baseline_v3_pilot.json). Roadmaps: [`ROADMAP_F1_90.md`](ROADMAP_F1_90.md), [`../experiments/MODEL_ROADMAP_F1_90.md`](../experiments/MODEL_ROADMAP_F1_90.md).

Re-evaluar SLO del checkpoint v3 (sin re-entrenar):

```powershell
python scripts/evaluate_mbert_official.py --model-dir versiones/v1/model --skip-official-f1-check
python scripts/evaluate_adversarial.py --model-dir versiones/v1/model --model-name mbert-v3-pilot --output-prefix mbert_slo
python scripts/compute_dialect_bias.py --model-dir versiones/v1/model --skip-xlmr
python scripts/evaluate_production_slo.py --model-dir versiones/v1/model --output reports/metrics/production_slo_baseline_v3_pilot.json
```

Próximo paso humano recomendado: recalibrar guía y re-anotar antes de escalar `legacy_train_real` / `new_candidates` a producción gold.
