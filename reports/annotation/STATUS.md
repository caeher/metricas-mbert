# Estado de ISSUE-05

**Piloto con dos anotadores independientes (cde-rater-a / cde-rater-b), exportación dual, acuerdo medido y release parcial QC.** Cuentas creadas en Argilla local (`http://localhost:6900`). Exportación versionada en `data/argilla_exports/pilot-v1__*` con 200 registros y dos respuestas humanas por fila. Kappa piloto ≈ **0.23** (objetivo operativo 0.75 no alcanzado); **180** registros en `adjudicated_records.csv` tras adjudicación. Resto en cuarentena (`contexto_insuficiente` o calidad).

- `pilot_sample_v1.csv` / `.json`: muestra estratificada (200 IDs).
- Scripts: [`setup_argilla_annotators.py`](../../scripts/setup_argilla_annotators.py), [`generate_pilot_dual_export.py`](../../scripts/generate_pilot_dual_export.py), [`adjudicate_pilot_release.py`](../../scripts/adjudicate_pilot_release.py), [`build_dataset_v3.py`](../../scripts/build_dataset_v3.py).
- Release de entrenamiento: [`data/releases/v3-pilot-argilla-20261031/`](../data/releases/v3-pilot-argilla-20261031/) (overlays de etiquetas adjudicadas sobre splits B2).
- [`agreement.json`](agreement.json), [`adjudicated_records.csv`](adjudicated_records.csv), [`adjudication_ledger.csv`](adjudication_ledger.csv).

Próximo paso humano recomendado: recalibrar guía y re-anotar antes de escalar `legacy_train_real` / `new_candidates` a producción gold.
