# Plan de modelo post-gold (hacia SLO 0.90)

**Estado:** plan · **Precondición:** release gold ISSUE-08 (tras [`ROADMAP_F1_90.md`](../annotation/ROADMAP_F1_90.md))

## Objetivo

Maximizar el **score operativo** definido en [`production_slo_v1.json`](../metrics/production_slo_v1.json), no solo F1-macro global en test histórico.

## Baseline de referencia

Checkpoint actual: `versiones/v1/model` (release `v3-pilot-argilla-20261031`).

Evaluar siempre:

```powershell
python scripts/evaluate_mbert_official.py --model-dir versiones/v1/model --skip-official-f1-check
python scripts/evaluate_production_slo.py --model-dir versiones/v1/model
```

Artefactos: [`production_slo_evaluation.json`](../metrics/production_slo_evaluation.json).

### Comparación v3 piloto vs v4-gold vs v5 Facebook Willian vs training-v3 CDE (2026-11-01)

| Checkpoint | Release | F1-macro test | F1 slice Facebook | Score operativo SLO v1 |
|------------|---------|---------------|-------------------|-------------------------|
| `versiones/v1/model` | `v3-pilot-argilla-20261031` | **0.7801** | — | 0.6866 |
| `versiones/v2/model` | `v4-gold-20261101` | **0.7641** | **0.6867** | 0.6867 |
| `versiones/v3/model` | `v5-facebook-willian-20261101` | **0.7678** | **0.6829** | 0.6829 |
| `versiones/v4/model` | `training-v3-20261101` | **0.7708** | **0.7016** | 0.7016 |

Notas v5: +464 gold Facebook en train; entrenamiento acortado con `MAX_STEPS=500` (init `mbert-sv-v4-gold`). Cuello de botella SLO sigue siendo slice Facebook; segunda oleada anotación + épocas completas recomendada antes de `platform-class-boost`.

Notas **training-v3**: solo CDE **v1.9.0** (`train` 21 585), init `bert-base-multilingual-cased`, 5 épocas completas, `platform-class-boost none`. Distinto de `mbert-sv-v3-pilot` (Argilla). Mejor score operativo que v4/v5 en slice Facebook (+0.015 vs v4-gold) con F1 test intermedio.

Trazabilidad: [`mbert_official_run_v3_pilot.json`](../metrics/mbert_official_run_v3_pilot.json), [`mbert_official_run_v4_gold.json`](../metrics/mbert_official_run_v4_gold.json), [`mbert_official_run_v5_facebook_willian.json`](../metrics/mbert_official_run_v5_facebook_willian.json), [`mbert_official_run_training_v3.json`](../metrics/mbert_official_run_training_v3.json), [`production_slo_baseline_v4_gold.json`](../metrics/production_slo_baseline_v4_gold.json), [`production_slo_baseline_v5_facebook_willian.json`](../metrics/production_slo_baseline_v5_facebook_willian.json), [`production_slo_baseline_training_v3.json`](../metrics/production_slo_baseline_training_v3.json).


1. Construir release inmutable:

   ```powershell
   python argilla/scripts/build_dataset_v3.py --release-id v4-gold-YYYYMMDD --adjudicated reports/annotation/adjudicated_records.csv
   ```

2. Baseline comparable (`train_mbert.py` con la release v4):

   ```powershell
   python scripts/train_mbert.py --release v2 --platform-class-boost none --output-dir versiones/v2/model --epochs 5
   python scripts/evaluate_mbert_official.py --model-dir versiones/v2/model --skip-official-f1-check
   ```

   (Sustituir `YYYYMMDD` cuando exista la release.)

3. Registrar ablación único factor: solo etiquetas gold vs v3 piloto (misma receta, misma semilla 42).

## Fase 2 — Receta F1 0.80 (ISSUE-11)

Meta intermedia configurada en [`config/settings.yaml`](../../config/settings.yaml) → `f1_target`.

```powershell
python scripts/run_f1_target_pipeline.py
python scripts/sweep_mbert_f1.py
python scripts/evaluate_f1_gates.py --model-dir models/mbert-sv-f1
```

Parámetros clave del plan 0.80:

- `synthetic_policy`: `b2_no_template_heavy`
- `loss_type`: `ordinal`, `ordinal_weight`: 0.3
- `layerwise_lr_decay`: 0.95
- Semillas: 42, 7, 13 (selección en **val**, no en test)

Tras cada candidato: `evaluate_production_slo.py --model-dir <dir>`.

## Fase 3 — Opcional ISSUE-12 / ISSUE-13

| Experimento | Script / módulo | Notas |
|-------------|-----------------|-------|
| DAPT solo train | `scripts/` DAPT + ISSUE-12 | Sin MLM en val/test |
| Ensamble | `f1_target.ensemble_candidates` en settings | Validar en val; reportar costo inferencia |
| XLM-R | `src/discurso_odio/models/xlmr.py`, `compute_slice_metrics.py` | Priorizar si tráfico producción es mayoritariamente X |

## Fase 4 — Cierre ISSUE-14

1. Congelar receta + hashes de release antes de abrir **test nuevo** (ISSUE-03).
2. Reportar macro-F1, cortes en `slice_metrics.csv`, constraints adversarial/FPR del SLO.
3. Entregar paquete en `reports/final_v3/` (checkpoint, tokenizer, `production_slo_evaluation.json`, guía repro).

## Criterios de adopción (SLO v1)

- `operational_score` ≥ **0.90** en [`production_slo_v1.json`](../metrics/production_slo_v1.json)
- Restricciones: adv75 ≥ 45/75, FPR dialectal ≤ 0.17
- Umbrales por slice/clase definidos en el mismo JSON
- IC bootstrap por slice documentado (ya en `slice_metrics.json`)

## Orden recomendado

```mermaid
flowchart LR
  Gold[Release gold ISSUE-08] --> Base[Baseline v4 ISSUE-09]
  Base --> F80[Pipeline F1 0.80 ISSUE-11]
  F80 --> Opt[DAPT / ensamble opcional]
  Opt --> SLO[evaluate_production_slo]
  SLO --> Final[ISSUE-14 entrega]
```

No lanzar sweep largo en GPU hasta kappa anotación ≥ 0.75 en muestra de control (ver roadmap anotación).
