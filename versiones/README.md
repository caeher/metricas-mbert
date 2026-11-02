# Versiones congeladas (dataset + checkpoint)

Cada carpeta `vN/` es autocontenida: `dataset/`, `model/`, `manifest.json` y `metrics/`.

| ID | Release legacy | Modelo legacy | Train |
|----|----------------|---------------|-------|
| `v1` | `v3-pilot-argilla-20261031` | `mbert-sv-v3-pilot` | 12 641 |
| `v2` | `v4-gold-20261101` | `mbert-sv-v4-gold` | 12 641 |
| `v3` | `v5-facebook-willian-20261101` | `mbert-sv-v5-facebook-willian` | 13 105 |
| `v4` | `training-v3-20261101` | `mbert-sv-training-v3` | 21 585 (CDE v1.9.0) |

Dataset **vigente** para entrenamiento/sync: `data/processed/` (mismo contenido que `v4/dataset/`).

Evaluación recomendada (línea CDE):

```powershell
python scripts/evaluate_mbert_official.py --model-dir versiones/v4/model --skip-official-f1-check
python scripts/evaluate_production_slo.py --model-dir versiones/v4/model
```

Entrenar sobre un release congelado:

```powershell
python scripts/train_mbert.py --release v2 --output-dir versiones/v2/model --epochs 5
```

Los IDs `v1`…`v4` y los `release_id` fechados son equivalentes en `load_release_split`.
