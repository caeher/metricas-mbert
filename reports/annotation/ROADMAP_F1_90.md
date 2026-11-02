# Plan de anotación hacia F1 operativo 0.90 (ISSUE-05)

**Estado:** plan de ejecución · **Depende de:** guía 2.0.0, Argilla local, [`production_slo_v1.json`](../metrics/production_slo_v1.json)

## Objetivo

Elevar la calidad gold antes de escalar entrenamiento. Kappa operativo **≥ 0.75** en muestras representativas; no sustituye el SLO de modelo pero evita un techo artificial por ruido de etiqueta.

## Evidencia de partida (piloto)

| Indicador | Valor | Fuente |
|-----------|-------|--------|
| Kappa piloto | ~0.23 | [`agreement.json`](agreement.json) |
| Meta operativa kappa | 0.75 | [`agreement.json`](agreement.json) |
| Registros adjudicados | 180 | [`adjudicated_records.csv`](adjudicated_records.csv) |
| Revisión cualitativa pendiente | 134 casos | [`qualitative_review.csv`](qualitative_review.csv) |
| Temas humanos codificados | 0 | [`qualitative_themes.csv`](qualitative_themes.csv) |

Ver también [`STATUS.md`](STATUS.md).

## Fases

### Fase A — Recalibración (1–2 semanas humanas)

1. Sesión de calibración con [`docs/annotation_guidelines_v2.md`](../../docs/annotation_guidelines_v2.md) y casos de [`docs/calibration_cases_v2.md`](../../docs/calibration_cases_v2.md).
2. Re-anotar la muestra piloto (200 IDs en [`pilot_sample_v1.csv`](pilot_sample_v1.csv)) con dos raters independientes.
3. Re-ejecutar [`scripts/analyze_annotation_exports.py`](../../scripts/analyze_annotation_exports.py) → actualizar `agreement.json`.
4. **Criterio de salida:** kappa piloto ≥ 0.75 o plan documentado de corrección de guía antes de escalar.

### Fase B — Adjudicación y cualitativa (paralelo a A)

1. Completar campos `qualitative_theme` / `qualitative_note` en [`adjudication_ledger.csv`](adjudication_ledger.csv) usando [`qualitative_review.csv`](qualitative_review.csv).
2. Ejecutar [`scripts/adjudicate_pilot_release.py`](../../scripts/adjudicate_pilot_release.py) preservando decisiones humanas existentes.
3. **Criterio de salida:** temas resumidos en `qualitative_themes.csv` y ledger sin filas críticas sin adjudicador.

### Fase C — Escala legacy_train_real (4–8 semanas)

1. Definir campaña Argilla `legacy_train_real` (muestreo estratificado por plataforma/clase; priorizar **Facebook** y **Amenazas/Violencia** por brecha SLO).
2. Importar candidatos con [`scripts/import_all_annotation_campaigns.py`](../../scripts/import_all_annotation_campaigns.py) / [`scripts/argilla_workflow.py`](../../scripts/argilla_workflow.py).
3. Dual annotation + adjudicación humana trazable (misma barra QC que [`agreement.json`](agreement.json) → `release.qc_gates`).
4. **Criterio de salida:** release adjudicada con kappa ≥ 0.75 en muestra de control mensual (n ≥ 200).

### Fase D — Candidatos nuevos (~2 324 textos)

1. Resolver duplicados/contradicciones documentados en [`docs/annotation_pilot_issue05.md`](../../docs/annotation_pilot_issue05.md).
2. Exigir `provenance_review_status=verified` antes de gold (ver [`README.md`](README.md)).
3. No mezclar en train hasta pasar Fase C en legacy.

## Entregables por fase

| Fase | Entregable |
|------|------------|
| A | `agreement.json` actualizado, acta de recalibración en Notion/docs |
| B | `qualitative_themes.csv`, ledger completo |
| C | Export Argilla + `adjudicated_records.csv` ampliado |
| D | Manifiesto de procedencia + release ISSUE-08 |

## Comandos de referencia

```powershell
python scripts/setup_argilla_annotators.py
python scripts/generate_pilot_dual_export.py
python scripts/analyze_annotation_exports.py
python scripts/adjudicate_pilot_release.py
python argilla/scripts/build_dataset_v3.py --release-id v4-gold-<fecha>
```

## Qué no hacer (ISSUE-14)

- No relabelar test/val con predicciones del modelo.
- No excluir casos difíciles para inflar kappa o F1.
- No tratar kappa alto como garantía de F1 0.90 sin evaluar [`production_slo_v1`](../metrics/production_slo_v1.json).
