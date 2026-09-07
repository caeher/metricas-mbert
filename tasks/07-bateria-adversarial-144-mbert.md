# T-07 — Batería adversarial extendida (144 casos, A–G)

**Responsable:** Cristian Escalante  
**Modelo:** mBERT (`bert-base-multilingual-cased`)  
**Prioridad:** alta  
**Origen:** Informe de Verificación de Modelos (1 sep 2026), sección 4

## Por qué

La verificación independiente evaluó ambos modelos sobre una batería de **144 casos** (familias A–G), incluyendo pares mínimos de emoji y falsos amigos dialectales. T-06 cubría solo la batería v1 ($n=75$). Sin la evaluación 144, los resultados de robustez no son comparables con el informe de verificación.

## Objetivo

Integrar la batería `test_adversarial_144.csv` en el pipeline oficial de mBERT, publicar métricas por familia A–G y documentar la comparación con la referencia XLM-R de la verificación.

## Qué se debe hacer

1. Mantener `data/processed/test_adversarial_144.csv` versionado con manifiesto SHA-256 (`adversarial_v2_144`).
2. Evaluar el checkpoint oficial (`checkpoint-1072`) con `evaluate_adversarial_144_battery()` o `scripts/evaluate_adversarial.py --battery-144`.
3. Exportar `mbert_v1_144_adversarial_{summary,family_metrics,predictions}.json/csv`.
4. Generar `tab_adversarial_144_families.tex` con mBERT vs referencia XLM-R.
5. Citar la batería 144 como evaluación principal; conservar la v1 ($n=75$) como histórico/diagnóstico.

## Criterios de aceptación

- [x] Existe manifiesto `adversarial_144_battery_manifest.json` con SHA-256 y familias A–G.
- [x] `evaluate_adversarial.py --battery-144` y `run_t06_pipeline.py --battery-144` ejecutan la evaluación 144.
- [x] El informe LaTeX incluye Tabla `tab:adversarial-144-families` y prosa que distingue 144 vs 75.
- [x] La verificación documenta discrepancias metodológicas si los aciertos globales difieren de la reejecución independiente.
- [ ] Con checkpoint local disponible: regenerar predicciones 144 y confirmar cifras en `mbert_v1_144_adversarial_summary.json`.

## Evidencia

| Artefacto | Rol |
|-----------|-----|
| `data/processed/test_adversarial_144.csv` | Batería A–G ($n=144$) |
| `reports/metrics/adversarial_144_battery_manifest.json` | Versión y SHA-256 |
| `reports/metrics/mbert_v1_144_adversarial_summary.json` | Resumen global y grupo G |
| `reports/latex/generated/tab_adversarial_144_families.tex` | Tabla mBERT vs XLM-R |

```powershell
python scripts/build_adversarial_144.py
python scripts/evaluate_adversarial.py --battery-144 --output-prefix mbert_v1 --model-name mbert-v1
python scripts/run_t06_pipeline.py --battery-144
python scripts/generate_latex_report.py
```

## Nota sobre reconciliación 69 vs 74

La verificación independiente (NumPy sobre `model.safetensors`) reportó **74/144** aciertos globales para mBERT. Una corrida previa del repo registró **69/144**. La diferencia puede deberse a versión de batería, criterio de familia F/G o variación numérica en inferencia. La tabla 144 cita la referencia XLM-R del informe de verificación; mBERT debe regenerarse desde el checkpoint oficial para auditoría.
