# T-06 — Evaluar robustez y equidad de mBERT

**Responsable:** Cristian Escalante  
**Modelo:** mBERT (`bert-base-multilingual-cased`)  
**Prioridad:** alta  
**Origen:** recomendaciones de robustez, emojis y sesgo dialectal de la auditoría

## Por qué

La auditoría no pudo someter mBERT a la batería adversarial por ausencia de pesos. XLM-R mostró fallas relevantes que también deben comprobarse en mBERT: amenazas ofuscadas (5/11 aciertos), emojis con asociación espuria (6/18 aciertos) y falsos positivos sobre jerga salvadoreña inofensiva (6/14, frente a 0/12 en español estándar). No es válido asumir que mBERT es robusto o equitativo sin medirlo.

## Objetivo

Medir y mejorar, si la evidencia lo exige, la robustez y la equidad dialectal de mBERT sin contaminar el conjunto de evaluación ni ocultar compensaciones en las métricas globales.

## Qué se debe hacer

1. Usar el checkpoint oficial de T-02 y aplicar una batería adversarial versionada, separada de train/val/test, con amenazas ofuscadas, emojis y pares dialectales semánticamente equivalentes.
2. Reportar resultados por familia de casos: aciertos, clase predicha, confianza, F1 de Amenazas/Violencia, falsos negativos de amenazas, FPR dialectal y F1-macro global.
3. Para dialecto, comparar frases inofensivas con jerga salvadoreña frente a equivalentes en español estándar, incluyendo `n` e intervalos de confianza cuando sea posible.
4. Si aparecen brechas, ampliar datos de entrenamiento con amenazas ofuscadas, textos no tóxicos con emojis y jerga salvadoreña no tóxica, sin reutilizar la batería como entrenamiento.
5. Reentrenar y comparar contra la línea base; documentar efectos sobre clases de alto riesgo y sobre la métrica global antes de adoptar una mitigación.

## Criterios de aceptación

- [x] mBERT se evalúa sobre una batería adversarial versionada y separada de los datos de entrenamiento.
- [x] El reporte identifica el checkpoint y muestra resultados desagregados para amenazas ofuscadas, emojis y jerga dialectal.
- [x] Las métricas incluyen F1 de Amenazas/Violencia, falsos negativos de amenazas, FPR por grupo dialectal, tamaños de muestra y F1-macro global.
- [x] Ninguna muestra de la batería adversarial se usa para entrenar, validar o seleccionar el modelo.
- [x] Si se aplica mitigación, se publican antes/después y las compensaciones; si no se aplica, se documentan los riesgos residuales.

## Evidencia T-06

| Artefacto | Rol |
|-----------|-----|
| `data/processed/test_adversarial.csv` | Batería v1 (n=75), diagnóstico |
| `reports/metrics/adversarial_battery_manifest.json` | Versión `adversarial_v1`, SHA-256, familias |
| `reports/metrics/mbert_adversarial_summary.json` | Baseline `checkpoint-1072`: 45/75, F1 diag=0.5314, FN amenazas=10 |
| `reports/metrics/mbert_adversarial_family_metrics.csv` | Por familia A–E: aciertos/FPR, confianza, IC Wilson |
| `reports/metrics/mbert_adversarial_predictions.csv` | Predicciones caso a caso (y_pred, confianza) |
| `reports/metrics/mbert_b2_adversarial_summary.json` | Candidato B2 (1 época): 59/75, F1 diag=0.7489, FN=0 |
| `reports/metrics/mbert_adversarial_before_after.csv` | Antes/después; **mitigación no adoptada** (F1 test 0.7777→0.7161) |
| `reports/latex/generated/tab_adversarial_*.tex` | Tablas y prosa generadas |
| `reports/latex/figures/11_adversarial_families.png` | Aciertos por familia vs XLM-R v1 |
| `tests/test_adversarial_eval.py` | Esquema, fugas val/test, SHA-256, checkpoint |

### Resultados clave (baseline oficial)

| Familia | mBERT (1072) | XLM-R v1 ref. |
|---------|--------------|---------------|
| C Amenazas ofuscadas | 6/11 | 5/11 |
| B Emojis | 7/18 | 6/18 |
| D Jerga inofensiva FPR | 12/14 (85.7%) | 6/14 |
| E Control FPR | 0/12 | 0/12 |
| Global | 45/75 | 48/75 |

### Mitigación B2

Se entrenó `models/mbert-sv-b2` (train B2, 7 filas solapadas excluidas, 1 época / MAX_STEPS=411). Mejora adversarial (59/75, grupo C 11/11, FPR D 4/14) pero **no se adopta**: F1-macro test cae >0.02 y F1 Amenazas/Violencia en test empeora. Riesgos residuales documentados en `adversarial_tradeoffs.tex`.

```powershell
python scripts/evaluate_adversarial.py --model-dir models/mbert-sv --eval-test
python scripts/train_mbert.py --output-dir models/mbert-sv-b2 --exclude-adversarial-overlap
python scripts/evaluate_adversarial.py --model-dir models/mbert-sv-b2 --model-name mbert-b2 --output-prefix mbert_b2 --eval-test
python scripts/run_t06_pipeline.py
python scripts/generate_latex_report.py
python -m pytest tests/test_adversarial_eval.py -q
```
