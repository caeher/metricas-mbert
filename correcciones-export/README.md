# Paquete de entrega — Correcciones mBERT (Cristian)

Entrega consolidada de las tareas T-01 a T-06 derivadas de la auditoría de modelos.
**Alcance:** solo mBERT / Cristian Escalante. Sin XLMR ni correcciones de Julio.

## Contenido

| Archivo | Descripción |
|---------|-------------|
| `mbert-cristian-colab.zip` | Código, datos, métricas y libreta Colab |
| `mbert-sv-audit.zip` | Checkpoint oficial (~630 MB, T-02) |
| `latex/main.docx` | Informe compilado (pandoc) |
| `notebooks/replicar_mbert_colab.ipynb` | Libreta única de replicación |
| `latex/` | Informe LaTeX de correcciones |

## Checklist T-01 … T-06

- [x] **T-01** Baseline clase mayoritaria: 0.128 (antes 0.087)
- [x] **T-02** Checkpoint reproducible + manifiesto SHA-256
- [x] **T-03** McNemar mBERT vs XLM-R: p=0.200 (no significativo)
- [x] **T-04** F1-macro oficial: **0.7777** (checkpoint-1072)
- [x] **T-05** Cortes por plataforma y contexto
- [x] **T-06** Batería adversarial; mitigación B2 **no adoptada**

## Cifras canónicas

| Métrica | Valor |
|---------|-------|
| F1-macro test | 0.7777 |
| Baseline mayoritaria | 0.128 |
| McNemar vs XLM-R | p = 0.200 |
| Adversarial global | 45/75 |
| Mitigación B2 | Rechazada (test 0.7161) |

## Uso en Google Colab

1. Runtime → GPU (T4).
2. Subir `mbert-cristian-colab.zip`.
3. Abrir y ejecutar `notebooks/replicar_mbert_colab.ipynb`.
4. **Opción A:** subir `mbert-sv-audit.zip` para verificar F1 = 0.7777 sin reentrenar.
5. **Opción B:** reentrenar (2–4 h GPU); el F1 puede variar ligeramente.

## Verificación local

```powershell
python scripts/verify_mbert_checkpoint.py --package artifacts/mbert-sv-audit.zip
python scripts/evaluate_mbert_official.py --model-dir models/mbert-sv
pytest tests/test_adversarial_eval.py -q
```

## Regenerar este paquete

```powershell
python scripts/build_correcciones_export.py
```

## Fuera de alcance

- Carpeta `XLMR/`
- Hallazgos y correcciones exclusivas de Julio / XLM-RoBERTa
- Modelo candidato B2 (documentado pero no adoptado)
