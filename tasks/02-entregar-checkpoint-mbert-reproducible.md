# T-02 — Entregar un checkpoint mBERT reproducible

**Responsable:** Cristian Escalante  
**Modelo:** mBERT (`bert-base-multilingual-cased`)  
**Prioridad:** alta  
**Hallazgo de auditoría:** 8

## Por qué

La auditoría no pudo reproducir las métricas de mBERT ni aplicarle la batería adversarial porque la entrega auditada no incluía los pesos entrenados. Sin el checkpoint exacto, tokenizer y configuración de inferencia, el F1 informado no puede verificarse de manera independiente.

## Objetivo

Entregar el modelo mBERT exacto que produjo las métricas oficiales junto con los metadatos necesarios para que un revisor pueda cargarlo y repetir la evaluación.

## Qué se debe hacer

1. Publicar mediante el canal de artefactos acordado —sin versionar binarios pesados en Git— un paquete con `model.safetensors`, `config.json`, `tokenizer.json`, `tokenizer_config.json` y los archivos auxiliares necesarios.
2. Incluir un manifiesto con commit, fecha, semilla, modelo base, checkpoint/época, hash SHA-256 de los archivos, `max_length=128`, función de normalización y versión de `test.csv`.
3. Incorporar una guía de inferencia que use `texto_modelo` y la misma configuración de entrenamiento/evaluación.
4. Entregar el vector de predicciones mBERT o un comando reproducible para generarlo sobre las 460 instancias del test.
5. Probar el paquete desde un entorno limpio y guardar la salida de verificación.

## Criterios de aceptación

- [x] El paquete permite cargar mBERT sin depender de archivos locales no declarados.
- [x] El manifiesto identifica de forma única el checkpoint, commit, hashes, semilla, normalización y `max_length`.
- [x] Una ejecución independiente sobre `data/processed/test.csv` reproduce el F1-macro oficial dentro de la tolerancia numérica declarada.
- [x] El paquete incluye o genera 460 predicciones alineadas con el test y con etiquetas de 0 a 3.
- [x] Los pesos siguen fuera del índice de Git, pero el informe registra dónde obtener el paquete de auditoría.

## Evidencia T-02

| Artefacto | Rol |
|-----------|-----|
| `artifacts/mbert-sv-audit.zip` | Paquete de auditoría (gitignored, ~630 MB) |
| `reports/metrics/mbert_checkpoint_manifest.json` | Commit, semilla 42, `checkpoint-1072`, SHA-256, contrato |
| `reports/metrics/mbert_test_predictions.csv` | 460 predicciones `id,y_pred_mbert` (etiquetas 0–3) |
| `reports/metrics/mbert_checkpoint_verify.json` | F1-macro 0.777719 vs oficial 0.7777, `passed: true` |
| `docs/guia-inferencia-mbert.md` | Guía de carga e inferencia |

```powershell
python scripts/package_mbert_checkpoint.py
python scripts/export_mbert_predictions.py
python scripts/verify_mbert_checkpoint.py --package artifacts/mbert-sv-audit.zip
```

