---
language:
  - es
license: apache-2.0
base_model: google-bert/bert-base-multilingual-cased
tags:
  - text-classification
  - hate-speech
  - toxicity
  - spanish
  - el-salvador
  - mbert
pipeline_tag: text-classification
library_name: transformers
metrics:
  - f1
  - accuracy
model-index:
  - name: mbert-sv
    results:
      - task:
          type: text-classification
          name: Toxicidad en español salvadoreño
        dataset:
          name: CDE v1.9.0 (test, n=460)
          type: custom
        metrics:
          - type: f1
            value: 0.7784
            name: F1-macro
          - type: accuracy
            value: 0.7804
            name: Accuracy
---

# mBERT-SV — Clasificación de toxicidad (El Salvador)

Fine-tuning de [`bert-base-multilingual-cased`](https://huggingface.co/google-bert/bert-base-multilingual-cased) para clasificación **multiclase** de toxicidad en redes sociales en español salvadoreño.

Checkpoint publicado: **`checkpoint-4050`** (release **v6**, init desde v4, semilla 42, dataset CDE **v1.9.0**, 21 585 ejemplos de entrenamiento).

## Etiquetas

| id | etiqueta |
|----|----------|
| 0 | No Tóxico |
| 1 | Lenguaje Ofensivo |
| 2 | Discurso de Odio |
| 3 | Amenazas/Violencia |

## Contrato de inferencia

| Campo | Valor |
|-------|-------|
| `max_length` | 128 |
| Columna de texto | `texto_modelo` |
| Normalizador | `normalize_for_model` v1.1 |

Si el texto no está preprocesado, hay que aplicar el mismo normalizador del repositorio de la tesina (`inference_contract.json`).

## Métricas oficiales (test, n = 460)

| Métrica | Valor |
|---------|-------|
| F1-macro | **0.7784** |
| Accuracy | 0.7804 |
| Precision-macro | 0.7800 |
| Recall-macro | 0.7776 |
| QWK | 0.7286 |

F1 por clase: No Tóxico 0.772 · Lenguaje Ofensivo 0.799 · Discurso de Odio 0.760 · Amenazas/Violencia 0.783.

## Uso

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

repo_id = "caeher/mbert-sv"
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForSequenceClassification.from_pretrained(repo_id)

text = "ejemplo de comentario"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
with torch.no_grad():
    logits = model(**inputs).logits
pred = int(logits.argmax(dim=-1))
print(model.config.id2label[str(pred)])
```

## Archivos

- `model.safetensors`, `config.json`, `tokenizer.json`, `tokenizer_config.json`
- `inference_contract.json` — contrato de preprocesamiento
- `test_metrics.json` / `train_metrics.json` — métricas del checkpoint
- `training_run.json` — receta y trazabilidad del entrenamiento v6

## Limitaciones

Modelo de tesina, no un moderador autónomo. El corpus es de El Salvador; el desempeño puede caer en otros dialectos o en jerga adversarial. FPR sobre jerga no tóxica y cortes por plataforma están documentados en el informe del proyecto.
