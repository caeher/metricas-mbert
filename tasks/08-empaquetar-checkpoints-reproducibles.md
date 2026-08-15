# T-08 — Empaquetar checkpoints reproducibles

**Prioridad:** alta  
**Responsable:** Cristian para mBERT; Julio para XLM-R  
**Hallazgo de auditoría:** 8

## Por qué

La auditoría no pudo reproducir mBERT ni ejecutarle la batería adversarial porque los pesos entrenados no formaban parte de la entrega. Además, el checkpoint XLM-R entregado correspondía a época 1 y no contenía los pesos necesarios. Una métrica sin el artefacto exacto que la produjo no puede auditarse de forma independiente.

## Objetivo

Entregar para cada modelo el checkpoint real evaluado, sus dependencias de inferencia y una cadena verificable hacia las métricas publicadas, sin incluir binarios pesados en Git.

## Qué se debe hacer

1. Preparar un paquete o ubicación de artefactos externa para cada modelo con `model.safetensors`, `config.json`, tokenizer, configuración de etiquetas y requisitos de inferencia.
2. Añadir un manifiesto con modelo base, época/checkpoint, commit, hash SHA-256 de cada archivo, semilla, `max_length`, normalizador y conjunto de prueba.
3. Incluir el vector de predicciones y/o un comando reproducible que regenere las métricas desde el paquete.
4. Probar el paquete en un entorno limpio, usando `texto_modelo`, y registrar las métricas obtenidas con tolerancia numérica declarada.
5. Enlazar el identificador del paquete en el informe y en los artefactos de métricas.

## Criterios de aceptación

- [ ] Cada modelo tiene pesos reales, configuración, tokenizer y mapeo de clases disponibles en un paquete verificable.
- [ ] El manifiesto identifica sin ambigüedad checkpoint, commit y hashes de los archivos.
- [ ] Una persona revisora puede cargar el paquete y reproducir F1-macro con la tolerancia especificada sobre `test.csv`.
- [ ] La época/checkpoint usada para las métricas coincide con la declarada en el informe.
- [ ] Los pesos no se añaden al repositorio Git; su entrega se realiza mediante el canal de artefactos acordado.

