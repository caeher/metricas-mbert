# Auditoría mBERT — Resultados y correcciones (Cristian Escalante)

> Extracción del [Informe de Auditoría de Modelos](../Informe_Auditoria_de_Modelos.docx) (UES, 14 ago 2026).  
> **Alcance:** solo mBERT / Cristian Escalante, más el hallazgo compartido McNemar mBERT vs XLM-R.  
> **No incluye** hallazgos ni correcciones exclusivas de Julio / XLM-RoBERTa.

| Campo | Valor |
|-------|-------|
| Modelo | mBERT (`bert-base-multilingual-cased`) |
| Responsable | Cristian Antonio Escalante Hernández |
| Conjunto de prueba | n = 460 (`data/processed/test.csv`) |
| Fecha del informe de auditoría | 14 de agosto de 2026 |

---

## Resumen ejecutivo

- **Limitación de la auditoría:** en la entrega auditada, mBERT **no incluía los pesos entrenados**. No se reprodujeron métricas de forma independiente ni se aplicó la batería adversarial de 75 casos. La revisión se limitó a código e informe.
- **Lo que la auditoría valida:** el código de evaluación de mBERT es sólido (cortes con variantes `SI`/`SÍ`, sonda de sesgo dialectal bien implementada). El informe de mBERT es metodológicamente más riguroso que el de XLM-R (bootstrap, McNemar vs baselines, calibración, benchmark externo).
- **Resultado global:** mBERT queda por debajo de XLM-R en el punto estimado (~2 pp de F1-macro), pero los intervalos de confianza al 95 % **se solapan**; con n = 460 la diferencia no es estadísticamente distinguible del ruido de muestreo.
- **Hallazgo crítico compartido (#4):** falta la prueba de McNemar pareada mBERT vs XLM-R. Cristian ya tiene la función; depende de que Julio exporte el vector de predicciones de XLM-R.
- **Hallazgo propio de Cristian (#5):** error en la fórmula del baseline trivial de clase mayoritaria (0.087 informado vs 0.128 correcto). Impacto práctico bajo, pero genera inconsistencia con el informe de preparación de datos v1.1 (0.129).
- **Entrega pendiente (#8):** incluir los pesos de `models/mbert-sv/` en la entrega para permitir verificación independiente y pruebas de robustez.

---

## Limitación de la auditoría sobre mBERT

El informe señala explícitamente:

> El modelo mBERT no incluye los pesos entrenados en la entrega, por lo que sus métricas no pudieron reproducirse de forma independiente ni someterse a la batería adversarial. La revisión de mBERT se limitó a su código e informe.

**Estado en este repositorio:** los pesos sí existen localmente en `resultados-metricas/models/mbert-sv/model.safetensors`, pero `models/mbert-sv/` está excluido de git (`.gitignore`). La corrección #8 es de **entrega/auditoría**, no de entrenamiento.

---

## Resultados de mBERT según la auditoría

### Métricas globales (Tabla 7 del informe)

| Modelo | F1-macro | IC 95 % | ¿Alcanza 0.80? |
|--------|----------|---------|----------------|
| XLM-RoBERTa | 0.8058 | [0.7665, 0.8420] | Sí, en el punto estimado |
| **mBERT** | **0.7862** | **[0.7473, 0.8220]** | **No** |
| LinearSVC (baseline) | 0.7645 | No reportado | No |
| LogisticRegression | 0.7576 | No reportado | No |
| MultinomialNB | 0.6501 | No reportado | No |

**Interpretación de la auditoría:**

- Diferencia mBERT vs XLM-R: ~2 puntos porcentuales de F1-macro.
- Los IC 95 % se solapan de forma sustancial → la diferencia **no es distinguible** del ruido de muestreo con n = 460.
- LinearSVC queda a ~0.042 puntos de XLM-R. La prueba de McNemar del informe de mBERT ya mostró que frente a los baselines lineales el margen **no es concluyente** — resultado que la tesina debe afrontar de frente.

### Cortes por plataforma (mBERT — referencia usada por error en XLM-R)

Las cifras de plataforma de Cristian fueron las que el informe de XLM-R copió por error (hallazgo 3 de Julio). Valores propios de mBERT según este repo:

| Corte | F1-macro | Fuente |
|-------|----------|--------|
| Plataforma X | 0.839 | `reports/metrics/slice_metrics.csv` |
| Plataforma Facebook | 0.673 | `reports/metrics/slice_metrics.csv` |

La auditoría cita valores redondeados ~0.8503 (X) y ~0.6896 (Facebook) en el informe de Cristian; el repo actual reporta 0.839 y 0.673 respectivamente (misma magnitud, redondeo distinto).

### Lo que la auditoría reconoce como correcto en mBERT

| Aspecto | Evidencia en el repo |
|---------|----------------------|
| Cortes con `SI` / `SÍ` | `src/discurso_odio/evaluation/slices.py` |
| Sonda de sesgo dialectal calculada (no hardcodeada) | `dialect_bias_probe()` → `reports/metrics/dialect_bias_probe.csv` |
| Bootstrap, McNemar vs baselines, calibración | `src/discurso_odio/evaluation/metrics.py`, `scripts/run_pipeline.py` |
| Benchmark externo | `scripts/run_benchmarks.py` |
| McNemar vs baselines ya ejecutado | `reports/metrics/mcnemar_tests.csv` |

**McNemar mBERT vs baselines (repo actual):**

| Baseline | p-value | Conclusión |
|----------|---------|------------|
| LogisticRegression | 0.278 | No significativo |
| LinearSVC | 0.520 | No significativo |
| MultinomialNB | 4.1 × 10⁻⁶ | Significativo |

---

## Hallazgos aplicables a Cristian / mBERT

### Hallazgo 4 — No hay McNemar pareado mBERT vs XLM-R

| Campo | Detalle |
|-------|---------|
| **Severidad** | Crítica |
| **Responsable** | Julio y Cristian |
| **Descripción** | El objetivo del proyecto incluye comparar mBERT y XLM-R, pero hoy solo se comparan dos números sin prueba de hipótesis pareada. |
| **Evidencia en repo** | `mcnemar_test()` existe en `src/discurso_odio/evaluation/metrics.py` (líneas 83–94). `scripts/run_pipeline.py` solo compara mBERT vs baselines TF-IDF (líneas 98–106). No hay fila `xlm-roberta` en `reports/metrics/mcnemar_tests.csv`. |
| **Estado** | Pendiente |

**Contrato del vector de predicciones XLM-R (para Julio):**

- 460 enteros, labels 0–3.
- Mismo orden que `data/processed/test.csv` (columna `label`).
- Export sugerido: CSV con columnas `id` (opcional) y `y_pred_xlmr`.

**Pasos para Cristian cuando Julio entregue el vector:**

1. Cargar `test.csv` y el CSV de predicciones XLM-R.
2. Obtener predicciones mBERT: `predict(test["texto_modelo"], MODELS_DIR / "mbert-sv")`.
3. Ejecutar `mcnemar_test(y_true, y_pred_mbert, y_pred_xlmr)`.
4. Añadir fila a `reports/metrics/mcnemar_tests.csv` y al informe LaTeX (`reports/latex/chapters/06_evaluacion_completa.tex`).

> La auditoría anticipa que la prueba **probablemente no será significativa**; ese resultado también es válido y defendible.

---

### Hallazgo 5 — Error en baseline trivial de clase mayoritaria

| Campo | Detalle |
|-------|---------|
| **Severidad** | Alta |
| **Responsable** | Cristian |
| **Descripción** | `trivial_baselines()` usa la proporción de la clase mayoritaria como precisión **y** recall. Para un clasificador que siempre predice la mayoritaria, el recall de esa clase es **1**, no su proporción. |
| **Evidencia en repo** | `src/discurso_odio/evaluation/metrics.py` (líneas 162–186) |

**Tabla de verificación (auditoría):**

| Baseline trivial | Informe mBERT | Valor correcto | Diferencia |
|------------------|---------------|----------------|------------|
| Clase mayoritaria | 0.087 | **0.128** | Subestimado ~32 % |
| Azar uniforme | 0.250 | 0.246 | Menor |
| Azar estratificado | 0.250 | 0.250 | Correcto |

**Fórmula correcta (F1-macro, clasificador siempre-majority):**

Con k = 4 clases y proporción p de la mayoritaria en test:

```
F1_macro = 2p / (k × (1 + p))
```

En test (n = 460): label 1 (Lenguaje Ofensivo) = 159/460 → p ≈ 0.3457 → **F1 = 0.128**.

El informe de preparación de datos v1.1 reporta **0.129** para la misma cantidad; la discrepancia 0.087 vs 0.128/0.129 es el problema señalado.

**Artefactos afectados hoy (valor incorrecto 0.087):**

| Archivo | Valor actual |
|---------|--------------|
| `reports/metrics/trivial_baselines.csv` | `clase_mayoritaria,0.087` |
| `reports/metrics/model_comparison.csv` | `trivial_clase_mayoritaria,0.087` |
| `reports/latex/generated/tab_trivial_baselines.tex` | `0.0870` |
| `reports/latex/chapters/04_baselines.tex` | Referencia al capítulo EDA con pisos triviales |

**Corrección de código (referencia, no aplicada en este documento):**

```python
# En trivial_baselines(): recall mayoritaria = 1.0, no props[majority_class]
p = props[majority_class]
f1_per_class = [2 * p * 1 / (p + 1) if c == majority_class else 0.0 for c in class_counts]
f1_majority = sum(f1_per_class) / k  # → 0.128 con test actual
```

**Tests:** añadir aserción en `tests/test_metrics.py` con counts del test real (159/460 mayoritaria → 0.128).

**Estado:** Pendiente.

---

### Hallazgo 8 — Entregar pesos de mBERT

| Campo | Detalle |
|-------|---------|
| **Severidad** | Alta |
| **Responsable** | Cristian |
| **Descripción** | Sin `model.safetensors` (y artefactos asociados) la auditoría no puede reproducir métricas ni correr robustez adversarial sobre mBERT. |
| **Evidencia en repo** | Pesos en `resultados-metricas/models/mbert-sv/`; `models/mbert-sv/` en `.gitignore` (línea 14). |
| **Estado** | Pendiente de entrega formal (ZIP/Drive/compartido con auditoría). |

**Checklist de entrega mínima:**

- [ ] `models/mbert-sv/model.safetensors`
- [ ] `models/mbert-sv/config.json`
- [ ] `models/mbert-sv/tokenizer.json` + `tokenizer_config.json`
- [ ] `models/mbert-sv/test_metrics.json` (opcional, trazabilidad)
- [ ] Instrucciones de inferencia (`texto_modelo`, `max_length=128`)

---

## Correcciones requeridas (solo Cristian / compartidas)

### Antes de entregar

| # | Corrección | Prioridad | Responsable | Criterio de cierre |
|---|------------|-----------|-------------|-------------------|
| 4 | Exportar predicciones XLM-R y ejecutar McNemar pareado vs mBERT | **Crítica** | Julio + Cristian | Fila `xlm-roberta` en `mcnemar_tests.csv`; párrafo en informe con estadístico y p-value |
| 5 | Corregir fórmula baseline clase mayoritaria → **0.128**; alinear con informe v1.1 | **Alta** | Cristian | `trivial_baselines()` corregida; CSV/LaTeX regenerados; test unitario |
| 8 | Entregar pesos `models/mbert-sv/` para verificación y robustez | **Alta** | Cristian | Auditoría confirma reproducción de F1-macro ± tolerancia numérica |

### Siguiente iteración (recomendaciones de la auditoría, aplicables a mBERT)

Aunque la batería adversarial no se corrió sobre mBERT, el informe recomienda para ambos modelos:

1. **Ampliar cobertura de amenazas ofuscadas** en el corpus (leetspeak, eufemismos locales como «dar piso»).
2. **Corregir asociación espuria emoji–toxicidad** (más textos no tóxicos con emojis).
3. **Mitigar sesgo dialectal** (más ejemplos no tóxicos con jerga salvadoreña).
4. **Reportar F1 desagregado por plataforma** — mBERT ya lo cubre parcialmente en `slice_metrics.csv`; mantener y citar en la tesina.

---

## Consistencia: cifras de la auditoría vs este repositorio

> **No mezclar columnas.** Usar una fuente por documento; esta tabla explica por qué difieren.

### F1-macro global mBERT

| Fuente | F1-macro | IC 95 % bootstrap | Notas |
|--------|----------|-------------------|-------|
| Informe de auditoría (Tabla 7) | **0.7862** | [0.7473, 0.8220] | Cifras citadas en la revisión independiente |
| Repo actual (`reports/metrics/mbert_test_metrics.csv`) | **0.7777** | [0.7358, 0.8144] | Pipeline local / Colab; redondeo informe LaTeX: 0.778 |
| Informe LaTeX (`reports/latex/chapters/06_evaluacion_completa.tex`) | **0.778** | [0.736, 0.814] | Versión redondeada del CSV |

**Posibles causas de la brecha auditoría (0.786) vs repo (0.778):**

- Pesos o checkpoint distinto en la entrega auditada vs `resultados-metricas/models/mbert-sv/`.
- Semilla, época de checkpoint o split no idéntico.
- La auditoría no pudo verificar mBERT; el 0.7862 proviene del informe de Cristian, no de reproducción independiente.

**Acción:** al entregar pesos (#8), documentar commit/checkpoint y regenerar métricas una sola vez; usar **una** cifra oficial en la tesina.

### Baseline trivial clase mayoritaria

| Fuente | Valor | Estado |
|--------|-------|--------|
| Informe mBERT (incorrecto) | 0.087 | Bug en `trivial_baselines()` |
| Auditoría / fórmula correcta | **0.128** | Con p = 159/460 en test |
| Informe preparación datos v1.1 | 0.129 | Referencia de alineación |
| Repo actual (`trivial_baselines.csv`) | 0.087 | **Pendiente corrección #5** |

### Sonda dialectal mBERT (repo — calculada, no hardcodeada)

| Métrica | Valor repo | Informe auditoría (XLM-R, referencia) |
|---------|------------|----------------------------------------|
| FPR No Tóxico sin jerga | 15.96 % | 15.96 % (recalculado) |
| FPR No Tóxico con jerga | 41.38 % | 44.83 % (XLM-R adversarial) |

Los valores de mBERT provienen de `dialect_bias_probe()` sobre predicciones reales del modelo, no de constantes en código.

---

## Mapa de archivos relevantes

| Ruta | Rol |
|------|-----|
| `src/discurso_odio/evaluation/metrics.py` | `trivial_baselines`, `mcnemar_test`, `bootstrap_ci` |
| `src/discurso_odio/evaluation/slices.py` | Cortes por plataforma, jerga, sarcasmo, ironía |
| `scripts/run_pipeline.py` | Evaluación test, export CSV, McNemar vs baselines |
| `scripts/train_mbert.py` | Entrenamiento mBERT |
| `data/processed/test.csv` | Conjunto de prueba (n = 460) |
| `models/mbert-sv/` | Pesos fine-tuned (gitignored; ver `resultados-metricas/`) |
| `reports/metrics/mbert_test_metrics.csv` | Métricas globales actuales |
| `reports/metrics/mcnemar_tests.csv` | McNemar vs baselines (falta XLM-R) |
| `reports/metrics/trivial_baselines.csv` | Pisos triviales (bug #5) |
| `reports/metrics/slice_metrics.csv` | F1 por plataforma y cortes |
| `reports/latex/chapters/06_evaluacion_completa.tex` | Capítulo de evaluación en informe |

---

## Fuera de alcance (Julio / XLM-R)

Estos ítems aparecen en el informe de auditoría pero **no son responsabilidad de Cristian**:

- Hallazgos 1–3: FPR jerga hardcodeado, cortes sarcasmo/ironía rotos, valores por plataforma copiados de mBERT.
- Hallazgos 6–7, 9–13: max_length 128 vs 256, falta bootstrap/McNemar en informe XLM-R, checkpoint época 1, robustez adversarial, id2label, .venv en git.
- Correcciones 1–3, 6–7, 9–10 de la Tabla 8 del informe.

---

## Referencia

- Documento fuente: `Informe_Auditoria_de_Modelos.docx` (raíz del repositorio).
- Fecha de extracción a Markdown: 15 de agosto de 2026.
