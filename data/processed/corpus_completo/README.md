# Dataset B2 — Corpus Salvadoreño de Toxicidad

Versión corregida y ampliada del dataset v1. **Para reentrenar mBERT y XLM-RoBERTa.**

---

## Qué cambia respecto al v1

| | v1 | **B2** |
|---|---|---|
| train | 2,144 | **3,293** (2,144 reales + 1,149 sintéticos) |
| val | 459 | 459 — **sin cambios** |
| test | 460 | 460 — **sin cambios** |
| Textos con emoji en train | 143 | **1,125** |
| Emoji concentrado en *Lenguaje Ofensivo* | 89% | **23%** |
| Catálogo de emojis | corrupto, ilegible | **51 entradas reconstruidas** |

**`val` y `test` son idénticos al v1, byte por byte.** Esto es deliberado: significa que el F1 de B2 es **directamente comparable** con el que ya midieron. Si sube, es porque el modelo mejoró, no porque la prueba se hizo más fácil.

---

## Por qué se añadieron datos sintéticos

La auditoría de los modelos v1 encontró cuatro huecos que ninguna arquitectura resuelve sin datos:

**1. El atajo del emoji.** El 89% de los textos con emoji del v1 eran *Lenguaje Ofensivo*. El modelo aprendió «hay emoji ⇒ ofensivo» en lugar del significado de cada uno. Un `❤️` en un texto amable lo empujaba a ofensivo.

**2. Emojis de ataque nunca vistos.** 🔫 🔪 ⚰️ 🐒 🧼 🐷 tenían **cero apariciones** en el v1, pese a que el catálogo del proyecto los documenta como vehículo de odio o amenaza.

**3. Amenazas ofuscadas.** «Te voy a m4t4r» se clasificaba como ofensivo, no como amenaza. Había muchas groserías deformadas en el corpus, pero casi ninguna amenaza deformada.

**4. Sesgo dialectal.** El 43% de frases inofensivas con jerga salvadoreña se marcaban como tóxicas, frente al 0% de sus equivalentes en español estándar.

---

## Composición de los 1,149 sintéticos

| Categoría | n | Qué corrige |
|---|---|---|
| `sintetico_relleno_emoji` | 269 | Garantiza ≥8 apariciones por emoji del catálogo |
| `sintetico_emoji_odio` | 179 | Emojis del catálogo, emparejados **coherentemente** con el grupo atacado |
| `sintetico_amenaza_ofuscada` | 138 | Leetspeak (`m4t4r`) y eufemismos salvadoreños (`dar piso`, `dar cuello`) |
| `sintetico_emoji_amenaza` | 130 | 🔫 🔪 ⚰️ 💀 🪦 💣 en amenazas |
| `sintetico_emoji_notoxico` | 125 | Rompe el atajo emoji ⇒ tóxico |
| `sintetico_emoji_ofensivo` | 112 | Mantiene el equilibrio natural |
| `sintetico_jerga_notoxico` | 100 | Jerga salvadoreña en contextos inofensivos |
| `sintetico_nacionalidad_notoxico` | 96 | Mencionar un país **no** es discurso de odio |

La categoría de nacionalidad merece explicación: la auditoría encontró que un bloque completo de contenido deportivo estaba etiquetado como odio solo por mencionar países. Estos 96 ejemplos enseñan explícitamente lo contrario.

**Coherencia grupo↔emoji.** Los emojis no se asignaron al azar. 🧼🇮🇱 solo aparece con «judíos», 🐒🇦🇷 solo con «argentinos», 🐋 solo con sobrepeso. Emparejar un emoji antisemita con «venezolanos» habría enseñado asociaciones falsas. Verificado: **0 incoherencias** grupo↔emoji.

**Contextos por emoji, no más emojis.** El criterio fue garantizar que cada emoji del catálogo aparezca **al menos 8 veces** en contextos y posiciones distintas. La mediana pasó de 2 a 8 apariciones por emoji. Añadir cientos de emojis nuevos vistos una o dos veces no habría enseñado nada.

**Posición variable.** En el corpus real los emojis van al inicio el 12% de las veces, en medio el 22% y al final el 66%. Los sintéticos reproducen esa distribución (11% / 27% / 62%). Si todos fueran al final, el modelo aprendería «emoji en posición final» como pista posicional: un atajo espurio sustituyendo a otro.

---

## Cómo usarlo

```python
import pandas as pd

tr = pd.read_csv("data/dataset_b2/train.csv")
va = pd.read_csv("data/dataset_b2/val.csv")
te = pd.read_csv("data/dataset_b2/test.csv")

X, y = tr["texto_modelo"], tr["label"]     # mismos campos que el v1
```

El esquema de columnas es **idéntico al v1** más dos campos nuevos:

| Campo nuevo | Valores |
|---|---|
| `origen` | `real` o `sintetico_*` |
| `plantilla_id` | Identificador de la plantilla generadora, vacío en los reales |

Para cualquier análisis que deba excluir sintéticos: `tr[tr.origen == "real"]`.

### Pesos de clase — usar los nuevos

```python
pesos = [0.9198, 0.9376, 0.9720, 1.2233]   # NT, LO, DO, AV
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(pesos, dtype=torch.float))
```

Están en `metadata.json`. **No reutilicen los del v1**: la distribución de train cambió.

---

## Conjunto diagnóstico

`test_adversarial.csv` — 75 casos de ofuscación, emojis, amenazas encubiertas y jerga inofensiva.

**No es un conjunto de prueba.** Es diagnóstico: se reporta **aparte**, nunca como métrica principal. Sirve para verificar que las correcciones funcionaron.

Resultado del modelo XLM-R entrenado con el v1, como línea base:

| Familia | Acierto v1 |
|---|---|
| Ofuscación de groserías | 85% |
| Jerga salvadoreña inofensiva | 57% |
| Amenazas ofuscadas | 45% |
| Emojis | 33% |
| **Global** | **48/75 (64%)** |

Si tras reentrenar con B2 estos números no suben —especialmente amenazas y emojis— el problema no eran los datos y hay que revisar la configuración.

---

## Reglas para no romper la comparabilidad

1. **No mover sintéticos a `val` ni `test`.** Están solo en train por diseño.
2. **Reportar siempre dos bloques:** métricas sobre `test.csv` (comparables con v1) y sobre `test_adversarial.csv` (robustez).
3. **Declarar el aumento sintético en la tesina.** Son 1,149 ejemplos generados por plantilla, un 35% del train. Es metodología estándar, pero debe decirse.
4. **Exportar las predicciones** de cada modelo sobre `test.csv` con dos columnas `ID,y_pred`, para poder correr la prueba pareada:

```bash
python src/evaluation/comparar_modelos.py \
       reports/preds_mbert.csv reports/preds_xlmr.csv \
       --nombres mBERT XLM-R --test data/dataset_b2/test.csv
```

---

## Reproducibilidad

```bash
python src/data/build_dataset_b2.py
```

Semilla fija (42). Módulos: `src/data/lexicon_emojis.py` (catálogo), `src/data/augmentation.py` (plantillas), `src/features/normalization.py` (normalización, sin cambios respecto al v1).

## Limitaciones

- Los sintéticos provienen de plantillas: son más regulares que el lenguaje real y no sustituyen recolección. La meta sigue siendo llegar a 771 ejemplos reales por clase.
- El catálogo de emojis se reconstruyó a partir de descripciones textuales; los glifos originales se perdieron por un problema de codificación y fueron reasignados manualmente.
- Los emojis de amenaza (🔫🔪⚰️) no aparecen en el corpus real salvadoreño recolectado. Su cobertura es enteramente sintética hasta que se recolecten casos reales.
