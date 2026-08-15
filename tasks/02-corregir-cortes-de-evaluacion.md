# T-02 — Corregir y ampliar los cortes de evaluación

**Prioridad:** crítica  
**Responsable:** Julio (XLM-R)  
**Hallazgo de auditoría:** 2

## Por qué

Los cortes de sarcasmo e ironía de XLM-R no se ejecutaron: el código buscaba columnas inexistentes y comparaba contra el entero `1` cuando los datos usan variantes de `SI`, `SÍ` y `NO`. Como resultado, la figura y el análisis omitieron precisamente los subconjuntos discutidos en el texto. La auditoría obtuvo F1-macro de 0.618 con sarcasmo frente a 0.816 sin él, y 0.619 con ironía frente a 0.804 sin ella.

## Objetivo

Calcular de forma robusta los cortes por plataforma, jerga, sarcasmo, ironía y contexto político para cada modelo y regenerar las visualizaciones a partir de resultados exportados.

## Qué se debe hacer

1. Corregir los nombres de columna a `Presencia_Sarcasmo`, `Presencia_Ironia` y `Contexto_Politico`.
2. Normalizar de forma centralizada los indicadores `SI`, `SÍ`, `NO` y sus equivalentes antes de construir máscaras booleanas.
3. Exportar para cada corte el tamaño de muestra, F1-macro, accuracy y, cuando el tamaño lo permita, intervalos de confianza.
4. Regenerar la Figura 4 de XLM-R incluyendo sarcasmo, ironía y contexto político, además de plataforma y jerga.
5. Añadir pruebas para variantes de acentuación, valores no previstos y cortes sin observaciones.

## Criterios de aceptación

- [x] Los tres nombres de columna se resuelven correctamente en el conjunto de prueba.
- [x] Los valores `SI` y `SÍ` se agrupan como positivos; `NO` como negativo.
- [x] La tabla de cortes contiene resultados y `n` para sarcasmo, ironía y contexto político cuando existan observaciones.
- [x] La figura y el texto del informe se regeneran desde la tabla exportada, sin barras o cifras manuales.
- [x] Las pruebas cubren los nombres de columna y las codificaciones de los indicadores.

