# Protocolo de evaluación v2.0

**Estado:** definido; la prueba nueva sellada sigue pendiente de recolección, deduplicación y adjudicación.  
**Política:** [`config/dataset_split_policy.yaml`](../config/dataset_split_policy.yaml)  
**Taxonomía:** [`config/taxonomy_v2.yaml`](../config/taxonomy_v2.yaml)  
**Snapshot histórico:** [`reports/splits/historical_manifest.json`](../reports/splits/historical_manifest.json) y [`historical_split_manifest.csv`](../reports/splits/historical_split_manifest.csv)

## Propósito y límite actual

El protocolo separa selección de modelos, calibración y evaluación final. El test legado se conserva para reproducir informes y comparar con el pasado; ya se consultó en análisis de errores, cortes por plataforma, auditorías y decisiones, así que no demuestra generalización independiente. Sus etiquetas tampoco se han reanotado bajo la taxonomía v2.

Al 2026-10-16, `train/val/test` tienen 3,293/459/460 filas; el train incluye 1,149 sintéticas. Las 2,328 filas candidatas nuevas proceden solo de X/Twitter y tienen `Validado=NO` y `Listo_Entrenamiento=NO`. No constituyen todavía datos de entrenamiento ni una prueba nueva. No hay cohorte sellada reservada.

El snapshot reproducible actual confirma cero textos exactos normalizados y cero autores compartidos entre los tres splits históricos. Los CSV históricos no contienen identificadores de hilo o evento; por ello, la comprobación no acredita separación por evento ni por casi duplicados. La cola de similitud y la adjudicación humana siguen siendo requisitos para cualquier release nuevo.

## Roles y usos permitidos

| Rol | Datos | Usos permitidos | Usos prohibidos |
|---|---|---|---|
| Train | Reales adjudicados y sintéticos aprobados | Ajustar pesos del modelo; aumentar sintéticos; DAPT | Incorporar filas sin etiqueta final o grupos de holdout |
| Desarrollo | Reales adjudicados separados por grupo | Early stopping, checkpoint, hiperparámetros y selección de receta | Ajuste de pesos; DAPT; generación sintética desde sus textos/grupos |
| Calibración | Reales adjudicados separados por grupo y reservados antes de buscar recetas | Calibración probabilística o ajuste de umbral predeclarado, después de congelar modelo e hiperparámetros | Entrenamiento, DAPT, sintéticos, selección de checkpoint/hiperparámetros |
| Test histórico | `data/raw/test.csv`, con copia de evaluación en `data/processed/test.csv` | Comparación histórica; comparación pareada de baseline y candidato con las mismas etiquetas | Elegir recetas, umbrales, hiperparámetros, checkpoints, DAPT o sintéticos |
| Test nuevo sellado | Publicaciones reales nuevas de Facebook y X, anotadas y agrupadas | Una evaluación final tras congelar la receta | Toda selección o ajuste; acceso a textos/grupos por parte del equipo de modelado antes del cierre |
| Diagnóstico | Baterías seleccionadas por fenómeno, palabra o clase | Análisis separado de errores y robustez | Métrica primaria o estimación de prevalencia/uso real |

Los casos de [`calibration_cases_v2.md`](calibration_cases_v2.md) son material de formación de anotadores; no son un split de calibración estadística ni una evaluación independiente. El `val.csv` histórico puede seguir apareciendo en reproducción de recetas antiguas, pero no se convierte retroactivamente en calibración independiente. Hace falta separar grupos reales adjudicados para ese rol.

Para DAPT se usarán solo textos reales elegibles de train y corpus de dominio admisible; los sintéticos quedan excluidos del preentrenamiento. El test histórico y el test nuevo tampoco se usan en MLM.

## Construcción de grupos y asignación

1. Congelar taxonomía y guía de anotación para la versión de etiquetas del release.
2. Resolver conflictos de etiqueta, URL/publicación, autor, hilo, evento y candidatos de casi duplicado. Los vínculos confirmados forman componentes conexos: si dos registros comparten una clave de agrupación, todo el componente queda en un único rol. Los candidatos no adjudicados se ponen en cuarentena o permanecen juntos de manera conservadora.
3. Crear el ID de componente de forma estable a partir de las claves normalizadas y versionadas. No usar el orden de las filas ni aleatoriedad sin semilla.
4. Retirar las cohortes de evaluación antes de repartir los grupos restantes. Asignar los grupos reales elegibles a train/desarrollo/calibración con `group-stratified-greedy-v1`, semilla 42, y objetivos 80/10/10 dentro del pool de desarrollo. La clase final y plataforma canónica guían la estratificación; grupos completos tienen prioridad sobre el balance. Registrar celdas sin soporte y no completarlas con sintéticos.
5. Incluir sintéticos solo en train, junto con plantilla/generador y grupo padre. Ninguna fuente de test, grupo padre o texto de evaluación puede ser prompt, ejemplo padre o destino de generación.
6. Escribir el manifiesto antes del entrenamiento con hashes de fuente, texto y etiqueta, IDs de grupo anonimizados, versión de taxonomía, semilla y código. Reproducirlo desde las mismas fuentes debe producir el mismo manifiesto.

Si falta autor/hilo/evento, se puede asignar un grupo singleton para ejecutar el algoritmo, pero el manifiesto debe marcar la cobertura de metadatos incompleta. La ausencia de metadatos no prueba independencia. Los duplicados exactos y por URL se agrupan automáticamente; candidatos de similitud solo pasan a una asignación definitiva tras revisión humana o agrupación conservadora.

## Prueba nueva, cobertura y tamaño

El plan de campo propone **800 ejemplos reales** como punto de partida para cobertura: 100 en cada celda de clase (0–3) por plataforma (Facebook/X). Es un presupuesto de planificación, no una garantía estadística ni una muestra representativa del uso. Una asignación equilibrada sirve para comparar cortes; no estima prevalencias.

Se mantendrán dos productos de evaluación separados:

- **Muestra representativa:** selección probabilística del universo de uso definido, sin filtrar por palabras ni etiquetas. Es la fuente de estimaciones primarias sensibles a prevalencia. Registrar probabilidades de inclusión y ponderar si la asignación no es proporcional.
- **Batería equilibrada de diagnóstico:** cuotas por clase y plataforma; cualquier búsqueda por palabras, fenómeno o clase se declara. Reportar por celda y no presentarla como distribución de uso real.

La meta de cuatro clases y ambas plataformas se verificará en la batería equilibrada. Para la muestra representativa, se informará la cobertura observada; si alguna celda necesaria para una conclusión no tiene suficiente soporte, se ampliará el muestreo o se marcará el corte como no estimable. No se sustituye una clase ausente por sintéticos.

Antes de recolectar la muestra representativa, el protocolo de estudio fijará la población, periodo, canales de cada plataforma, cortes que se reportarán y semiancho deseado de los intervalos. El tamaño final se calculará para el corte con menor soporte esperado y para las métricas primarias, considerando agrupación por autor/evento, ponderaciones, multiplicidad y población finita cuando aplique. El n=800 inicial no determina por sí mismo precisión de F1, precisión, recall ni macro-F1.

## Sellado y evaluación final

Una persona custodio distinta del equipo que selecciona recetas retiene textos, etiquetas, IDs de grupo y manifiesto del test nuevo. Antes de abrirlo, se archivan y hashean: commit de código, fuentes y manifiestos de train, sintéticos/prompts, configuración, tokenizer, normalizador, receta, checkpoint, umbrales e contrato de inferencia. El custodio registra esos hashes y habilita una corrida final. Las predicciones y métricas no disparan una nueva selección de receta sobre ese test.

En cada cierre de release, ejecutar una auditoría de solapamiento entre evaluación y todos los corpus de ajuste, incluyendo DAPT no etiquetado y ejemplos padres de sintéticos. Comparar texto exacto normalizado, URL, autor, hilo, evento y componente de casi duplicados; guardar recuentos, excepciones adjudicadas y hashes en `reports/splits/release_overlap_audit.json`. Un solapamiento de evaluación bloquea el release hasta resolverlo o excluir conservadoramente el componente.

## Versiones de etiquetas y presentación de resultados

El snapshot actual conserva etiquetas históricas y resultados ya publicados. Si se corrige la guía o se cambia taxonomía, no se sobrescriben las etiquetas ni métricas históricas:

1. Publicar nueva versión de etiquetas, guía y manifiesto con hashes y racionales de adjudicación.
2. Reevaluar el baseline congelado y el candidato sobre exactamente las mismas etiquetas nuevas y el mismo conjunto; publicar predicciones e incertidumbre pareadas.
3. Conservar en una tabla distinta los resultados históricos con sus etiquetas originales. La diferencia entre ambos reportes es cambio de anotación/evaluación y no una mejora atribuible al entrenamiento.

Todo reporte identifica versión del split, versión de taxonomía, origen de la muestra, si la selección fue por clase/palabra, tamaño por clase y plataforma, método de intervalos y exclusiones. Presentar por separado: (a) comparabilidad histórica, (b) muestra representativa, (c) batería equilibrada de diagnóstico. No promediar ni combinar esas métricas.

## Artefactos disponibles

- [`historical_split_manifest.csv`](../reports/splits/historical_split_manifest.csv): inventario por fila con ID, clase heredada, hashes de texto/archivo y grupo de autor seudonimizado cuando existe.
- [`historical_manifest.json`](../reports/splits/historical_manifest.json): resumen, conteos y verificación de copias históricas.
- [`new_test_sealed_manifest.json`](../reports/splits/new_test_sealed_manifest.json): marcador sin registros; documenta explícitamente que el test aún no está reservado.
- [`scripts/build_split_manifests.py`](../scripts/build_split_manifests.py): regeneración del snapshot histórico.

Comando desde la raíz del proyecto:

```powershell
python scripts/build_split_manifests.py
```

Estos artefactos versionan el estado histórico; no reemplazan la deduplicación/adjudicación pendiente ni crean una prueba independiente.
