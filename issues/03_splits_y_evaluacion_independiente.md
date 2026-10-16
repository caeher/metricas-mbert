# ISSUE-03 — Versionar splits por grupos y reservar una prueba nueva

**Estado:** en curso — política, protocolo y snapshot histórico publicados; faltan adjudicación, división por grupos del nuevo release y recolección/reserva del test sellado · **Prioridad:** P0  
**Dependencias:** [ISSUE-01](01_taxonomia_y_correcciones.md), [ISSUE-02](02_auditoria_fuentes_y_duplicados.md)

## Problema y evidencia

El test histórico ya aparece en análisis de errores y decisiones anteriores. Es útil para comparabilidad histórica, pero evaluar repetidamente allí no acredita generalización a datos nuevos. Las fuentes nuevas están seleccionadas por clase y son únicamente X, por lo que no representan automáticamente el uso real.

## Trabajo a realizar

1. Guardar hashes, etiquetas y manifiestos de evaluación histórica. Definir protocolo y rol de desarrollo, calibración, test histórico y test nuevo sellado antes de aumentación o selección de recetas.

2. Tras deduplicación/adjudicación, dividir registros reales por grupos de autor/hilo/evento/casi duplicados, estratificando por clase/plataforma donde el soporte lo permita. Los sintéticos pertenecen exclusivamente a train.

3. Reservar prueba real nueva con las cuatro clases y Facebook/X. Proponer inicialmente 800 textos para planificar cobertura y calcular tamaño final por precisión de intervalos y cortes; ese número no es una garantía estadística ni se completa con sintéticos.

4. Distinguir una muestra representativa del uso de una batería equilibrada para diagnóstico; declarar selección por palabras o clases. No presentar una batería equilibrada como estimación de distribución real.

5. Si la guía modifica etiquetas del test, publicar una nueva versión y reevaluar baseline y candidato sobre las mismas etiquetas. Mantener aparte el resultado histórico para no atribuir a entrenamiento una mejora de anotación.

6. Prohibir textos/grupos de evaluación en generación sintética, DAPT, ajuste de umbrales, selección de hiperparámetros o checkpoints. Auditar solapamiento al cierre de cada release.

## Entregables

[`config/dataset_split_policy.yaml`](../config/dataset_split_policy.yaml),
[`docs/evaluation_protocol_v2.md`](../docs/evaluation_protocol_v2.md),
[`reports/splits/historical_manifest.json`](../reports/splits/historical_manifest.json),
[`reports/splits/historical_split_manifest.csv`](../reports/splits/historical_split_manifest.csv)
y [`reports/splits/new_test_sealed_manifest.json`](../reports/splits/new_test_sealed_manifest.json).
El histórico es reproducible, pero el marcador del test nuevo está vacío hasta
que se recolecten y adjudiquen datos reales de Facebook y X.

## Criterios de aceptación

- [ ] Splits reproducibles, evaluación real y cero grupos compartidos según las reglas declaradas.
- [ ] Test final accesible para evaluación solo después de congelar la receta; no se selecciona por su desempeño.
- [ ] Todo cambio de taxonomía se refleja en versiones y en la reevaluación del baseline.

