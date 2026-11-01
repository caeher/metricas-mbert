# Piloto y revisión humana — ISSUE-05

## Estado de ejecución

El piloto tiene muestra reproducible (200), cuentas **cde-rater-a** / **cde-rater-b**, exportación dual versionada bajo `data/argilla_exports/`, acuerdo en `reports/annotation/agreement.json` y release parcial en `adjudicated_records.csv` (ver [`STATUS.md`](../reports/annotation/STATUS.md)). Las cohortes completas se importan con `scripts/import_all_annotation_campaigns.py`. Kappa piloto puede quedar por debajo del objetivo 0.75: recalibrar guía antes de escalar.

La muestra contiene un bloque balanceado de 120 textos históricos reales de `train` (15 por cada combinación de las cuatro etiquetas antiguas y Facebook/X), 18 textos exclusivos por cada una de las dos fuentes nuevas, las cuatro colisiones de rótulos compartidos encontradas en la cola canónica y 40 casos adicionales de frontera. Algunos casos de frontera vuelven a subir el total de históricos o de una fuente sobre su cuota base. Los estratos históricos solo sirven para muestreo y no se envían a Argilla. La muestra no estima prevalencia ni sustituye revisar los corpus completos.

La cola actual contiene **cuatro** registros con `human_adjudication_required`, aunque el texto de ISSUE-05 menciona tres conflictos. El manifiesto incluye los cuatro.

El importador en seco selecciona ahora **2,324** candidatos canónicos nuevos, una unidad menos que los 2,325 textos únicos del recuento exploratorio descrito en la issue. Usa la vista canónica como población operativa y reconcilia esa diferencia antes de cerrar el release de datos.

## Preparar y cargar el piloto

Ejecuta desde la raíz `ml-metricas`:

```powershell
python scripts/prepare_annotation_pilot.py
python scripts/import_argilla.py --campaign pilot --dry-run
python scripts/import_argilla.py --campaign pilot
```

El manifiesto `reports/annotation/pilot_sample_v1.csv` es para el importador y contiene IDs junto con estratos de muestreo. No lo distribuyas a los anotadores. Argilla recibe solamente los IDs elegidos, texto original/normalizado y procedencia técnica; la campaña `pilot` no acepta sugerencias y no muestra rótulos previos como respuestas. Crea dos cuentas humanas distintas en el workspace antes de abrir la cola; ambos anotan cada registro independientemente.

## Revisar los corpus completos

Después del piloto y la recalibración necesaria, importa cada cohorte por separado. Mantén val y test en el workspace restringido de auditoría indicado en `docs/argilla_workflow.md`:

```powershell
python scripts/import_argilla.py --campaign legacy_train_real --dry-run
python scripts/import_argilla.py --campaign new_candidates --dry-run
python scripts/import_argilla.py --campaign legacy_train_synthetic --dry-run
python scripts/import_argilla.py --campaign historical_val_audit --dry-run
python scripts/import_argilla.py --campaign historical_test_audit --dry-run
```

Quita `--dry-run` para crear las campañas cuando Argilla y los dos anotadores estén configurados. No combines `corpus_completo.csv` con sus splits. Los sintéticos se revisan en su propia campaña. Todo candidato nuevo permanece en cuarentena hasta tener anotación, decisión final y revisión humana de procedencia.

## Exportar, medir y adjudicar

Exporta cada campaña desde Argilla con `scripts/export_argilla.py --campaign <campaña>`. Conserva cada carpeta de exportación versionada. Después calcula acuerdo y actualiza el ledger:

```powershell
python scripts/analyze_annotation_exports.py --exports "data/argilla_exports/ruta-export-pilot"
```

Pasa varias carpetas después de `--exports` para recalcular varias campañas juntas. En ejecuciones posteriores el script conserva decisiones manuales de otras campañas si la huella del registro canónico sigue igual.

El análisis informa acuerdo porcentual, Cohen kappa nominal si participan dos anotadores, o alfa nominal de Krippendorff si participan más de dos, junto con n, matriz, clase, fuente y cortes provisionales. El umbral 0.75 es una meta operativa inicial y no garantiza F1.

En `reports/annotation/adjudication_ledger.csv`, conserva sin cambios las respuestas originales. Para cada discrepancia, completa `final_annotation_status`, `final_label` (solo 0–3 cuando se resuelva), `final_quality`, `decision_type=adjudicated`, `decision_reason` y `adjudicator`. Codifica el análisis cualitativo en `qualitative_theme` y `qualitative_note`; el archivo `qualitative_review.csv` reúne razones y desacuerdos para revisión. Si falta contexto, registra `final_annotation_status=contexto_insuficiente`, deja `final_label` vacío y explica qué contexto falta. Para candidatos nuevos, registra `provenance_review_status=verified` solo tras revisión humana de la fuente; si no se confirma, conserva la cuarentena.

Vuelve a ejecutar el análisis con las exportaciones y el ledger editado. `adjudicated_records.csv` incluirá solo registros con dos o más respuestas humanas enviadas, estado y clase final, calidad `usable`, justificación por respuesta, guía 2.0.0 y decisión trazable; además exige procedencia verificada para fuentes candidatas nuevas. Los casos incompletos o insuficientes permanecen fuera del release.

Los reportes `agreement.*`, las matrices, los cortes por fuente/clase y los cambios provisional→final se crean a partir de las exportaciones humanas. No se crean métricas de acuerdo vacías ni se simulan decisiones antes de recibir esas exportaciones.
