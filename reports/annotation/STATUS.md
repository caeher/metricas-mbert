# Estado de ISSUE-05

**Piloto anotado por una persona; release de entrenamiento bloqueado por control de calidad.** El stack Argilla local está arriba en `http://localhost:6900` y el dataset `cde_annotation_pilot_v1` contiene 200 registros. El 2026-10-28 se exportaron 1,680 respuestas de una sola persona. El analizador encontró cero registros con dos anotadores humanos enviados y cero filas elegibles para `adjudicated_records.csv`; por tanto no se calcularon acuerdo/kappa ni existe un release listo para reentrenar. Hace falta una segunda anotación humana independiente y la adjudicación/revisión de calidad que corresponda.

- `pilot_sample_v1.csv` y `.json`: muestra determinista de 200 textos reales para el piloto ciego.
- [Protocolo del piloto](../../docs/annotation_pilot_issue05.md): comandos y flujo de ejecución, anotación y adjudicación.
- [Analizador](../../scripts/analyze_annotation_exports.py): al recibir exportaciones, genera `agreement.json`, cortes, matriz, resultados por clase, cola cualitativa, ledger, cambios provisional→final y `adjudicated_records.csv` solo con registros que pasan los controles de calidad.
- El último análisis local está en `reports/annotation/retrain_20261028/`; se mantiene fuera de Git porque contiene respuestas y texto anotado.

El ledger conserva las respuestas recibidas, pero ninguna respuesta individual se trata como consenso. No deben interpretarse los archivos de muestreo ni la importación de registros como anotaciones.
