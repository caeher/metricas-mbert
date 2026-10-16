# Estado de ISSUE-05

**Campaña local lista; falta anotación humana.** El stack Argilla local está arriba en `http://localhost:6900` y se importaron 200 registros al dataset `cde_annotation_pilot_v1`. Las credenciales iniciales se guardan en `.env.argilla`, ignorado por Git. Aún no existen dos cuentas humanas de anotación ni exportaciones con respuestas; por eso no se han calculado acuerdo/kappa, cambios de clase ni registros adjudicados.

- `pilot_sample_v1.csv` y `.json`: muestra determinista de 200 textos reales para el piloto ciego.
- [Protocolo del piloto](../../docs/annotation_pilot_issue05.md): comandos y flujo de ejecución, anotación y adjudicación.
- [Analizador](../../scripts/analyze_annotation_exports.py): al recibir exportaciones, genera `agreement.json`, cortes, matriz, resultados por clase, cola cualitativa, ledger, cambios provisional→final y `adjudicated_records.csv` solo con registros que pasan los controles de calidad.

El ledger y los reportes con resultados se crearán después de exportar respuestas humanas desde Argilla. No deben interpretarse los archivos de muestreo ni la importación de registros como anotaciones.
