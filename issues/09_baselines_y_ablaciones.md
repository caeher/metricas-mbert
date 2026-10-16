# ISSUE-09 — Reproducir baseline y aislar efectos de datos y etiquetas

**Estado:** pendiente · **Prioridad:** P1  
**Dependencias:** [ISSUE-03](03_splits_y_evaluacion_independiente.md), [ISSUE-05](05_anotacion_y_adjudicacion.md), [ISSUE-08](08_builder_dataset_y_pesos.md)

## Problema y evidencia

mbert_official_run.json registra checkpoint-1644, F1 macro=0.7610126 y n_test=460. Esto es evidencia histórica; no demuestra que pesos/runtime estén disponibles localmente. La meta configurada es 0.80. error_analysis_b2.json registra 107 errores y brechas de plataforma.

## Trabajo a realizar

1. Verificar disponibilidad/hashes del paquete oficial con scripts/package_mbert_checkpoint.py y verify_mbert_checkpoint.py. Si faltan pesos, registrar la limitación y reproducir desde receta B2 identificada, sin prometer igualdad exacta.

2. Congelar un baseline bert-base-multilingual-cased con revisión del modelo, dataset, normalizador, longitud, semilla y presupuesto. Verificar dependencias instalables en el runtime real.

3. Comparar datos con receta fija: reales antiguos adjudicados; antiguos+nuevos adjudicados; esa mezcla con sintéticos antiguos aceptados; mezcla con sintéticos nuevos aceptados. Mantener el mismo desarrollo y grupos elegibles.

4. Separar mejoras de etiquetas, cantidad, sintéticos y receta. Si cambian taxonomía o evaluación, reevaluar baseline compatible en lugar de usar directamente el 0.7610 histórico como control.

5. Ejecutar control técnico pequeño y luego ablaciones completas. Confirmar finalistas en semillas 42,7,13; no escoger la semilla por test.

6. Registrar F1 macro, clases/plataformas, recall amenazas, FPR neutral con jerga, tiempo/VRAM y tamaño efectivo. Definir presupuesto de corridas antes de lanzarlas.

## Entregables

reports/experiments/baselines_and_data_ablation.*, manifiestos por corrida y receta de control; reutilizar train_mbert.py y evaluadores existentes.

## Criterios de aceptación

- [ ] Cada comparación declara el único factor cambiado y usa evaluación compatible.
- [ ] Se reportan media/variación por semilla y diferencias entre métricas históricas y nuevas.
- [ ] Se elige base de datos/receta en desarrollo y costo; el test nuevo no interviene.

