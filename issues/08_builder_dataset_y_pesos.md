# ISSUE-08 — Construir dataset versionado y pesos dinámicos de entrenamiento

**Estado:** pendiente · **Prioridad:** P0  
**Dependencias:** [ISSUE-02](02_auditoria_fuentes_y_duplicados.md), [ISSUE-03](03_splits_y_evaluacion_independiente.md), [ISSUE-05](05_anotacion_y_adjudicacion.md)

## Problema y evidencia

src/discurso_odio/data/loaders.py carga splits fijos y get_b2_class_weights() lee pesos B2 estáticos. El filtrado sintético depende de orígenes conocidos. Los datos nuevos requieren un contrato de origen y pesos recalculados.

## Trabajo a realizar

1. Crear scripts/build_dataset_v3.py desde exportaciones adjudicadas y manifiestos, evitando concatenar CSV crudos. Preservar texto original, rótulos previos/finales, estados, hashes y versiones.

2. Actualizar loaders para seleccionar una release explícita. Separar tipo de origen de familia sintética y no depender de una lista cerrada de nombres sintetico_*. Validar class final, no etiqueta provisional.

3. Calcular pesos solo con train de cada corrida/fold. Comparar sin ponderar, por clase y por plataforma/clase. No aplicar ciegamente pesos B2 a la nueva distribución.

4. Controlar la multiplicación entre sobremuestreo, pesos de clase, refuerzo de plataforma y downweight sintético. Guardar suma de pesos/tamaño efectivo por clase, plataforma y origen.

5. Generar metadata con distribución real/sintética, exclusiones, grupos y parámetros. Producir releases atómicas y reproducibles bajo data/releases/<version>/, manteniendo B2 accesible para comparaciones.

6. Añadir controles de esquema, IDs únicos, labels 0-3, ausencia de pendientes y exclusión de grupos protegidos, incluyendo sintéticos.

## Entregables

Builder, releases inmutables con manifiesto/hashes y ampliación compatible de loaders; nuevas rutas propuestas.

## Criterios de aceptación

- [ ] La misma entrada, política y semilla produce los mismos splits/hashes.
- [ ] Solo registros adjudicados elegibles entran a entrenamiento; pesos nunca usan val/test.
- [ ] El pipeline puede cargar B2 y la nueva release sin sobrescritura ni confusión de pesos.

