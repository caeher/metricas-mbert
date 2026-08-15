# T-10 — Retirar artefactos pesados del control de versiones

**Prioridad:** media  
**Responsable:** Julio para XLM-R; revisión del equipo  
**Hallazgo de auditoría:** 13

## Por qué

La auditoría encontró un entorno virtual y pesos del modelo dentro del proyecto versionado de XLM-R. Esto infla el repositorio, dificulta revisiones, hace frágil la reproducibilidad y confunde el código fuente con los artefactos de ejecución. A la vez, los pesos deben entregarse para auditarse; eliminar de Git no significa dejar de entregar.

## Objetivo

Separar código, dependencias reproducibles y checkpoints, conservando una referencia verificable a los artefactos evaluados.

## Qué se debe hacer

1. Verificar con `git ls-files` los entornos virtuales, cachés y pesos actualmente rastreados en el proyecto de XLM-R.
2. Eliminar esos archivos del índice de Git de forma no destructiva y completar `.gitignore` para entornos, cachés y archivos de pesos.
3. Mantener requisitos bloqueados o documentados (`requirements`/lockfile) en vez de versionar el entorno.
4. Publicar pesos y resultados grandes por medio del paquete con manifiesto definido en T-08.
5. Documentar el flujo de restauración: crear entorno, instalar dependencias, descargar artefacto y ejecutar evaluación.

## Criterios de aceptación

- [ ] `git ls-files` no devuelve `.venv`, directorios equivalentes ni archivos de pesos/checkpoints.
- [ ] `.gitignore` cubre los artefactos excluidos sin ocultar código, configuración ni manifiestos necesarios.
- [ ] El repositorio conserva un archivo de dependencias reproducible y un enlace/identificador al paquete de T-08.
- [ ] Una clonación limpia puede preparar el entorno y ejecutar el flujo documentado.

