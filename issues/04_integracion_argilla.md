# ISSUE-04 — Integrar Argilla con campañas, preguntas y trazabilidad

**Estado:** integración preparada; falta ejecutar piloto con dos anotadores · **Prioridad:** P0  
**Dependencias:** [ISSUE-01](01_taxonomia_y_correcciones.md), [ISSUE-02](02_auditoria_fuentes_y_duplicados.md), [ISSUE-03](03_splits_y_evaluacion_independiente.md)

## Problema y evidencia

No se encontró integración Argilla en el proyecto. Hace falta convertir revisión de etiquetas, sugerencias y respuestas humanas en un flujo reproducible para los datos antiguos, nuevos y sintéticos.

## Trabajo a realizar

1. Elegir y fijar versiones compatibles del servidor/SDK; documentar Docker local o Space, persistencia, backups y restauración. Mantener claves en variables de entorno y acceso acorde con los datos. Usar dependencias opcionales para no bloquear inferencia.

2. Crear campañas para reales antiguos de entrenamiento, candidatos nuevos y sintéticos antiguos/nuevos. Mantener val/test históricos en campaña restringida de auditoría, fuera de adquisición/entrenamiento. No importar dos veces los registros al combinar corpus y splits.

3. Configurar texto original/normalizado y preguntas de label 0-3, subclase, grupo objetivo, dialecto, cita/negación, violencia, calidad, contexto insuficiente y motivo. Publicar la guía versionada de ISSUE-01.

4. Configurar mínimo dos respuestas antes de iniciar anotación. Separar sugerencias del modelo de respuestas de personas; guardar checkpoint/agent/score. Ocultar rótulos previos y sugerencias en calibración/auditoría independiente para reducir anclaje.

5. Implementar scripts/import_argilla.py y export_argilla.py con IDs externos estables, simulación previa, idempotencia, actualización incremental y reanudación. Exportar todas las respuestas por usuario; una última respuesta no equivale automáticamente a adjudicación.

6. Versionar exportaciones portables locales y su restauración. Si se usa Hub, registrar revisión y visibilidad; no hace falta publicación pública. Verificar APIs contra la versión realmente instalada.

## Entregables

Configuración Argilla, importador/exportador, dependencias opcionales y docs/argilla_workflow.md; rutas nuevas propuestas.

Implementación: [compose local](../infra/argilla/docker-compose.yml), [extra opcional del SDK](../pyproject.toml), [.env de ejemplo](../.env.argilla.example), [importador](../scripts/import_argilla.py), [exportador/restaurador](../scripts/export_argilla.py) y [guía del flujo](../docs/argilla_workflow.md). Se fijaron SDK/servidor 2.8.0 y se verificaron firmas de la API del SDK instalado aislado. El piloto con respuestas de dos cuentas aún requiere que el usuario configure el servidor y que dos personas anoten.

## Criterios de aceptación

- [ ] Un piloto se importa dos veces sin duplicación, recibe dos respuestas independientes y se restaura/exporta conservando autores y procedencia.
- [x] La selección de campañas excluye val/test de train/candidatos; los secretos usan `.env.argilla`, ignorado por Git.
- [x] Las sugerencias usan el objeto separado de Argilla, la exportación conserva cada respuesta con autor/estado y no hay adjudicación automática.

