# ISSUE-01 — Consolidar la taxonomía y las correcciones del usuario

**Estado:** resuelto — política v2.0 y calibración preparadas; la revisión/exportación del corpus sigue el flujo de ISSUE-05 · **Prioridad:** P0  
**Dependencias:** Ninguna.

**Entregables:** [guía de anotación](../docs/annotation_guidelines_v2.md), [taxonomía versionada](../config/taxonomy_v2.yaml), [registro de decisiones](../docs/taxonomy_decisions_v2.md) y [40 casos propuestos de calibración](../docs/calibration_cases_v2.md).

## Problema y evidencia

Las correcciones de los documentos aclaran que viejo/vieja puede ser neutral y que lárgate/vete/fuera no es ofensivo por sí solo. En odio se insiste en que nacionalidades, religiones, discapacidad y clase social requieren contexto; se separaron feminazi/zorra/perra como ataques sexistas. Sigue abierto el tratamiento de política, clasismo y odio con violencia.

## Trabajo a realizar

1. Crear docs/annotation_guidelines_v2.md y config/taxonomy_v2.yaml (rutas propuestas). Conservar cuatro labels: 0 No Tóxico, 1 Lenguaje Ofensivo, 2 Discurso de Odio, 3 Amenazas/Violencia. Trazar cada regla hasta los comentarios de docs/lexico_taxonomia/ y registrar las decisiones.

2. Incorporar ejemplos completos: vieja como referencia neutral frente a vieja idiota; fuera como petición de salir frente a exclusión por identidad. No asumir que mención de grupo más cualquier grosería implica odio; identificar si el ataque se basa en esa identidad.

3. Aplicar la indicación del usuario para feminazi/zorra/perra en ataques sexistas dirigidos a personas, conservando distinción con usos literales de animales, citas y reapropiación. No transformar esta indicación en coincidencia automática de palabras.

4. Resolver el alcance de discriminación política y clasismo y la precedencia ante odio con amenaza. Conservar objetivo identitario/violencia como atributos adicionales. El odio puede expresarse sin groserías.

5. Preparar 40 casos de calibración propuestos, diez por clase, incluyendo jerga amistosa/hostil, negación, cita, noticia, amenaza y crítica política. Usarlos para formación, sin presentarlos como prueba independiente.

## Entregables

Guía de anotación, taxonomía versionada y registro de decisiones con ejemplos contextualizados.

## Criterios de aceptación

- [x] Todas las correcciones del usuario están trazadas; las decisiones pendientes se resuelven antes de exportar etiquetas finales.
- [x] Existe opción de contexto insuficiente y no se convierte automáticamente en clase 0.
- [x] Dos anotadores pueden aplicar el mismo árbol de decisión; las listas originales comentadas se conservan.

