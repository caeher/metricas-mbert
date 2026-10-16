# Guía breve de anotación Argilla v2.0.0

La guía completa y el registro de decisiones están en `docs/annotation_guidelines_v2.md` y `docs/taxonomy_decisions_v2.md` del repositorio. Esta versión breve conserva las reglas operativas para la interfaz Argilla.

## Etiqueta primaria

- `0 · No Tóxico`: habla cotidiana, afecto, descripción, crítica de ideas o actuación, noticia no aprobatoria; sin ataque discriminatorio ni promoción de daño físico.
- `1 · Lenguaje Ofensivo`: insulto dirigido, vulgaridad como ataque o descalificación, sin base identitaria discriminatoria ni promoción de daño físico.
- `2 · Discurso de Odio`: ataque, exclusión o trato desigual por una identidad o afiliación cubierta por la política del proyecto. No requiere groserías.
- `3 · Amenazas/Violencia`: amenaza, exhortación, deseo inequívoco o aprobación de daño físico.

El estado `contexto_insuficiente` no es una quinta clase. Úsalo cuando falte contexto para saber destinatario, sentido, postura ante una cita, base del ataque o intención de daño; deja la etiqueta vacía y explica qué falta. Nunca conviertas incertidumbre en clase 0.

## Orden de decisión

1. Comprueba si el mensaje y el contexto disponible permiten entender el acto y la postura del autor.
2. Si el autor promueve daño físico, etiqueta 3.
3. Si degrada, excluye o discrimina por identidad, etiqueta 2.
4. Si dirige un insulto o descalificación sin base identitaria ni violencia, etiqueta 1.
5. En otro caso, etiqueta 0.

Precedencia: `3 > 2 > 1 > 0`. Si una amenaza se dirige a un grupo identitario, conserva los atributos de odio además de la etiqueta primaria 3.

## Fronteras que requieren contexto

- `viejo/vieja`, `lárgate`, `vete` y `fuera` no son tóxicos por sí solos. Una descalificación dirigida puede ser 1; exclusión por identidad, 2; expulsión mediante violencia, 3.
- `maje`, `cerote`, `chero`, `cipote` y otras formas locales pueden ser amistosas, descriptivas o insultantes. Usa frase, destinatario y tono; si no alcanzan, marca contexto insuficiente.
- Una grosería como interjección o crítica de una situación no es automáticamente un ataque a una persona.
- Mencionar nacionalidad, religión, raza/etnia, género, sexualidad, discapacidad o clase no basta para etiquetar 2. Busca evidencia de que el motivo del ataque o trato desigual sea esa identidad.
- La crítica de partidos, figuras públicas, políticas, ideas o desempeño no es odio. Afiliación política cuenta como base operativa de clase 2 cuando se propone persecución, exclusión, pérdida de derechos/servicios o trato desigual por esa afiliación. Un insulto sin esa discriminación puede ser 1.
- La pobreza o clase descriptiva no es odio. Un insulto clasista puede ser 1; niega derechos/servicios o excluye por clase de manera explícita o inequívoca para 2.
- `feminazi`, `zorra`, `perra`, `puta/puto` y términos similares dependen del uso. Asigna 2 solo con base sexista o de género contextual; insulto sin base demostrable puede ser 1. Uso literal, cita neutral/condenatoria o reapropiación afirmativa no es por sí solo odio.
- Amenaza directa, exhortación, deseo inequívoco o aprobación de violencia física es 3. Una noticia, cita condenada, negación o metáfora no lo es por sí sola. Si no se puede distinguir la postura, usa contexto insuficiente.

## Respuestas

Registra etiqueta, estado, subclase, grupos objetivo, dialecto, cita/reporte, negación, tipo de violencia, calidad, motivo de contexto insuficiente y una justificación contextual breve. Usa `unknown` cuando no haya evidencia para afirmar o negar un atributo. No infieras intención por perfil, nombre o ubicación.

Dos personas anotan independientemente. No veas rótulos previos ni sugerencias en calibración/piloto ciego. Las respuestas de otra persona no son una adjudicación: un tercer criterio humano registra etiqueta final, motivo y guía. Si el contexto no se recupera, deja la clase final vacía y conserva la cuarentena.
