# Guía de anotación v2.0

**Estado:** política operativa para calibración y revisión humana; no es un conjunto de etiquetas gold.  
**Taxonomía:** [`config/taxonomy_v2.yaml`](../config/taxonomy_v2.yaml)  
**Registro de decisiones y trazabilidad:** [`taxonomy_decisions_v2.md`](taxonomy_decisions_v2.md)  
**Casos de calibración:** [`calibration_cases_v2.md`](calibration_cases_v2.md)

## 1. Propósito y unidad de anotación

Asigne una sola etiqueta primaria al acto comunicativo del autor del texto, usando el contexto disponible del mensaje. Las cuatro etiquetas conservan los identificadores del proyecto:

| ID | Etiqueta | Criterio resumido |
|---:|---|---|
| 0 | No Tóxico | No hay ataque, discriminación ni promoción de daño físico; incluye habla cotidiana, crítica de ideas y reportes no aprobatorios. |
| 1 | Lenguaje Ofensivo | Hay insulto, vulgaridad dirigida o descalificación, sin base identitaria discriminatoria ni promoción de daño físico. |
| 2 | Discurso de Odio | Hay ataque, exclusión o trato discriminatorio basado en identidad o afiliación expresamente cubierta por esta política. No requiere groserías. |
| 3 | Amenazas/Violencia | Hay amenaza, exhortación, deseo o aprobación inequívoca de daño físico. |

La salida de anotación también tiene un **estado**. Cuando falte contexto necesario, use `contexto_insuficiente`, deje la etiqueta primaria vacía/null y anote qué contexto falta. No convierta la falta de evidencia en clase 0. Esa opción no añade una quinta clase al modelo: es un estado de revisión y esos registros no se exportan como etiquetas finales hasta adjudicarlos.

Anote el texto completo del post y, si está disponible y hace falta para interpretar deixis, ironía, cita o respuesta, el mensaje al que responde. No infiera intención a partir del perfil, la ubicación atribuida, el nombre del usuario ni una palabra aislada. Las fuentes nuevas descritas en `docs/lexico_taxonomia/README.md` siguen siendo candidatas sin validar.

## 2. Árbol de decisión

Siga los pasos en orden. Registre los atributos aunque una regla de precedencia determine la etiqueta primaria.

1. **¿Se entiende el acto con el contexto disponible?** Si una ambigüedad relevante no se resuelve (destinatario, sentido literal/figurado, postura ante una cita, base del ataque o intención de daño), marque `contexto_insuficiente`; no asigne 0 por defecto. Si el contexto sí alcanza, continúe.
2. **¿El autor promueve daño físico?** Asigne 3 si amenaza con hacerlo, exhorta a hacerlo, expresa un deseo inequívoco de que ocurra o lo aprueba explícitamente. Registre tipo, destinatario y cualquier base identitaria. Una noticia, una cita condenada, una negación o un uso metafórico no cumplen por sí solos este paso.
3. **¿El ataque/discriminación se basa en identidad?** Asigne 2 si degrada, deshumaniza, excluye o propone trato desigual por una identidad cubierta (incluidas las extensiones operativas de clase socioeconómica y afiliación política descritas abajo). Debe haber evidencia de que el motivo es esa identidad; la mera mención de un grupo y una grosería cercana no basta.
4. **¿Hay insulto o descalificación sin esa base?** Asigne 1 si el autor dirige una ofensa a una persona/grupo, o usa una vulgaridad como ataque. Insultos sobre una conducta, una idea, un partido o una persona no pasan a 2 sin evidencia de discriminación por identidad.
5. **Si nada de lo anterior aplica**, asigne 0: comunicación neutral, afectiva, descriptiva, crítica política de ideas/actuaciones, noticia no aprobatoria, negación o cita condenatoria.

### Precedencia cuando hay solapamiento

La etiqueta primaria es mutuamente excluyente y sigue **Amenazas/Violencia > Discurso de Odio > Lenguaje Ofensivo > No Tóxico (3 > 2 > 1 > 0)**. Si una amenaza se dirige a personas por su identidad, la etiqueta es 3 y se conserva `identity_basis: true` y `identity_targets`. Así se registra la dimensión de odio sin ocultar la violencia. Si hay odio con insultos, pero sin violencia física promovida, la etiqueta es 2. Los atributos nunca reemplazan ni crean una quinta clase.

## 3. Criterios por frontera

### 3.1 Neutralidad y ofensa cotidiana

- `viejo/vieja` puede ser una referencia de edad, parentesco o persona; por sí sola no es ofensiva. «La vieja de la tienda ya abrió» es 0 si el tono es descriptivo. «Esa vieja idiota no deja hablar a nadie» es 1: `idiota` es una descalificación dirigida, y la edad no es el motivo del ataque.
- `lárgate`, `vete` y `fuera` no son ofensivos por sí solos. «Fuera de mi casa, por favor; ya cerramos» es una petición de salir y es 0. «Fuera los hondureños del barrio por ser hondureños» es exclusión por nacionalidad y es 2. «Voy a sacar a golpes a los hondureños» es 3, con objetivo identitario adicional.
- La jerga amistosa (`maje`, `cerote`, `chero`, `cipote`, entre otras) puede ser afectiva, descriptiva o insultante. Determine uso, dirección y tono en la frase; si una forma ambigua carece de contexto, use contexto insuficiente.
- Una vulgaridad como interjección («¡qué mierda de tráfico!») no es automáticamente un ataque a una persona. En ausencia de otro criterio, es 0. Si descalifica a alguien, 1; si lo degrada por identidad, 2.

### 3.2 Identidad, grupo y groserías

Para la clase 2, responda afirmativamente ambas preguntas: **¿a quién afecta el ataque?** y **¿se le ataca o discrimina por una identidad?** La presencia de nacionalidad, religión, raza/etnia, género, orientación sexual o discapacidad no basta. Tampoco basta que el texto contenga una grosería cerca del nombre de un grupo.

- «El hondureño de la oficina es un cerote porque llegó tarde» critica a una persona por su conducta: 1 si el insulto es dirigido; `identity_basis: false`.
- «Los hondureños son inferiores y no deberían alquilar aquí» generaliza y excluye por nacionalidad: 2, aunque no incluya una grosería.
- Las menciones de nacionalidad, religión, raza/etnia, discapacidad o clase pueden ser descriptivas, críticas de una conducta individual, citas, autodenominaciones o ataques. Anote el contexto y no convierta el sustantivo grupal en una señal de odio automática.
- Un estereotipo o una exclusión explícita basada en identidad puede ser 2 aunque se exprese con lenguaje formal o cortés. No se exige insulto ni amenaza.

### 3.3 Sexismo y términos señalados por el usuario

Cuando `feminazi`, `zorra` o `perra` se usan como ataque sexista dirigido a una persona, asigne 2. La decisión depende del uso y del motivo, nunca de una coincidencia automática:

- Si el término se usa para degradar a una mujer/persona por género, sexualidad o por desafiar estereotipos de género, es 2.
- Si es insulto dirigido pero el contexto no muestra base sexista, puede ser 1. Si no se puede determinar si es insulto, uso literal, cita o reapropiación, use contexto insuficiente.
- El uso literal para un animal («mi perra tuvo cachorros»), un título o mención lingüística, una cita neutral/condenatoria y la reapropiación afirmativa no son por sí mismos odio. Si el autor adopta o aprueba el ataque citado, clasifique el acto respaldado.
- `puta/puto`, `maricón/marica` y otras formas también requieren determinar destinatario y base; no active 2 solo por el lema.

### 3.4 Política y clasismo (alcance resuelto para este proyecto)

- **Crítica política:** criticar políticas, decisiones, ideologías, partidos, figuras públicas o desempeño no es odio. «La alcaldía gestionó mal el presupuesto» es 0. Un insulto contra una persona por su conducta o contra partidarios como descalificación sin discriminación es 1.
- **Discriminación por afiliación política:** en esta taxonomía de proyecto, afiliación política cuenta como base identitaria operativa solo si el texto propone exclusión, persecución, privación de derechos/servicios o trato desigual contra personas por pertenecer o apoyar esa afiliación. Eso es 2. La opinión o pertenencia partidaria, por sí sola, no activa 2. La violencia física promovida contra ellos tiene precedencia 3 y conserva el atributo de objetivo político.
- **Clasismo:** la pobreza o clase social descriptiva no es odio. Insulto clasista a una persona («sos un pobre cerote») es 1 si no propone trato desigual por clase. Asigne 2 cuando se degrade, excluya o niegue derechos/servicios a personas por ser pobres o pertenecer a una clase social. La base de clase debe ser explícita o inequívoca.
- Estas dos extensiones son reglas de anotación del proyecto; no afirman que exista una categoría legal universal de odio.

### 3.5 Amenaza, violencia, negación y reporte

- Asigne 3 a amenaza directa («te voy a golpear»), exhortación a ejercer daño físico («vayan a quemarles la casa»), deseo inequívoco de daño («ojalá lo maten») o aprobación explícita de violencia («estuvo bien que los atacaran»).
- Una amenaza requiere sentido físico y una intención/promoción atribuible al autor. La literalidad del verbo no basta: «el debate mató la propuesta» es metafórico; «el periodista informó que hubo un ataque» es reporte.
- Una negación clara no es una amenaza: «No voy a golpear a nadie» es 0, salvo que otra parte del texto contenga un ataque distinto.
- Las citas y noticias se anotan por la postura del autor del post. Reproducir una cita para condenarla, analizarla o informar neutralmente es 0 (o la clase que corresponda a otro contenido propio); aprobar o amplificar la amenaza cuenta como 3. Si no se distingue postura, contexto insuficiente.
- El lenguaje amenazante puede ser odio a la vez. Use 3 como primaria y guarde `identity_basis`, grupo/objetivo y motivo. El deseo/exhortación de violencia contra un grupo puede ser 3 aunque no se dirija a una persona identificable.

## 4. Atributos y registro de anotación

Por cada registro, guarde al menos:

| Campo | Valores/uso |
|---|---|
| `label` | `0`, `1`, `2`, `3`; null solo mientras `annotation_status=contexto_insuficiente`. |
| `annotation_status` | `anotado` o `contexto_insuficiente`. No equivale a una clase. |
| `insufficient_context_reason` | Contexto faltante concreto o null. |
| `directed_at_person` | `true`, `false` o `unknown`. |
| `identity_basis` | `true`, `false` o `unknown`; true solo con evidencia contextual. |
| `identity_targets` | Lista de grupo(s) explícitos: origen/nacionalidad, raza/etnia, religión, género/sexualidad, discapacidad, clase socioeconómica o afiliación política; lista vacía si no aplica. |
| `violence_type` | `none`, `direct_threat`, `incitement`, `wish`, `endorsement`, `report`, `quoted_condemned`, `metaphor` u otro tipo explicado. |
| `quoted_or_reported` | Indica si reproduce o reporta palabras/hechos de otra persona. |
| `negated` | Indica negación explícita de la amenaza/ataque relevante. |
| `reclaimed_or_affirming` | Indica reapropiación/autodenominación afirmativa, no ataque. |
| `rationale` | Una frase con evidencia contextual que justifique etiqueta y atributos. |

No infiera un atributo desconocido como falso. Las etiquetas de calibración en [`calibration_cases_v2.md`](calibration_cases_v2.md) son ejemplos propuestos para formación, no filas recolectadas, etiquetas gold ni una prueba independiente.

## 5. Flujo entre anotadores y exportación

1. Dos anotadores aplican la misma versión de la guía de forma independiente y registran etiqueta, atributos y justificación.
2. Si cualquiera marca contexto insuficiente o hay desacuerdo, el caso permanece en revisión; un adjudicador humano resuelve usando contexto disponible y registra la decisión y su motivo. No sobrescriba las respuestas originales.
3. Si el contexto necesario no puede recuperarse, conserve `contexto_insuficiente`, sin clase primaria final, y excluya el caso de exportación de etiquetas finales.
4. Registre `guideline_version: 2.0.0`, procedencia y trazabilidad de corrección. No use el lexicón como sustituto de anotación contextual.
5. Los casos de calibración pueden usarse para entrenamiento y discusión. Manténgalos fuera de la evaluación independiente y no los presente como datos reales.

