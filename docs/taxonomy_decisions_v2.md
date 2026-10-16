# Registro de decisiones de taxonomía v2.0

**Versión de política:** 2.0.0  
**Estado:** decisiones operativas resueltas para calibración y anotación; no significa que los corpus históricos ya estén reetiquetados.  
**Convención:** se conservan las cuatro clases principales; `contexto_insuficiente` es estado de anotación con `label: null`, no una quinta clase.

| ID | Decisión operativa | Trazabilidad a comentarios existentes |
|---|---|---|
| D01 | Se mantienen 0 No Tóxico, 1 Lenguaje Ofensivo, 2 Discurso de Odio y 3 Amenazas/Violencia. | `docs/lexico_taxonomia/README.md`, “Criterios operativos propuestos”; nombres e IDs ya usados en `config/settings.yaml`. |
| D02 | La suficiencia de contexto se registra fuera de las cuatro etiquetas. Si no se puede decidir sentido, postura o base, `label` queda null y el caso no se exporta como etiqueta final. | `04_ambiguas.md`, “no les asigno una clase fija” y “términos ambiguos”; `issues/05_anotacion_y_adjudicacion.md`, paso 5 (poner en cuarentena contexto insuficiente). |
| D03 | `viejo/vieja` no implica ofensa; una descalificación combinada como “vieja idiota” es 1 si el motivo es conducta/persona y no identidad. | `01_lenguaje_ofensivo.md`, fila `viejo`, comentario del usuario: puede ser palabra no tóxica y se vuelve ofensiva con “vieja idiota”/“vieja imbécil”. |
| D04 | `lárgate/vete/fuera` no son ofensivos por sí solos. Petición de salir sin ataque = 0; exclusión por identidad = 2; amenaza de daño al expulsar = 3. | `01_lenguaje_ofensivo.md`, fila `lárgate, vete, fuera`, comentario del usuario: “no es ofensiva, por sí sola, es como pedir que alguien se salga”; `02_discurso_de_odio.md`, nacionalidades como referencias no peyorativas por sí mismas. |
| D05 | Mención de grupo + grosería no basta para 2; debe probarse que el agravio o trato desigual se basa en esa identidad. El odio puede ser formal y no contener groserías. | `02_discurso_de_odio.md`, notas de nacionalidad, religión, discapacidad, mujer/feminista, clase y política: mención por sí sola no indica odio; señales de generalización/exclusión; `04_ambiguas.md`, fila con identidades y pregunta “¿hay generalización, degradación, exclusión o discriminación?”. |
| D06 | `feminazi/zorra/perra` dirigidas como ataque sexista a una persona = 2; no se activa por coincidencia léxica. Literal animal, cita no respaldada, mención metalingüística y reapropiación afirmativa se distinguen por contexto. Un insulto sin base sexista puede ser 1. | Corrección del usuario reflejada en `01_lenguaje_ofensivo.md`, fila `perra/perro/zorra` (posible insulto, literal/metafórico), y `02_discurso_de_odio.md`, fila `feminazi/zorra/perra`; `04_ambiguas.md`, fila `perro/a` y recomendación de revisar objetivo. |
| D07 | Nacionalidad, religión, raza/etnia, discapacidad y género/sexualidad pueden ser descriptores neutrales; exija ataque o discriminación por ese motivo. | `00_no_toxico.md`, fila de gentilicios/identidades; `02_discurso_de_odio.md`, filas correspondientes a cada familia; `04_ambiguas.md`, fila de identidades. |
| D08 | Crítica política de políticas, partidos, ideologías y desempeño es 0; insulto personal/grupal no discriminatorio es 1. Afiliación política solo entra en 2 cuando se propone exclusión, persecución, privación de derechos/servicios o trato desigual por afiliación. | `02_discurso_de_odio.md`, fila política (`arenero`, `terengo`, `foca`, FMLN/ARENA, comunistas/bukelistas): crítica no es automáticamente odio; `04_ambiguas.md`, fila política pregunta por alcance; `01_lenguaje_ofensivo.md`, fila `narco/dictador/ladrón`. |
| D09 | El clasismo entra en 2 cuando hay degradación/exclusión/privación de derechos por clase socioeconómica; una descripción de pobreza es 0 y un insulto personal sin trato desigual por clase es 1. | `02_discurso_de_odio.md`, fila `pobre/s, clasista, ignorante/s` reconoce ambigüedad y deja la política por resolver; `04_ambiguas.md`, fila `pobre/ignorante` solicita definir alcance. |
| D10 | Ante solapamiento, la jerarquía primaria es **Amenazas/Violencia > Discurso de Odio > Lenguaje Ofensivo > No Tóxico (3 > 2 > 1 > 0)**; se conserva el objetivo identitario/político/clasista y `identity_basis`. Odio sin violencia = 2 y puede no tener groserías. | `03_amenaza_violencia.md`, “si el destinatario es además un grupo identitario, guardar ese atributo aunque la clase principal sea 3”; `README.md`, regla de solapamiento pendiente; `04_ambiguas.md`, regla de precedencia recomendada. |
| D11 | 3 requiere promoción atribuible de daño físico: amenaza, exhortación, deseo inequívoco o aprobación. Reporte, cita condenada, negación y metáfora no cuentan por sí solos. | `03_amenaza_violencia.md`, tabla advierte que verbo no basta; subtipos 1–5 (amenaza, exhortación, deseo, reporte, metáfora); `04_ambiguas.md`, fila de verbos de daño pregunta por agente, destinatario, modalidad y negación. |
| D12 | La postura del autor ante una cita o noticia decide la etiqueta: informar/condenar no es respaldar; si se desconoce la postura y cambia la decisión, contexto insuficiente. | `01_lenguaje_ofensivo.md`, cita/interjección puede requerir otra decisión; `03_amenaza_violencia.md`, noticia/cita/referencia histórica no es automáticamente amenaza; `04_ambiguas.md`, fila de matar/muerte y negación. |
| D13 | Las listas originales comentadas continúan como bancos candidatos y evidencia de trazabilidad, no como reglas automáticas ni etiquetas gold. | `docs/lexico_taxonomia/README.md`, fuentes nuevas sin validar y regla de no entrenar por coincidencia léxica; estados de validación descritos en listas. |

## Ejemplos de adjudicación contextual

| Texto contextualizado | Primaria | Atributos/razón |
|---|---:|---|
| “La vieja de la tienda ya abrió; voy por pan.” | 0 | Referencia descriptiva, sin ataque. `identity_basis: false`. |
| “Esa vieja idiota volvió a insultar a todos.” | 1 | Insulto personal dirigido; edad no es la base. |
| “Fuera de mi casa, por favor; ya cerramos.” | 0 | Petición de salir, sin ataque. |
| “Fuera los hondureños del barrio por ser hondureños.” | 2 | Exclusión explícita por nacionalidad; no requiere insulto. `identity_targets: [origen_nacionalidad]`. |
| “El hondureño de la oficina es un cerote porque llegó tarde.” | 1 | El motivo declarado es conducta individual; grupo mencionado no es la base. |
| “No alquilen a los pobres: no merecen vivir aquí.” | 2 | Exclusión basada explícitamente en clase socioeconómica. |
| “A los bukelistas hay que pegarles por apoyar al presidente.” | 3 | Exhortación de daño físico; afiliación política se conserva como objetivo/base. |
| “La nota citó ‘hay que pegarles’ y condenó el llamado.” | 0 | Reporte y rechazo, no aprobación propia. `quoted_or_reported: true`. |
| “María es una feminazi; las mujeres que exigen igualdad merecen que las saquen.” | 2 | Ataque sexista y exclusión basada en género; sin amenaza física. |
| “Mi perra se llama Zorra; así la nombró mi hija.” | 0 | Referencia literal a animales, no ataque a una persona. |
| “Leí ‘zorra’ en el comentario; no lo repito como insulto, lo estoy denunciando.” | 0 | Cita condenada; el acto del autor es denunciar. |
| “En la marcha, varias mujeres usan ‘zorra’ como consigna propia para celebrar que deciden sobre su vida.” | 0 | Reapropiación afirmativa, no ataque a otra persona. |
| “El gobierno publicó cifras falsas sobre el desempleo.” | 0 | Crítica política basada en actuación, no en identidad partidaria. |
| “No voy a matar a nadie; estoy molesto, pero me voy a calmar.” | 0 | Negación clara de violencia y desescalada. |
| “Te voy a matar por ser musulmán.” | 3 | Amenaza física con base religiosa; conservar `identity_basis: true`, `identity_targets: [religion]`. |
| “Vieja.” (sin texto previo, destinatario ni conversación) | — | `contexto_insuficiente`; no distinguir referencia, vocativo o insulto con el texto disponible. |

