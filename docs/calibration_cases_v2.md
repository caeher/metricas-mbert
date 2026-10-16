# Casos propuestos de calibración v2.0

**Uso:** discusión y formación de anotadores. Son ejemplos construidos; no son textos recolectados, etiquetas gold ni una prueba independiente. Úselos para aplicar la guía [`annotation_guidelines_v2.md`](annotation_guidelines_v2.md), comparar razones y revisar desacuerdos. La clase principal es la propuesta esperada bajo la versión 2.0.0.

Campos abreviados: `B` = `identity_basis`; `T` = `identity_targets`; `V` = `violence_type`; `Q` = cita/reporte; `N` = negación; `R` = reapropiación afirmativa. Salvo donde se indique, `annotation_status=anotado`.

## Clase 0 — No Tóxico (10)

| ID | Texto y contexto | Primaria | Atributos y razón |
|---|---|---:|---|
| C0-01 | «Mi chero, guardame campo; ya voy llegando.» | 0 | Jerga amistosa; B=false, V=none. |
| C0-02 | «La vieja de la tienda ya abrió; voy por pan.» | 0 | Referencia neutral a una persona; sin insulto. |
| C0-03 | «Fuera de mi casa, por favor; ya cerramos el local.» | 0 | Petición de salir, sin descalificación ni identidad. |
| C0-04 | «¡Púchica, qué golazo hizo la Selecta!» | 0 | Interjección expresiva, no dirigida como insulto. |
| C0-05 | «No voy a golpear a nadie; mejor me retiro de la discusión.» | 0 | Negación explícita y desescalada; N=true, V=none. |
| C0-06 | «El comentario decía “zorra”; lo cito para denunciar el insulto.» | 0 | Cita condenada; Q=true, R=false. |
| C0-07 | «La alcaldía publicó tarde el presupuesto y debe explicar el retraso.» | 0 | Crítica de actuación pública, no de identidad política. |
| C0-08 | «La noticia informa que hubo un ataque anoche; la policía investiga.» | 0 | Reporte neutral de violencia; Q=true, V=report. |
| C0-09 | «Mi perra tuvo cachorros; vamos a buscarles casa.» | 0 | Uso literal del animal, no insulto a una persona. |
| C0-10 | «Mi vecina hondureña me prestó una escalera.» | 0 | Mención descriptiva de nacionalidad; B=false. |

## Clase 1 — Lenguaje Ofensivo (10)

| ID | Texto y contexto | Primaria | Atributos y razón |
|---|---|---:|---|
| C1-01 | «Maje, sos un cerote; por tu culpa perdimos el bus y ya no te aguanto.» | 1 | Insulto vulgar dirigido y hostil; no hay base identitaria ni daño promovido. |
| C1-02 | «Esa vieja idiota volvió a interrumpir la reunión.» | 1 | Descalificación dirigida; `vieja` sola sería neutral, `idiota` concreta el insulto. |
| C1-03 | «El hondureño de la oficina es un cerote porque nunca entrega el informe a tiempo.» | 1 | El motivo explícito es conducta individual, no nacionalidad; B=false. |
| C1-04 | «Qué mierda de conductor sos; casi atropellás a la gente por ir viendo el teléfono.» | 1 | Descalificación dirigida por conducta, sin amenaza ni objetivo identitario. |
| C1-05 | «El vocero del partido es un ladrón sinvergüenza por ocultar las facturas.» | 1 | Insulto/acusación sobre conducta; no discrimina por afiliación. |
| C1-06 | «Sos un pendejo por publicar sin leer la nota.» | 1 | Insulto dirigido por conducta. |
| C1-07 | «El comentarista llamó “imbécil” al árbitro y añadió: “estoy de acuerdo con él”.» | 1 | Cita respaldada de un insulto no identitario; Q=true, se adopta la descalificación. |
| C1-08 | «Ese árbitro es un perro inútil; dejó perder el partido.» | 1 | Metáfora insultante dirigida a una persona, no ataque identitario. |
| C1-09 | «Qué publicación más estúpida; no entendiste la pregunta.» | 1 | Descalificación personal/dirigida, sin identidad ni violencia. |
| C1-10 | «Maje, no tenés cerebro para entender el dato; dejá de estorbar.» | 1 | Insulto personal explícito; jerga no es el único criterio. |

## Clase 2 — Discurso de Odio (10)

| ID | Texto y contexto | Primaria | Atributos y razón |
|---|---|---:|---|
| C2-01 | «Los hondureños son inferiores y no deberían alquilar casas en este barrio.» | 2 | Generalización y exclusión por nacionalidad, sin necesidad de grosería; B=true, T=origen_nacionalidad. |
| C2-02 | «Esos hondureños son una basura por ser hondureños y no deberían vivir aquí.» | 2 | Grosería y exclusión explícitamente basadas en nacionalidad; B=true, T=origen_nacionalidad. |
| C2-03 | «No contraten musulmanes; por su religión no son de fiar.» | 2 | Estereotipo y trato desigual por religión; B=true, T=religion. |
| C2-04 | «Las personas con discapacidad no deberían recibir educación pública.» | 2 | Negación de servicio por discapacidad, aunque el tono sea formal; T=discapacidad. |
| C2-05 | «Las mujeres que piden igualdad son unas feminazis y hay que quitarles voz en el trabajo.» | 2 | `feminazis` funciona como ataque sexista y se propone exclusión por género; T=genero_sexualidad. |
| C2-06 | «A María le digo zorra porque una mujer decente no sale con quien quiere.» | 2 | Slur dirigido a una persona por estereotipo sexual/género; no es coincidencia automática. |
| C2-07 | «Esas perras no sirven para dirigir; las mujeres deben quedarse en casa.» | 2 | Ataque sexista y generalización sobre mujeres; no hay promoción de daño físico. |
| C2-08 | «No alquilen a los pobres: no merecen vivir aquí con gente decente.» | 2 | Exclusión y degradación por clase socioeconómica; T=clase_socioeconomica. |
| C2-09 | «Los areneros deberían perder el derecho al voto solo por apoyar a ARENA.» | 2 | Privación de derecho por afiliación política; extensión operativa del proyecto, T=afiliacion_politica. |
| C2-10 | «Por ser indígenas, no deberían poder ocupar cargos en la alcaldía.» | 2 | Restricción de participación por pertenencia étnica; no requiere grosería, T=raza_etnia. |

## Clase 3 — Amenazas/Violencia (10)

| ID | Texto y contexto | Primaria | Atributos y razón |
|---|---|---:|---|
| C3-01 | «Te voy a encontrar al salir y te voy a quebrar las piernas.» | 3 | Amenaza directa de daño físico; V=direct_threat, T=[]. |
| C3-02 | «Vayan a quemarles la casa esta noche para que aprendan.» | 3 | Exhortación a violencia física; V=incitement. |
| C3-03 | «Ojalá maten al periodista por publicar esa investigación.» | 3 | Deseo inequívoco de daño mortal; V=wish. |
| C3-04 | «Te voy a matar por ser musulmán.» | 3 | Amenaza + base religiosa; B=true, T=religion, V=direct_threat. Prevalece 3. |
| C3-05 | «Hay que sacar a golpes a todos los hondureños del barrio.» | 3 | Exhortación violenta y exclusión por nacionalidad; B=true, T=origen_nacionalidad, V=incitement. |
| C3-06 | «Comparto lo que escribió el diputado: “hay que pegarles a los bukelistas hasta que se callen”.» | 3 | Cita aprobada por el autor; violencia física contra afiliación política; Q=true, B=true, T=afiliacion_politica, V=endorsement. |
| C3-07 | «Si volvés a romper mi ventana, te voy a romper la cara.» | 3 | Amenaza física condicional y directa; V=direct_threat. |
| C3-08 | «Después del partido te voy a dar bala por ese gol.» | 3 | Amenaza física de disparar; V=direct_threat. |
| C3-09 | «Lo golpearon entre varios y se lo merecía; ojalá le vuelvan a dar.» | 3 | Aprobación de violencia física y deseo de que continúe; V=endorsement. |
| C3-10 | «A los pobres hay que pegarles para que se vayan del centro.» | 3 | Exhortación de violencia física; base de clase se conserva, B=true, T=clase_socioeconomica, V=incitement. Prevalece 3. |

## Contrastes para discusión (fuera del cupo de 40)

Estos controles no forman parte de los 40 casos contabilizados. «El hondureño de la oficina es un cerote porque llegó tarde» es 1: conducta individual, no nacionalidad. «Mi vecina hondureña me prestó una escalera» es 0: mención descriptiva. Contrastan con C2-02, donde la exclusión se basa explícitamente en nacionalidad. «En la marcha, varias mujeres usan “zorra” como consigna propia para celebrar que deciden sobre su vida» es 0 por reapropiación afirmativa, no un ataque. Contrástese con C2-06.

## Casos que requieren contexto adicional (fuera del cupo de 40)

Estos controles enseñan el estado de anotación, no añaden una clase: «Vieja» aislado no permite distinguir referencia, vocativo o insulto; «Feminazi» sin frase ni destinatario no demuestra por sí solo un ataque sexista; «mueran» sin sujeto, modalidad ni postura puede ser cita, noticia, deseo o exhortación. En los tres, use `contexto_insuficiente`, `label: null` y explique qué contexto falta.

