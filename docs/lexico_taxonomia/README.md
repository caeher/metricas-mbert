# Banco léxico inicial por taxonomía

**Versión:** 0.1 · **Idioma:** español salvadoreño y español de redes  
**Estado:** propuestas para revisión humana; no son etiquetas gold ni una lista automática de bloqueo.

La política contextual versionada para anotar estas listas está en [Guía de anotación v2.0](../annotation_guidelines_v2.md); las decisiones y su trazabilidad están en el [registro v2.0](../taxonomy_decisions_v2.md). Estas listas originales se conservan como evidencia y vocabulario candidato.

## Archivos

- [No tóxicas (label 0)](00_no_toxico.md)
- [Lenguaje ofensivo (label 1)](01_lenguaje_ofensivo.md)
- [Discurso de odio (label 2)](02_discurso_de_odio.md)
- [Amenazas y violencia (label 3)](03_amenaza_violencia.md)
- [Términos ambiguos y decisiones pendientes](04_ambiguas.md)

## Qué se usó

1. **Corpus existente:** términos observados en filas reales de `data/raw/train.csv` (`origen=real`); se excluyeron filas sintéticas al extraer candidatos.
2. **Fuentes recién recolectadas:** `data/raw/lenguaje_ofensivo.csv` (1,328 filas) y `data/raw/thenewdata_X.csv` (1,000 filas). Ambas tienen `Validado=NO` y `Listo_Entrenamiento=NO`. Se usan solo para proponer vocabulario. En la fuente de X, 497 notas dicen que la ubicación salvadoreña se infiere y no está confirmada por el perfil; no asumir que esos textos sean salvadoreños ni tratarlos como verdad de referencia.
3. **Fuentes de internet:** entradas del [Diccionario de americanismos de ASALE](https://www.asale.org/damer/) para usos regionales y la definición operativa amplia de [discurso de odio de Naciones Unidas](https://www.un.org/en/node/165070).

Las descripciones de las tablas son **notas de anotación para este proyecto**, no definiciones legales. Los términos extraídos de fuentes sin validar aparecen como candidatos: que un término sea frecuente en una subclase provisional no prueba que esa asignación sea correcta.

## Criterios operativos propuestos

- **0 — No tóxica:** comunicación neutral, descriptiva, coloquial o amistosa sin ataque dirigido ni promoción de daño. La jerga salvadoreña no debe contar como toxicidad por sí sola.
- **1 — Lenguaje ofensivo:** insulto, vulgaridad dirigida o descalificación, sin ataque basado en identidad ni amenaza de daño físico. Una grosería citada o usada como interjección puede requerir otra decisión.
- **2 — Discurso de odio:** ataque o lenguaje discriminatorio contra una persona o grupo por identidad (por ejemplo, origen, nacionalidad, raza, religión, género, orientación sexual o discapacidad). La ONU advierte que no hay una definición jurídica universal; esta es una guía de anotación, no una determinación legal.
- **3 — Amenaza/violencia:** amenaza concreta, deseo o exhortación de daño físico, o defensa explícita de ejercer violencia. Una noticia, cita o descripción histórica de violencia no se convierte automáticamente en amenaza.

**Regla de solapamiento pendiente de aprobación:** cuando un texto amenaza con dañar a un grupo identitario, conservar el objetivo de identidad como atributo adicional y decidir con el equipo si la etiqueta principal será 3. No ocultar la dimensión de odio al forzar una sola etiqueta.

## Cómo revisar las listas

Para cada forma, asigna `label`, `subclase`, `uso local`, `ejemplo contextual`, `fuente`, `estado` (`candidata`, `validada` o `descartada`) y, cuando aplique, `grupo_objetivo`. Valida primero frases completas y ambas fuentes por separado. Añade variantes ortográficas y formas ofuscadas solo después de encontrar ejemplos reales.

No entrenar un clasificador con estas palabras como etiquetas ni usar una coincidencia de lexicón para sustituir el contexto. El banco sirve para guiar la revisión, localizar casos difíciles y mejorar las instrucciones de anotación. Para Argilla, cada fila se puede convertir en un registro con una pregunta de clase y otra de contexto/ambigüedad.

