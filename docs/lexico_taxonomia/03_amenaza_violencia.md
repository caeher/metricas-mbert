# 3 — Amenazas y violencia

Esta lista reúne verbos de daño, amenazas dirigidas, deseos de daño y eufemismos candidatos. La presencia de un verbo no basta: una noticia, una cita, una referencia histórica, una metáfora o un uso idiomático puede no ser una amenaza.

| Forma o patrón | Descripción para revisión | Procedencia / nota |
|---|---|---|
| `matar`, `matarte`, `te voy a matar`, `te mato` | Daño mortal; la segunda persona y la intención pueden volverlo amenaza directa. | `matar` y `m4t4r` aparecen en la batería/vocabulario del proyecto. Revisar sujeto, destinatario, temporalidad y cita. |
| `mueran`, `que se muera`, `ojalá te mueras` | Deseo o exhortación de muerte; distinguir de una descripción o una cita. | `mueran` y `muerte` observados en corpus real. |
| `asesinar`, `asesino/a`, `ejecutar` | Referencias a homicidio o ejecución; puede ser acusación, noticia o amenaza. | `asesinos` observado en corpus real; infinitivos como candidatos morfológicos. |
| `balear`, `disparar`, `dar bala`, `dar plomo`, `correr bala` | Acciones o alusiones a disparos. | Candidatos para búsqueda; validar formas y uso en las dos fuentes. |
| `apuñalar`, `degollar`, `cuchillo`, `machete` | Acciones/instrumentos de daño físico; un nombre de objeto no implica amenaza. | Candidatos de búsqueda que requieren contexto. |
| `golpear`, `reventar`, `torturar`, `quemar`, `colgar` | Verbos de agresión física o tortura; distinguir hipérbole de intención violenta. | Candidatos de búsqueda; añadir variantes flexionadas tras revisión. |
| `fusilar`, `masacrar`, `exterminar`, `desaparecer` | Violencia extrema o eliminación; pueden aparecer en discusión histórica/política. | `exterminio` observado en corpus real; los demás son candidatos por familia semántica. |
| `dar piso`, `dar cuello` | Eufemismos de daño/asesinato incluidos por el proyecto como ejemplos de amenaza ofuscada salvadoreña. | Documentados en `data/raw/README.md`; validar su uso real, variante y sentido en cada fuente. |
| `quebrar`, `plomear`, `hacer mierda` | Posibles eufemismos/expresiones de daño; algunos también son metafóricos. | Candidatos exploratorios, no confirmados como salvadoreñismos por una fuente lexicográfica en este banco. |
| `m4t4r`, `m@tar`, `m*t*r`, `m a t a r`, letras repetidas | Ofuscación ortográfica de una forma de daño. | `m4t4r` ya aparece como ejemplo en la documentación local del dataset. Buscar sin asumir que cada variante sea real o frecuente. |
| `marero/a`, `pandillero/a`, `muerte`, `cárcel`, `CECOT` | Vocabulario de contexto salvadoreño sobre pandillas/seguridad; no es una amenaza por sí mismo. | `mareros`, `pandilleros`, `muerte`, `matar`, `CECOT` aparecen en corpus real. Puede ser una noticia, una descripción o una exhortación violenta. |

## Subtipos útiles para anotar aparte

1. **Amenaza directa:** intención de dañar a un destinatario identificable.
2. **Exhortación:** se pide a terceros que ejerzan violencia.
3. **Deseo de daño:** se desea que alguien resulte herido o muera.
4. **Descripción/reporte:** se cuenta que ocurrió violencia, sin promoverla ni amenazar.
5. **Metáfora/uso idiomático:** vocabulario violento usado sin sentido físico.

Separar estos subtipos ayuda a revisar errores y a no confundir una amenaza con un insulto fuerte. Si el destinatario es además un grupo identitario, guardar ese atributo aunque la clase principal sea 3.

