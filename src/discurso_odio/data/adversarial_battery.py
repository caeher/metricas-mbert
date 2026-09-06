"""Construcción de la batería adversarial extendida (A–G, n=144)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from discurso_odio.paths import TEST_ADVERSARIAL_144_CSV, TEST_ADVERSARIAL_CSV

BATTERY_144_VERSION = "adversarial_v2_144"

GROUP_F = "F. Falsos amigos dialectales"
GROUP_G = "G. Pares mínimos de emoji"

# 34 casos: jerga salvadoreña inofensiva y contrastes semánticos (informe verificación §4.3)
F_CASES: list[dict[str, Any]] = [
    {"base": "pajero mentiroso", "variante": "jerga", "texto_original": "Ese maje es bien pajero, nunca cumple", "texto_modelo": "ese maje es bien pajero nunca cumple", "clase": "No Tóxico", "label": 0},
    {"base": "perro vecino", "variante": "jerga", "texto_original": "El chucho de la vecina ladra toda la noche", "texto_modelo": "el chucho de la vecina ladra toda la noche", "clase": "No Tóxico", "label": 0},
    {"base": "prestamo dinero", "variante": "jerga", "texto_original": "Prestame pisto para el pasaje", "texto_modelo": "prestame pisto para el pasaje", "clase": "No Tóxico", "label": 0},
    {"base": "amigo cerote", "variante": "jerga", "texto_original": "Mi cerote me ayudó con la mudanza", "texto_modelo": "mi cerote me ayudo con la mudanza", "clase": "No Tóxico", "label": 0},
    {"base": "auto chivo", "variante": "jerga", "texto_original": "Qué chivo te quedó el carro", "texto_modelo": "que chivo te quedo el carro", "clase": "No Tóxico", "label": 0},
    {"base": "muchacho bicho", "variante": "jerga", "texto_original": "No sea bicho, ayudeme con la tarea", "texto_modelo": "no sea bicho ayudeme con la tarea", "clase": "No Tóxico", "label": 0},
    {"base": "cita cabal", "variante": "jerga", "texto_original": "Vaya pues, nos vemos mañana cabal", "texto_modelo": "vaya pues nos vemos manana cabal", "clase": "No Tóxico", "label": 0},
    {"base": "concierto vergon", "variante": "jerga", "texto_original": "Está bien vergón el concierto", "texto_modelo": "esta bien vergon el concierto", "clase": "No Tóxico", "label": 0},
    {"base": "saludo maje", "variante": "jerga", "texto_original": "Qué onda maje, todo tranquilo", "texto_modelo": "que onda maje todo tranquilo", "clase": "No Tóxico", "label": 0},
    {"base": "hijo cipote", "variante": "jerga", "texto_original": "Mi cipote ya cumplió los quince", "texto_modelo": "mi cipote ya cumplio los quince", "clase": "No Tóxico", "label": 0},
    {"base": "borracho pedo", "variante": "jerga", "texto_original": "Andaba bien pedo pero llegó bien", "texto_modelo": "andaba bien pedo pero llego bien", "clase": "No Tóxico", "label": 0},
    {"base": "sorpresa puchica", "variante": "jerga", "texto_original": "Puchica, se me olvidó la cita", "texto_modelo": "puchica se me olvido la cita", "clase": "No Tóxico", "label": 0},
    {"base": "comida pupusas", "variante": "jerga", "texto_original": "Vamos a comer pupusas al mercado", "texto_modelo": "vamos a comer pupusas al mercado", "clase": "No Tóxico", "label": 0},
    {"base": "torpe bayunco", "variante": "jerga", "texto_original": "Ese bayunco no sabe ni manejar", "texto_modelo": "ese bayunco no sabe ni manejar", "clase": "No Tóxico", "label": 0},
    {"base": "alegre verga", "variante": "jerga", "texto_original": "Qué verga maje, qué alegre verte", "texto_modelo": "que verga maje que alegre verte", "clase": "No Tóxico", "label": 0},
    {"base": "rico puta", "variante": "jerga", "texto_original": "Puta, qué rico estuvo el almuerzo", "texto_modelo": "puta que rico estuvo el almuerzo", "clase": "No Tóxico", "label": 0},
    {"base": "si a huevo", "variante": "jerga", "texto_original": "A huevo que sí, vamos a la playa", "texto_modelo": "a huevo que si vamos a la playa", "clase": "No Tóxico", "label": 0},
    {"base": "amigo chero", "variante": "jerga", "texto_original": "Ese chero siempre llega tarde pero llega", "texto_modelo": "ese chero siempre llega tarde pero llega", "clase": "No Tóxico", "label": 0},
    {"base": "fiesta guaro", "variante": "jerga", "texto_original": "Ya casi está el guaro para el convivio", "texto_modelo": "ya casi esta el guaro para el convivio", "clase": "No Tóxico", "label": 0},
    {"base": "lluvia chuco", "variante": "jerga", "texto_original": "Qué chuco quedó el patio con la lluvia", "texto_modelo": "que chuco quedo el patio con la lluvia", "clase": "No Tóxico", "label": 0},
    {"base": "mentiroso estandar", "variante": "contraste", "texto_original": "Ese muchacho es bien mentiroso, nunca cumple", "texto_modelo": "ese muchacho es bien mentiroso nunca cumple", "clase": "No Tóxico", "label": 0},
    {"base": "perro estandar", "variante": "contraste", "texto_original": "El perro de la vecina ladra toda la noche", "texto_modelo": "el perro de la vecina ladra toda la noche", "clase": "No Tóxico", "label": 0},
    {"base": "dinero estandar", "variante": "contraste", "texto_original": "Prestame dinero para el pasaje", "texto_modelo": "prestame dinero para el pasaje", "clase": "No Tóxico", "label": 0},
    {"base": "amigo estandar", "variante": "contraste", "texto_original": "Mi amigo me ayudó con la mudanza", "texto_modelo": "mi amigo me ayudo con la mudanza", "clase": "No Tóxico", "label": 0},
    {"base": "carro estandar", "variante": "contraste", "texto_original": "Qué bonito te quedó el carro", "texto_modelo": "que bonito te quedo el carro", "clase": "No Tóxico", "label": 0},
    {"base": "ayuda estandar", "variante": "contraste", "texto_original": "Por favor ayudeme con la tarea", "texto_modelo": "por favor ayudeme con la tarea", "clase": "No Tóxico", "label": 0},
    {"base": "cita estandar", "variante": "contraste", "texto_original": "Está bien, nos vemos mañana", "texto_modelo": "esta bien nos vemos manana", "clase": "No Tóxico", "label": 0},
    {"base": "concierto estandar", "variante": "contraste", "texto_original": "Estuvo excelente el concierto", "texto_modelo": "estuvo excelente el concierto", "clase": "No Tóxico", "label": 0},
    {"base": "saludo estandar", "variante": "contraste", "texto_original": "Hola, todo tranquilo", "texto_modelo": "hola todo tranquilo", "clase": "No Tóxico", "label": 0},
    {"base": "hijo estandar", "variante": "contraste", "texto_original": "Mi hijo ya cumplió los quince", "texto_modelo": "mi hijo ya cumplio los quince", "clase": "No Tóxico", "label": 0},
    {"base": "olvido estandar", "variante": "contraste", "texto_original": "Se me olvidó la cita", "texto_modelo": "se me olvido la cita", "clase": "No Tóxico", "label": 0},
    {"base": "comida estandar", "variante": "contraste", "texto_original": "Vamos a comer al mercado", "texto_modelo": "vamos a comer al mercado", "clase": "No Tóxico", "label": 0},
    {"base": "torpe estandar", "variante": "contraste", "texto_original": "Ese muchacho no sabe ni manejar", "texto_modelo": "ese muchacho no sabe ni manejar", "clase": "No Tóxico", "label": 0},
    {"base": "alegre estandar", "variante": "contraste", "texto_original": "Qué bueno verte de nuevo", "texto_modelo": "que bueno verte de nuevo", "clase": "No Tóxico", "label": 0},
]

# 35 casos: mismo texto con emoji distinto (informe verificación §4.1)
G_CASES: list[dict[str, Any]] = [
    {"base": "familia bonita", "variante": "corazon_inocuo", "texto_original": "Qué bonita familia ❤️", "texto_modelo": "que bonita familia :corazón_rojo:", "clase": "No Tóxico", "label": 0},
    {"base": "vecino nuevo", "variante": "saludo_inocuo", "texto_original": "Ese es el vecino nuevo 👋", "texto_modelo": "ese es el vecino nuevo :saludo_con_la_mano:", "clase": "No Tóxico", "label": 0},
    {"base": "grupo siempre", "variante": "manos_inocuo", "texto_original": "Ahí viene el grupo de siempre 🙌", "texto_modelo": "ahi viene el grupo de siempre :manos_levantadas_celebrando:", "clase": "No Tóxico", "label": 0},
    {"base": "te espera pastel", "variante": "pastel_inocuo", "texto_original": "Ya sabés lo que te espera 🎂", "texto_modelo": "ya sabes lo que te espera :tarta_de_cumpleaños:", "clase": "No Tóxico", "label": 0},
    {"base": "vienen del sur", "variante": "balon_inocuo", "texto_original": "Los que vienen del sur ⚽", "texto_modelo": "los que vienen del sur :balón_de_fútbol:", "clase": "No Tóxico", "label": 0},
    {"base": "alegre verte", "variante": "corazon_inocuo", "texto_original": "Qué alegre verte de nuevo ❤️", "texto_modelo": "que alegre verte de nuevo :corazón_rojo:", "clase": "No Tóxico", "label": 0},
    {"base": "rico almuerzo", "variante": "risa_inocuo", "texto_original": "Qué rico estuvo el almuerzo 😂", "texto_modelo": "que rico estuvo el almuerzo :cara_llorando_de_risa:", "clase": "No Tóxico", "label": 0},
    {"base": "llegamos bien", "variante": "pulgar_inocuo", "texto_original": "Ya llegamos bien a la casa 👍", "texto_modelo": "ya llegamos bien a la casa :pulgar_hacia_arriba:", "clase": "No Tóxico", "label": 0},
    {"base": "te espera pistola", "variante": "pistola_amenaza", "texto_original": "Ya sabés lo que te espera 🔫", "texto_modelo": "ya sabes lo que te espera :pistola_de_agua:", "clase": "Amenazas/Violencia", "label": 3},
    {"base": "buscar manana", "variante": "cuchillo_amenaza", "texto_original": "Te voy a buscar mañana 🔪", "texto_modelo": "te voy a buscar manana :cuchillo_de_cocina:", "clase": "Amenazas/Violencia", "label": 3},
    {"base": "espero afuera", "variante": "ataud_amenaza", "texto_original": "Te espero afuera ⚰️", "texto_modelo": "te espero afuera :ataúd:", "clase": "Amenazas/Violencia", "label": 3},
    {"base": "espero afuera", "variante": "sin_emoji", "texto_original": "Te espero afuera", "texto_modelo": "te espero afuera", "clase": "Amenazas/Violencia", "label": 3},
    {"base": "familia bonita", "variante": "sin_emoji", "texto_original": "Qué bonita familia", "texto_modelo": "que bonita familia", "clase": "No Tóxico", "label": 0},
    {"base": "vecino nuevo", "variante": "sin_emoji", "texto_original": "Ese es el vecino nuevo", "texto_modelo": "ese es el vecino nuevo", "clase": "No Tóxico", "label": 0},
    {"base": "grupo siempre", "variante": "sin_emoji", "texto_original": "Ahí viene el grupo de siempre", "texto_modelo": "ahi viene el grupo de siempre", "clase": "No Tóxico", "label": 0},
    {"base": "te espera pastel", "variante": "sin_emoji", "texto_original": "Ya sabés lo que te espera mañana", "texto_modelo": "ya sabes lo que te espera manana", "clase": "No Tóxico", "label": 0},
    {"base": "vienen del sur", "variante": "sin_emoji", "texto_original": "Los que vienen del sur", "texto_modelo": "los que vienen del sur", "clase": "No Tóxico", "label": 0},
    {"base": "alegre verte", "variante": "sin_emoji", "texto_original": "Qué alegre verte de nuevo", "texto_modelo": "que alegre verte de nuevo", "clase": "No Tóxico", "label": 0},
    {"base": "rico almuerzo", "variante": "sin_emoji", "texto_original": "Qué rico estuvo el almuerzo", "texto_modelo": "que rico estuvo el almuerzo", "clase": "No Tóxico", "label": 0},
    {"base": "llegamos bien", "variante": "sin_emoji", "texto_original": "Ya llegamos bien a la casa", "texto_modelo": "ya llegamos bien a la casa", "clase": "No Tóxico", "label": 0},
    {"base": "buscar manana", "variante": "sin_emoji", "texto_original": "Te voy a buscar mañana", "texto_modelo": "te voy a buscar manana", "clase": "Amenazas/Violencia", "label": 3},
    {"base": "espero afuera", "variante": "pistola_amenaza", "texto_original": "Te espero afuera 🔫", "texto_modelo": "te espero afuera :pistola_de_agua:", "clase": "Amenazas/Violencia", "label": 3},
    {"base": "espero afuera", "variante": "cuchillo_amenaza", "texto_original": "Te espero afuera 🔪", "texto_modelo": "te espero afuera :cuchillo_de_cocina:", "clase": "Amenazas/Violencia", "label": 3},
    {"base": "espero afuera", "variante": "calavera_amenaza", "texto_original": "Te voy a buscar mañana 💀", "texto_modelo": "te voy a buscar manana :calavera:", "clase": "Amenazas/Violencia", "label": 3},
    {"base": "insulto payaso", "variante": "payaso_ofensivo", "texto_original": "Mira a este tipo 🤡", "texto_modelo": "mira a este tipo :cara_de_payaso:", "clase": "Lenguaje Ofensivo", "label": 1},
    {"base": "insulto payaso", "variante": "sin_emoji", "texto_original": "Mira a este tipo", "texto_modelo": "mira a este tipo", "clase": "Lenguaje Ofensivo", "label": 1},
    {"base": "odio mono", "variante": "mono_odio", "texto_original": "Los jugadores de ese equipo 🐒", "texto_modelo": "los jugadores de ese equipo :mono:", "clase": "Discurso de Odio", "label": 2},
    {"base": "odio mono", "variante": "sin_emoji", "texto_original": "Los jugadores de ese equipo", "texto_modelo": "los jugadores de ese equipo", "clase": "Discurso de Odio", "label": 2},
    {"base": "decoracion casa", "variante": "confeti_inocuo", "texto_original": "Le quedó preciosa la decoración de la casa 🎉", "texto_modelo": "le quedo preciosa la decoracion de la casa :cañón_de_confeti:", "clase": "No Tóxico", "label": 0},
    {"base": "decoracion casa", "variante": "sin_emoji", "texto_original": "Le quedó preciosa la decoración de la casa", "texto_modelo": "le quedo preciosa la decoracion de la casa", "clase": "No Tóxico", "label": 0},
    {"base": "concierto increible", "variante": "corazon_inocuo", "texto_original": "El concierto estuvo increíble ❤️", "texto_modelo": "el concierto estuvo increible :corazón_rojo:", "clase": "No Tóxico", "label": 0},
    {"base": "concierto increible", "variante": "sin_emoji", "texto_original": "El concierto estuvo increíble", "texto_modelo": "el concierto estuvo increible", "clase": "No Tóxico", "label": 0},
    {"base": "dar piso", "variante": "sin_emoji", "texto_original": "Te voy a dar piso", "texto_modelo": "te voy a dar piso", "clase": "Amenazas/Violencia", "label": 3},
    {"base": "dar piso", "variante": "bomba_amenaza", "texto_original": "Te voy a dar piso 💣", "texto_modelo": "te voy a dar piso :bomba:", "clase": "Amenazas/Violencia", "label": 3},
    {"base": "dar cuello", "variante": "sin_emoji", "texto_original": "Le van a dar cuello a ese", "texto_modelo": "le van a dar cuello a ese", "clase": "Amenazas/Violencia", "label": 3},
]


def _rows_for_group(group: str, cases: list[dict[str, Any]], start_idx: int) -> list[dict[str, Any]]:
    rows = []
    for offset, case in enumerate(cases):
        rows.append(
            {
                "ID": f"ADV-{start_idx + offset:04d}",
                "grupo": group,
                "base": case["base"],
                "variante": case["variante"],
                "texto_original": case["texto_original"],
                "texto_modelo": case["texto_modelo"],
                "clase": case["clase"],
                "label": case["label"],
            }
        )
    return rows


def build_adversarial_144_dataframe() -> pd.DataFrame:
    base_df = pd.read_csv(TEST_ADVERSARIAL_CSV)
    f_rows = _rows_for_group(GROUP_F, F_CASES, start_idx=len(base_df) + 1)
    g_rows = _rows_for_group(GROUP_G, G_CASES, start_idx=len(base_df) + len(f_rows) + 1)
    extended = pd.DataFrame(f_rows + g_rows)
    return pd.concat([base_df, extended], ignore_index=True)


def write_adversarial_144_csv(path: Path | None = None) -> Path:
    path = path or TEST_ADVERSARIAL_144_CSV
    df = build_adversarial_144_dataframe()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


FAMILY_SPECS_144: dict[str, dict[str, Any]] = {
    "A. Ofuscación de groserías": {"key": "A", "n": 20, "metric": "accuracy"},
    "B. Emojis": {"key": "B", "n": 18, "metric": "accuracy"},
    "C. Amenazas ofuscadas": {"key": "C", "n": 11, "metric": "accuracy"},
    "D. Jerga salvadoreña inofensiva": {"key": "D", "n": 14, "metric": "fpr"},
    "E. Control sin jerga": {"key": "E", "n": 12, "metric": "fpr"},
    GROUP_F: {"key": "F", "n": 34, "metric": "fpr"},
    GROUP_G: {"key": "G", "n": 35, "metric": "emoji_pair"},
}

BATTERY_144_N = 144
