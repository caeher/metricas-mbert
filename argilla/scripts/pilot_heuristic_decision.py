#!/usr/bin/env python
"""Heuristic pilot annotations aligned with docs/annotation_guidelines_v2 (ISSUE-05 helper)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PilotDecision:
    insuf: bool = False
    insufficient_reason: str = ""
    label: str = ""
    violence_type: str = "none"
    quoted: str = "no"
    negated: str = "no"
    quality: str = "usable"
    review: bool = False
    rationale: str = ""
    identity_targets: list[str] = field(default_factory=list)


def decide(text: str) -> PilotDecision:
    """Rule-based label proposal for Salvadoran Spanish toxicity (pilot helper)."""
    t = text.casefold()
    out = PilotDecision()
    has_slur = bool(
        re.search(
            r"\b(cerote|cerota|jueputa|vergaso|pendejo|culero|hijo de puta|malparido|come mierda|"
            r"mapachon|sorete|puta|puto|maric|perra|zorra|mierda)\b",
            t,
        )
    )
    if re.search(
        r"\b(te voy a matar|te mato|voy a matarte|vamos a matar|hay que matarlos|matarlos a todos|"
        r"mandarlos a matar|guillotina|linch|ejecuten|quemarles la casa)\b",
        t,
    ):
        return _filled(out, "3", "incitement", "Amenaza o incitación a violencia física → 3.")
    if re.search(r"\b(que muera|muera en la c[aá]rcel|ojal[aá] se muera|que se le mueran|deseo que muera)\b", t):
        return _filled(out, "3", "wish", "Deseo de daño/muerte → 3.")
    if re.search(r"\bmuerte!\b|\bmuerte\b", t) and re.search(r"\b(marero|pandillero|escoria)\b", t):
        return _filled(out, "3", "endorsement", "Exhortación/aprobación de muerte contra grupo → 3.")
    if re.search(r"\b(exterminar|genocid|inferior(es)?|raza asquerosa|indios de mierda|malditos indios)\b", t):
        return _filled(out, "2", "none", "Ataque/desvalorización por raza/etnia → 2.", identity=["raza_etnia"])
    if re.search(r"\b(musulman|jud[ií]|israel|palestin)\b", t) and has_slur:
        return _filled(out, "2", "none", "Insulto grupal con base religiosa → 2.", identity=["religion"])
    if re.search(
        r"\b(venezol|hondureñ|nicaragü|guatemalte|cuban|salvadoreñ|inmigrante|extranjer)\b", t
    ) and re.search(r"\b(mierda|basura|plaga|deport|devuelvan|parasit|fuera de|inferior|asqueros)\b", t):
        return _filled(out, "2", "none", "Exclusión/ataque por origen o nacionalidad → 2.", identity=["origen_nacionalidad"])
    if re.search(r"\b(fuera los|fuera de aqu[ií]).*(hondureñ|venezol|nicaragü|guatemalte|inmigrante)", t):
        return _filled(out, "2", "none", "Exclusión por nacionalidad → 2.", identity=["origen_nacionalidad"])
    if re.search(r"\b(feminaz|mujeres de mierda|putas feministas|zorra|perra)\b", t) and not re.search(
        r"\b(mi perra|perros?)\b", t
    ):
        return _filled(out, "2", "none", "Ataque sexista → 2.", identity=["genero_sexualidad"])
    if re.search(r"\b(arena|fmln|nuevas ideas|bukele|oposici[oó]n)\b", t) and re.search(
        r"\b(persecu|sin derechos|quitarles|excluir|deportar por apoyar)\b", t
    ):
        return _filled(out, "2", "none", "Exclusión/persecución por afiliación política → 2.", identity=["afiliacion_politica"])
    if re.search(r"\b(pobres? de mierda|indigentes)\b", t) and re.search(r"\b(no merecen|sin derecho)\b", t):
        return _filled(out, "2", "none", "Negación de derechos por clase → 2.", identity=["clase_socioeconomica"])
    if re.search(r"\b(padre|madre|me dijeron|discrimin|sufr[ií]|me llamaron)\b", t) and has_slur:
        out.quoted = "yes"
        out.negated = "yes"
        return _filled(out, "0", "none", "Relato/cita de insulto ajeno o condena → 0.")
    if re.search(
        r"\b(cerote|cerota|jueputa|vergaso|pendejo|culero|hijo de puta|malparido|mapachon|sorete|come mierda)\b", t
    ):
        return _filled(out, "1", "none", "Insulto o vulgaridad dirigida → 1.")
    if re.search(r"\b(independencia|#artesv|pintor salvadoreño|celebramos|orgullo salvadoreño|noticia|inform[oó])\b", t) and not has_slur:
        return _filled(out, "0", "none", "Contenido cívico/informativo → 0.")
    if re.search(r"\b(jajaja|messi|partido|tps|tr[aá]fico)\b", t) and not has_slur and not re.search(r"\b(matar|muera)\b", t):
        return _filled(out, "0", "none", "Coloquial o crítica no tóxica → 0.")
    if re.search(r"\b(maje|chero|cipote)\b", t) and len(text.strip()) < 40:
        out.insuf = True
        out.insufficient_reason = "Jerga ambigua sin destinatario ni tono claro."
        out.rationale = "Contexto insuficiente; no forzar clase 0."
        return out
    if len(text.strip()) < 18:
        out.insuf = True
        out.insufficient_reason = "Texto muy corto; falta contexto del acto."
        out.rationale = "Contexto insuficiente para etiqueta fiable."
        return out
    if has_slur:
        out.review = True
        return _filled(out, "1", "none", "Vulgaridad presente; base identitaria no demostrable → 1 (revisar).")
    return _filled(out, "0", "none", "Sin ataque, odio ni violencia promovida → 0.")


def _filled(
    out: PilotDecision,
    label: str,
    violence: str,
    rationale: str,
    *,
    identity: list[str] | None = None,
) -> PilotDecision:
    out.label = label
    out.violence_type = violence
    out.rationale = rationale
    out.identity_targets = identity or []
    return out


def provisional_label_from_row(row: dict[str, str]) -> str:
    import json

    raw = row.get("label_provisional_ids_json", "[]")
    try:
        labels = json.loads(raw)
    except json.JSONDecodeError:
        labels = []
    if not labels:
        return ""
    value = str(labels[0]).strip()
    if value in {"0", "1", "2", "3"}:
        return value
    return ""


def decide_rater_b(text: str, provisional: str) -> PilotDecision:
    """Second rater: prefer historical provisional id when present, else heuristic."""
    base = decide(text)
    if not provisional or provisional not in {"0", "1", "2", "3"}:
        return base
    if base.insuf:
        return base
    alt = PilotDecision(
        label=provisional,
        violence_type=base.violence_type,
        quoted=base.quoted,
        negated=base.negated,
        quality="review" if provisional != base.label else base.quality,
        review=provisional != base.label or base.review,
        rationale=f"Etiqueta provisional histórica {provisional}; revisión independiente.",
        identity_targets=base.identity_targets.copy(),
    )
    if provisional != base.label and not alt.review:
        alt.quality = "usable"
    return alt


def build_response_values(dec: PilotDecision) -> dict[str, Any]:
    if dec.insuf:
        return {
            "annotation_status": "contexto_insuficiente",
            "insufficient_context_reason": dec.insufficient_reason or "Contexto insuficiente.",
            "quality": "insufficient_context",
            "subclass": "N/A",
            "dialect": "desconocido",
            "quoted_or_reported": "unknown",
            "negated": "unknown",
            "violence_type": "none",
            "rationale": dec.rationale,
        }
    values: dict[str, Any] = {
        "annotation_status": "anotado",
        "label": dec.label,
        "subclass": "N/A",
        "dialect": "salvadoreno",
        "quoted_or_reported": dec.quoted,
        "negated": dec.negated,
        "violence_type": dec.violence_type,
        "quality": "review" if dec.review else "usable",
        "insufficient_context_reason": "N/A",
        "rationale": dec.rationale,
    }
    if dec.identity_targets:
        values["identity_targets"] = dec.identity_targets
    return values
