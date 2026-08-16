"""Pipeline de normalización — fuente única de verdad.

Preserva tildes, mayúsculas y puntuación (criterio del informe v1.1).
Normaliza: Unicode NFC, caracteres invisibles, espacios, menciones, URLs, emojis.
"""

from __future__ import annotations

import re
import unicodedata

import emoji
import ftfy

MENTION_PATTERN = re.compile(r"@\w+")
URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+|\bt\.co/\S+",
    re.IGNORECASE,
)
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE_PATTERN = re.compile(r"\+?\d[\d\s\-()]{7,}\d")
INVISIBLE_CHARS = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]")
MULTI_SPACE = re.compile(r"\s+")


def fix_encoding(text: str) -> str:
    return ftfy.fix_text(text) if text else ""


def normalize_unicode(text: str) -> str:
    text = fix_encoding(text)
    text = INVISIBLE_CHARS.sub("", text)
    return unicodedata.normalize("NFC", text)


def collapse_whitespace(text: str) -> str:
    return MULTI_SPACE.sub(" ", text).strip()


def replace_mentions(text: str, marker: str = "@usuario") -> str:
    return MENTION_PATTERN.sub(marker, text)


def replace_urls(text: str, marker: str = "http://url") -> str:
    text = URL_PATTERN.sub(marker, text)
    text = EMAIL_PATTERN.sub("[email]", text)
    text = PHONE_PATTERN.sub("[telefono]", text)
    return text


def replace_emojis(text: str) -> str:
    def _demojize(char: str, _data=None) -> str:
        desc = emoji.demojize(char, delimiters=("", "")).replace("_", " ")
        return f" {desc} " if desc else " "

    return emoji.replace_emoji(text, _demojize)


def normalize_for_model(text: str) -> str:
    """Variante texto_modelo: preservación máxima + anonimización estructural."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = normalize_unicode(text)
    text = replace_urls(text)
    text = replace_mentions(text)
    text = replace_emojis(text)
    return collapse_whitespace(text)


def normalize_aggressive(text: str) -> str:
    """Variante texto_agresivo: minúsculas sin tildes (grupo de control ablación)."""
    text = normalize_for_model(text)
    text = text.lower()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    return collapse_whitespace(text)


def has_accents_or_enye(text: str) -> bool:
    return bool(re.search(r"[áéíóúñÁÉÍÓÚÑ]", text))


def has_uppercase(text: str) -> bool:
    return any(c.isupper() for c in text)


def has_punctuation(text: str) -> bool:
    return bool(re.search(r"[^\w\s]", text, re.UNICODE))


def has_mention_literal(text: str) -> bool:
    return bool(MENTION_PATTERN.search(text))


def has_url_literal(text: str) -> bool:
    return bool(URL_PATTERN.search(text) or EMAIL_PATTERN.search(text))


def has_emoji_raw(text: str) -> bool:
    return bool(emoji.emoji_count(text))
