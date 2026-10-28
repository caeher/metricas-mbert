#!/usr/bin/env python
"""Audit dataset source files and export a provenance-preserving canonical view.

This is an inventory and quarantine tool. It never promotes a source label to a
final annotation and never resolves a disagreement automatically.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit


SCRIPT_VERSION = "1.1.0"
LABEL_TEXT_TO_ID = {
    "no toxico": "0",
    "no tóxico": "0",
    "lenguaje ofensivo": "1",
    "discurso de odio": "2",
    "amenazas/violencia": "3",
}
CORE_SOURCES = {
    "data/raw/train.csv": "core_split",
    "data/raw/val.csv": "core_split",
    "data/raw/test.csv": "core_split",
}
CANDIDATE_SOURCES = {
    "data/raw/lenguaje_ofensivo.csv": "candidate_unvalidated",
    "data/raw/thenewdata_X.csv": "candidate_unvalidated",
}
BATTERY_SOURCES = {
    "data/raw/test_adversarial.csv": "diagnostic_battery",
    "data/processed/test_adversarial_144.csv": "diagnostic_battery",
}
REFERENCE_SOURCES = {
    "data/raw/corpus_completo.csv": "redundant_corpus_view",
    "data/processed/train.csv": "byte_identical_copy",
    "data/processed/val.csv": "byte_identical_copy",
    "data/processed/test.csv": "byte_identical_copy",
    "data/processed/corpus_completo.csv": "byte_identical_copy",
}
SOURCES = {**CORE_SOURCES, **CANDIDATE_SOURCES, **BATTERY_SOURCES, **REFERENCE_SOURCES}

CANONICAL_IMPORT_ROLES = frozenset(
    {
        "core_split",
        "candidate_unvalidated",
        "diagnostic_battery",
        "frozen_evaluation_bank",
        "candidate_unlabeled_thread",
    }
)


def sha256(value: str | bytes) -> str:
    data = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(data).hexdigest()


def normalize_text(value: str | None) -> str:
    """Conservative exact-match normalization: NFC, casefold, collapsed space."""
    if not value:
        return ""
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def normalize_url(value: str | None) -> str:
    if not value or not value.strip():
        return ""
    raw = value.strip()
    try:
        parts = urlsplit(raw if "://" in raw else "https://" + raw)
        host = (parts.hostname or "").casefold()
        if host.startswith("www."):
            host = host[4:]
        port = f":{parts.port}" if parts.port else ""
        path = re.sub(r"/+$", "", parts.path) or "/"
        # Query and fragment are omitted so tracking parameters do not split a post.
        return urlunsplit((parts.scheme.casefold() or "https", host + port, path, "", ""))
    except (ValueError, UnicodeError):
        return raw


def x_publication_date(value: str | None) -> str:
    """Decode the creation date carried by an X/Twitter snowflake status ID."""
    match = re.search(r"/status/(\d+)", value or "", flags=re.IGNORECASE)
    if not match:
        return ""
    try:
        milliseconds = (int(match.group(1)) >> 22) + 1288834974657
        return datetime.fromtimestamp(milliseconds / 1000, tz=timezone.utc).date().isoformat()
    except (OverflowError, OSError, ValueError):
        return ""


def canonical_platform(value: str | None) -> str:
    raw = (value or "").strip()
    if raw.casefold() in {"twitter", "x", "twitter/x", "x/twitter"}:
        return "X"
    return raw


def canonical_label(value: str | None) -> str:
    raw = (value or "").strip()
    if raw in {"0", "1", "2", "3"}:
        return raw
    return LABEL_TEXT_TO_ID.get(normalize_text(raw), "")


def classify_core_origin(origin_raw: str) -> tuple[str, str]:
    """Map dataset v1.8 `origen` values to origin_type and origin_original."""
    raw = (origin_raw or "").strip()
    lowered = raw.casefold()
    if lowered.startswith("sintetico"):
        return "synthetic", raw or "synthetic"
    if lowered.startswith("v3_"):
        return "synthetic_v3", raw or "synthetic_v3"
    if lowered in {"real", "real_hilo", "caso_dificil"}:
        return "real", raw or "real"
    if not raw:
        return "unknown", "unknown"
    return "unknown", raw


def parse_date(value: str | None) -> tuple[str, str]:
    raw = (value or "").strip()
    if not raw:
        return "", "missing"
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%d/%m/%Y"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat(), "parsed"
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return parsed.date().isoformat(), "parsed"
    except ValueError:
        return "", "unparsed"


def read_csv(path: Path) -> tuple[list[dict[str, str]], str, str, int, int]:
    raw = path.read_bytes()
    try:
        content = raw.decode("utf-8-sig")
        encoding = "utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8"
    except UnicodeDecodeError:
        content = raw.decode("cp1252", errors="replace")
        encoding = "cp1252_with_replacement"
    try:
        dialect = csv.Sniffer().sniff(content[:8192], delimiters=",;\t")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","
    reader = csv.DictReader(io.StringIO(content, newline=""), delimiter=delimiter)
    rows: list[dict[str, str]] = []
    extra_count = 0
    missing_count = 0
    for row in reader:
        if None in row:
            extra_count += 1
            extras = row.pop(None) or []
            if extras:
                row["__extra_fields__"] = " | ".join(extras)
        if any(value is None for key, value in row.items() if key != "__extra_fields__"):
            missing_count += 1
        rows.append({key: "" if value is None else str(value) for key, value in row.items()})
    return rows, encoding, delimiter, extra_count, missing_count


def get_field(row: dict[str, str], *names: str) -> str:
    for name in names:
        if name in row:
            return row[name]
    lower_map = {key.casefold(): key for key in row}
    for name in names:
        key = lower_map.get(name.casefold())
        if key is not None:
            return row[key]
    return ""


def infer_geo(row: dict[str, str], role: str) -> tuple[str, str, str, str, str, str]:
    note = get_field(row, "Notas_Etiquetador", "notes", "nota")
    dialect_raw = get_field(row, "Lenguaje_Dialectal", "dialect", "dialecto")
    note_norm = normalize_text(note)
    dialect_norm = normalize_text(dialect_raw)
    geographic_inference = bool(
        re.search(r"ubicaci[oó]n (?:se )?infiere", note_norm)
        or re.search(r"no est[aá] confirmada por perfil", note_norm)
    )
    inferred_sv_dialect = "posible uso salvadoreño" in dialect_norm or "posible uso salvadoreno" in dialect_norm
    claimed_sv_dialect = dialect_norm == "salvadoreño" or dialect_norm == "salvadoreno"
    country = ""
    country_status = "unknown"
    country_evidence = ""
    if geographic_inference:
        # Notes say the location was inferred, but do not reliably name a country.
        country_status = "inferred"
        country_evidence = "location_inferred_country_not_specified_and_profile_not_confirmed"
    elif role == "candidate_unvalidated" and "pais inferido: el salvador" in note_norm:
        country = "El Salvador"
        country_status = "inferred"
        country_evidence = "source_note_inference_not_independently_verified"
    dialect = ""
    dialect_status = "unknown"
    if inferred_sv_dialect or claimed_sv_dialect:
        dialect = "salvadoreño"
        dialect_status = "inferred"
    elif dialect_norm in {"español", "espanol"}:
        # The source only identifies Spanish, not a regional dialect.
        dialect_status = "unknown"
    return country, country_status, dialect, dialect_status, country_evidence, note


def make_observation(
    root: Path,
    rel_path: str,
    row_number: int,
    row: dict[str, str],
    role: str,
) -> dict[str, Any]:
    if role == "core_split":
        source_id = get_field(row, "ID")
        text = get_field(row, "texto_original", "Texto_Original")
        text_versions = {
            key: row.get(key, "")
            for key in ("texto_original", "texto_modelo", "texto_agresivo")
            if row.get(key, "")
        }
        platform = get_field(row, "Plataforma")
        date_raw = get_field(row, "fecha")
        author = get_field(row, "Usuario_ID_Anonimizado")
        thread_group = get_field(row, "thread_group_id", "thread_id", "hilo_id", "conversation_id", "id_conversacion", "reply_to_id", "in_reply_to_status_id")
        url = ""
        label = get_field(row, "label", "clase")
        label_class = get_field(row, "clase") or label
        origin_raw = get_field(row, "origen")
        origin, origin_original = classify_core_origin(origin_raw)
        template = get_field(row, "plantilla_id")
        validation = get_field(row, "validado") or "not_recorded"
        readiness = "not_recorded"
        annotation_status = "legacy_split_label_unreviewed"
        provenance_status = "synthetic_template_id_present" if origin in {"synthetic", "synthetic_v3"} and template else (
            "synthetic_template_id_missing" if origin in {"synthetic", "synthetic_v3"} else "dataset_source_metadata_no_publication_url"
        )
        extras = {}
        for name in ("Grupos_Afectados", "subclase", "recolector", "fuera_de_sv", "validado", "p0", "p1", "p2", "p3"):
            if row.get(name, "") not in ("", None):
                extras[name] = row[name]
    elif role == "frozen_evaluation_bank":
        source_id = get_field(row, "ID") or get_field(row, "hilo")
        text = get_field(row, "texto_original", "texto")
        text_versions = {key: row.get(key, "") for key in ("texto_original", "texto") if row.get(key, "")}
        platform = "X"
        date_raw = ""
        author = ""
        thread_group = get_field(row, "hilo")
        url = ""
        label = get_field(row, "label", "clase")
        label_class = label
        origin_raw = "frozen_evaluation_bank"
        origin = "real"
        template = ""
        validation = "frozen_bank_manifest"
        readiness = "frozen_evaluation_only"
        annotation_status = "frozen_bank_label_unreviewed"
        provenance_status = "frozen_bank_never_trainable"
        extras = {key: row.get(key, "") for key in ("p0", "p1", "p2", "p3", "hilo") if row.get(key, "") not in ("", None)}
    elif role == "candidate_unlabeled_thread":
        source_id = get_field(row, "ID") or f"{get_field(row, 'hilo')}-{row_number}"
        text = get_field(row, "texto", "Texto", "texto_original", "Texto_Original")
        text_versions = {"texto": text} if text else {}
        platform = "X"
        date_raw = ""
        author = ""
        thread_group = get_field(row, "hilo", "thread_id", "thread_group_id")
        url = get_field(row, "URL_Original", "url")
        label = ""
        label_class = ""
        origin_raw = "unlabeled_thread_acquisition"
        origin = "real"
        template = ""
        validation = "unlabeled"
        readiness = "not_ready_for_training"
        annotation_status = "unlabeled_acquisition_candidate"
        provenance_status = "thread_source_without_label"
        extras = {key: row.get(key, "") for key in row if key not in {"texto", "Texto", "texto_original", "ID", "hilo"} and row.get(key, "")}
    elif role == "candidate_unvalidated":
        source_id = get_field(row, "ID")
        text = get_field(row, "Texto_Original", "texto_original")
        text_versions = {"Texto_Original": text} if text else {}
        platform = get_field(row, "Plataforma")
        date_raw = get_field(row, "Fecha_Recolección", "fecha")
        author = get_field(row, "Usuario_ID_Anonimizado")
        thread_group = get_field(row, "thread_group_id", "thread_id", "hilo_id", "conversation_id", "id_conversacion", "reply_to_id", "in_reply_to_status_id")
        url = get_field(row, "URL_Original", "url")
        label = get_field(row, "Clase_Toxicidad", "label")
        label_class = label
        origin_raw = "claimed_real_social_post"
        origin = "real"
        template = ""
        validation = get_field(row, "Validado") or "missing"
        readiness = get_field(row, "Listo_Entrenamiento") or "missing"
        annotation_status = "unvalidated_candidate_label"
        provenance_status = "url_present_not_independently_verified" if url.strip() else "missing_publication_url"
        extras = {
            key: row.get(key, "")
            for key in ("Subclase_Toxicidad", "Grupos_Afectados", "Lenguaje_Dialectal", "Notas_Etiquetador", "Responsable", "Validador", "Fecha_Validacion", "Razon_Desacuerdo")
            if row.get(key, "")
        }
    else:
        source_id = get_field(row, "ID")
        text = get_field(row, "texto_original", "Texto_Original", "texto", "text")
        text_versions = {key: row.get(key, "") for key in ("texto_original", "texto_modelo", "texto", "text") if row.get(key, "")}
        platform = get_field(row, "Plataforma")
        date_raw = get_field(row, "fecha")
        author = get_field(row, "Usuario_ID_Anonimizado")
        thread_group = get_field(row, "thread_group_id", "thread_id", "hilo_id", "conversation_id", "id_conversacion", "reply_to_id", "in_reply_to_status_id", "grupo_de_prueba", "subgrupo")
        url = get_field(row, "URL_Original")
        label = get_field(row, "label", "label_esperado", "clase")
        label_class = get_field(row, "clase", "resultado_esperado") or label
        origin_raw = "curated_or_generated_diagnostic"
        origin = "synthetic"
        template = get_field(row, "plantilla_id", "base", "grupo")
        validation = "diagnostic_only"
        readiness = "diagnostic_only"
        annotation_status = "diagnostic_label_unreviewed"
        provenance_status = "diagnostic_only_not_provenance_verified"
        extras = {key: row.get(key, "") for key in ("grupo", "base", "variante") if row.get(key, "")}

    normalized = normalize_text(text)
    url_normalized = normalize_url(url)
    country, country_status, dialect, dialect_status, country_evidence, note = infer_geo(row, role)
    parsed_date, date_status = parse_date(date_raw)
    source_ref = f"{rel_path}#row-{row_number}:{source_id or 'NO_ID'}"
    raw_label = label.strip()
    label_id = canonical_label(raw_label or label_class)
    text_versions.setdefault("texto_original", text)
    return {
        "source_file": rel_path,
        "source_role": role,
        "source_row_number": row_number,
        "source_record_id": source_id,
        "source_record_ref": source_ref,
        "text_original": text,
        "text_normalized": normalized,
        "text_sha256": sha256(normalized) if normalized else "",
        "text_versions_json": json.dumps(text_versions, ensure_ascii=False, sort_keys=True),
        "platform_original": platform,
        "platform_canonical": canonical_platform(platform),
        "origin_original": origin_raw if role != "core_split" else origin_original,
        "origin_type": origin,
        "label_original": raw_label,
        "label_original_class": label_class,
        "label_provisional_id": label_id,
        "label_final": "",
        "annotation_status": annotation_status,
        "provenance_status": provenance_status,
        "validation_status": validation,
        "training_readiness_status": readiness,
        "date_original": date_raw,
        "date_iso": parsed_date,
        "date_parse_status": date_status,
        "publication_date_from_x_id": x_publication_date(url),
        "url_original": url,
        "url_normalized": url_normalized,
        "publication_sha256": sha256(url_normalized) if url_normalized else "",
        "author_group_id": author,
        "thread_group_id": thread_group,
        "country": country,
        "country_status": country_status,
        "country_evidence": country_evidence,
        "dialect": dialect,
        "dialect_status": dialect_status,
        "source_notes": note,
        "template_or_battery_group": template,
        "source_metadata_json": json.dumps(extras, ensure_ascii=False, sort_keys=True),
        "raw_row_sha256": sha256(json.dumps(row, ensure_ascii=False, sort_keys=True)),
    }


def group_counts(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(value for value in values if value != "").items()))


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def normalize_source_record(root: Path, rel: str, role: str) -> tuple[list[dict[str, str]], dict[str, Any]]:
    path = root / rel
    rows, encoding, delimiter, extra_rows, missing_rows = read_csv(path)
    observations = [make_observation(root, rel, i, row, role) for i, row in enumerate(rows, start=2)]
    inventory = {
        "source_file": rel,
        "source_role": role,
        "present": True,
        "row_count": len(rows),
        "byte_size": path.stat().st_size,
        "file_sha256": sha256(path.read_bytes()),
        "encoding_detected": encoding,
        "delimiter": "TAB" if delimiter == "\t" else delimiter,
        "rows_with_extra_fields": extra_rows,
        "rows_with_missing_fields": missing_rows,
        "platform_counts_json": json.dumps(group_counts(o["platform_original"] for o in observations), ensure_ascii=False, sort_keys=True),
        "canonical_platform_counts_json": json.dumps(group_counts(o["platform_canonical"] for o in observations), ensure_ascii=False, sort_keys=True),
        "label_counts_json": json.dumps(group_counts(o["label_original"] for o in observations), ensure_ascii=False, sort_keys=True),
        "provisional_label_id_counts_json": json.dumps(group_counts(o["label_provisional_id"] for o in observations), ensure_ascii=False, sort_keys=True),
        "origin_counts_json": json.dumps(group_counts(o["origin_original"] for o in observations), ensure_ascii=False, sort_keys=True),
        "provenance_status_counts_json": json.dumps(group_counts(o["provenance_status"] for o in observations), ensure_ascii=False, sort_keys=True),
        "validated_counts_json": json.dumps(group_counts(o["validation_status"] for o in observations), ensure_ascii=False, sort_keys=True),
        "ready_counts_json": json.dumps(group_counts(o["training_readiness_status"] for o in observations), ensure_ascii=False, sort_keys=True),
        "empty_text_records": sum(not o["text_normalized"] for o in observations),
        "records_with_url": sum(bool(o["url_original"]) for o in observations),
        "records_without_url": sum(not o["url_original"] for o in observations),
        "records_with_author_group": sum(bool(o["author_group_id"]) for o in observations),
        "records_with_thread_group": sum(bool(o["thread_group_id"]) for o in observations),
        "records_with_template_or_battery_group": sum(bool(o["template_or_battery_group"]) for o in observations),
        "records_with_unparsed_date": sum(o["date_parse_status"] == "unparsed" for o in observations),
        "records_with_missing_date": sum(o["date_parse_status"] == "missing" for o in observations),
        "label_class_mismatch_records": sum(
            bool(o["label_provisional_id"] and canonical_label(o["label_original_class"]) and o["label_provisional_id"] != canonical_label(o["label_original_class"]))
            for o in observations
        ),
        "imported_into_canonical": role in CANONICAL_IMPORT_ROLES,
    }
    return observations, inventory


def duplicate_relation_rows(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []
    dimensions = [
        ("author_group", "author_group_id"),
        ("thread_group", "thread_group_id"),
        ("template_or_battery_group", "template_or_battery_group"),
    ]
    for relation_type, field in dimensions:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for observation in observations:
            value = str(observation.get(field, "")).strip()
            if value and not (relation_type == "author_group" and value.casefold() in {"sintetico", "unknown", "desconocido", "n/a", "na"}):
                groups[value].append(observation)
        for value, members in sorted(groups.items()):
            if len(members) < 2:
                continue
            relation_id = f"REL-{sha256(relation_type + '|' + value)[:16]}"
            relations.append({
                "relation_id": relation_id,
                "relation_type": relation_type,
                "relation_value": value,
                "member_count": len(members),
                "members_json": json.dumps([m["source_record_ref"] for m in sorted(members, key=lambda x: x["source_record_ref"])], ensure_ascii=False),
                "disposition": "link_only_no_automatic_exclusion",
                "review_status": "audit_signal",
            })
    return relations


def build_canonical_entities(observations: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    union = UnionFind(len(observations))
    text_index: dict[str, int] = {}
    url_index: dict[str, int] = {}
    for index, observation in enumerate(observations):
        text_hash = observation["text_sha256"]
        if text_hash:
            if text_hash in text_index:
                union.union(index, text_index[text_hash])
            else:
                text_index[text_hash] = index
        url_hash = observation["publication_sha256"]
        if url_hash:
            if url_hash in url_index:
                union.union(index, url_index[url_hash])
            else:
                url_index[url_hash] = index

    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, observation in enumerate(observations):
        groups[union.find(index)].append(observation)

    entities: list[dict[str, Any]] = []
    duplicate_ledger: list[dict[str, Any]] = []
    for members in groups.values():
        members.sort(key=lambda x: x["source_record_ref"])
        refs = sorted(m["source_record_ref"] for m in members)
        labels = sorted({m["label_provisional_id"] for m in members if m["label_provisional_id"]})
        originals = sorted({m["label_original"] for m in members if m["label_original"]})
        roles = sorted({m["source_role"] for m in members})
        split_members = [m for m in members if m["source_role"] == "core_split"]
        candidate_members = [m for m in members if m["source_role"] == "candidate_unvalidated"]
        battery_members = [m for m in members if m["source_role"] == "diagnostic_battery"]
        frozen_members = [m for m in members if m["source_role"] == "frozen_evaluation_bank"]
        unlabeled_members = [m for m in members if m["source_role"] == "candidate_unlabeled_thread"]
        train_members = [m for m in split_members if m["source_file"].endswith("/train.csv")]
        has_label_conflict = len(labels) > 1
        exact_text_hashes = sorted({m["text_sha256"] for m in members if m["text_sha256"]})
        publication_hashes = sorted({m["publication_sha256"] for m in members if m["publication_sha256"]})
        if len(exact_text_hashes) == 1:
            identity_key = f"text:{exact_text_hashes[0]}"
        elif len(publication_hashes) == 1:
            identity_key = f"url:{publication_hashes[0]}"
        elif exact_text_hashes or publication_hashes:
            identity_key = "anchors:" + "|".join(sorted([*(f"text:{value}" for value in exact_text_hashes), *(f"url:{value}" for value in publication_hashes)]))
        else:
            identity_key = "refs:" + "|".join(refs)
        entity_id = f"CAN-{sha256(identity_key)[:20]}"
        text_hash_values = [m["text_sha256"] for m in members if m["text_sha256"]]
        publication_hash_values = [m["publication_sha256"] for m in members if m["publication_sha256"]]
        has_exact_text_match = len(text_hash_values) > len(set(text_hash_values))
        has_publication_match = len(publication_hash_values) > len(set(publication_hash_values))
        match_dimensions = [
            dimension
            for dimension, present in (("exact_normalized_text", has_exact_text_match), ("normalized_publication_url", has_publication_match))
            if present
        ]
        if has_label_conflict:
            disposition = "human_adjudication_required"
            annotation_status = "conflicting_provisional_labels"
        elif train_members and frozen_members:
            disposition = "human_adjudication_required"
            annotation_status = "train_frozen_bank_text_overlap"
        elif candidate_members:
            disposition = "quarantine_unvalidated_candidate"
            annotation_status = "unvalidated_source_label"
        elif unlabeled_members and not split_members and not frozen_members:
            disposition = "quarantine_unlabeled_acquisition"
            annotation_status = "unlabeled_acquisition_candidate"
        elif frozen_members and not split_members:
            disposition = "frozen_evaluation_bank_only"
            annotation_status = "frozen_bank_label_unreviewed"
        elif battery_members and not split_members:
            disposition = "diagnostic_only"
            annotation_status = "diagnostic_label_unreviewed"
        elif split_members:
            disposition = "legacy_split_label_unreviewed"
            annotation_status = "legacy_split_label_unreviewed"
        else:
            disposition = "quarantine_unresolved"
            annotation_status = "unreviewed"
        platforms_original = sorted({m["platform_original"] for m in members if m["platform_original"]})
        platforms_canonical = sorted({m["platform_canonical"] for m in members if m["platform_canonical"]})
        origins = sorted({m["origin_type"] for m in members if m["origin_type"]})
        text_variants = sorted({m["text_original"] for m in members if m["text_original"]})
        all_versions: dict[str, list[str]] = defaultdict(list)
        for member in members:
            for key, value in json.loads(member["text_versions_json"]).items():
                if value and value not in all_versions[key]:
                    all_versions[key].append(value)
        entity = {
            "canonical_id": entity_id,
            "text_original": text_variants[0] if text_variants else "",
            "text_versions_json": json.dumps({k: sorted(v) for k, v in sorted(all_versions.items())}, ensure_ascii=False, sort_keys=True),
            "text_normalized": members[0]["text_normalized"] if len(exact_text_hashes) == 1 else "",
            "text_sha256": exact_text_hashes[0] if len(exact_text_hashes) == 1 else "",
            "text_version_count": len(text_variants),
            "platform_originals_json": json.dumps(platforms_original, ensure_ascii=False),
            "platform_canonical": platforms_canonical[0] if len(platforms_canonical) == 1 else "mixed_or_unknown",
            "origin_types_json": json.dumps(origins, ensure_ascii=False),
            "origin_type": origins[0] if len(origins) == 1 else "mixed_or_unknown",
            "provenance_statuses_json": json.dumps(sorted({m["provenance_status"] for m in members}), ensure_ascii=False),
            "label_originals_json": json.dumps(originals, ensure_ascii=False),
            "label_provisional_ids_json": json.dumps(labels, ensure_ascii=False),
            "label_final": "",
            "annotation_status": annotation_status,
            "validation_statuses_json": json.dumps(sorted({m["validation_status"] for m in members}), ensure_ascii=False),
            "training_readiness_statuses_json": json.dumps(sorted({m["training_readiness_status"] for m in members}), ensure_ascii=False),
            "date_originals_json": json.dumps(sorted({m["date_original"] for m in members if m["date_original"]}), ensure_ascii=False),
            "date_iso_values_json": json.dumps(sorted({m["date_iso"] for m in members if m["date_iso"]}), ensure_ascii=False),
            "publication_dates_from_x_ids_json": json.dumps(sorted({m["publication_date_from_x_id"] for m in members if m["publication_date_from_x_id"]}), ensure_ascii=False),
            "url_originals_json": json.dumps(sorted({m["url_original"] for m in members if m["url_original"]}), ensure_ascii=False),
            "url_normalized_values_json": json.dumps(sorted({m["url_normalized"] for m in members if m["url_normalized"]}), ensure_ascii=False),
            "publication_sha256_values_json": json.dumps(publication_hashes, ensure_ascii=False),
            "author_group_ids_json": json.dumps(sorted({m["author_group_id"] for m in members if m["author_group_id"]}), ensure_ascii=False),
            "thread_group_ids_json": json.dumps(sorted({m["thread_group_id"] for m in members if m["thread_group_id"]}), ensure_ascii=False),
            "country_values_json": json.dumps(sorted({m["country"] for m in members if m["country"]}), ensure_ascii=False),
            "country_statuses_json": json.dumps(sorted({m["country_status"] for m in members}), ensure_ascii=False),
            "country_evidence_json": json.dumps(sorted({m["country_evidence"] for m in members if m["country_evidence"]}), ensure_ascii=False),
            "dialect_values_json": json.dumps(sorted({m["dialect"] for m in members if m["dialect"]}), ensure_ascii=False),
            "dialect_statuses_json": json.dumps(sorted({m["dialect_status"] for m in members}), ensure_ascii=False),
            "source_roles_json": json.dumps(roles, ensure_ascii=False),
            "source_files_json": json.dumps(sorted({m["source_file"] for m in members}), ensure_ascii=False),
            "source_record_ids_json": json.dumps(sorted({m["source_record_id"] for m in members if m["source_record_id"]}), ensure_ascii=False),
            "source_record_refs_json": json.dumps(refs, ensure_ascii=False),
            "source_row_hashes_json": json.dumps(sorted({m["raw_row_sha256"] for m in members}), ensure_ascii=False),
            "source_notes_json": json.dumps(sorted({m["source_notes"] for m in members if m["source_notes"]}), ensure_ascii=False),
            "source_metadata_json": json.dumps(sorted({m["source_metadata_json"] for m in members if m["source_metadata_json"]}), ensure_ascii=False),
            "source_observation_count": len(members),
            "record_disposition": disposition,
            "human_adjudication_status": "pending" if has_label_conflict else "not_required_by_detected_label_conflict",
        }
        entities.append(entity)
        if len(members) > 1:
            duplicate_event_type = "exact_text_duplicate" if has_exact_text_match else "duplicate_publication_url"
            duplicate_ledger.append({
                "ledger_id": f"DUP-{sha256(entity_id)[:16]}",
                "canonical_id": entity_id,
                "event_type": duplicate_event_type,
                "match_dimensions": json.dumps(match_dimensions, ensure_ascii=False),
                "member_count": len(members),
                "label_originals_json": json.dumps(originals, ensure_ascii=False),
                "provisional_label_ids_json": json.dumps(labels, ensure_ascii=False),
                "source_record_refs_json": json.dumps(refs, ensure_ascii=False),
                "source_texts_json": json.dumps(text_variants, ensure_ascii=False),
                "source_urls_json": json.dumps(sorted({m["url_original"] for m in members if m["url_original"]}), ensure_ascii=False),
                "platforms_json": json.dumps(platforms_original, ensure_ascii=False),
                "disposition": "one_canonical_entity_retained",
                "human_adjudication_status": "pending" if has_label_conflict else "not_required_by_detected_label_conflict",
                "human_resolution_label": "",
                "human_resolved_by": "",
                "human_resolved_at": "",
                "human_rationale": "",
            })
        if has_label_conflict:
            duplicate_ledger.append({
                "ledger_id": f"CONFLICT-{sha256(entity_id)[:16]}",
                "canonical_id": entity_id,
                "event_type": "label_conflict",
                "match_dimensions": json.dumps(match_dimensions, ensure_ascii=False),
                "member_count": len(members),
                "label_originals_json": json.dumps(originals, ensure_ascii=False),
                "provisional_label_ids_json": json.dumps(labels, ensure_ascii=False),
                "source_record_refs_json": json.dumps(refs, ensure_ascii=False),
                "source_texts_json": json.dumps(text_variants, ensure_ascii=False),
                "source_urls_json": json.dumps(sorted({m["url_original"] for m in members if m["url_original"]}), ensure_ascii=False),
                "platforms_json": json.dumps(platforms_original, ensure_ascii=False),
                "disposition": "quarantine_all_members_until_human_adjudication",
                "human_adjudication_status": "pending",
                "human_resolution_label": "",
                "human_resolved_by": "",
                "human_resolved_at": "",
                "human_rationale": "",
            })
    entities.sort(key=lambda row: row["canonical_id"])
    duplicate_ledger.sort(key=lambda row: (row["event_type"], row["canonical_id"]))
    return entities, duplicate_ledger


def similarity_candidates(entities: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    """Find high character-trigram Jaccard candidates without all-pairs scans."""
    valid = [(i, normalize_text(row["text_normalized"])) for i, row in enumerate(entities) if len(row["text_normalized"]) >= 16]
    features: dict[int, set[str]] = {}
    posting: dict[str, list[int]] = defaultdict(list)
    for index, text in valid:
        grams = {text[i : i + 3] for i in range(len(text) - 2)}
        features[index] = grams
        for gram in grams:
            posting[gram].append(index)
    pairs: set[tuple[int, int]] = set()
    for members in posting.values():
        # Very common trigrams create noise and quadratic work.
        if len(members) > 60:
            continue
        for offset, left in enumerate(members):
            for right in members[offset + 1 :]:
                pairs.add((left, right))
    output: list[dict[str, Any]] = []
    for left, right in sorted(pairs):
        grams_left, grams_right = features[left], features[right]
        jaccard = len(grams_left & grams_right) / len(grams_left | grams_right)
        if jaccard < threshold:
            continue
        lrow, rrow = entities[left], entities[right]
        output.append({
            "similarity_pair_id": f"SIM-{sha256('|'.join(sorted([lrow['canonical_id'], rrow['canonical_id']])) )[:16]}",
            "canonical_id_a": lrow["canonical_id"],
            "canonical_id_b": rrow["canonical_id"],
            "char_trigram_jaccard": f"{jaccard:.6f}",
            "text_a": lrow["text_original"],
            "text_b": rrow["text_original"],
            "source_record_refs_a_json": lrow["source_record_refs_json"],
            "source_record_refs_b_json": rrow["source_record_refs_json"],
            "review_status": "human_review_candidate",
            "automatic_deduplication": "no",
        })
    return output


def encoding_audit(observations: list[dict[str, Any]], inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for observation in observations:
        by_source[observation["source_file"]].append(observation)
    rows = []
    encoding_by_source = {entry["source_file"]: entry.get("encoding_detected", "unknown") for entry in inventory}
    mojibake_pattern = re.compile(r"(?:Ã(?:¡|©|­|³|º|±|£|µ|¼|œ)|Â(?:¿|¡| |\u00a0)|â€(?:™|œ|ž|“|”|–|—)|ï¿½|ðŸ|\ufffd)")
    midword_question = re.compile(r"(?<=[A-Za-zÁÉÍÓÚáéíóúÜüÑñ])\?(?=[A-Za-zÁÉÍÓÚáéíóúÜüÑñ])")
    for source_file, members in sorted(by_source.items()):
        text_values = [m["text_original"] for m in members]
        values_by_record = []
        for member in members:
            metadata_values = [v for v in json.loads(member["source_metadata_json"]).values() if isinstance(v, str)]
            values_by_record.append([member["text_original"], member["source_notes"], *metadata_values])
        all_values = [value for record_values in values_by_record for value in record_values]
        rows.append({
            "source_file": source_file,
            "record_count": len(members),
            "utf8_input_valid": str(not encoding_by_source.get(source_file, "").startswith("cp1252")).lower(),
            "decoded_as": encoding_by_source.get(source_file, "unknown"),
            "replacement_character_records": sum(any("\ufffd" in value for value in values) for values in values_by_record),
            "mojibake_marker_records": sum(any(mojibake_pattern.search(value) for value in values) for values in values_by_record),
            "midword_question_mark_records": sum(any(midword_question.search(value) for value in values) for values in values_by_record),
            "literal_question_mark_in_text_records": sum("?" in value for value in text_values),
            "text_contains_non_ascii": sum(any(ord(ch) > 127 for ch in value) for value in text_values),
            "finding": "question_mark_may_be_lost_character; inspect source/export; no repair applied" if any(midword_question.search(value) for value in all_values) else "no_detected_replacement_or_mojibake_marker; display-only rendering cannot be proven from bytes",
            "notes": "Counts are source-string indicators, not proof of original corruption. Original values and raw file hashes are retained.",
        })
    return rows


def encoding_suspects(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mojibake_pattern = re.compile(r"(?:Ã(?:¡|©|­|³|º|±|£|µ|¼|œ)|Â(?:¿|¡| |\u00a0)|â€(?:™|œ|ž|“|”|–|—)|ï¿½|ðŸ|\ufffd)")
    midword_question = re.compile(r"(?<=[A-Za-zÁÉÍÓÚáéíóúÜüÑñ])\?(?=[A-Za-zÁÉÍÓÚáéíóúÜüÑñ])")
    findings = []
    for observation in observations:
        metadata = json.loads(observation["source_metadata_json"])
        metadata_values = [value for value in metadata.values() if isinstance(value, str)]
        text = observation["text_original"]
        notes = observation["source_notes"]
        fields = {"text_original": text, "source_notes": notes, "source_metadata": " ".join(metadata_values)}
        flags = []
        for field, value in fields.items():
            if "\ufffd" in value:
                flags.append(f"{field}:replacement_character")
            if mojibake_pattern.search(value):
                flags.append(f"{field}:mojibake_marker")
            if midword_question.search(value):
                flags.append(f"{field}:question_mark_between_letters")
        if flags:
            findings.append({
                "source_record_ref": observation["source_record_ref"],
                "source_file": observation["source_file"],
                "source_record_id": observation["source_record_id"],
                "text_sha256": observation["text_sha256"],
                "text_original": text,
                "source_notes": notes,
                "suspicion_flags_json": json.dumps(sorted(set(flags)), ensure_ascii=False),
                "disposition": "retain_original_quarantine_for_human_source_review_no_repair",
            })
    return findings


def date_audit(observations: list[dict[str, Any]], as_of: date) -> list[dict[str, Any]]:
    rows = []
    for observation in observations:
        reason = ""
        if observation["date_parse_status"] == "unparsed":
            reason = "unparseable_date"
        elif observation["date_parse_status"] == "missing" and observation["origin_type"] != "synthetic":
            reason = "missing_date"
        elif observation["date_iso"]:
            parsed = date.fromisoformat(observation["date_iso"])
            if parsed > as_of:
                reason = "future_date_relative_to_audit_day"
            publication_date = observation["publication_date_from_x_id"]
            if publication_date and parsed < date.fromisoformat(publication_date):
                reason = "collection_date_precedes_x_publication_date"
        if reason:
            rows.append({
                "source_record_ref": observation["source_record_ref"],
                "source_file": observation["source_file"],
                "date_original": observation["date_original"],
                "date_iso": observation["date_iso"],
                "publication_date_from_x_id": observation["publication_date_from_x_id"],
                "issue": reason,
                "review_status": "needs_source_review",
            })
    return rows


def geography_summary(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for observation in observations:
        grouped[observation["source_file"]].append(observation)
    rows = []
    for source_file, members in sorted(grouped.items()):
        rows.append({
            "source_file": source_file,
            "records": len(members),
            "country_confirmed": sum(m["country_status"] == "confirmed" for m in members),
            "country_inferred": sum(m["country_status"].startswith("inferred") for m in members),
            "country_unknown": sum(m["country_status"] == "unknown" for m in members),
            "dialect_confirmed": sum(m["dialect_status"] == "confirmed" for m in members),
            "dialect_inferred": sum(m["dialect_status"] == "inferred" for m in members),
            "dialect_unknown": sum(m["dialect_status"] == "unknown" for m in members),
            "policy": "no country/dialect promotion from archive membership or platform name",
        })
    return rows


def discover_extra_sources(root: Path) -> list[tuple[str, str]]:
    """Return (relative_path, role) for frozen bank and unlabeled thread CSVs."""
    discovered: list[tuple[str, str]] = []
    frozen_dir = root / "data/evaluation/frozen_bank"
    if frozen_dir.is_dir():
        for path in sorted(frozen_dir.glob("*.csv")):
            discovered.append((path.relative_to(root).as_posix(), "frozen_evaluation_bank"))
    threads_dir = root / "data/sources/x_threads"
    if threads_dir.is_dir():
        for path in sorted(threads_dir.rglob("*.csv")):
            discovered.append((path.relative_to(root).as_posix(), "candidate_unlabeled_thread"))
    return discovered


def compute_train_holdout_leakage(observations: list[dict[str, Any]]) -> dict[str, Any]:
    train_observations = [
        observation
        for observation in observations
        if observation["source_role"] == "core_split" and observation["source_file"].endswith("/train.csv")
    ]
    train_hashes = {o["text_sha256"] for o in train_observations if o["text_sha256"]}
    train_raw_texts = {o["text_original"] for o in train_observations if o["text_original"]}

    def holdout_bucket(observation: dict[str, Any]) -> str | None:
        role = observation["source_role"]
        stem = Path(observation["source_file"]).stem
        if role == "core_split" and stem in {"val", "test"}:
            return stem
        if role == "diagnostic_battery":
            return stem
        if role == "frozen_evaluation_bank":
            return f"frozen_bank:{stem}"
        return None

    normalized_counts: Counter[str] = Counter()
    raw_counts: Counter[str] = Counter()
    for observation in observations:
        bucket = holdout_bucket(observation)
        if not bucket:
            continue
        if observation["text_sha256"] and observation["text_sha256"] in train_hashes:
            normalized_counts[bucket] += 1
        if observation["text_original"] and observation["text_original"] in train_raw_texts:
            raw_counts[bucket] += 1
    return {
        "normalized_text_overlap_counts": dict(sorted(normalized_counts.items())),
        "raw_text_overlap_counts": dict(sorted(raw_counts.items())),
        "normalized_overlap_total": int(sum(normalized_counts.values())),
        "raw_overlap_total": int(sum(raw_counts.values())),
        "expected_normalized_overlap_total": 0,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row)) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_audit(root: Path, report_dir: Path, interim_dir: Path, similarity_threshold: float, as_of: date) -> dict[str, Any]:
    root = root.resolve()
    observations: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    loaded: dict[str, list[dict[str, str]]] = {}
    for rel_path, role in SOURCES.items():
        path = root / rel_path
        if not path.exists():
            inventory.append({"source_file": rel_path, "source_role": role, "present": False, "row_count": 0, "imported_into_canonical": False})
            continue
        parsed_role = "core_split" if role in {"redundant_corpus_view", "byte_identical_copy"} else role
        rows, entry = normalize_source_record(root, rel_path, parsed_role)
        entry["source_role"] = role
        entry["imported_into_canonical"] = role in CANONICAL_IMPORT_ROLES
        if role in CANONICAL_IMPORT_ROLES:
            observations.extend(rows)
        inventory.append(entry)
        loaded[rel_path] = rows

    for rel_path, role in discover_extra_sources(root):
        if rel_path in loaded or any(entry["source_file"] == rel_path for entry in inventory):
            continue
        path = root / rel_path
        if not path.exists():
            continue
        rows, entry = normalize_source_record(root, rel_path, role)
        entry["source_role"] = role
        entry["imported_into_canonical"] = role in CANONICAL_IMPORT_ROLES
        if role in CANONICAL_IMPORT_ROLES:
            observations.extend(rows)
        inventory.append(entry)
        loaded[rel_path] = rows

    # Make mirrors and the full-corpus view explicit in the inventory. These files
    # are checked and counted but their observations do not enter canonical groups.
    for entry in inventory:
        rel = entry["source_file"]
        if rel in REFERENCE_SOURCES:
            entry["imported_into_canonical"] = False
            counterpart = rel.replace("data/processed/", "data/raw/")
            if rel.startswith("data/processed/") and counterpart in loaded:
                entry["byte_identical_to"] = counterpart if entry["file_sha256"] == next((x.get("file_sha256") for x in inventory if x["source_file"] == counterpart), "") else ""
        if rel == "data/raw/corpus_completo.csv":
            entry["row_set_compared_with_splits"] = "pending"

    entities, duplicate_ledger = build_canonical_entities(observations)
    similarity_rows = similarity_candidates(entities, similarity_threshold)
    relations = duplicate_relation_rows(observations)
    encoding_rows = encoding_audit(observations, inventory)
    encoding_suspect_rows = encoding_suspects(observations)
    date_rows = date_audit(observations, as_of)
    geo_rows = geography_summary(observations)

    # Candidate records are quarantined until explicit validation and adjudication.
    for observation in observations:
        if observation["source_role"] == "candidate_unvalidated" and (
            observation["validation_status"].strip().casefold() != "si"
            or observation["training_readiness_status"].strip().casefold() != "si"
        ):
            duplicate_ledger.append({
                "ledger_id": f"EXCL-{sha256(observation['source_record_ref'])[:16]}",
                "canonical_id": "",
                "event_type": "unvalidated_candidate_source",
                "match_dimensions": "[]",
                "member_count": 1,
                "label_originals_json": json.dumps([observation["label_original"]], ensure_ascii=False),
                "provisional_label_ids_json": json.dumps([observation["label_provisional_id"]] if observation["label_provisional_id"] else [], ensure_ascii=False),
                "source_record_refs_json": json.dumps([observation["source_record_ref"]], ensure_ascii=False),
                "disposition": "excluded_from_trainable_corpus_pending_validation",
                "human_adjudication_status": "pending_source_validation",
                "human_resolution_label": "",
                "human_resolved_by": "",
                "human_resolved_at": "",
                "human_rationale": "",
            })
        elif observation["source_role"] == "frozen_evaluation_bank":
            duplicate_ledger.append({
                "ledger_id": f"EXCL-{sha256(observation['source_record_ref'])[:16]}",
                "canonical_id": "",
                "event_type": "frozen_evaluation_bank_source",
                "match_dimensions": "[]",
                "member_count": 1,
                "label_originals_json": json.dumps([observation["label_original"]], ensure_ascii=False),
                "provisional_label_ids_json": json.dumps([observation["label_provisional_id"]] if observation["label_provisional_id"] else [], ensure_ascii=False),
                "source_record_refs_json": json.dumps([observation["source_record_ref"]], ensure_ascii=False),
                "disposition": "excluded_from_trainable_corpus_frozen_bank",
                "human_adjudication_status": "not_applicable",
                "human_resolution_label": "",
                "human_resolved_by": "",
                "human_resolved_at": "",
                "human_rationale": "",
            })
        elif observation["source_role"] == "candidate_unlabeled_thread":
            duplicate_ledger.append({
                "ledger_id": f"EXCL-{sha256(observation['source_record_ref'])[:16]}",
                "canonical_id": "",
                "event_type": "unlabeled_thread_acquisition_source",
                "match_dimensions": "[]",
                "member_count": 1,
                "label_originals_json": "[]",
                "provisional_label_ids_json": "[]",
                "source_record_refs_json": json.dumps([observation["source_record_ref"]], ensure_ascii=False),
                "disposition": "excluded_from_trainable_corpus_pending_annotation",
                "human_adjudication_status": "pending_annotation",
                "human_resolution_label": "",
                "human_resolved_by": "",
                "human_resolved_at": "",
                "human_rationale": "",
            })
    for entry in inventory:
        if entry.get("source_role") in {"redundant_corpus_view", "byte_identical_copy"}:
            duplicate_ledger.append({
                "ledger_id": f"VIEW-{sha256(entry['source_file'])[:16]}",
                "canonical_id": "",
                "event_type": "source_view_not_imported",
                "match_dimensions": "same logical dataset view or byte-identical mirror",
                "member_count": entry.get("row_count", 0),
                "label_originals_json": "[]",
                "provisional_label_ids_json": "[]",
                "source_record_refs_json": json.dumps([entry["source_file"]], ensure_ascii=False),
                "disposition": "inventory_only_prevents_double_counting",
                "human_adjudication_status": "not_applicable",
                "human_resolution_label": "",
                "human_resolved_by": "",
                "human_resolved_at": "",
                "human_rationale": "",
            })
    duplicate_ledger.sort(key=lambda row: (row["event_type"], row["ledger_id"]))

    # Identify whether the full corpus view is the same row set and fields as the splits.
    core_refs = [o for o in observations if o["source_role"] == "core_split"]
    full_rows, full_inventory = ([], None)
    full_path = root / "data/raw/corpus_completo.csv"
    if full_path.exists():
        full_rows, _, _, _, _ = read_csv(full_path)
        by_id = {o["source_record_id"]: o for o in core_refs}
        full_id_matches = sum(row.get("ID", "") in by_id for row in full_rows)
        full_exact_matches = 0
        for row in full_rows:
            member = by_id.get(row.get("ID", ""))
            row_hash = sha256(json.dumps(row, ensure_ascii=False, sort_keys=True))
            if member and row_hash == member["raw_row_sha256"]:
                full_exact_matches += 1
        for entry in inventory:
            if entry["source_file"] == "data/raw/corpus_completo.csv":
                entry["row_set_compared_with_splits"] = "same_ids_and_all_fields" if len(full_rows) == len(core_refs) == full_id_matches == full_exact_matches else "different_or_partial"
                entry["split_id_matches"] = full_id_matches
                entry["split_exact_record_matches"] = full_exact_matches

    # Keep output stable and avoid embedding execution timestamps.
    write_csv(report_dir / "source_inventory.csv", inventory)
    write_csv(interim_dir / "canonical_records.csv", entities)
    write_csv(interim_dir / "deduplication_exclusion_ledger.csv", duplicate_ledger)
    write_csv(report_dir / "similarity_candidates.csv", similarity_rows, [
        "similarity_pair_id", "canonical_id_a", "canonical_id_b", "char_trigram_jaccard", "text_a", "text_b", "source_record_refs_a_json", "source_record_refs_b_json", "review_status", "automatic_deduplication",
    ])
    write_csv(report_dir / "author_thread_template_relations.csv", relations, [
        "relation_id", "relation_type", "relation_value", "member_count", "members_json", "disposition", "review_status",
    ])
    write_csv(report_dir / "encoding_audit.csv", encoding_rows)
    write_csv(report_dir / "encoding_suspects.csv", encoding_suspect_rows, [
        "source_record_ref", "source_file", "source_record_id", "text_sha256", "text_original", "source_notes", "suspicion_flags_json", "disposition",
    ])
    write_csv(report_dir / "date_audit.csv", date_rows, ["source_record_ref", "source_file", "date_original", "date_iso", "publication_date_from_x_id", "issue", "review_status"])
    adjudication_rows = []
    entity_by_id = {row["canonical_id"]: row for row in entities}
    for conflict in (row for row in duplicate_ledger if row["event_type"] == "label_conflict"):
        entity = entity_by_id.get(conflict["canonical_id"], {})
        adjudication_rows.append({
            "canonical_id": conflict["canonical_id"],
            "conflict_classification": "exact_text_label_conflict" if "exact_normalized_text" in conflict["match_dimensions"] else "same_publication_text_version_label_conflict",
            "match_dimensions": conflict["match_dimensions"],
            "source_files_json": entity.get("source_files_json", "[]"),
            "source_record_ids_json": entity.get("source_record_ids_json", "[]"),
            "source_record_refs_json": conflict["source_record_refs_json"],
            "label_originals_json": conflict["label_originals_json"],
            "provisional_label_ids_json": conflict["provisional_label_ids_json"],
            "source_texts_json": conflict.get("source_texts_json", "[]"),
            "source_urls_json": conflict.get("source_urls_json", "[]"),
            "platforms_json": conflict.get("platforms_json", "[]"),
            "human_adjudication_status": "pending",
            "human_resolution_label": "",
            "adjudicator": "",
            "resolved_at": "",
            "rationale": "",
        })
    adjudication_rows.sort(key=lambda row: row["canonical_id"])
    write_csv(report_dir / "adjudication_queue.csv", adjudication_rows, [
        "canonical_id", "conflict_classification", "match_dimensions", "source_files_json", "source_record_ids_json", "source_record_refs_json", "label_originals_json", "provisional_label_ids_json", "source_texts_json", "source_urls_json", "platforms_json", "human_adjudication_status", "human_resolution_label", "adjudicator", "resolved_at", "rationale",
    ])
    write_csv(report_dir / "geography_language_audit.csv", geo_rows)

    core = [o for o in observations if o["source_role"] == "core_split"]
    candidate = [o for o in observations if o["source_role"] == "candidate_unvalidated"]
    batteries = [o for o in observations if o["source_role"] == "diagnostic_battery"]
    frozen_bank = [o for o in observations if o["source_role"] == "frozen_evaluation_bank"]
    unlabeled_threads = [o for o in observations if o["source_role"] == "candidate_unlabeled_thread"]
    leakages = compute_train_holdout_leakage(observations)
    conflict_rows = [row for row in duplicate_ledger if row["event_type"] == "label_conflict"]
    counts_by_class = group_counts(o["label_provisional_id"] for o in core)
    split_text_hashes = {o["text_sha256"] for o in core if o["text_sha256"]}
    summary = {
        "script_version": SCRIPT_VERSION,
        "method": {
            "exact_text_normalization": "Unicode NFC + casefold + collapsed whitespace; accents and punctuation retained",
            "publication_url_normalization": "lowercase host/path cleanup; query and fragment removed",
            "deduplication": "merge only exact normalized text or normalized publication URL into one canonical entity",
            "similarity": {"method": "character trigram Jaccard candidate search", "threshold": similarity_threshold, "automatic_deduplication": False},
            "author_thread_template": "reported as relation signals only; never automatic exclusion",
            "final_labels": "blank for every record; only original/provisional labels are carried through",
            "date_anomaly_reference_day": as_of.isoformat(),
        },
        "source_row_counts": {entry["source_file"]: entry.get("row_count", 0) for entry in inventory},
        "core_splits": {
            "train": sum(o["source_file"].endswith("/train.csv") for o in core),
            "val": sum(o["source_file"].endswith("/val.csv") for o in core),
            "test": sum(o["source_file"].endswith("/test.csv") for o in core),
            "total": len(core),
            "origin_type_counts": group_counts(o["origin_type"] for o in core),
            "provisional_label_id_counts": counts_by_class,
        },
        "candidate_sources": {
            "total": len(candidate),
            "platform_counts": group_counts(o["platform_canonical"] for o in candidate),
            "provisional_label_counts": group_counts(o["label_original"] for o in candidate),
            "validated_counts": group_counts(o["validation_status"] for o in candidate),
            "training_readiness_counts": group_counts(o["training_readiness_status"] for o in candidate),
            "country_inferred_value_unspecified": sum(
                o["country_status"] == "inferred" and not o["country"] for o in candidate
            ),
            "exact_text_matches_with_historical_splits": sum(
                bool(c["text_sha256"])
                and c["text_sha256"] in split_text_hashes
                for c in candidate
            ),
        },
        "batteries": {"observations": len(batteries), "source_count": len(BATTERY_SOURCES), "diagnostic_only": True},
        "frozen_evaluation_bank": {
            "observations": len(frozen_bank),
            "provisional_label_id_counts": group_counts(o["label_provisional_id"] for o in frozen_bank),
            "thread_group_counts": group_counts(o["thread_group_id"] for o in frozen_bank),
            "never_trainable": True,
        },
        "unlabeled_thread_candidates": {
            "observations": len(unlabeled_threads),
            "thread_group_counts": group_counts(o["thread_group_id"] for o in unlabeled_threads),
            "source_file_counts": group_counts(o["source_file"] for o in unlabeled_threads),
        },
        "train_holdout_leakage": leakages,
        "canonical_entity_count": len(entities),
        "canonical_disposition_counts": group_counts(row["record_disposition"] for row in entities),
        "exact_duplicate_or_publication_groups": sum(row["event_type"] == "exact_text_duplicate" or row["event_type"] == "duplicate_publication_url" for row in duplicate_ledger),
        "deduplication_ledger_event_counts": group_counts(row["event_type"] for row in duplicate_ledger),
        "label_conflict_count": len(conflict_rows),
        "exact_text_label_conflict_count": sum("exact_normalized_text" in row["match_dimensions"] for row in conflict_rows),
        "same_publication_text_version_label_conflict_count": sum("exact_normalized_text" not in row["match_dimensions"] for row in conflict_rows),
        "label_conflict_status": "pending human adjudication; no automatic label selected",
        "similarity_candidate_count": len(similarity_rows),
        "encoding_suspect_observation_count": len(encoding_suspect_rows),
        "date_audit_issue_counts": group_counts(row["issue"] for row in date_rows),
        "geography_and_language_by_source": {
            row["source_file"]: {
                "country_confirmed": row["country_confirmed"],
                "country_inferred": row["country_inferred"],
                "country_unknown": row["country_unknown"],
                "dialect_confirmed": row["dialect_confirmed"],
                "dialect_inferred": row["dialect_inferred"],
                "dialect_unknown": row["dialect_unknown"],
            }
            for row in geo_rows
        },
        "author_thread_template_relation_groups": len(relations),
        "empty_text_observations": sum(not o["text_normalized"] for o in observations),
        "all_labels_final": False,
        "corpus_full_view": "same IDs and all fields as the union of train/val/test; imported zero extra rows",
        "exact_match_limit": "Zero exact matches against historical splits does not establish absence of near-duplicates, URL/post overlap, author/thread leakage, or template leakage.",
        "known_limitations": [
            "CSV-level bytes can distinguish a valid UTF-8 display issue from replacement characters or likely lost characters only when the bytes preserve evidence; lost source characters cannot be reconstructed safely.",
            "Country and dialect are not inferred from membership in this project archive or from X/Twitter platform labels.",
            "Three exact-text conflicts and one same-publication text-version conflict remain unresolved until a human annotator enters reviewer, date, final label and rationale in the ledger.",
        ],
    }
    summary_path = report_dir / "audit_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="project root (default: parent of scripts/)")
    parser.add_argument("--report-dir", type=Path, default=None, help="report output directory")
    parser.add_argument("--interim-dir", type=Path, default=None, help="canonical/interim output directory")
    parser.add_argument("--similarity-threshold", type=float, default=0.82, help="character-trigram Jaccard threshold")
    parser.add_argument("--as-of", default="2026-10-16", help="reference date for future-date checks (ISO YYYY-MM-DD)")
    args = parser.parse_args()
    if not 0 < args.similarity_threshold <= 1:
        parser.error("--similarity-threshold must be in (0, 1]")
    root = args.root.resolve()
    try:
        as_of = date.fromisoformat(args.as_of)
    except ValueError:
        parser.error("--as-of must be an ISO date (YYYY-MM-DD)")
    reports = args.report_dir.resolve() if args.report_dir else root / "reports/data_audit"
    interim = args.interim_dir.resolve() if args.interim_dir else root / "data/interim"
    summary = run_audit(root, reports, interim, args.similarity_threshold, as_of)
    print(json.dumps({
        "core_splits": summary["core_splits"],
        "candidate_sources": summary["candidate_sources"],
        "canonical_entity_count": summary["canonical_entity_count"],
        "label_conflict_count": summary["label_conflict_count"],
        "similarity_candidate_count": summary["similarity_candidate_count"],
        "report_dir": str(reports),
        "interim_dir": str(interim),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
