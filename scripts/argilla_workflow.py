#!/usr/bin/env python
"""Shared Argilla configuration and data preparation for ISSUE-04."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORDS = ROOT / "data/interim/canonical_records.csv"
GUIDELINES_PATH = ROOT / "docs/annotation_guidelines_v2.md"
SDK_VERSION = "2.8.0"
SCHEMA_VERSION = "argilla-issue04-v1"
TAXONOMY_VERSION = "2.0.0"
MIN_SUBMITTED = 2
EVALUATION_FILES = {"data/raw/val.csv", "data/raw/test.csv"}
TRAIN_FILE = "data/raw/train.csv"
CANDIDATE_FILES = {
    "data/raw/lenguaje_ofensivo.csv",
    "data/raw/thenewdata_X.csv",
}

LABELS = {
    "0": "0 · No Tóxico",
    "1": "1 · Lenguaje Ofensivo",
    "2": "2 · Discurso de Odio",
    "3": "3 · Amenazas/Violencia",
}
IDENTITY_TARGETS = {
    "origen_nacionalidad": "Origen / nacionalidad",
    "raza_etnia": "Raza / etnia",
    "religion": "Religión",
    "genero_sexualidad": "Género / sexualidad",
    "discapacidad": "Discapacidad",
    "clase_socioeconomica": "Clase socioeconómica",
    "afiliacion_politica": "Afiliación política",
}
VIOLENCE_TYPES = {
    "direct_threat": "Amenaza directa",
    "incitement": "Incitación",
    "wish": "Deseo de daño",
    "endorsement": "Aprobación de violencia",
    "report": "Reporte / noticia",
    "quoted_condemned": "Cita condenada",
    "metaphor": "Metáfora",
    "none": "Ninguna",
}
YES_NO_UNKNOWN = {"yes": "Sí", "no": "No", "unknown": "Desconocido"}


@dataclass(frozen=True)
class Campaign:
    slug: str
    title: str
    dataset_name: str
    restricted: bool = False
    suggestions_allowed: bool = True


CAMPAIGNS = {
    "legacy_train_real": Campaign(
        "legacy-train-real", "Reales antiguos de train", "cde_legacy_train_real_tax2"
    ),
    "legacy_train_synthetic": Campaign(
        "legacy-train-synthetic", "Sintéticos antiguos de train", "cde_legacy_synthetic_tax2"
    ),
    "new_candidates": Campaign(
        "new-candidates", "Candidatos nuevos en cuarentena", "cde_new_candidates_tax2"
    ),
    "new_synthetic": Campaign(
        "new-synthetic", "Sintéticos nuevos", "cde_new_synthetic_tax2"
    ),
    "calibration": Campaign(
        "calibration-v2-0", "Calibración de la guía v2.0.0", "cde_calibration_tax2", False, False
    ),
    "historical_val_audit": Campaign(
        "historical-val-audit", "Auditoría restringida: val histórico", "cde_audit_val_tax2", True, False
    ),
    "historical_test_audit": Campaign(
        "historical-test-audit", "Auditoría restringida: test histórico", "cde_audit_test_tax2", True, False
    ),
}


class WorkflowError(RuntimeError):
    """A user-correctable error in an import/export operation."""


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE entries without evaluating shell syntax."""
    if not path.exists():
        return
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise WorkflowError(f"{path}:{line_number}: se esperaba KEY=VALUE")
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if value[:1] == value[-1:] and value.startswith(("'", '"')):
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def json_list(row: dict[str, str], field: str) -> list[str]:
    value = row.get(field, "")
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise WorkflowError(f"Valor JSON inválido en {field} para {row.get('canonical_id')}") from exc
    if not isinstance(parsed, list):
        raise WorkflowError(f"{field} debe ser una lista JSON para {row.get('canonical_id')}")
    return [str(item) for item in parsed]


def load_canonical_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise WorkflowError(f"No existe el archivo canónico: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "canonical_id",
            "text_original",
            "text_normalized",
            "origin_types_json",
            "source_files_json",
            "source_record_refs_json",
            "source_record_ids_json",
            "source_row_hashes_json",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise WorkflowError(f"Faltan columnas en {path}: {', '.join(sorted(missing))}")
        rows = list(reader)
    ids = [row["canonical_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise WorkflowError(f"canonical_id duplicado en {path}; regenera ISSUE-02 antes de importar")
    return rows


def source_files(row: dict[str, str]) -> set[str]:
    return set(json_list(row, "source_files_json"))


def source_roles(row: dict[str, str]) -> set[str]:
    return set(json_list(row, "source_roles_json"))


def is_synthetic(row: dict[str, str]) -> bool:
    origins = [origin.casefold() for origin in json_list(row, "origin_types_json")]
    return any(origin.startswith(("sintetico", "synthetic")) for origin in origins)


def campaign_rows(rows: Iterable[dict[str, str]], name: str) -> list[dict[str, str]]:
    if name not in CAMPAIGNS:
        raise WorkflowError(f"Campaña desconocida: {name}")
    if name == "calibration":
        raise WorkflowError("La campaña calibration se carga desde docs/calibration_cases_v2.md")
    selected: list[dict[str, str]] = []
    for row in rows:
        files = source_files(row)
        has_eval = bool(files.intersection(EVALUATION_FILES))
        has_train = TRAIN_FILE in files
        has_candidate = bool(files.intersection(CANDIDATE_FILES))
        roles = source_roles(row)
        is_diagnostic = "diagnostic_battery" in roles
        synthetic = is_synthetic(row)

        # Evaluation membership always wins, even if a duplicate has a train reference.
        if (
            name == "historical_val_audit"
            and "data/raw/val.csv" in files
            and "data/raw/test.csv" not in files
        ):
            selected.append(row)
        elif (
            name == "historical_test_audit"
            and "data/raw/test.csv" in files
            and "data/raw/val.csv" not in files
        ):
            selected.append(row)
        elif has_eval:
            continue
        elif name == "legacy_train_real" and has_train and not synthetic:
            selected.append(row)
        elif name == "legacy_train_synthetic" and has_train and synthetic:
            selected.append(row)
        elif name == "new_candidates" and has_candidate and not has_train and not synthetic:
            selected.append(row)
        elif (
            name == "new_synthetic"
            and synthetic
            and not has_train
            and not has_candidate
            and not is_diagnostic
            and "candidate_unvalidated" not in roles
        ):
            # Future synthetic releases can enter once ISSUE-02 adds them to the canonical view.
            selected.append(row)
    selected.sort(key=lambda item: item["canonical_id"])
    return selected


def load_calibration_cases() -> list[dict[str, str]]:
    """Read the 40 proposed calibration examples without exposing expected labels."""
    if not GUIDELINES_PATH.with_name("calibration_cases_v2.md").exists():
        raise WorkflowError("No existe docs/calibration_cases_v2.md")
    path = GUIDELINES_PATH.with_name("calibration_cases_v2.md")
    rows: list[dict[str, str]] = []
    pattern = re.compile(r"^\|\s*(C[0-3]-\d{2})\s*\|")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not pattern.match(line):
            continue
        cells = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            raise WorkflowError(f"Fila de calibración mal formada: {line}")
        case_id, text_and_context = cells[0], cells[1]
        text_sha = hashlib.sha256(text_and_context.encode("utf-8")).hexdigest()
        canonical_id = "CAL-" + hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:20]
        normalized = " ".join(unicodedata.normalize("NFC", text_and_context).casefold().split())
        rows.append(
            {
                "canonical_id": canonical_id,
                "text_original": text_and_context,
                "text_normalized": normalized,
                "text_sha256": text_sha,
                "origin_types_json": '["sintetico_calibracion"]',
                "source_files_json": json.dumps(["docs/calibration_cases_v2.md"], ensure_ascii=False),
                "source_record_refs_json": json.dumps([f"docs/calibration_cases_v2.md#{case_id}"], ensure_ascii=False),
                "source_record_ids_json": json.dumps([case_id], ensure_ascii=False),
                "source_row_hashes_json": json.dumps([text_sha]),
                "source_roles_json": '["calibration_case"]',
                "source_observation_count": "1",
                "platform_originals_json": "[]",
                "platform_canonical": "",
                "url_originals_json": "[]",
                "author_group_ids_json": "[]",
                "thread_group_ids_json": "[]",
            }
        )
    if len(rows) != 40:
        raise WorkflowError(f"Se esperaban 40 casos propuestos; se encontraron {len(rows)}")
    return sorted(rows, key=lambda row: row["canonical_id"])


def external_id(campaign_name: str, canonical_id: str) -> str:
    campaign = CAMPAIGNS[campaign_name]
    return f"cde-issue04-{campaign.slug}-{canonical_id}"


def metadata_for(row: dict[str, str], campaign_name: str, suggestion: dict[str, Any] | None) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "campaign": CAMPAIGNS[campaign_name].slug,
        "canonical_id": row["canonical_id"],
        "taxonomy_version": TAXONOMY_VERSION,
        "guideline_version": TAXONOMY_VERSION,
        "text_sha256": row.get("text_sha256", ""),
        "source_files_json": row.get("source_files_json", "[]"),
        "source_record_refs_json": row.get("source_record_refs_json", "[]"),
        "source_record_ids_json": row.get("source_record_ids_json", "[]"),
        "source_row_hashes_json": row.get("source_row_hashes_json", "[]"),
        "source_roles_json": row.get("source_roles_json", "[]"),
        "source_observation_count": row.get("source_observation_count", ""),
        "source_fingerprint": source_fingerprint(row),
        "platform_originals_json": row.get("platform_originals_json", "[]"),
        "platform_canonical": row.get("platform_canonical", ""),
        "origin_types_json": row.get("origin_types_json", "[]"),
        "url_originals_json": row.get("url_originals_json", "[]"),
        "author_group_ids_json": row.get("author_group_ids_json", "[]"),
        "thread_group_ids_json": row.get("thread_group_ids_json", "[]"),
    }
    if suggestion:
        metadata.update(
            {
                "suggestion_checkpoint": suggestion["checkpoint"],
                "suggestion_agent": suggestion["agent"],
                "suggestion_score": str(suggestion["score"]),
            }
        )
    return metadata


def source_fingerprint(row: dict[str, str]) -> str:
    parts = [
        row.get("canonical_id", ""),
        row.get("text_original", ""),
        row.get("text_normalized", ""),
        row.get("text_sha256", ""),
        row.get("source_files_json", "[]"),
        row.get("source_record_refs_json", "[]"),
        row.get("source_record_ids_json", "[]"),
        row.get("source_row_hashes_json", "[]"),
    ]
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def load_suggestions(path: Path | None, campaign_name: str) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    if not CAMPAIGNS[campaign_name].suggestions_allowed:
        raise WorkflowError("Las campañas de auditoría/calibración ciega no aceptan sugerencias")
    if not path.exists():
        raise WorkflowError(f"No existe el archivo de sugerencias: {path}")
    suggestions: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise WorkflowError(f"JSONL inválido en {path}:{line_number}") from exc
            canonical_id = str(item.get("canonical_id", "")).strip()
            label = str(item.get("label", item.get("label_pred", ""))).strip()
            agent = str(item.get("agent", "")).strip()
            checkpoint = str(item.get("checkpoint", "")).strip()
            try:
                score = float(item["score"])
            except (KeyError, TypeError, ValueError) as exc:
                raise WorkflowError(f"score numérico requerido en {path}:{line_number}") from exc
            if not canonical_id or label not in LABELS or not agent or not checkpoint:
                raise WorkflowError(
                    f"Se requieren canonical_id, label 0..3, agent y checkpoint en {path}:{line_number}"
                )
            if not 0.0 <= score <= 1.0:
                raise WorkflowError(f"score fuera de [0,1] en {path}:{line_number}")
            if canonical_id in suggestions:
                raise WorkflowError(f"Sugerencia duplicada para {canonical_id} en {path}:{line_number}")
            suggestions[canonical_id] = {
                "label": label,
                "agent": agent,
                "checkpoint": checkpoint,
                "score": score,
            }
    return suggestions


def make_argilla_settings(rg: Any, campaign_name: str, client: Any | None = None) -> Any:
    campaign = CAMPAIGNS[campaign_name]
    guidelines = GUIDELINES_PATH.read_text(encoding="utf-8")
    campaign_options = [item.slug for item in CAMPAIGNS.values()]
    suggestion_guidance = (
        "Las etiquetas de origen y decisiones anteriores están ocultas. Si aparece una sugerencia, "
        "Argilla la marca como propuesta de modelo; no es una respuesta humana ni una adjudicación. "
        "Registre su decisión de forma independiente."
        if campaign.suggestions_allowed
        else "Las etiquetas de origen, sugerencias del modelo y decisiones anteriores están ocultas. "
        "Registre cada respuesta de forma independiente; no adjudique desde esta interfaz."
    )
    settings = rg.Settings(
        guidelines=(
            f"Guía de anotación del proyecto, versión {TAXONOMY_VERSION}; esquema {SCHEMA_VERSION}.\n\n"
            f"Campaña: {campaign.title}. Responda usando el texto y el contexto mostrado. "
            f"{suggestion_guidance}\n\n"
            f"{guidelines}"
        ),
        fields=[
            rg.TextField(name="original_text", title="Texto original", required=True, client=client),
            rg.TextField(name="normalized_text", title="Texto normalizado", required=True, client=client),
        ],
        questions=[
            rg.LabelQuestion(
                name="annotation_status",
                title="Estado de anotación",
                labels={"anotado": "Anotado", "contexto_insuficiente": "Contexto insuficiente"},
                description="Marque contexto insuficiente si falta información necesaria; no lo convierta en clase 0.",
                required=True,
                client=client,
            ),
            rg.LabelQuestion(
                name="label",
                title="Etiqueta primaria",
                labels=LABELS,
                description="Elija solo una etiqueta cuando el estado sea anotado. Deje sin respuesta si falta contexto.",
                required=False,
                client=client,
            ),
            rg.TextQuestion(name="subclass", title="Subclase", required=False, client=client),
            rg.MultiLabelQuestion(
                name="identity_targets",
                title="Grupo(s) objetivo",
                labels=IDENTITY_TARGETS,
                description="Marque solo los grupos atacados por identidad; puede elegir varios.",
                required=False,
                client=client,
            ),
            rg.LabelQuestion(
                name="dialect",
                title="Dialecto / variedad",
                labels={
                    "salvadoreno": "Español salvadoreño",
                    "centroamericano_otro": "Otra variedad centroamericana",
                    "espanol_otro": "Otra variedad del español",
                    "otro_idioma": "Otro idioma",
                    "desconocido": "Desconocido / sin señal suficiente",
                },
                required=False,
                client=client,
            ),
            rg.LabelQuestion(
                name="quoted_or_reported",
                title="¿Cita o reporte?",
                labels=YES_NO_UNKNOWN,
                required=False,
                client=client,
            ),
            rg.LabelQuestion(
                name="negated", title="¿Hay negación relevante?", labels=YES_NO_UNKNOWN, required=False, client=client
            ),
            rg.LabelQuestion(
                name="violence_type",
                title="Tipo de violencia",
                labels=VIOLENCE_TYPES,
                required=False,
                client=client,
            ),
            rg.LabelQuestion(
                name="quality",
                title="Calidad del registro",
                labels={
                    "usable": "Utilizable",
                    "review": "Requiere revisión de fuente/contexto",
                    "insufficient_context": "Contexto insuficiente",
                    "duplicate_conflict": "Posible duplicado o conflicto",
                },
                required=False,
                client=client,
            ),
            rg.TextQuestion(
                name="insufficient_context_reason",
                title="Motivo de contexto insuficiente",
                description="Indique qué información concreta falta, si aplica.",
                required=False,
                client=client,
            ),
            rg.TextQuestion(
                name="rationale",
                title="Justificación contextual",
                description="Explique en una frase la evidencia y decisión, sin adjudicar respuestas ajenas.",
                required=False,
                client=client,
            ),
        ],
        metadata=[
            rg.TermsMetadataProperty(
                name="campaign",
                title="Campaña",
                options=campaign_options,
                visible_for_annotators=False,
                client=client,
            )
        ],
        allow_extra_metadata=True,
        distribution=rg.TaskDistribution(min_submitted=MIN_SUBMITTED),
    )
    return settings


def argilla_client(rg: Any) -> Any:
    configured_version = getattr(rg, "__version__", "")
    if configured_version != SDK_VERSION:
        raise WorkflowError(
            f"Se requiere argilla=={SDK_VERSION}; se encontró {configured_version or 'versión desconocida'}. "
            "Instale el extra opcional: pip install -e '.[argilla]'"
        )
    api_url = os.environ.get("ARGILLA_API_URL", "").strip()
    api_key = os.environ.get("ARGILLA_API_KEY", "").strip()
    if not api_url or not api_key:
        raise WorkflowError("Configure ARGILLA_API_URL y ARGILLA_API_KEY en el entorno o .env.argilla")
    return rg.Argilla(api_url=api_url, api_key=api_key)


def workspace_for(campaign_name: str, explicit: str | None = None) -> str:
    campaign = CAMPAIGNS[campaign_name]
    if campaign.restricted:
        key = "ARGILLA_AUDIT_WORKSPACE"
        workspace = os.environ.get(key, "").strip()
        if not workspace:
            raise WorkflowError(f"Configure {key}; auditorías restringidas no aceptan un workspace arbitrario")
        if explicit and explicit != workspace:
            raise WorkflowError(f"--workspace debe coincidir con {key} para campañas de auditoría")
        return workspace
    if explicit:
        return explicit
    key = "ARGILLA_WORKSPACE"
    workspace = os.environ.get(key, "").strip()
    if not workspace:
        raise WorkflowError(f"Configure {key}; no se usará el workspace general por defecto")
    return workspace


def get_or_create_dataset(rg: Any, client: Any, campaign_name: str, workspace: str) -> tuple[Any, bool]:
    campaign = CAMPAIGNS[campaign_name]
    dataset = client.datasets(name=campaign.dataset_name, workspace=workspace)
    if dataset is None:
        dataset = rg.Dataset(
            name=campaign.dataset_name,
            workspace=workspace,
            settings=make_argilla_settings(rg, campaign_name, client=client),
            client=client,
        ).create()
        return dataset, True

    dataset.settings.get()
    distribution = dataset.settings.distribution
    if distribution is None or distribution.min_submitted != MIN_SUBMITTED:
        raise WorkflowError(
            f"{campaign.dataset_name} no tiene min_submitted={MIN_SUBMITTED}; "
            "no importe ni abra la campaña hasta corregir la distribución antes de anotar"
        )
    field_names = {field.name for field in dataset.settings.fields}
    question_names = {question.name for question in dataset.settings.questions}
    required_fields = {"original_text", "normalized_text"}
    required_questions = {
        "annotation_status",
        "label",
        "subclass",
        "identity_targets",
        "dialect",
        "quoted_or_reported",
        "negated",
        "violence_type",
        "quality",
        "insufficient_context_reason",
        "rationale",
    }
    if not required_fields.issubset(field_names) or not required_questions.issubset(question_names):
        raise WorkflowError(
            f"El esquema de {campaign.dataset_name} no coincide con {SCHEMA_VERSION}; "
            "cree una nueva campaña con sufijo de versión"
        )
    if SCHEMA_VERSION not in (dataset.settings.guidelines or "") or TAXONOMY_VERSION not in (
        dataset.settings.guidelines or ""
    ):
        raise WorkflowError(
            f"La guía de {campaign.dataset_name} no coincide con taxonomía/esquema vigente; "
            "cree una nueva campaña con sufijo de versión"
        )
    return dataset, False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

