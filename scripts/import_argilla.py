#!/usr/bin/env python
"""Import a disjoint project campaign into Argilla 2.8.0."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from argilla_workflow import (
    CAMPAIGNS,
    DEFAULT_RECORDS,
    GUIDELINES_PATH,
    MIN_SUBMITTED,
    SCHEMA_VERSION,
    SDK_VERSION,
    TAXONOMY_VERSION,
    WorkflowError,
    argilla_client,
    campaign_rows,
    external_id,
    get_or_create_dataset,
    load_calibration_cases,
    load_canonical_rows,
    load_env_file,
    load_suggestions,
    metadata_for,
    source_fingerprint,
    utc_now,
    workspace_for,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", choices=sorted(CAMPAIGNS), help="Campaña que se importa.")
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS, help="CSV canónico de ISSUE-02.")
    parser.add_argument("--suggestions", type=Path, help="Predicciones opcionales en JSONL, una por canonical_id.")
    parser.add_argument("--workspace", help="Workspace Argilla; se infiere de .env.argilla si se omite.")
    parser.add_argument("--env-file", type=Path, default=Path(".env.argilla"), help="Archivo local ignorado por Git.")
    parser.add_argument("--limit", type=int, help="Limita la campaña para un piloto; conserva IDs estables.")
    parser.add_argument("--dry-run", action="store_true", help="Imprime el plan sin conectar con Argilla.")
    parser.add_argument("--restore-export", type=Path, help="Restaura una exportación portable JSONL.")
    parser.add_argument("--batch-size", type=int, default=256, help="Tamaño de lote del SDK (1–1000).")
    return parser.parse_args()


def suggestion_for_record(rg: Any, suggestion: dict[str, Any] | None) -> list[Any]:
    if suggestion is None:
        return []
    return [
        rg.Suggestion(
            question_name="label",
            value=suggestion["label"],
            score=suggestion["score"],
            agent=suggestion["agent"],
            type="model",
        )
    ]


def import_record(rg: Any, campaign_name: str, row: dict[str, str], suggestion: dict[str, Any] | None) -> Any:
    fields = {
        "original_text": row.get("text_original", ""),
        "normalized_text": row.get("text_normalized", ""),
    }
    if not fields["original_text"]:
        raise WorkflowError(f"Texto original vacío en {row.get('canonical_id')}; no se importó")
    metadata = metadata_for(row, campaign_name, suggestion)
    metadata["source_fingerprint"] = source_fingerprint(row)
    return rg.Record(
        id=external_id(campaign_name, row["canonical_id"]),
        fields=fields,
        metadata=metadata,
        suggestions=suggestion_for_record(rg, suggestion),
    )


def response_users(record: Any) -> set[str]:
    return {str(response.user_id) for response in record.responses}


def existing_suggestion(record: Any) -> dict[str, Any] | None:
    try:
        suggestion = record.suggestions["label"]
    except (KeyError, TypeError):
        return None
    return {"label": str(suggestion.value), "agent": suggestion.agent, "score": suggestion.score}


def same_suggestion(record: Any, candidate: dict[str, Any] | None) -> bool:
    current = existing_suggestion(record)
    if candidate is None:
        return True
    if current is None:
        return False
    try:
        return (
            current["label"] == candidate["label"]
            and current["agent"] == candidate["agent"]
            and float(current["score"]) == float(candidate["score"])
        )
    except (TypeError, ValueError):
        return False


def apply_import(dataset: Any, rg: Any, campaign_name: str, rows: list[dict[str, str]], suggestions: dict[str, dict[str, Any]], batch_size: int) -> dict[str, int]:
    campaign = CAMPAIGNS[campaign_name]
    existing = {
        str(record.id): record
        for record in dataset.records(with_suggestions=True, with_responses=True)
    }
    new_records: list[Any] = []
    updated_records: list[Any] = []
    unchanged = 0
    preserved = 0

    for row in rows:
        canonical_id = row["canonical_id"]
        candidate_suggestion = suggestions.get(canonical_id)
        desired = import_record(rg, campaign_name, row, candidate_suggestion)
        current = existing.get(str(desired.id))
        if current is None:
            new_records.append(desired)
            continue

        current_metadata = current.metadata.to_dict()
        current_fingerprint = current_metadata.get("source_fingerprint", "")
        desired_fingerprint = desired.metadata.to_dict()["source_fingerprint"]
        has_responses = bool(list(current.responses))

        if has_responses and current_fingerprint != desired_fingerprint:
            authors = ", ".join(sorted(response_users(current))) or "autor no disponible"
            raise WorkflowError(
                f"{canonical_id} ya tiene respuestas ({authors}) y cambió su procedencia/texto. "
                "No se sobrescribió; cree una nueva campaña/version del registro para revisar el cambio."
            )

        if (
            not has_responses
            and current_fingerprint != desired_fingerprint
            and existing_suggestion(current) is not None
            and candidate_suggestion is None
        ):
            raise WorkflowError(
                f"{canonical_id} cambió y conserva una sugerencia anterior. "
                "Pase la sugerencia vigente para actualizarla o cree una nueva campaña."
            )

        if has_responses:
            preserved += 1
            # Human answers and the suggestion shown during annotation are immutable here.
            if current_fingerprint == desired_fingerprint:
                unchanged += 1
            continue

        if current_fingerprint == desired_fingerprint and same_suggestion(current, candidate_suggestion):
            unchanged += 1
            continue

        for name, value in desired.fields.to_dict().items():
            current.fields[name] = value
        for name, value in desired.metadata.to_dict().items():
            current.metadata[name] = value
        if candidate_suggestion is not None:
            current.suggestions.add(suggestion_for_record(rg, candidate_suggestion)[0])
        updated_records.append(current)

    created = 0
    updated = 0
    for offset in range(0, len(new_records), batch_size):
        batch = new_records[offset : offset + batch_size]
        dataset.records.log(records=batch, batch_size=batch_size)
        created += len(batch)
    for offset in range(0, len(updated_records), batch_size):
        batch = updated_records[offset : offset + batch_size]
        dataset.records.log(records=batch, batch_size=batch_size)
        updated += len(batch)
    return {
        "selected": len(rows),
        "created": created,
        "updated": updated,
        "unchanged": unchanged,
        "with_human_responses_preserved": preserved,
        "minimum_submitted": MIN_SUBMITTED,
    }


def restore_records(rg: Any, dataset: Any, export_dir: Path, batch_size: int) -> dict[str, int]:
    manifest_path = export_dir / "manifest.json"
    records_path = export_dir / "records.jsonl"
    if not manifest_path.exists() or not records_path.exists():
        raise WorkflowError(f"Exportación incompleta: se requieren {manifest_path.name} y {records_path.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    campaign_name = manifest.get("campaign")
    if campaign_name not in CAMPAIGNS:
        raise WorkflowError("La campaña de la exportación no coincide con ISSUE-04")
    if manifest.get("workflow_schema_version") != SCHEMA_VERSION:
        raise WorkflowError("La exportación usa otro esquema; restaure con la versión compatible del proyecto")
    if manifest.get("taxonomy_version") != TAXONOMY_VERSION:
        raise WorkflowError("La versión de taxonomía de la exportación no coincide con la campaña solicitada")
    records_bytes = records_path.read_bytes()
    expected_sha = manifest.get("records_sha256")
    if not expected_sha or hashlib.sha256(records_bytes).hexdigest() != expected_sha:
        raise WorkflowError("El SHA-256 de records.jsonl no coincide con manifest.json")

    existing_records = {
        str(record.id): record
        for record in dataset.records(with_suggestions=True, with_responses=True)
    }
    restored: list[Any] = []
    updated_records: list[Any] = []
    already_present = 0
    responses_added = 0
    seen_export_ids: set[str] = set()
    exported_count = 0
    for line_number, line in enumerate(records_bytes.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        record_id = str(payload.get("external_id", ""))
        if not record_id or record_id in seen_export_ids:
            raise WorkflowError(f"ID externo vacío o duplicado en {records_path}:{line_number}")
        seen_export_ids.add(record_id)
        exported_count += 1
        expected_prefix = f"cde-issue04-{CAMPAIGNS[campaign_name].slug}-"
        if not record_id.startswith(expected_prefix):
            raise WorkflowError(f"ID externo ajeno a la campaña en {records_path}:{line_number}")
        responses = [
            rg.Response(
                question_name=item["question_name"],
                value=item["value"],
                user_id=uuid.UUID(str(item["user_id"])),
                status=item.get("status"),
            )
            for item in payload.get("responses", [])
        ]
        suggestions = [
            rg.Suggestion(
                question_name=item["question_name"],
                value=item["value"],
                score=item.get("score"),
                agent=item.get("agent"),
                type=item.get("type"),
            )
            for item in payload.get("suggestions", [])
        ]
        current = existing_records.get(record_id)
        if current is not None:
            already_present += 1
            current_responses = list(current.responses)
            current_has_responses = bool(current_responses)
            current_fields = current.fields.to_dict()
            current_metadata = current.metadata.to_dict()
            incoming_fields = payload.get("fields", {})
            incoming_metadata = payload.get("metadata", {})
            fields_changed = any(current_fields.get(key) != value for key, value in incoming_fields.items())
            metadata_changed = any(current_metadata.get(key) != value for key, value in incoming_metadata.items())
            if current_has_responses and (fields_changed or metadata_changed):
                raise WorkflowError(
                    f"{record_id} ya tiene respuestas y difiere del snapshot en texto o procedencia; "
                    "no se modificó la anotación existente"
                )
            record_changed = False
            if fields_changed:
                for name, value in incoming_fields.items():
                    current.fields[name] = value
                record_changed = True
            if metadata_changed:
                for name, value in incoming_metadata.items():
                    current.metadata[name] = value
                record_changed = True

            response_keys = {
                (answer.question_name, str(answer.user_id)): answer
                for answer in current_responses
            }
            for incoming in responses:
                key = (incoming.question_name, str(incoming.user_id))
                existing_response = response_keys.get(key)
                if existing_response is not None:
                    same_value = json.dumps(existing_response.value, ensure_ascii=False, sort_keys=True) == json.dumps(
                        incoming.value, ensure_ascii=False, sort_keys=True
                    )
                    existing_status = getattr(existing_response.status, "value", existing_response.status)
                    incoming_status = getattr(incoming.status, "value", incoming.status)
                    if not same_value or existing_status != incoming_status:
                        raise WorkflowError(
                            f"Respuesta conflictiva para {record_id}, pregunta {incoming.question_name}, "
                            f"usuario {incoming.user_id}; no se sobrescribió"
                        )
                    continue
                current.responses.add(incoming)
                response_keys[key] = incoming
                responses_added += 1
                record_changed = True

            current_suggestions = {item.question_name: item for item in current.suggestions}
            for incoming in suggestions:
                existing_suggestion_value = current_suggestions.get(incoming.question_name)
                if existing_suggestion_value is None:
                    current.suggestions.add(incoming)
                    record_changed = True
                elif (
                    existing_suggestion_value.value != incoming.value
                    or existing_suggestion_value.score != incoming.score
                    or existing_suggestion_value.agent != incoming.agent
                    or existing_suggestion_value.type != incoming.type
                ):
                    if current_has_responses or responses:
                        raise WorkflowError(
                            f"Sugerencia conflictiva para {record_id}; el registro ya tiene respuestas humanas"
                        )
                    current.suggestions.add(incoming)
                    record_changed = True

            if record_changed:
                updated_records.append(current)
            continue

        restored.append(
            rg.Record(
                id=record_id,
                fields=payload.get("fields", {}),
                metadata=payload.get("metadata", {}),
                vectors=payload.get("vectors", {}),
                responses=responses,
                suggestions=suggestions,
            )
        )

    if exported_count != manifest.get("record_count"):
        raise WorkflowError("record_count de manifest.json no coincide con records.jsonl")

    created = 0
    for offset in range(0, len(restored), batch_size):
        batch = restored[offset : offset + batch_size]
        dataset.records.log(records=batch, batch_size=batch_size)
        created += len(batch)
    updated = 0
    for offset in range(0, len(updated_records), batch_size):
        batch = updated_records[offset : offset + batch_size]
        dataset.records.log(records=batch, batch_size=batch_size)
        updated += len(batch)
    return {
        "restored": created,
        "updated_existing": updated,
        "already_present": already_present,
        "responses_added": responses_added,
        "minimum_submitted": MIN_SUBMITTED,
    }


def main() -> int:
    args = parse_args()
    try:
        if args.batch_size < 1 or args.batch_size > 1000:
            raise WorkflowError("--batch-size debe estar entre 1 y 1000")
        load_env_file(args.env_file if args.env_file.is_absolute() else (Path.cwd() / args.env_file))

        if args.restore_export:
            manifest = json.loads((args.restore_export / "manifest.json").read_text(encoding="utf-8"))
            campaign_name = manifest.get("campaign")
            if campaign_name not in CAMPAIGNS:
                raise WorkflowError("No se pudo determinar la campaña de la exportación")
            workspace = workspace_for(campaign_name, args.workspace)
            if args.dry_run:
                print(
                    json.dumps(
                        {
                            "mode": "restore",
                            "export": str(args.restore_export),
                            "campaign": campaign_name,
                            "workspace": workspace,
                            "records": manifest.get("record_count"),
                            "dry_run": True,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                return 0
            import argilla as rg

            client = argilla_client(rg)
            dataset, created_dataset = get_or_create_dataset(rg, client, campaign_name, workspace)
            result = restore_records(rg, dataset, args.restore_export, args.batch_size)
            result.update({"campaign": campaign_name, "dataset": dataset.name, "workspace": workspace})
            result["dataset_created"] = created_dataset
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0

        if not args.campaign:
            raise WorkflowError("Indique --campaign o --restore-export")
        source_path = args.records
        if args.campaign == "calibration":
            rows = load_calibration_cases()
            source_path = GUIDELINES_PATH.with_name("calibration_cases_v2.md")
        else:
            rows = campaign_rows(load_canonical_rows(args.records), args.campaign)
        if args.limit is not None:
            if args.limit < 1:
                raise WorkflowError("--limit debe ser positivo")
            rows = rows[: args.limit]
        suggestions = load_suggestions(args.suggestions, args.campaign)
        selected_ids = {row["canonical_id"] for row in rows}
        unmatched = sorted(set(suggestions).difference(selected_ids))
        if unmatched:
            raise WorkflowError(
                "El archivo de sugerencias contiene IDs fuera de esta campaña: " + ", ".join(unmatched[:5])
            )

        workspace = workspace_for(args.campaign, args.workspace)
        campaign = CAMPAIGNS[args.campaign]
        if args.dry_run:
            source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
            sample = [
                {
                    "canonical_id": row["canonical_id"],
                    "external_id": external_id(args.campaign, row["canonical_id"]),
                    "has_suggestion": row["canonical_id"] in suggestions,
                }
                for row in rows[:5]
            ]
            print(
                json.dumps(
                    {
                        "mode": "import",
                        "campaign": args.campaign,
                        "dataset": campaign.dataset_name,
                        "workspace": workspace,
                        "source": str(source_path),
                        "source_sha256": source_sha,
                        "selected_records": len(rows),
                        "suggested_records": len(suggestions),
                        "minimum_submitted": MIN_SUBMITTED,
                        "taxonomy_version": TAXONOMY_VERSION,
                        "workflow_schema_version": SCHEMA_VERSION,
                        "external_id_examples": sample,
                        "dry_run": True,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        if not rows:
            raise WorkflowError("La campaña no contiene filas; no se creó ni modificó un dataset")

        import argilla as rg

        client = argilla_client(rg)
        dataset, created_dataset = get_or_create_dataset(rg, client, args.campaign, workspace)
        result = apply_import(dataset, rg, args.campaign, rows, suggestions, args.batch_size)
        result.update(
            {
                "campaign": args.campaign,
                "dataset": campaign.dataset_name,
                "workspace": workspace,
                "dataset_created": created_dataset,
                "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                "schema_version": SCHEMA_VERSION,
                "timestamp_utc": utc_now(),
            }
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (WorkflowError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
