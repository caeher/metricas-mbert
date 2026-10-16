#!/usr/bin/env python
"""Export every Argilla response and suggestion to an immutable local JSONL release."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from argilla_workflow import (
    CAMPAIGNS,
    SCHEMA_VERSION,
    SDK_VERSION,
    TAXONOMY_VERSION,
    WorkflowError,
    argilla_client,
    load_env_file,
    utc_now,
    workspace_for,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", choices=sorted(CAMPAIGNS), help="Campaña que se exporta.")
    parser.add_argument("--workspace", help="Workspace Argilla; se infiere de .env.argilla si se omite.")
    parser.add_argument("--env-file", type=Path, default=Path(".env.argilla"), help="Archivo local ignorado por Git.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/argilla_exports"),
        help="Carpeta para exportaciones portables y versionadas.",
    )
    parser.add_argument("--restore", type=Path, help="Exportación a restaurar; delega en import_argilla.py.")
    parser.add_argument("--dry-run", action="store_true", help="Muestra el destino sin descargar registros.")
    return parser.parse_args()


def status_value(status: Any) -> Any:
    return getattr(status, "value", status)


def serialize_record(record: Any) -> dict[str, Any]:
    responses = [
        {
            "question_name": response.question_name,
            "value": response.value,
            "user_id": str(response.user_id),
            "status": status_value(response.status),
        }
        for response in record.responses
    ]
    suggestions = [
        {
            "question_name": suggestion.question_name,
            "value": suggestion.value,
            "score": suggestion.score,
            "agent": suggestion.agent,
            "type": suggestion.type,
        }
        for suggestion in record.suggestions
    ]
    return {
        "external_id": str(record.id),
        "fields": record.fields.to_dict(),
        "metadata": record.metadata.to_dict(),
        "responses": responses,
        "suggestions": suggestions,
        "vectors": record.vectors.to_dict(),
        "status": record.status,
    }


def main() -> int:
    args = parse_args()
    try:
        load_env_file(args.env_file if args.env_file.is_absolute() else (Path.cwd() / args.env_file))
        if args.restore:
            # Keep restore logic alongside the importer's ID and schema guards.
            command = [sys.executable, str(Path(__file__).with_name("import_argilla.py")), "--restore-export", str(args.restore)]
            if args.workspace:
                command.extend(["--workspace", args.workspace])
            if args.env_file:
                command.extend(["--env-file", str(args.env_file)])
            if args.dry_run:
                command.append("--dry-run")
            import subprocess

            return subprocess.call(command)

        if not args.campaign:
            raise WorkflowError("Indique --campaign o --restore")
        campaign = CAMPAIGNS[args.campaign]
        workspace = workspace_for(args.campaign, args.workspace)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        export_id = f"{campaign.slug}__{stamp}"
        export_dir = args.output_dir / export_id
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "campaign": args.campaign,
                        "dataset": campaign.dataset_name,
                        "workspace": workspace,
                        "output": str(export_dir),
                        "dry_run": True,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        import argilla as rg

        client = argilla_client(rg)
        dataset = client.datasets(name=campaign.dataset_name, workspace=workspace)
        if dataset is None:
            raise WorkflowError(f"No existe la campaña {campaign.dataset_name} en {workspace}")

        # One JSON object per record retains all independent users and every response status.
        records = list(dataset.records(with_suggestions=True, with_responses=True))
        export_dir.mkdir(parents=True, exist_ok=False)
        records_path = export_dir / "records.jsonl"
        response_count = 0
        suggestion_count = 0
        annotators: set[str] = set()
        with records_path.open("w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                item = serialize_record(record)
                response_count += len(item["responses"])
                suggestion_count += len(item["suggestions"])
                annotators.update(response["user_id"] for response in item["responses"])
                stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

        records_sha = hashlib.sha256(records_path.read_bytes()).hexdigest()
        manifest = {
            "export_format_version": "1.0.0",
            "export_id": export_id,
            "created_at_utc": utc_now(),
            "campaign": args.campaign,
            "dataset": campaign.dataset_name,
            "workspace": workspace,
            "record_count": len(records),
            "response_count": response_count,
            "suggestion_count": suggestion_count,
            "response_user_ids": sorted(annotators),
            "argilla_sdk_version": SDK_VERSION,
            "argilla_server_version_expected": SDK_VERSION,
            "taxonomy_version": TAXONOMY_VERSION,
            "guideline_version": TAXONOMY_VERSION,
            "workflow_schema_version": SCHEMA_VERSION,
            "records_file": records_path.name,
            "records_sha256": records_sha,
            "restoration": {
                "method": "python scripts/import_argilla.py --restore-export <directory>",
                "preserves": ["external_id", "fields", "provenance metadata", "all responses and authors", "suggestions"],
                "requires": "the target Argilla instance has matching workspace users and dataset schema",
            },
        }
        (export_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"export_dir": str(export_dir), **{k: manifest[k] for k in (
            "record_count", "response_count", "suggestion_count", "records_sha256"
        )}}, ensure_ascii=False, indent=2))
        return 0
    except (WorkflowError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
