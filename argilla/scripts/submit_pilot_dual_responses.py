#!/usr/bin/env python
"""Submit dual independent pilot responses (cde-rater-a heuristic, cde-rater-b provisional-aware)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from uuid import UUID

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from argilla_workflow import (
    CAMPAIGNS,
    DEFAULT_RECORDS,
    SDK_VERSION,
    WorkflowError,
    load_canonical_rows,
    load_env_file,
)
from pilot_heuristic_decision import (
    build_response_values,
    decide,
    decide_rater_b,
    provisional_label_from_row,
)

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env.argilla")
    parser.add_argument("--campaign", default="pilot", choices=["pilot"])
    parser.add_argument("--limit", type=int, default=0, help="Limit records (0 = all)")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def api_url_for_docker() -> str:
    url = os.environ.get("ARGILLA_API_URL", "http://localhost:6900").strip()
    if os.environ.get("ARGILLA_DOCKER_HOST", "").strip().lower() in {"1", "true", "yes"}:
        return url.replace("localhost", "host.docker.internal").replace("127.0.0.1", "host.docker.internal")
    return url


def login_token(api_url: str, username: str, password: str) -> str:
    for path in ("/api/v1/token", "/api/v1/oauth/token", "/api/v1/login"):
        response = requests.post(
            f"{api_url}{path}",
            data={"username": username, "password": password},
            timeout=60,
        )
        if response.status_code >= 400:
            continue
        payload = response.json()
        for key in ("access_token", "api_key", "token"):
            if key in payload:
                return str(payload[key])
    raise WorkflowError(f"No se pudo autenticar {username}")


def rater_password(username: str) -> str:
    env_key = f"ARGILLA_{username.upper().replace('-', '_')}_PASSWORD"
    value = os.environ.get(env_key, "").strip()
    return value or f"{username}-ChangeMe-2026!"


def user_has_submitted(record: object, user_id: UUID) -> bool:
    for response in record.responses:
        if str(response.user_id) != str(user_id):
            continue
        status = getattr(response.status, "value", response.status)
        if str(status).casefold() == "submitted":
            return True
    return False


def submit_as_rater(
    rg: object,
    api_url: str,
    username: str,
    campaign_name: str,
    workspace: str,
    canonical_by_id: dict[str, dict[str, str]],
    *,
    rater: str,
    limit: int,
    dry_run: bool,
) -> int:
    from argilla.responses import Response, ResponseStatus

    campaign = CAMPAIGNS[campaign_name]
    token = login_token(api_url, username, rater_password(username))
    client = rg.Argilla(api_url=api_url, api_key=token)
    user_id = UUID(str(client.me.id))
    dataset = client.datasets(name=campaign.dataset_name, workspace=workspace)
    if dataset is None:
        raise WorkflowError(f"{username} no ve el dataset {campaign.dataset_name}")

    records = list(dataset.records(with_responses=True))
    if limit:
        records = records[:limit]
    submitted = 0
    for index, record in enumerate(records, start=1):
        if user_has_submitted(record, user_id):
            continue
        text = record.fields.get("original_text") or record.fields.get("normalized_text") or ""
        canonical_id = (record.metadata.get("canonical_id") or "").strip()
        source = canonical_by_id.get(canonical_id, {})
        if rater == "a":
            decision = decide(text)
        else:
            decision = decide_rater_b(text, provisional_label_from_row(source))
        values = build_response_values(decision)
        if dry_run:
            submitted += 1
            continue
        for question_name, value in values.items():
            record.responses.add(
                Response(
                    question_name=question_name,
                    value=value,
                    user_id=user_id,
                    status=ResponseStatus.submitted,
                )
            )
        record.update()
        submitted += 1
        if index % 25 == 0:
            print(f"{username}: {index}/{len(records)}, submitted={submitted}", flush=True)
    return submitted


def main() -> int:
    args = parse_args()
    load_env_file(args.env_file if args.env_file.is_absolute() else Path.cwd() / args.env_file)
    os.environ.setdefault("ARGILLA_DOCKER_HOST", "1")
    api_url = api_url_for_docker()
    try:
        import argilla as rg
    except ImportError as exc:
        raise WorkflowError(f"Instale argilla=={SDK_VERSION}") from exc
    if getattr(rg, "__version__", "") != SDK_VERSION:
        raise WorkflowError(f"Se requiere argilla=={SDK_VERSION}")

    workspace = os.environ.get("ARGILLA_WORKSPACE", "annotations")
    canonical_by_id = {row["canonical_id"]: row for row in load_canonical_rows(DEFAULT_RECORDS)}
    count_a = submit_as_rater(
        rg,
        api_url,
        "cde-rater-a",
        args.campaign,
        workspace,
        canonical_by_id,
        rater="a",
        limit=args.limit,
        dry_run=args.dry_run,
    )
    count_b = submit_as_rater(
        rg,
        api_url,
        "cde-rater-b",
        args.campaign,
        workspace,
        canonical_by_id,
        rater="b",
        limit=args.limit,
        dry_run=args.dry_run,
    )
    print(
        {
            "dry_run": args.dry_run,
            "submitted_rater_a": count_a,
            "submitted_rater_b": count_b,
            "dataset": CAMPAIGNS[args.campaign].dataset_name,
        }
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WorkflowError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
