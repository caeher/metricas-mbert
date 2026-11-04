#!/usr/bin/env python
"""Submit heuristic dual responses for legacy_train_real retrain priority batch."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from argilla_workflow import CAMPAIGNS, DEFAULT_RECORDS, SDK_VERSION, WorkflowError, load_canonical_rows, load_env_file
from pilot_heuristic_decision import build_response_values, decide, decide_rater_b, provisional_label_from_row
from submit_pilot_dual_responses import login_token, rater_password

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "reports" / "annotation" / "legacy_train_real_retrain_priority.csv"
PROGRESS_PATH = ROOT / "reports" / "annotation" / "annotation_progress.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env.argilla")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--rater", choices=["a", "b", "both"], default="a")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-progress", action="store_true", default=True)
    return parser.parse_args()


def manifest_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return {row["canonical_id"].strip() for row in csv.DictReader(stream) if row.get("canonical_id")}


def user_has_submitted_sdk(record: object, user_id: object) -> bool:
    from uuid import UUID

    uid = str(user_id)
    for response in record.responses:
        if str(response.user_id) != uid:
            continue
        status = getattr(response.status, "value", response.status)
        if str(status).casefold() == "submitted":
            return True
    return False


def submit_rater_sdk(
    rg: object,
    api_url: str,
    username: str,
    campaign_name: str,
    workspace: str,
    canonical_by_id: dict[str, dict[str, str]],
    allowed_ids: set[str],
    *,
    rater: str,
    limit: int,
    dry_run: bool,
) -> tuple[int, list[str]]:
    from uuid import UUID

    from argilla.responses import Response, ResponseStatus

    campaign = CAMPAIGNS[campaign_name]
    token = login_token(api_url, username, rater_password(username))
    client = rg.Argilla(api_url=api_url, api_key=token)
    user_id = UUID(str(client.me.id))
    dataset = client.datasets(name=campaign.dataset_name, workspace=workspace)
    if dataset is None:
        raise WorkflowError(f"{username} no ve el dataset {campaign.dataset_name}")

    records = [
        record
        for record in dataset.records(with_responses=True)
        if str((record.metadata.get("canonical_id") or "")).strip() in allowed_ids
    ]
    if limit:
        records = records[:limit]
    submitted = 0
    annotated_ids: list[str] = []
    for index, record in enumerate(records, start=1):
        canonical_id = str((record.metadata.get("canonical_id") or "")).strip()
        if user_has_submitted_sdk(record, user_id):
            continue
        text = record.fields.get("original_text") or record.fields.get("normalized_text") or ""
        source = canonical_by_id.get(canonical_id, {})
        if rater == "a":
            decision = decide(text)
        else:
            decision = decide_rater_b(text, provisional_label_from_row(source))
        values = build_response_values(decision)
        if dry_run:
            submitted += 1
            annotated_ids.append(canonical_id)
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
        annotated_ids.append(canonical_id)
        if index % 25 == 0:
            print(f"{username}: {index}/{len(records)}, submitted={submitted}", flush=True)
    return submitted, annotated_ids


def update_progress(rater: str, count: int, ids: list[str], *, manifest: Path) -> None:
    payload: dict = {}
    if PROGRESS_PATH.is_file():
        payload = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    try:
        manifest_rel = manifest.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        manifest_rel = str(manifest)
    payload.setdefault("sessions", []).append(
        {
            "rater": f"cde-rater-{rater}",
            "manifest": manifest_rel,
            "method": "sdk_heuristic_guidelines_v2",
            "browser": "bulk submit vía Argilla SDK (helper piloto ISSUE-05)",
            "submitted_count": count,
            "canonical_ids": ids,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    PROGRESS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    load_env_file(args.env_file if args.env_file.is_absolute() else Path.cwd() / args.env_file)
    if os.environ.get("ARGILLA_DOCKER_HOST", "").strip().lower() in {"1", "true", "yes"}:
        os.environ.pop("ARGILLA_DOCKER_HOST", None)
    api_url = os.environ.get("ARGILLA_API_URL", "http://localhost:6900").strip()
    try:
        import argilla as rg
    except ImportError as exc:
        raise WorkflowError(f"Instale pip install -e '.[argilla]'") from exc
    if getattr(rg, "__version__", "") != SDK_VERSION:
        raise WorkflowError(f"Se requiere argilla=={SDK_VERSION}")

    allowed = manifest_ids(args.manifest)
    workspace = os.environ.get("ARGILLA_WORKSPACE", "annotations")
    canonical_by_id = {row["canonical_id"]: row for row in load_canonical_rows(DEFAULT_RECORDS)}
    campaign = "legacy_train_real"
    results: dict[str, int] = {}

    if args.rater in {"a", "both"}:
        count, ids = submit_rater_sdk(
            rg,
            api_url,
            "cde-rater-a",
            campaign,
            workspace,
            canonical_by_id,
            allowed,
            rater="a",
            limit=args.limit,
            dry_run=args.dry_run,
        )
        results["submitted_rater_a"] = count
        if args.write_progress and not args.dry_run:
            update_progress("a", count, ids, manifest=args.manifest.resolve())

    if args.rater in {"b", "both"}:
        count, ids = submit_rater_sdk(
            rg,
            api_url,
            "cde-rater-b",
            campaign,
            workspace,
            canonical_by_id,
            allowed,
            rater="b",
            limit=args.limit,
            dry_run=args.dry_run,
        )
        results["submitted_rater_b"] = count
        if args.write_progress and not args.dry_run:
            update_progress("b", count, ids, manifest=args.manifest.resolve())

    print({"dry_run": args.dry_run, "manifest_ids": len(allowed), **results})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WorkflowError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
