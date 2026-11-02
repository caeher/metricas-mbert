#!/usr/bin/env python
"""Ensure independent Argilla annotator accounts exist in the annotation workspace."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from argilla_workflow import SDK_VERSION, WorkflowError, argilla_client, load_env_file

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RATERS = (
    ("cde-rater-a", "ARGILLA_RATER_A_PASSWORD"),
    ("cde-rater-b", "ARGILLA_RATER_B_PASSWORD"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env.argilla")
    return parser.parse_args()


def api_url_for_docker() -> str:
    url = os.environ.get("ARGILLA_API_URL", "http://localhost:6900").strip()
    if os.environ.get("ARGILLA_DOCKER_HOST", "").strip().lower() in {"1", "true", "yes"}:
        return url.replace("localhost", "host.docker.internal").replace("127.0.0.1", "host.docker.internal")
    return url


def default_password(username: str) -> str:
    return f"{username}-ChangeMe-2026!"


def ensure_users(rg: object, workspace_name: str) -> list[tuple[str, str]]:
    client = argilla_client(rg)
    workspace = client.workspaces(name=workspace_name)
    if workspace is None:
        workspace = rg.Workspace(name=workspace_name).create()  # type: ignore[attr-defined]
    audit_name = os.environ.get("ARGILLA_AUDIT_WORKSPACE", "restricted-audit").strip()
    if audit_name:
        audit = client.workspaces(name=audit_name)
        if audit is None:
            rg.Workspace(name=audit_name).create()  # type: ignore[attr-defined]
    created: list[tuple[str, str]] = []
    for username, env_key in DEFAULT_RATERS:
        password = os.environ.get(env_key, "").strip() or default_password(username)
        user = next((item for item in client.users if item.username == username), None)
        if user is None:
            user = rg.User(  # type: ignore[attr-defined]
                username=username,
                password=password,
                role="annotator",
                first_name=username,
            ).create()
            created.append((username, password))
        try:
            user.add_to_workspace(workspace)
        except Exception as exc:
            if "Conflict" not in type(exc).__name__ and "409" not in str(exc):
                raise
    return created


def main() -> int:
    args = parse_args()
    load_env_file(args.env_file if args.env_file.is_absolute() else Path.cwd() / args.env_file)
    os.environ["ARGILLA_API_URL"] = api_url_for_docker()
    try:
        import argilla as rg
    except ImportError as exc:
        raise WorkflowError(f"Instale el extra argilla ({SDK_VERSION}): pip install -e '.[argilla]'") from exc
    workspace = os.environ.get("ARGILLA_WORKSPACE", "annotations")
    created = ensure_users(rg, workspace)
    if created:
        print("Created annotators (store passwords securely):")
        for username, password in created:
            print(f"  {username}: {password}")
    else:
        print("Annotators already present:", ", ".join(u for u, _ in DEFAULT_RATERS))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WorkflowError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
