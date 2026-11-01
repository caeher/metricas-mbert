#!/usr/bin/env python
"""REST helpers for Argilla annotator sessions (Bearer token from /api/v1/token)."""

from __future__ import annotations

from typing import Any

import httpx
import requests

from argilla_workflow import WorkflowError


def login_bearer(api_url: str, username: str, password: str) -> str:
    for path in ("/api/v1/token", "/api/v1/oauth/token", "/api/v1/login"):
        response = requests.post(
            f"{api_url.rstrip('/')}{path}",
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


def bearer_client(api_url: str, token: str) -> httpx.Client:
    return httpx.Client(
        base_url=api_url.rstrip("/"),
        headers={"Authorization": f"Bearer {token}"},
        timeout=120.0,
    )


def get_me(client: httpx.Client) -> dict[str, Any]:
    response = client.get("/api/v1/me")
    response.raise_for_status()
    return response.json()


def get_dataset(client: httpx.Client, workspace: str, dataset_name: str) -> dict[str, Any]:
    response = client.get(f"/api/v1/me/datasets?name={dataset_name}&workspace={workspace}")
    if response.status_code >= 400:
        response = client.get(f"/api/v1/datasets?name={dataset_name}&workspace={workspace}")
    response.raise_for_status()
    payload = response.json()
    items = payload.get("items") or payload
    if isinstance(items, list):
        for item in items:
            if item.get("name") == dataset_name:
                return item
        if items:
            return items[0]
    raise WorkflowError(f"Dataset {dataset_name} no encontrado en {workspace}")


def list_records(client: httpx.Client, dataset_id: str, *, limit: int = 500) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    offset = 0
    page = 100
    while offset < limit:
        batch = min(page, limit - offset)
        response = client.get(
            f"/api/v1/datasets/{dataset_id}/records",
            params={"offset": offset, "limit": batch, "include": ["responses"]},
        )
        response.raise_for_status()
        items = response.json().get("items", [])
        if not items:
            break
        records.extend(items)
        offset += len(items)
        if len(items) < batch:
            break
    return records


def user_response_payload(user_id: str, values: dict[str, Any]) -> dict[str, Any]:
    return {
        "user_id": user_id,
        "status": "submitted",
        "values": {name: {"value": value} for name, value in values.items()},
    }


def upsert_user_responses(
    client: httpx.Client,
    record: dict[str, Any],
    user_id: str,
    values: dict[str, Any],
) -> None:
    record_id = record["id"]
    existing = list(record.get("responses") or [])
    filtered = [item for item in existing if str(item.get("user_id")) != str(user_id)]
    filtered.append(user_response_payload(user_id, values))
    body = {
        "fields": record.get("fields") or {},
        "metadata": record.get("metadata") or {},
        "responses": filtered,
    }
    response = client.patch(f"/api/v1/records/{record_id}", json=body)
    response.raise_for_status()
