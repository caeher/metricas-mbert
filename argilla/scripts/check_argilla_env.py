#!/usr/bin/env python
"""Verify Argilla SDK install and optional server reachability."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from argilla_workflow import SDK_VERSION, load_env_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env.argilla")
    parser.add_argument("--ping-server", action="store_true", help="GET ARGILLA_API_URL/health")
    args = parser.parse_args()

    try:
        import argilla as rg
    except ImportError:
        print("error: instale pip install -e '.[argilla]'", file=sys.stderr)
        return 1

    installed = getattr(rg, "__version__", "unknown")
    print(f"argilla SDK: {installed} (expected {SDK_VERSION})")
    if installed != SDK_VERSION:
        print("warning: versión distinta a la fijada en argilla_workflow.py", file=sys.stderr)

    if args.env_file.exists():
        load_env_file(args.env_file)
    api_url = os.environ.get("ARGILLA_API_URL", "http://localhost:6900").rstrip("/")
    print(f"ARGILLA_API_URL: {api_url}")

    if args.ping_server:
        import requests

        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            response.raise_for_status()
            print("server: ok")
        except requests.RequestException as exc:
            print(f"server: unreachable ({exc})", file=sys.stderr)
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
