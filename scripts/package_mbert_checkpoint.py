#!/usr/bin/env python
"""Empaqueta el checkpoint mBERT para auditoría (ZIP + manifiesto SHA-256)."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.models.mbert import ensure_checkpoint_contract
from discurso_odio.models.mbert_artifacts import (
    ARTIFACTS_DIR,
    MANIFEST_REPORT_PATH,
    PACKAGE_ROOT_NAME,
    ZIP_NAME,
    apply_id2label,
    build_manifest,
    prepare_staging,
    resolve_mbert_source,
)
from discurso_odio.paths import REPORTS_METRICS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def write_zip(staged_dir: Path, zip_path: Path) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(staged_dir.rglob("*")):
            if not file.is_file():
                continue
            arcname = f"{PACKAGE_ROOT_NAME}/{file.relative_to(staged_dir).as_posix()}"
            archive.write(file, arcname)
    return zip_path


def package_mbert_checkpoint(
    *,
    source_dir: Path | None = None,
    output_zip: Path | None = None,
    artifact_url: str = "",
) -> dict:
    source = resolve_mbert_source(source_dir)
    logger.info("Checkpoint fuente: %s", source)

    ensure_checkpoint_contract(source)
    apply_id2label(source / "config.json")

    staging_root = ARTIFACTS_DIR / "_staging"
    staged_dir = staging_root / PACKAGE_ROOT_NAME
    if staging_root.exists():
        shutil.rmtree(staging_root)

    names = prepare_staging(source, staged_dir)
    manifest = build_manifest(
        staged_dir=staged_dir,
        source_dir=source,
        file_names=names,
        artifact_url=artifact_url,
    )

    manifest_path = staged_dir / "MANIFEST.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    REPORTS_METRICS.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    zip_path = output_zip or (ARTIFACTS_DIR / ZIP_NAME)
    write_zip(staged_dir, zip_path)
    logger.info("Paquete: %s (%.1f MB)", zip_path, zip_path.stat().st_size / (1024 * 1024))
    logger.info("Manifiesto: %s", MANIFEST_REPORT_PATH)
    shutil.rmtree(staging_root, ignore_errors=True)
    return {"zip": zip_path, "manifest": MANIFEST_REPORT_PATH, "source": source}


def main():
    parser = argparse.ArgumentParser(description="Empaqueta el checkpoint mBERT de auditoría")
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Alias de --source-dir (directorio del checkpoint).",
    )
    parser.add_argument("--output-zip", type=Path, default=None)
    parser.add_argument("--artifact-url", type=str, default="", help="URL de Drive u otro canal")
    args = parser.parse_args()
    source = args.model_dir or args.source_dir
    package_mbert_checkpoint(
        source_dir=source,
        output_zip=args.output_zip,
        artifact_url=args.artifact_url,
    )


if __name__ == "__main__":
    main()
