#!/usr/bin/env python
"""Report tokenizer fertility and truncation on train/val (ISSUE-10)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.config import load_settings
from discurso_odio.data.loaders import load_split
from discurso_odio.evaluation.predictions import git_commit
from discurso_odio.models.inference_contract import resolve_inference_contract
from discurso_odio.paths import REPORTS_METRICS

JSON_OUT = REPORTS_METRICS / "tokenizer_fertility.json"
CSV_OUT = REPORTS_METRICS / "tokenizer_fertility_by_slice.csv"


def word_count(text: str) -> int:
    return len(text.split()) if text.strip() else 0


def token_stats(
    texts: list[str],
    tokenizer,
    max_length: int,
) -> dict[str, float]:
    lengths: list[int] = []
    word_counts: list[int] = []
    char_counts: list[int] = []
    truncated = 0
    fertility_values: list[float] = []

    for text in texts:
        words = word_count(text)
        chars = len(text)
        ids = tokenizer.encode(text, add_special_tokens=True, truncation=False)
        full_len = len(ids)
        if full_len > max_length:
            truncated += 1
        enc = tokenizer(text, truncation=True, max_length=max_length, add_special_tokens=True)
        lengths.append(len(enc["input_ids"]))
        word_counts.append(words)
        char_counts.append(chars)
        if words > 0:
            fertility_values.append(full_len / words)

    n = max(len(texts), 1)
    arr = np.array(lengths, dtype=float)
    fert = np.array(fertility_values, dtype=float) if fertility_values else np.array([0.0])
    return {
        "n": len(texts),
        "max_length": max_length,
        "tokens_mean": float(arr.mean()),
        "tokens_median": float(np.median(arr)),
        "tokens_p95": float(np.percentile(arr, 95)),
        "tokens_max": float(arr.max()) if len(arr) else 0.0,
        "truncated_pct": float(truncated / n * 100),
        "truncated_count": truncated,
        "chars_per_token_mean": float(np.mean(np.array(char_counts) / np.maximum(arr, 1))),
        "fertility_tokens_per_word_mean": float(fert.mean()),
        "fertility_tokens_per_word_median": float(np.median(fert)),
        "words_mean": float(np.mean(word_counts)) if word_counts else 0.0,
    }


def slice_rows(df: pd.DataFrame, text_col: str, label_col: str, tokenizer, max_length: int) -> list[dict]:
    rows: list[dict] = []
    overall = token_stats(df[text_col].fillna("").astype(str).tolist(), tokenizer, max_length)
    rows.append({"slice_type": "overall", "slice_value": "all", **overall})

    for label, group in df.groupby(label_col):
        stats = token_stats(group[text_col].fillna("").astype(str).tolist(), tokenizer, max_length)
        rows.append({"slice_type": "label", "slice_value": str(label), **stats})

    if "plataforma" in df.columns:
        for platform, group in df.groupby("plataforma"):
            stats = token_stats(group[text_col].fillna("").astype(str).tolist(), tokenizer, max_length)
            rows.append({"slice_type": "platform", "slice_value": str(platform), **stats})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Checkpoint con tokenizer (default: bert-base-multilingual-cased)",
    )
    parser.add_argument("--max-length", type=int, default=None)
    args = parser.parse_args()

    settings = load_settings()
    text_col = settings["model"]["text_column"]
    label_col = settings["model"]["label_column"]

    if args.model_dir and args.model_dir.is_dir():
        contract = resolve_inference_contract(args.model_dir)
        max_length = contract.validate_max_length(args.max_length)
        tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
        tokenizer_source = str(args.model_dir.resolve())
    else:
        model_name = settings["model"]["name"]
        max_length = args.max_length or settings["inference_contract"]["max_length"]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer_source = model_name

    train_df = load_split("train")
    val_df = load_split("val")
    combined = pd.concat([train_df, val_df], ignore_index=True)

    slice_data = slice_rows(combined, text_col, label_col, tokenizer, max_length)
    train_stats = token_stats(train_df[text_col].fillna("").astype(str).tolist(), tokenizer, max_length)
    val_stats = token_stats(val_df[text_col].fillna("").astype(str).tolist(), tokenizer, max_length)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "tokenizer_source": tokenizer_source,
        "text_column": text_col,
        "train": train_stats,
        "val": val_stats,
        "combined_train_val": slice_data[0],
        "slices": slice_data,
    }

    REPORTS_METRICS.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pd.DataFrame(slice_data).to_csv(CSV_OUT, index=False)
    print(json.dumps({"ok": True, "json": str(JSON_OUT), "csv": str(CSV_OUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
