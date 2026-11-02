"""Metadatos de corrida de entrenamiento mBERT (reproducibilidad ISSUE-14)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import TrainingArguments

from discurso_odio.config import load_settings
from discurso_odio.evaluation.predictions import git_commit
from discurso_odio.models.mbert_artifacts import read_checkpoint_identity, sha256_file
from discurso_odio.paths import DATA_PROCESSED, PROJECT_ROOT


def count_model_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": int(total), "trainable_params": int(trainable)}


def training_args_snapshot(args: TrainingArguments) -> dict[str, Any]:
    fields = (
        "output_dir",
        "num_train_epochs",
        "max_steps",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "learning_rate",
        "warmup_ratio",
        "weight_decay",
        "seed",
        "fp16",
        "eval_strategy",
        "save_strategy",
        "load_best_model_at_end",
        "metric_for_best_model",
    )
    return {name: getattr(args, name, None) for name in fields}


def runtime_from_trainer_state(state_path: Path) -> dict[str, Any]:
    if not state_path.is_file():
        return {}
    with state_path.open(encoding="utf-8") as handle:
        state = json.load(handle)
    runtime: dict[str, Any] = {
        "global_step": state.get("global_step"),
        "epoch": state.get("epoch"),
        "best_metric": state.get("best_metric"),
        "best_model_checkpoint": state.get("best_model_checkpoint"),
    }
    for entry in reversed(state.get("log_history") or []):
        if "train_runtime" in entry:
            runtime["train_runtime_seconds"] = float(entry["train_runtime"])
            runtime["train_samples_per_second"] = entry.get("train_samples_per_second")
            runtime["train_steps_per_second"] = entry.get("train_steps_per_second")
            break
    return runtime


def resolve_trainer_state_path(model_dir: Path) -> Path | None:
    identity = read_checkpoint_identity(model_dir)
    best = identity.get("best_model_checkpoint")
    if best:
        candidate = model_dir / str(best) / "trainer_state.json"
        if candidate.is_file():
            return candidate
    states = sorted(model_dir.glob("checkpoint-*/trainer_state.json"))
    if states:
        return states[-1]
    root = model_dir / "trainer_state.json"
    return root if root.is_file() else None


def dataset_hashes() -> dict[str, Any]:
    meta_path = DATA_PROCESSED / "metadata.json"
    result: dict[str, Any] = {"dataset_metadata_path": str(meta_path.relative_to(PROJECT_ROOT))}
    if meta_path.is_file():
        with meta_path.open(encoding="utf-8") as handle:
            meta = json.load(handle)
        result["dataset_version"] = meta.get("version")
        result["train_total"] = meta.get("train_total")
    for name in ("train.csv", "val.csv", "test.csv"):
        path = DATA_PROCESSED / name
        if path.is_file():
            result[f"{name}_sha256"] = sha256_file(path)
    return result


def build_training_run_record(
    *,
    model_dir: Path,
    trainer: Any | None = None,
    training_recipe: dict[str, Any] | None = None,
    eval_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings = load_settings()
    model_dir = Path(model_dir)
    identity = read_checkpoint_identity(model_dir)
    state_path = resolve_trainer_state_path(model_dir)
    runtime = runtime_from_trainer_state(state_path) if state_path else {}

    params: dict[str, int] = {}
    args_snapshot: dict[str, Any] = {}
    if trainer is not None:
        params = count_model_parameters(trainer.model)
        args_snapshot = training_args_snapshot(trainer.args)
        if not runtime and hasattr(trainer, "state"):
            for entry in reversed(getattr(trainer.state, "log_history", []) or []):
                if "train_runtime" in entry:
                    runtime["train_runtime_seconds"] = float(entry["train_runtime"])
                    runtime["train_samples_per_second"] = entry.get("train_samples_per_second")
                    runtime["train_steps_per_second"] = entry.get("train_steps_per_second")
                    break
            runtime.setdefault("global_step", trainer.state.global_step)

    record: dict[str, Any] = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "base_model": settings["model"]["name"],
        "model_dir": str(model_dir.resolve()),
        "seed": settings["project"]["seed"],
        "checkpoint_identity": identity,
        "parameters": params,
        "training_arguments": args_snapshot,
        "runtime": runtime,
        "dataset": dataset_hashes(),
        "training_recipe": training_recipe or {},
        "eval_metrics": eval_metrics or {},
    }
    return record


def write_training_run(
    model_dir: Path,
    *,
    trainer: Any | None = None,
    training_recipe: dict[str, Any] | None = None,
    eval_metrics: dict[str, Any] | None = None,
) -> Path:
    model_dir = Path(model_dir)
    record = build_training_run_record(
        model_dir=model_dir,
        trainer=trainer,
        training_recipe=training_recipe,
        eval_metrics=eval_metrics,
    )
    out_path = model_dir / "training_run.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return out_path
