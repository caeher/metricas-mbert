"""Continued pretraining MLM (DAPT) sobre train+val — Fase 4."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from discurso_odio.config import get_training_config, load_settings
from discurso_odio.data.loaders import load_split
from discurso_odio.models.inference_contract import get_inference_contract, write_inference_contract
from discurso_odio.utils.seed import set_seed


def build_dapt_texts(*, include_val: bool = True) -> list[str]:
    """Textos para MLM: train + val (nunca test)."""
    train_df = load_split("train")
    texts = train_df["texto_modelo"].fillna("").astype(str).tolist()
    if include_val:
        val_df = load_split("val")
        texts.extend(val_df["texto_modelo"].fillna("").astype(str).tolist())
    return [t for t in texts if t.strip()]


def train_mbert_dapt(
    output_dir: str | Path,
    *,
    model_name: str | None = None,
    num_train_epochs: float = 2.0,
    learning_rate: float = 5e-5,
    seed: int | None = None,
    include_val: bool = True,
) -> Path:
    """Preentrena mBERT con MLM sobre corpus local."""
    settings = load_settings()
    train_cfg = get_training_config()
    seed = seed or settings["project"]["seed"]
    set_seed(seed)

    model_name = model_name or settings["model"]["name"]
    contract = get_inference_contract()
    max_length = contract.max_length
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    texts = build_dapt_texts(include_val=include_val)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    ds = Dataset.from_dict({"text": texts}).map(tokenize, batched=True, remove_columns=["text"])

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        learning_rate=learning_rate,
        warmup_ratio=0.06,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_steps=100,
        seed=seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    write_inference_contract(output_dir)

    meta = {
        "model_name": model_name,
        "n_texts": len(texts),
        "include_val": include_val,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "seed": seed,
        "output_dir": str(output_dir),
    }
    (output_dir / "dapt_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output_dir
