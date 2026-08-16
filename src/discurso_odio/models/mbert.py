"""Fine-tuning de mBERT con Hugging Face Transformers 5.x."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from discurso_odio.config import get_max_steps, get_training_config, load_settings
from discurso_odio.data.loaders import get_class_weights
from discurso_odio.models.inference_contract import (
    apply_normalization,
    get_inference_contract,
    resolve_inference_contract,
    write_inference_contract,
)
from discurso_odio.paths import MODELS_DIR
from discurso_odio.utils.seed import set_seed


def default_mbert_model_dir() -> Path:
    return MODELS_DIR / "mbert-sv"


def ensure_checkpoint_contract(model_dir: str | Path | None = None) -> Path:
    """Escribe inference_contract.json en el directorio del checkpoint."""
    model_dir = Path(model_dir) if model_dir else default_mbert_model_dir()
    return write_inference_contract(model_dir, get_inference_contract())


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_tokenizer(model_name: str | None = None):
    settings = load_settings()
    model_name = model_name or settings["model"]["name"]
    return AutoTokenizer.from_pretrained(model_name)


def build_model(model_name: str | None = None, num_labels: int | None = None):
    settings = load_settings()
    model_name = model_name or settings["model"]["name"]
    num_labels = num_labels or settings["labels"]["num_classes"]
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)


def make_hf_dataset(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    tokenizer,
    max_length: int,
) -> Dataset:
    texts = df[text_col].fillna("").astype(str).tolist()
    labels = df[label_col].astype(int).tolist()

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    ds = Dataset.from_dict({"text": texts, "labels": labels})
    return ds.map(tokenize, batched=True, remove_columns=["text"])


def make_weighted_loss_fn(class_weights: dict[int, float], device: torch.device):
    weights = torch.tensor(
        [class_weights[i] for i in sorted(class_weights)],
        dtype=torch.float32,
        device=device,
    )

    def compute_loss_func(outputs, labels, num_items_in_batch=None):
        loss_fn = CrossEntropyLoss(weight=weights)
        return loss_fn(outputs.logits, labels)

    return compute_loss_func


def build_trainer(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: str | Path,
    text_col: str | None = None,
    label_col: str | None = None,
    model_name: str | None = None,
) -> tuple[Trainer, AutoTokenizer]:
    settings = load_settings()
    train_cfg = get_training_config()
    max_steps = get_max_steps()
    set_seed(settings["project"]["seed"])

    text_col = text_col or settings["model"]["text_column"]
    label_col = label_col or settings["model"]["label_column"]
    model_name = model_name or settings["model"]["name"]
    contract = get_inference_contract()
    max_length = contract.max_length
    device = get_device()

    tokenizer = build_tokenizer(model_name)
    model = build_model(model_name)
    model.to(device)

    train_ds = make_hf_dataset(train_df, text_col, label_col, tokenizer, max_length)
    val_ds = make_hf_dataset(val_df, text_col, label_col, tokenizer, max_length)

    class_weights = get_class_weights(train_df, label_col)
    compute_loss_func = make_weighted_loss_fn(class_weights, device)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            "accuracy": float((preds == labels).mean()),
        }

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=50,
        seed=settings["project"]["seed"],
        report_to="none",
        max_steps=max_steps if max_steps > 0 else -1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        compute_loss_func=compute_loss_func,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=train_cfg["early_stopping_patience"])],
    )
    return trainer, tokenizer


def train_mbert(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: str | Path | None = None,
    text_col: str | None = None,
) -> Trainer:
    settings = load_settings()
    output_dir = Path(output_dir or settings["model"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer, tokenizer = build_trainer(train_df, val_df, output_dir, text_col=text_col)
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    write_inference_contract(output_dir)

    eval_metrics = trainer.evaluate()
    with open(output_dir / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)

    return trainer


def predict(
    texts: list[str],
    model_dir: str | Path | None = None,
    batch_size: int = 16,
    *,
    max_length: int | None = None,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (predicciones, probabilidades) usando el contrato del checkpoint."""
    model_dir = Path(model_dir) if model_dir else default_mbert_model_dir()
    if not model_dir.exists():
        raise FileNotFoundError(f"Checkpoint mBERT no encontrado: {model_dir}")

    contract = resolve_inference_contract(model_dir)
    max_length = contract.validate_max_length(max_length)
    device = get_device()

    if normalize:
        texts = apply_normalization(texts, contract)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    all_preds: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        all_preds.append(np.argmax(probs, axis=-1))

    return np.concatenate(all_preds), np.concatenate(all_probs)
