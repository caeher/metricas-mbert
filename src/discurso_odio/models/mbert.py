"""Fine-tuning de mBERT con Hugging Face Transformers 5.x."""

from __future__ import annotations

import json
from dataclasses import dataclass
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
from discurso_odio.data.loaders import (
    compute_sample_loss_weights,
    get_b2_class_weights,
    get_class_weights,
)
from discurso_odio.models.inference_contract import (
    apply_normalization,
    get_inference_contract,
    resolve_inference_contract,
    write_inference_contract,
)
from discurso_odio.paths import MODELS_DIR
from discurso_odio.utils.seed import set_seed


def default_mbert_model_dir() -> Path:
    b2 = MODELS_DIR / "mbert-sv-b2"
    if (b2 / "model.safetensors").exists():
        return b2
    return MODELS_DIR / "mbert-sv"


def ensure_checkpoint_contract(model_dir: str | Path | None = None) -> Path:
    """Escribe inference_contract.json en el directorio del checkpoint."""
    model_dir = Path(model_dir) if model_dir else default_mbert_model_dir()
    return write_inference_contract(model_dir, get_inference_contract())


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_label_maps() -> tuple[dict[int, str], dict[str, int]]:
    settings = load_settings()
    names = list(settings["labels"]["names"])
    id2label = {i: name for i, name in enumerate(names)}
    label2id = {name: i for i, name in enumerate(names)}
    return id2label, label2id


def build_tokenizer(model_name: str | None = None):
    settings = load_settings()
    model_name = model_name or settings["model"]["name"]
    return AutoTokenizer.from_pretrained(model_name)


def build_model(model_name: str | None = None, num_labels: int | None = None):
    settings = load_settings()
    model_name = model_name or settings["model"]["name"]
    num_labels = num_labels or settings["labels"]["num_classes"]
    id2label, label2id = build_label_maps()
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )


def make_hf_dataset(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    tokenizer,
    max_length: int,
    sample_weights: pd.Series | np.ndarray | None = None,
) -> Dataset:
    texts = df[text_col].fillna("").astype(str).tolist()
    labels = df[label_col].astype(int).tolist()
    data: dict[str, list] = {"text": texts, "labels": labels}
    if sample_weights is not None:
        data["sample_weight"] = np.asarray(sample_weights, dtype=np.float32).tolist()

    def tokenize(batch):
        tokenized = tokenizer(batch["text"], truncation=True, max_length=max_length)
        if "sample_weight" in batch:
            tokenized["sample_weight"] = batch["sample_weight"]
        return tokenized

    ds = Dataset.from_dict(data)
    return ds.map(tokenize, batched=True, remove_columns=["text"])


@dataclass
class SampleWeightCollator(DataCollatorWithPadding):
    """Collator que conserva sample_weight para la pérdida ponderada por fila."""

    def __call__(self, features):
        weights = [float(f.pop("sample_weight", 1.0)) for f in features]
        batch = super().__call__(features)
        batch["sample_weight"] = torch.tensor(weights, dtype=torch.float32)
        return batch


class MbertTrainer(Trainer):
    """Trainer con pérdida ponderada por clase, fila y label smoothing."""

    def __init__(
        self,
        *args,
        class_weights: dict[int, float] | None = None,
        label_smoothing: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        sample_weight = inputs.pop("sample_weight", None)
        outputs = model(**inputs)
        logits = outputs.logits

        weight_tensor = None
        if self.class_weights:
            weight_tensor = torch.tensor(
                [self.class_weights[i] for i in sorted(self.class_weights)],
                dtype=torch.float32,
                device=logits.device,
            )

        loss_fn = CrossEntropyLoss(
            weight=weight_tensor,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        if sample_weight is not None:
            loss = (loss * sample_weight.to(logits.device)).mean()
        else:
            loss = loss.mean()

        return (loss, outputs) if return_outputs else loss


def build_trainer(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: str | Path,
    text_col: str | None = None,
    label_col: str | None = None,
    model_name: str | None = None,
    *,
    class_weights: dict[int, float] | None = None,
    sample_weights: pd.Series | np.ndarray | None = None,
    label_smoothing: float = 0.0,
    use_b2_class_weights: bool = False,
) -> tuple[MbertTrainer, AutoTokenizer]:
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

    if class_weights is None:
        class_weights = get_b2_class_weights() if use_b2_class_weights else get_class_weights(train_df, label_col)

    train_ds = make_hf_dataset(
        train_df, text_col, label_col, tokenizer, max_length, sample_weights=sample_weights
    )
    val_ds = make_hf_dataset(val_df, text_col, label_col, tokenizer, max_length)

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

    trainer = MbertTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=SampleWeightCollator(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        label_smoothing=label_smoothing,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=train_cfg["early_stopping_patience"])],
    )
    return trainer, tokenizer


def train_mbert(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: str | Path | None = None,
    text_col: str | None = None,
    *,
    class_weights: dict[int, float] | None = None,
    sample_weights: pd.Series | np.ndarray | None = None,
    label_smoothing: float = 0.0,
    use_b2_class_weights: bool = False,
) -> MbertTrainer:
    settings = load_settings()
    output_dir = Path(output_dir or settings["model"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer, tokenizer = build_trainer(
        train_df,
        val_df,
        output_dir,
        text_col=text_col,
        class_weights=class_weights,
        sample_weights=sample_weights,
        label_smoothing=label_smoothing,
        use_b2_class_weights=use_b2_class_weights,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    write_inference_contract(output_dir)
    from discurso_odio.models.mbert_artifacts import apply_id2label

    apply_id2label(output_dir / "config.json")

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
