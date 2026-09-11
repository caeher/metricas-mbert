"""Fine-tuning de mBERT con Hugging Face Transformers 5.x."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
from discurso_odio.models.losses import combined_ordinal_loss
from discurso_odio.paths import MODELS_DIR
from discurso_odio.utils.seed import set_seed

LossType = Literal["ce", "focal", "ordinal"]


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


def cuda_training_kwargs() -> dict[str, bool]:
    """Opciones de Trainer que aceleran el entrenamiento en GPU NVIDIA."""
    use_cuda = torch.cuda.is_available()
    return {
        "fp16": use_cuda,
        "dataloader_pin_memory": use_cuda,
    }


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


def build_model(
    model_name: str | None = None,
    num_labels: int | None = None,
    *,
    classifier_dropout: float | None = None,
):
    settings = load_settings()
    model_name = model_name or settings["model"]["name"]
    num_labels = num_labels or settings["labels"]["num_classes"]
    id2label, label2id = build_label_maps()
    kwargs: dict = {
        "num_labels": num_labels,
        "id2label": id2label,
        "label2id": label2id,
    }
    if classifier_dropout is not None:
        kwargs["classifier_dropout"] = classifier_dropout
    return AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)


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
        loss_type: LossType = "ce",
        ordinal_weight: float = 0.3,
        focal_gamma: float = 2.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type
        self.ordinal_weight = ordinal_weight
        self.focal_gamma = focal_gamma

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

        if self.loss_type == "ce":
            loss_fn = CrossEntropyLoss(
                weight=weight_tensor,
                reduction="none",
                label_smoothing=self.label_smoothing,
            )
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        else:
            flat_logits = logits.view(-1, logits.size(-1))
            flat_labels = labels.view(-1)
            gamma = self.focal_gamma if self.loss_type == "focal" else 0.0
            ord_w = self.ordinal_weight if self.loss_type == "ordinal" else 0.0
            if self.loss_type == "focal":
                ord_w = 0.0
            loss = combined_ordinal_loss(
                flat_logits,
                flat_labels,
                ordinal_weight=ord_w,
                gamma=gamma,
                class_weights=weight_tensor,
                label_smoothing=self.label_smoothing,
            )

        if sample_weight is not None:
            loss = (loss * sample_weight.to(logits.device)).mean()
        else:
            loss = loss.mean()

        return (loss, outputs) if return_outputs else loss


@dataclass
class MbertTrainingConfig:
    """Hiperparámetros configurables para barrido Fase 1–2."""

    learning_rate: float | None = None
    num_train_epochs: int | None = None
    early_stopping_patience: int | None = None
    seed: int | None = None
    label_smoothing: float = 0.0
    loss_type: LossType = "ce"
    ordinal_weight: float = 0.3
    focal_gamma: float = 2.0
    classifier_dropout: float | None = None
    layerwise_lr_decay: float | None = None
    warmup_ratio: float | None = None
    weight_decay: float | None = None


def _build_layerwise_optimizer(model, trainer_args, layerwise_lr_decay: float):
    """Asigna LR decreciente por capas del encoder BERT."""
    no_decay = ["bias", "LayerNorm.weight"]
    base_lr = trainer_args.learning_rate
    encoder = getattr(model, "bert", None) or getattr(model, "roberta", None)
    if encoder is None:
        return None

    param_groups: list[dict] = []
    n_layers = len(encoder.encoder.layer)
    for layer_idx, layer in enumerate(encoder.encoder.layer):
        lr = base_lr * (layerwise_lr_decay ** (n_layers - layer_idx))
        param_groups.append(
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": trainer_args.weight_decay,
                "lr": lr,
            }
        )
        param_groups.append(
            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr,
            }
        )

    head_prefixes = ("classifier", "dropout")
    head_params = [
        p for n, p in model.named_parameters()
        if n.startswith(head_prefixes) and not any(nd in n for nd in no_decay)
    ]
    head_params_nd = [
        p for n, p in model.named_parameters()
        if n.startswith(head_prefixes) and any(nd in n for nd in no_decay)
    ]
    if head_params:
        param_groups.append({"params": head_params, "weight_decay": trainer_args.weight_decay, "lr": base_lr})
    if head_params_nd:
        param_groups.append({"params": head_params_nd, "weight_decay": 0.0, "lr": base_lr})

    embed_params = [p for n, p in encoder.embeddings.named_parameters()]
    if embed_params:
        embed_lr = base_lr * (layerwise_lr_decay ** (n_layers + 1))
        param_groups.append({"params": embed_params, "weight_decay": trainer_args.weight_decay, "lr": embed_lr})

    optimizer = torch.optim.AdamW(param_groups)
    num_training_steps = trainer_args.max_steps if trainer_args.max_steps > 0 else (
        len(trainer_args.__dict__.get("_train_dataloader", []))  # fallback unused
    )
    return optimizer


class LayerwiseTrainer(MbertTrainer):
    """Trainer con optimizador layer-wise LR decay."""

    def __init__(self, *args, layerwise_lr_decay: float = 0.95, **kwargs):
        self.layerwise_lr_decay = layerwise_lr_decay
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        opt = _build_layerwise_optimizer(self.model, self.args, self.layerwise_lr_decay)
        if opt is not None:
            self.optimizer = opt
            return self.optimizer
        return super().create_optimizer()


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
    training_config: MbertTrainingConfig | None = None,
    init_model_dir: str | Path | None = None,
) -> tuple[MbertTrainer, AutoTokenizer]:
    settings = load_settings()
    train_cfg = get_training_config()
    cfg = training_config or MbertTrainingConfig()
    max_steps = get_max_steps()
    seed = cfg.seed if cfg.seed is not None else settings["project"]["seed"]
    set_seed(seed)

    text_col = text_col or settings["model"]["text_column"]
    label_col = label_col or settings["model"]["label_column"]
    model_name = model_name or settings["model"]["name"]
    contract = get_inference_contract()
    max_length = contract.max_length
    device = get_device()

    tokenizer = build_tokenizer(model_name)
    if init_model_dir:
        from transformers import AutoModelForMaskedLM

        init_path = str(init_model_dir)
        model = build_model(model_name, classifier_dropout=cfg.classifier_dropout)
        try:
            model = AutoModelForSequenceClassification.from_pretrained(init_path)
        except (OSError, ValueError):
            mlm = AutoModelForMaskedLM.from_pretrained(init_path)
            encoder = getattr(model, "bert", None) or getattr(model, "roberta", None)
            mlm_encoder = getattr(mlm, "bert", None) or getattr(mlm, "roberta", None)
            if encoder is not None and mlm_encoder is not None:
                encoder.load_state_dict(mlm_encoder.state_dict(), strict=False)
    else:
        model = build_model(model_name, classifier_dropout=cfg.classifier_dropout)
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
        num_train_epochs=cfg.num_train_epochs or train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        learning_rate=cfg.learning_rate or train_cfg["learning_rate"],
        warmup_ratio=cfg.warmup_ratio if cfg.warmup_ratio is not None else train_cfg["warmup_ratio"],
        weight_decay=cfg.weight_decay if cfg.weight_decay is not None else train_cfg["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=50,
        seed=seed,
        report_to="none",
        max_steps=max_steps if max_steps > 0 else -1,
        **cuda_training_kwargs(),
    )

    patience = cfg.early_stopping_patience or train_cfg["early_stopping_patience"]
    trainer_cls = LayerwiseTrainer if cfg.layerwise_lr_decay else MbertTrainer
    trainer_kwargs: dict = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "processing_class": tokenizer,
        "data_collator": SampleWeightCollator(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
        "class_weights": class_weights,
        "label_smoothing": cfg.label_smoothing if training_config else label_smoothing,
        "loss_type": cfg.loss_type,
        "ordinal_weight": cfg.ordinal_weight,
        "focal_gamma": cfg.focal_gamma,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=patience)],
    }
    if cfg.layerwise_lr_decay:
        trainer_kwargs["layerwise_lr_decay"] = cfg.layerwise_lr_decay

    trainer = trainer_cls(**trainer_kwargs)
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
    training_config: MbertTrainingConfig | None = None,
    init_model_dir: str | Path | None = None,
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
        training_config=training_config,
        init_model_dir=init_model_dir,
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
