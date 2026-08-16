#!/usr/bin/env python
"""Inferencia interactiva mBERT con el contrato de auditoría."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discurso_odio.models.inference_contract import (
    apply_normalization_single,
    resolve_inference_contract,
)
from discurso_odio.models.mbert import default_mbert_model_dir

LABELS = ["No Tóxico", "Lenguaje Ofensivo", "Discurso de Odio", "Amenazas/Violencia"]


def cargar_modelo(model_dir: Path):
    if not model_dir.exists():
        raise FileNotFoundError(f"No se encontró el checkpoint: {model_dir}")
    contract = resolve_inference_contract(model_dir)
    print(f"Cargando tokenizador y modelo desde: {model_dir}")
    print(
        f"Contrato: max_length={contract.max_length}, "
        f"text_column={contract.text_column}, "
        f"normalizer={contract.normalizer} v{contract.normalizer_version}"
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Dispositivo activo: {device}")
    return tokenizer, model, device, contract


def predecir_toxicidad(texto: str, tokenizer, model, device, contract):
    texto_norm = apply_normalization_single(texto, contract)
    inputs = tokenizer(
        texto_norm,
        return_tensors="pt",
        truncation=True,
        max_length=contract.max_length,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    pred_idx = int(torch.argmax(probs).item())
    return LABELS[pred_idx], float(probs[pred_idx].item()), probs, pred_idx


def main():
    parser = argparse.ArgumentParser(description="Inferencia mBERT (texto_modelo / max_length=128)")
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--text", type=str, default=None, help="Si se omite, entra al modo interactivo")
    args = parser.parse_args()
    model_dir = args.model_dir or default_mbert_model_dir()
    tokenizer, model, device, contract = cargar_modelo(model_dir)

    if args.text:
        clase, conf, probs, idx = predecir_toxicidad(args.text, tokenizer, model, device, contract)
        print(f"-> {idx} {clase} ({conf:.2%})")
        for i, label in enumerate(LABELS):
            print(f"   - {label}: {probs[i].item():.2%}")
        return

    ejemplos = [
        "Hola a todos, feliz inicio de semana",
        "Ese maje no sabe lo que esta haciendo",
        "Te voy a ir a buscar a la salida para romperte la cara",
    ]
    print("\n=== EJEMPLOS ===")
    for texto in ejemplos:
        clase, conf, _, idx = predecir_toxicidad(texto, tokenizer, model, device, contract)
        print(f"\nTexto: {texto}\nPredicción: {idx} {clase} ({conf:.2%})")

    print("\n--- MODO INTERACTIVO (salir/exit para terminar) ---")
    while True:
        entrada = input("\nTexto > ")
        if entrada.lower() in {"salir", "exit", "quit"}:
            break
        if not entrada.strip():
            continue
        clase, conf, probs, idx = predecir_toxicidad(entrada, tokenizer, model, device, contract)
        print(f"-> {idx} {clase} ({conf:.2%})")
        for i, label in enumerate(LABELS):
            print(f"   - {label}: {probs[i].item():.2%}")


if __name__ == "__main__":
    main()
