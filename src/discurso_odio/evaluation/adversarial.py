"""Evaluación de robustez sobre la batería adversarial versionada (T-06)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from discurso_odio.config import load_settings
from discurso_odio.data.loaders import load_split
from discurso_odio.evaluation.predictions import git_commit
from discurso_odio.evaluation.slices import _wilson_ci
from discurso_odio.models.mbert import predict
from discurso_odio.models.mbert_artifacts import sha256_file
from discurso_odio.data.adversarial_battery import (
    BATTERY_144_N,
    BATTERY_144_VERSION,
    FAMILY_SPECS_144,
    GROUP_F,
    GROUP_G,
)
from discurso_odio.paths import (
    ADVERSARIAL_144_BATTERY_MANIFEST_PATH,
    ADVERSARIAL_BATTERY_MANIFEST_PATH,
    REPORTS_METRICS,
    TEST_ADVERSARIAL_144_CSV,
    TEST_ADVERSARIAL_CSV,
)

BATTERY_VERSION = "adversarial_v1"
BATTERY_N = 75

FAMILY_SPECS: dict[str, dict[str, Any]] = {
    "A. Ofuscación de groserías": {"key": "A", "n": 20, "metric": "accuracy"},
    "B. Emojis": {"key": "B", "n": 18, "metric": "accuracy"},
    "C. Amenazas ofuscadas": {"key": "C", "n": 11, "metric": "accuracy"},
    "D. Jerga salvadoreña inofensiva": {"key": "D", "n": 14, "metric": "fpr"},
    "E. Control sin jerga": {"key": "E", "n": 12, "metric": "fpr"},
}

XLMR_V1_BASELINE = {
    "A. Ofuscación de groserías": {"correct": 17, "n": 20},
    "B. Emojis": {"correct": 6, "n": 18},
    "C. Amenazas ofuscadas": {"correct": 5, "n": 11},
    "D. Jerga salvadoreña inofensiva": {"fp": 6, "n": 14},
    "E. Control sin jerga": {"fp": 0, "n": 12},
    "global": {"correct": 48, "n": 75},
}

XLMR_V1_BASELINE_144 = {
    **XLMR_V1_BASELINE,
    GROUP_F: {"fp": 26, "n": 34},
    GROUP_G: {"correct": 12, "n": 35},
    "global": {"correct": 74, "n": 144},
}

MITIGATION_THRESHOLDS = {
    "group_c_max_correct": 6,
    "group_c_min_fn": 4,
    "group_b_max_correct": 9,
    "fpr_gap_min": 0.15,
}


def adversarial_text_set(df: pd.DataFrame | None = None) -> set[str]:
    """Conjunto normalizado de textos de la batería (modelo + original)."""
    if df is None:
        df = load_split("adversarial")
    texts: set[str] = set()
    for col in ("texto_modelo", "texto_original"):
        if col in df.columns:
            texts |= set(df[col].astype(str).str.strip().str.lower()) - {"", "nan"}
    return texts


def build_battery_manifest(*, write: bool = True) -> dict[str, Any]:
    """Construye o valida el manifiesto versionado de la batería adversarial."""
    adv_path = TEST_ADVERSARIAL_CSV
    adv_df = pd.read_csv(adv_path)
    manifest: dict[str, Any] = {
        "battery_version": BATTERY_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": load_settings()["project"]["seed"],
        "path": str(adv_path.relative_to(adv_path.parents[2])),
        "n_cases": len(adv_df),
        "sha256": sha256_file(adv_path),
        "families": {name: spec["n"] for name, spec in FAMILY_SPECS.items()},
        "rule": "diagnóstico: se reporta aparte, nunca como métrica principal",
        "git_commit": git_commit(),
    }
    if write:
        ADVERSARIAL_BATTERY_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ADVERSARIAL_BATTERY_MANIFEST_PATH, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return manifest


def load_battery_manifest() -> dict[str, Any]:
    if ADVERSARIAL_BATTERY_MANIFEST_PATH.exists():
        with open(ADVERSARIAL_BATTERY_MANIFEST_PATH, encoding="utf-8") as handle:
            return json.load(handle)
    return build_battery_manifest(write=True)


def _normalized_texts_from_split(df: pd.DataFrame) -> set[str]:
    texts: set[str] = set()
    for col in ("texto_modelo", "texto_original"):
        if col in df.columns:
            texts |= set(df[col].astype(str).str.strip().str.lower()) - {"", "nan"}
    return texts


def find_split_overlaps(
    adversarial_df: pd.DataFrame | None = None,
    *,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> pd.DataFrame:
    """Devuelve filas de la batería que coinciden textualmente con un split."""
    adversarial_df = adversarial_df if adversarial_df is not None else load_split("adversarial")
    adv_texts = adversarial_text_set(adversarial_df)
    rows: list[dict[str, Any]] = []
    for split in splits:
        split_df = load_split(split)
        split_texts = _normalized_texts_from_split(split_df)
        overlap = adv_texts & split_texts
        for text in sorted(overlap):
            adv_rows = adversarial_df[
                adversarial_df["texto_modelo"].astype(str).str.strip().str.lower().eq(text)
                | adversarial_df["texto_original"].astype(str).str.strip().str.lower().eq(text)
            ]
            for _, adv_row in adv_rows.iterrows():
                rows.append(
                    {
                        "split": split,
                        "adv_id": adv_row["ID"],
                        "grupo": adv_row["grupo"],
                        "text": text[:120],
                    }
                )
    return pd.DataFrame(rows)


def find_train_overlaps(train_df: pd.DataFrame | None = None) -> pd.DataFrame:
    return _find_train_overlaps(train_df)


def _find_train_overlaps(train_df: pd.DataFrame | None = None) -> pd.DataFrame:
    adversarial_df = load_split("adversarial")
    train_df = train_df if train_df is not None else load_split("train")
    adv_texts = adversarial_text_set(adversarial_df)
    rows: list[dict[str, Any]] = []
    for _, row in train_df.iterrows():
        text = str(row["texto_modelo"]).strip().lower()
        if text in adv_texts:
            adv_match = adversarial_df[
                adversarial_df["texto_modelo"].astype(str).str.strip().str.lower().eq(text)
            ]
            adv_id = adv_match.iloc[0]["ID"] if len(adv_match) else ""
            adv_grupo = adv_match.iloc[0]["grupo"] if len(adv_match) else ""
            rows.append(
                {
                    "split": "train",
                    "adv_id": adv_id,
                    "grupo": adv_grupo,
                    "text": text[:120],
                    "origen": row.get("origen", ""),
                }
            )
    return pd.DataFrame(rows)


def assert_adversarial_not_in_splits(
    *,
    splits: tuple[str, ...] = ("val", "test"),
    fail_on_train_overlap: bool = False,
) -> pd.DataFrame:
    """Verifica que la batería no contamine val/test (y opcionalmente train)."""
    overlaps = find_split_overlaps(splits=splits)
    if not overlaps.empty:
        sample = overlaps.head(5).to_dict(orient="records")
        raise ValueError(
            f"La batería adversarial coincide con {splits}: {len(overlaps)} casos. "
            f"Muestra: {sample}"
        )
    train_overlaps = find_train_overlaps()
    if fail_on_train_overlap and not train_overlaps.empty:
        raise ValueError(
            f"La batería adversarial coincide con train en {len(train_overlaps)} filas."
        )
    return train_overlaps


def _family_key(grupo: str) -> str:
    return FAMILY_SPECS.get(grupo, {}).get("key", grupo[:1])


def _accuracy_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    n = len(y_true)
    correct = int((y_true == y_pred).sum())
    acc = correct / n if n else float("nan")
    ci_lo, ci_hi = _wilson_ci(correct, n) if n else (float("nan"), float("nan"))
    return {"n": n, "correct": correct, "accuracy": acc, "acc_ci_lo": ci_lo, "acc_ci_hi": ci_hi}


def _fpr_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    mask = y_true == 0
    n = int(mask.sum())
    fp = int(((y_pred >= 1) & mask).sum()) if n else 0
    fpr = fp / n if n else float("nan")
    ci_lo, ci_hi = _wilson_ci(fp, n) if n else (float("nan"), float("nan"))
    return {"n": n, "fp": fp, "fpr": fpr, "fpr_ci_lo": ci_lo, "fpr_ci_hi": ci_hi}


def _threat_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    mask = y_true == 3
    n = int(mask.sum())
    if n == 0:
        return {
            "n_threats": 0,
            "threat_fn": 0,
            "threat_fn_rate": float("nan"),
            "f1_amenazas": float("nan"),
        }
    fn = int(((y_pred != 3) & mask).sum())
    f1 = float(f1_score(y_true[mask], y_pred[mask], labels=[3], average="macro", zero_division=0))
    return {
        "n_threats": n,
        "threat_fn": fn,
        "threat_fn_rate": fn / n,
        "f1_amenazas": f1,
    }


def _confidence_stats(confidence: np.ndarray, correct: np.ndarray) -> dict[str, float]:
    if len(confidence) == 0:
        return {
            "conf_mean": float("nan"),
            "conf_mean_correct": float("nan"),
            "conf_mean_incorrect": float("nan"),
        }
    return {
        "conf_mean": float(np.mean(confidence)),
        "conf_mean_correct": float(np.mean(confidence[correct])) if correct.any() else float("nan"),
        "conf_mean_incorrect": float(np.mean(confidence[~correct])) if (~correct).any() else float("nan"),
    }


def evaluate_adversarial_battery(
    *,
    model_dir: Path,
    model_name: str = "mbert",
    checkpoint: str = "",
    run_id: str | None = None,
    output_prefix: str = "mbert",
) -> dict[str, Any]:
    """Evalúa un checkpoint mBERT sobre test_adversarial.csv."""
    manifest = load_battery_manifest()
    current_sha = sha256_file(TEST_ADVERSARIAL_CSV)
    if manifest.get("sha256") and manifest["sha256"] != current_sha:
        raise ValueError(
            f"SHA-256 de test_adversarial.csv ({current_sha}) no coincide con el manifiesto "
            f"({manifest['sha256']})."
        )

    train_overlaps = assert_adversarial_not_in_splits(splits=("val", "test"))

    adv_df = load_split("adversarial")
    if len(adv_df) != BATTERY_N:
        raise ValueError(f"Se esperaban {BATTERY_N} casos adversariales, hay {len(adv_df)}.")

    y_pred, y_prob = predict(adv_df["texto_modelo"].tolist(), model_dir)
    y_true = adv_df["label"].values.astype(int)
    y_pred = np.asarray(y_pred, dtype=int)
    confidence = np.max(y_prob, axis=1)
    correct = y_true == y_pred

    pred_df = adv_df.copy()
    pred_df["y_true"] = y_true
    pred_df["y_pred"] = y_pred
    pred_df["confianza"] = confidence
    pred_df["acierto"] = correct

    family_rows: list[dict[str, Any]] = []
    for grupo, spec in FAMILY_SPECS.items():
        mask = adv_df["grupo"].values == grupo
        if not mask.any():
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        conf = confidence[mask]
        corr = correct[mask]
        row: dict[str, Any] = {
            "model": model_name,
            "checkpoint": checkpoint,
            "grupo": grupo,
            "family_key": spec["key"],
            "metric_type": spec["metric"],
            **_confidence_stats(conf, corr),
        }
        if spec["metric"] == "accuracy":
            row.update(_accuracy_metrics(yt, yp))
        else:
            row.update(_fpr_metrics(yt, yp))
        pred_dist = pd.Series(yp).value_counts().sort_index()
        for label in range(4):
            row[f"pred_label_{label}"] = int(pred_dist.get(label, 0))
        family_rows.append(row)

    family_df = pd.DataFrame(family_rows)
    threat = _threat_metrics(y_true, y_pred)
    global_acc = _accuracy_metrics(y_true, y_pred)
    global_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    d_row = family_df[family_df["family_key"] == "D"]
    e_row = family_df[family_df["family_key"] == "E"]
    fpr_gap = float("nan")
    ci_overlap = True
    if len(d_row) and len(e_row):
        fpr_d = float(d_row.iloc[0]["fpr"])
        fpr_e = float(e_row.iloc[0]["fpr"])
        fpr_gap = fpr_d - fpr_e
        ci_overlap = not (
            float(d_row.iloc[0]["fpr_ci_lo"]) > float(e_row.iloc[0]["fpr_ci_hi"])
            or float(e_row.iloc[0]["fpr_ci_lo"]) > float(d_row.iloc[0]["fpr_ci_hi"])
        )

    mitigation = assess_mitigation_gap(family_df, fpr_gap=fpr_gap, ci_overlap=ci_overlap)

    summary: dict[str, Any] = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "model": model_name,
        "checkpoint": checkpoint,
        "run_id": run_id,
        "battery_version": manifest["battery_version"],
        "battery_sha256": manifest["sha256"],
        "split": TEST_ADVERSARIAL_CSV.name,
        "n_cases": len(adv_df),
        "f1_macro_diagnostic": global_f1,
        "global_correct": global_acc["correct"],
        "global_accuracy": global_acc["accuracy"],
        "global_acc_ci_lo": global_acc["acc_ci_lo"],
        "global_acc_ci_hi": global_acc["acc_ci_hi"],
        **threat,
        "fpr_dialectal_gap": fpr_gap,
        "fpr_ci_overlap": ci_overlap,
        "mitigation_required": mitigation["mitigation_required"],
        "mitigation_reasons": mitigation["reasons"],
        "train_overlap_n": int(len(train_overlaps)),
        "train_overlap_note": (
            "Sin solapamiento con val/test. Filas de train B2 con texto idéntico "
            "se excluyen al reentrenar (T-06)."
            if len(train_overlaps) > 0
            else "Sin solapamiento textual con train/val/test."
        ),
        "xlmr_v1_baseline": XLMR_V1_BASELINE,
        "is_diagnostic_metric": True,
    }

    export_adversarial_artifacts(
        pred_df=pred_df,
        family_df=family_df,
        summary=summary,
        output_prefix=output_prefix,
    )
    return summary


def assess_mitigation_gap(
    family_df: pd.DataFrame,
    *,
    fpr_gap: float,
    ci_overlap: bool,
) -> dict[str, Any]:
    reasons: list[str] = []
    c_row = family_df[family_df["family_key"] == "C"]
    b_row = family_df[family_df["family_key"] == "B"]

    if len(c_row):
        correct = int(c_row.iloc[0]["correct"])
        n = int(c_row.iloc[0]["n"])
        fn = n - correct
        if correct <= MITIGATION_THRESHOLDS["group_c_max_correct"]:
            reasons.append(f"amenazas_ofuscadas_correct_{correct}_of_{n}")
        if fn > MITIGATION_THRESHOLDS["group_c_min_fn"]:
            reasons.append(f"amenazas_fn_{fn}_of_{n}")

    if len(b_row):
        correct = int(b_row.iloc[0]["correct"])
        n = int(b_row.iloc[0]["n"])
        if correct <= MITIGATION_THRESHOLDS["group_b_max_correct"]:
            reasons.append(f"emojis_correct_{correct}_of_{n}")

    if not np.isnan(fpr_gap):
        if fpr_gap >= MITIGATION_THRESHOLDS["fpr_gap_min"] or not ci_overlap:
            reasons.append(f"fpr_dialectal_gap_{fpr_gap:.4f}_ci_overlap_{ci_overlap}")

    return {"mitigation_required": bool(reasons), "reasons": reasons}


def export_adversarial_artifacts(
    *,
    pred_df: pd.DataFrame,
    family_df: pd.DataFrame,
    summary: dict[str, Any],
    output_prefix: str,
) -> tuple[Path, Path, Path]:
    REPORTS_METRICS.mkdir(parents=True, exist_ok=True)
    pred_path = REPORTS_METRICS / f"{output_prefix}_adversarial_predictions.csv"
    family_path = REPORTS_METRICS / f"{output_prefix}_adversarial_family_metrics.csv"
    summary_path = REPORTS_METRICS / f"{output_prefix}_adversarial_summary.json"

    export_cols = [
        "ID",
        "grupo",
        "base",
        "variante",
        "clase",
        "label",
        "y_pred",
        "confianza",
        "acierto",
    ]
    pred_df[export_cols].to_csv(pred_path, index=False)
    family_df.to_csv(family_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return pred_path, family_path, summary_path


def export_before_after(
    baseline_prefix: str = "mbert",
    candidate_prefix: str = "mbert_b2",
    *,
    mitigation_adopted: bool | None = None,
    adoption_notes: str = "",
) -> Path:
    """Tabla antes/después entre checkpoint oficial y candidato B2."""
    baseline_summary_path = REPORTS_METRICS / f"{baseline_prefix}_adversarial_summary.json"
    candidate_summary_path = REPORTS_METRICS / f"{candidate_prefix}_adversarial_summary.json"
    baseline_test_path = REPORTS_METRICS / f"{baseline_prefix}_parallel_test_metrics.csv"
    if not baseline_test_path.exists():
        baseline_test_path = REPORTS_METRICS / "mbert_test_metrics.csv"
    candidate_test_path = REPORTS_METRICS / f"{candidate_prefix}_parallel_test_metrics.csv"
    baseline_family_path = REPORTS_METRICS / f"{baseline_prefix}_adversarial_family_metrics.csv"
    candidate_family_path = REPORTS_METRICS / f"{candidate_prefix}_adversarial_family_metrics.csv"

    if not candidate_summary_path.exists():
        raise FileNotFoundError(f"No existe evaluación del candidato: {candidate_summary_path}")

    with open(baseline_summary_path, encoding="utf-8") as handle:
        baseline = json.load(handle)
    with open(candidate_summary_path, encoding="utf-8") as handle:
        candidate = json.load(handle)

    def _family_metric(path: Path, key: str, field: str) -> float | int:
        df = pd.read_csv(path)
        row = df[df["family_key"] == key]
        if row.empty or field not in row.columns:
            return float("nan")
        return row.iloc[0][field]

    baseline_test = pd.read_csv(baseline_test_path) if baseline_test_path.exists() else pd.DataFrame()
    candidate_test = pd.read_csv(candidate_test_path) if candidate_test_path.exists() else pd.DataFrame()

    rows = [
        {
            "stage": "baseline",
            "model": baseline.get("model", baseline_prefix),
            "checkpoint": baseline.get("checkpoint", ""),
            "test_f1_macro": baseline_test.iloc[0]["f1_macro"] if len(baseline_test) else float("nan"),
            "test_f1_amenazas": baseline_test.iloc[0]["f1_Amenazas/Violencia"] if len(baseline_test) else float("nan"),
            "adv_global_correct": baseline.get("global_correct"),
            "adv_n": baseline.get("n_cases"),
            "adv_f1_macro_diagnostic": baseline.get("f1_macro_diagnostic"),
            "adv_f1_amenazas": baseline.get("f1_amenazas"),
            "adv_threat_fn": baseline.get("threat_fn"),
            "group_c_correct": _family_metric(baseline_family_path, "C", "correct"),
            "group_b_correct": _family_metric(baseline_family_path, "B", "correct"),
            "fpr_d": _family_metric(baseline_family_path, "D", "fpr"),
            "fpr_e": _family_metric(baseline_family_path, "E", "fpr"),
            "fpr_gap": baseline.get("fpr_dialectal_gap"),
            "mitigation_adopted": False,
        },
        {
            "stage": "candidate_b2",
            "model": candidate.get("model", candidate_prefix),
            "checkpoint": candidate.get("checkpoint", ""),
            "test_f1_macro": candidate_test.iloc[0]["f1_macro"] if len(candidate_test) else float("nan"),
            "test_f1_amenazas": candidate_test.iloc[0]["f1_Amenazas/Violencia"] if len(candidate_test) else float("nan"),
            "adv_global_correct": candidate.get("global_correct"),
            "adv_n": candidate.get("n_cases"),
            "adv_f1_macro_diagnostic": candidate.get("f1_macro_diagnostic"),
            "adv_f1_amenazas": candidate.get("f1_amenazas"),
            "adv_threat_fn": candidate.get("threat_fn"),
            "group_c_correct": _family_metric(candidate_family_path, "C", "correct"),
            "group_b_correct": _family_metric(candidate_family_path, "B", "correct"),
            "fpr_d": _family_metric(candidate_family_path, "D", "fpr"),
            "fpr_e": _family_metric(candidate_family_path, "E", "fpr"),
            "fpr_gap": candidate.get("fpr_dialectal_gap"),
            "mitigation_adopted": mitigation_adopted,
        },
    ]
    out_df = pd.DataFrame(rows)
    out_path = REPORTS_METRICS / "mbert_adversarial_before_after.csv"
    out_df.to_csv(out_path, index=False)

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mitigation_adopted": mitigation_adopted,
        "adoption_notes": adoption_notes,
        "rows": out_df.to_dict(orient="records"),
    }
    with open(REPORTS_METRICS / "mbert_adversarial_before_after.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)
    return out_path


def decide_mitigation_adoption(before_after_df: pd.DataFrame) -> tuple[bool, str]:
    """Criterio del plan: mejora adversarial sin degradar test oficial."""
    if len(before_after_df) < 2:
        return False, "Faltan filas baseline/candidato."
    base = before_after_df.iloc[0]
    cand = before_after_df.iloc[1]

    c_improved = cand["group_c_correct"] > base["group_c_correct"] or cand["adv_threat_fn"] < base["adv_threat_fn"]
    gap_improved = cand["fpr_gap"] < base["fpr_gap"]
    f1_ok = pd.isna(base["test_f1_macro"]) or (
        cand["test_f1_macro"] >= base["test_f1_macro"] - 0.02
    )
    amen_ok = pd.isna(base["test_f1_amenazas"]) or (
        cand["test_f1_amenazas"] >= base["test_f1_amenazas"]
    )

    if c_improved and gap_improved and f1_ok and amen_ok:
        return True, "Mejora en amenazas ofuscadas y brecha FPR dialectal sin degradar test."
    reasons = []
    if not c_improved:
        reasons.append("sin mejora en grupo C/FN amenazas")
    if not gap_improved:
        reasons.append("sin reducción de brecha FPR D-E")
    if not f1_ok:
        reasons.append("F1-macro test cae >0.02")
    if not amen_ok:
        reasons.append("F1 Amenazas/Violencia en test empeora")
    return False, "; ".join(reasons)


def evaluate_test_metrics_parallel(
    *,
    model_dir: Path,
    model_name: str,
    checkpoint: str,
    output_prefix: str,
) -> dict[str, Any]:
    """Métricas sobre test.csv sin validar F1 oficial T-04."""
    from discurso_odio.evaluation.metrics import compute_all_metrics

    test = load_split("test")
    y_true = test["label"].values.astype(int)
    y_pred, y_prob = predict(test["texto_modelo"].tolist(), model_dir)
    metrics = compute_all_metrics(y_true, np.asarray(y_pred), y_prob)
    metrics_row = {
        "model": model_name,
        "checkpoint": checkpoint,
        "split": "test.csv",
        "n": len(test),
        "f1_macro": metrics["f1_macro"],
        "f1_Amenazas/Violencia": metrics.get("f1_Amenazas/Violencia", float("nan")),
        "accuracy": metrics["accuracy"],
    }
    out_path = REPORTS_METRICS / f"{output_prefix}_parallel_test_metrics.csv"
    if output_prefix == "mbert" and (REPORTS_METRICS / "mbert_test_metrics.csv").exists():
        # No sobrescribir artefactos T-04; reutilizar métricas oficiales en before/after.
        official = pd.read_csv(REPORTS_METRICS / "mbert_test_metrics.csv")
        if len(official):
            metrics_row["f1_macro"] = float(official.iloc[0].get("f1_macro", metrics_row["f1_macro"]))
            for col in official.columns:
                if col.startswith("f1_") and col in metrics_row:
                    metrics_row[col] = float(official.iloc[0][col])
    pd.DataFrame([metrics_row]).to_csv(out_path, index=False)
    return metrics_row


def build_battery_144_manifest(*, write: bool = True) -> dict[str, Any]:
    """Manifiesto versionado de la batería extendida A–G (n=144)."""
    adv_path = TEST_ADVERSARIAL_144_CSV
    if not adv_path.exists():
        from discurso_odio.data.adversarial_battery import write_adversarial_144_csv

        write_adversarial_144_csv(adv_path)
    adv_df = pd.read_csv(adv_path)
    manifest: dict[str, Any] = {
        "battery_version": BATTERY_144_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": load_settings()["project"]["seed"],
        "path": str(adv_path.relative_to(adv_path.parents[2])),
        "n_cases": len(adv_df),
        "sha256": sha256_file(adv_path),
        "families": {name: spec["n"] for name, spec in FAMILY_SPECS_144.items()},
        "rule": "diagnóstico: se reporta aparte, nunca como métrica principal",
        "git_commit": git_commit(),
    }
    if write:
        ADVERSARIAL_144_BATTERY_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ADVERSARIAL_144_BATTERY_MANIFEST_PATH, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return manifest


def load_battery_144_manifest() -> dict[str, Any]:
    if ADVERSARIAL_144_BATTERY_MANIFEST_PATH.exists():
        with open(ADVERSARIAL_144_BATTERY_MANIFEST_PATH, encoding="utf-8") as handle:
            return json.load(handle)
    return build_battery_144_manifest(write=True)


def _emoji_pair_metrics(
    adv_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """Métricas del grupo G: emoji inocuo que dispara toxicidad y emoji de ataque ignorado."""
    mask_g = adv_df["grupo"].values == GROUP_G
    if not mask_g.any():
        return {}

    g_df = adv_df.loc[mask_g].copy()
    g_df["y_true"] = y_true[mask_g]
    g_df["y_pred"] = y_pred[mask_g]

    innocuous_variants = {
        "corazon_inocuo",
        "saludo_inocuo",
        "manos_inocuo",
        "pastel_inocuo",
        "balon_inocuo",
        "risa_inocuo",
        "pulgar_inocuo",
        "confeti_inocuo",
    }
    attack_variants = {
        "pistola_amenaza",
        "cuchillo_amenaza",
        "ataud_amenaza",
        "calavera_amenaza",
        "bomba_amenaza",
    }

    inoc_mask = g_df["variante"].isin(innocuous_variants) & (g_df["y_true"] == 0)
    attack_mask = g_df["variante"].isin(attack_variants) & (g_df["y_true"] == 3)

    inoc_fp = int(((g_df.loc[inoc_mask, "y_pred"] >= 1).sum())) if inoc_mask.any() else 0
    inoc_n = int(inoc_mask.sum())
    attack_fn = int(((g_df.loc[attack_mask, "y_pred"] != 3).sum())) if attack_mask.any() else 0
    attack_n = int(attack_mask.sum())
    emoji_offensive = int(((g_df["y_pred"] == 1).sum()))

    return {
        "group_g_innocuous_fp": inoc_fp,
        "group_g_innocuous_n": inoc_n,
        "group_g_attack_fn": attack_fn,
        "group_g_attack_n": attack_n,
        "group_g_pred_offensive_rate": emoji_offensive / len(g_df) if len(g_df) else float("nan"),
        "group_g_pred_class3": int((g_df["y_pred"] == 3).sum()),
    }


def evaluate_adversarial_144_battery(
    *,
    model_dir: Path,
    model_name: str = "mbert",
    checkpoint: str = "",
    run_id: str | None = None,
    output_prefix: str = "mbert",
) -> dict[str, Any]:
    """Evalúa un checkpoint mBERT sobre test_adversarial_144.csv (A–G)."""
    manifest = load_battery_144_manifest()
    adv_path = TEST_ADVERSARIAL_144_CSV
    current_sha = sha256_file(adv_path)
    if manifest.get("sha256") and manifest["sha256"] != current_sha:
        raise ValueError(
            f"SHA-256 de test_adversarial_144.csv ({current_sha}) no coincide con el manifiesto "
            f"({manifest['sha256']})."
        )

    train_overlaps = assert_adversarial_not_in_splits(splits=("val", "test"))

    adv_df = load_split("adversarial_144")
    if len(adv_df) != BATTERY_144_N:
        raise ValueError(f"Se esperaban {BATTERY_144_N} casos adversariales, hay {len(adv_df)}.")

    y_pred, y_prob = predict(adv_df["texto_modelo"].tolist(), model_dir)
    y_true = adv_df["label"].values.astype(int)
    y_pred = np.asarray(y_pred, dtype=int)
    confidence = np.max(y_prob, axis=1)
    correct = y_true == y_pred

    pred_df = adv_df.copy()
    pred_df["y_true"] = y_true
    pred_df["y_pred"] = y_pred
    pred_df["confianza"] = confidence
    pred_df["acierto"] = correct

    family_rows: list[dict[str, Any]] = []
    for grupo, spec in FAMILY_SPECS_144.items():
        mask = adv_df["grupo"].values == grupo
        if not mask.any():
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        conf = confidence[mask]
        corr = correct[mask]
        row: dict[str, Any] = {
            "model": model_name,
            "checkpoint": checkpoint,
            "grupo": grupo,
            "family_key": spec["key"],
            "metric_type": spec["metric"],
            **_confidence_stats(conf, corr),
        }
        if spec["metric"] == "accuracy":
            row.update(_accuracy_metrics(yt, yp))
        elif spec["metric"] == "fpr":
            row.update(_fpr_metrics(yt, yp))
        elif spec["metric"] == "emoji_pair":
            row.update(_accuracy_metrics(yt, yp))
        pred_dist = pd.Series(yp).value_counts().sort_index()
        for label in range(4):
            row[f"pred_label_{label}"] = int(pred_dist.get(label, 0))
        family_rows.append(row)

    family_df = pd.DataFrame(family_rows)
    threat = _threat_metrics(y_true, y_pred)
    global_acc = _accuracy_metrics(y_true, y_pred)
    global_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    emoji_metrics = _emoji_pair_metrics(adv_df, y_true, y_pred)

    d_row = family_df[family_df["family_key"] == "D"]
    e_row = family_df[family_df["family_key"] == "E"]
    f_row = family_df[family_df["family_key"] == "F"]
    fpr_gap = float("nan")
    ci_overlap = True
    if len(d_row) and len(e_row):
        fpr_d = float(d_row.iloc[0]["fpr"])
        fpr_e = float(e_row.iloc[0]["fpr"])
        fpr_gap = fpr_d - fpr_e
        ci_overlap = not (
            float(d_row.iloc[0]["fpr_ci_lo"]) > float(e_row.iloc[0]["fpr_ci_hi"])
            or float(e_row.iloc[0]["fpr_ci_lo"]) > float(d_row.iloc[0]["fpr_ci_hi"])
        )

    fpr_f = float(f_row.iloc[0]["fpr"]) if len(f_row) else float("nan")
    mitigation = assess_mitigation_gap(family_df, fpr_gap=fpr_gap, ci_overlap=ci_overlap)

    summary: dict[str, Any] = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "model": model_name,
        "checkpoint": checkpoint,
        "run_id": run_id,
        "battery_version": manifest["battery_version"],
        "battery_sha256": manifest["sha256"],
        "split": adv_path.name,
        "n_cases": len(adv_df),
        "f1_macro_diagnostic": global_f1,
        "global_correct": global_acc["correct"],
        "global_accuracy": global_acc["accuracy"],
        "global_acc_ci_lo": global_acc["acc_ci_lo"],
        "global_acc_ci_hi": global_acc["acc_ci_hi"],
        **threat,
        "fpr_dialectal_gap": fpr_gap,
        "fpr_dialectal_f": fpr_f,
        "fpr_ci_overlap": ci_overlap,
        "battery_pred_class3": int((y_pred == 3).sum()),
        "mitigation_required": mitigation["mitigation_required"],
        "mitigation_reasons": mitigation["reasons"],
        "train_overlap_n": int(len(train_overlaps)),
        "xlmr_v1_baseline": XLMR_V1_BASELINE_144,
        "is_diagnostic_metric": True,
        **emoji_metrics,
    }

    export_adversarial_artifacts(
        pred_df=pred_df,
        family_df=family_df,
        summary=summary,
        output_prefix=f"{output_prefix}_144",
    )
    return summary


def decide_mitigation_adoption_multiobjective(
    before_after_df: pd.DataFrame,
    *,
    baseline_f1: float = 0.7777,
    bootstrap_overlap: bool = False,
) -> tuple[bool, str]:
    """Criterio multiobjetivo del informe de verificación (B2 vs v1)."""
    if len(before_after_df) < 2:
        return False, "Faltan filas baseline/candidato."
    base = before_after_df.iloc[0]
    cand = before_after_df.iloc[1]

    c_improved = (
        cand.get("group_c_correct", 0) > base.get("group_c_correct", 0)
        or cand.get("adv_threat_fn", 999) < base.get("adv_threat_fn", 999)
    )
    gap_improved = cand.get("fpr_gap", 1) < base.get("fpr_gap", 1)
    f1_delta = float(cand.get("test_f1_macro", 0)) - float(base.get("test_f1_macro", baseline_f1))
    f1_ok = f1_delta >= -0.02 or bootstrap_overlap

    g_inoc_fp_base = base.get("group_g_innocuous_fp", float("nan"))
    g_inoc_fp_cand = cand.get("group_g_innocuous_fp", float("nan"))
    g_improved = pd.isna(g_inoc_fp_base) or (
        not pd.isna(g_inoc_fp_cand) and g_inoc_fp_cand < g_inoc_fp_base
    )

    pred3_base = base.get("battery_pred_class3", float("nan"))
    pred3_cand = cand.get("battery_pred_class3", float("nan"))
    pred3_ok = pd.isna(pred3_base) or (
        not pd.isna(pred3_cand) and pred3_cand <= pred3_base + 5
    )

    adv_improved = cand.get("adv_global_correct", 0) > base.get("adv_global_correct", 0)

    if adv_improved and c_improved and gap_improved and f1_ok and g_improved and pred3_ok:
        return True, "Mejora adversarial 144 con F1 test dentro de tolerancia y sin sobre-predicción de Amenazas."

    reasons: list[str] = []
    if not adv_improved:
        reasons.append("sin mejora global adversarial")
    if not c_improved:
        reasons.append("sin mejora grupo C/FN amenazas")
    if not gap_improved:
        reasons.append("sin reducción brecha FPR D-E")
    if not f1_ok:
        reasons.append(f"F1-macro test cae >0.02 (delta={f1_delta:.4f})")
    if not g_improved:
        reasons.append("emoji inocuo sigue disparando toxicidad (grupo G)")
    if not pred3_ok:
        reasons.append(f"sobre-predicción clase 3 en batería ({pred3_cand} vs {pred3_base})")
    return False, "; ".join(reasons)
