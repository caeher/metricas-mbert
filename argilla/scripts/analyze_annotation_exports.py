#!/usr/bin/env python
"""Measure human agreement and build a traceable, QC-gated annotation release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from typing import Any

from argilla_workflow import (
    DEFAULT_RECORDS,
    TAXONOMY_VERSION,
    WorkflowError,
    load_canonical_rows,
    source_fingerprint,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "reports/annotation"
LABELS = ("0", "1", "2", "3")
MANUAL_FIELDS = (
    "final_annotation_status",
    "final_label",
    "final_quality",
    "decision_type",
    "decision_reason",
    "adjudicator",
    "provenance_review_status",
    "provenance_review_reason",
    "qualitative_theme",
    "qualitative_note",
)
LEDGER_FIELDS = [
    "canonical_id", "campaign", "external_id", "source_files_json", "source_roles_json",
    "platform", "provisional_labels_json", "provisional_subclasses_json", "record_disposition", "source_fingerprint",
    "submitted_rater_count", "original_responses_json", "answer_signature",
    "annotation_status_a", "label_a", "quality_a", "rationale_a", "user_id_a",
    "annotation_status_b", "label_b", "quality_b", "rationale_b", "user_id_b",
    *MANUAL_FIELDS, "guideline_version", "release_eligible", "release_blockers",
]


def json_value(value: Any, default: Any = None) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(str(value))
    except (json.JSONDecodeError, TypeError):
        return default


def list_value(value: Any) -> list[Any]:
    parsed = json_value(value, [])
    return parsed if isinstance(parsed, list) else []


def scalar(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(scalar(item) for item in value if scalar(item))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "" if value is None else str(value).strip()


def label_value(value: Any) -> str:
    value = scalar(value)
    match = re.match(r"^\s*([0-3])(?:\b|\s|·|$)", value)
    return match.group(1) if match else ""


def response_status(value: Any) -> str:
    return scalar(getattr(value, "value", value)).casefold()


def load_export(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if path.is_file():
        records_path = path
        manifest_path = path.with_name("manifest.json")
    else:
        manifest_path = path / "manifest.json"
        records_path = path / "records.jsonl"
    if not records_path.exists():
        raise WorkflowError(f"No existe la exportación JSONL: {records_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if manifest.get("records_sha256"):
        actual_sha = hashlib.sha256(records_path.read_bytes()).hexdigest()
        if actual_sha != manifest["records_sha256"]:
            raise WorkflowError(f"SHA-256 de records.jsonl no coincide con {manifest_path}")
    records: list[dict[str, Any]] = []
    with records_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise WorkflowError(f"JSONL inválido en {records_path}:{line_number}") from exc
            if not isinstance(item, dict) or not item.get("external_id"):
                raise WorkflowError(f"Falta external_id en {records_path}:{line_number}")
            records.append(item)
    if manifest and manifest.get("record_count") != len(records):
        raise WorkflowError(f"record_count no coincide en {manifest_path}")
    return manifest, records


def submitted_answers(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = defaultdict(dict)
    for response in record.get("responses", []):
        if not isinstance(response, dict) or response_status(response.get("status")) != "submitted":
            continue
        user_id = scalar(response.get("user_id"))
        question = scalar(response.get("question_name"))
        if not user_id or not question:
            continue
        value = response.get("value")
        if question in grouped[user_id] and grouped[user_id][question] != value:
            grouped[user_id]["_duplicate_conflict"] = True
        grouped[user_id][question] = value
    return dict(grouped)


def annotation_status(answer: dict[str, Any]) -> str:
    return scalar(answer.get("annotation_status")).casefold()


def valid_rating(answer: dict[str, Any]) -> str:
    if answer.get("_duplicate_conflict"):
        return ""
    if annotation_status(answer) != "anotado":
        return ""
    label = label_value(answer.get("label"))
    return label if label in LABELS else ""


def nominal_metrics(items: list[list[tuple[str, str]]]) -> dict[str, Any]:
    eligible = [[(user, label) for user, label in item if label in LABELS] for item in items]
    eligible = [item for item in eligible if len(item) >= 2]
    if not eligible:
        return {
            "n": 0, "pairwise_comparisons": 0, "percent_agreement": None,
            "estimator": "cohen_kappa", "kappa": None, "observed_agreement": None,
            "expected_agreement": None, "annotators": [],
        }

    annotators = sorted({user for item in eligible for user, _ in item})
    total_pairs = 0
    agreeing_pairs = 0
    category_pair_counts = Counter()
    for item in eligible:
        for i in range(len(item)):
            for j in range(i + 1, len(item)):
                left, right = item[i][1], item[j][1]
                total_pairs += 1
                agreeing_pairs += left == right
                category_pair_counts[(left, right)] += 1
                category_pair_counts[(right, left)] += 1
    observed = agreeing_pairs / total_pairs if total_pairs else None
    if len(annotators) == 2 and all(len(item) == 2 for item in eligible):
        ordered = []
        for item in eligible:
            by_user = {user: label for user, label in item}
            if set(by_user) == set(annotators):
                ordered.append((by_user[annotators[0]], by_user[annotators[1]]))
        n = len(ordered)
        if not n:
            kappa = expected = None
        else:
            row_counts = Counter(left for left, _ in ordered)
            col_counts = Counter(right for _, right in ordered)
            expected = sum(row_counts[label] * col_counts[label] for label in LABELS) / (n * n)
            kappa = (observed - expected) / (1 - expected) if expected is not None and expected < 1 else None
        estimator = "cohen_kappa"
    else:
        # Krippendorff's nominal alpha supports three or more raters and varying rater counts.
        category_totals = Counter(label for item in eligible for _, label in item)
        rating_count = sum(category_totals.values())
        expected = (
            1 - sum(count * (count - 1) for count in category_totals.values()) / (rating_count * (rating_count - 1))
            if rating_count > 1 else None
        )
        observed_disagreement = 1 - observed if observed is not None else None
        kappa = (
            1 - observed_disagreement / expected
            if expected is not None and expected > 0 and observed_disagreement is not None else None
        )
        estimator = "krippendorff_alpha_nominal"
    return {
        "n": len(eligible),
        "pairwise_comparisons": total_pairs,
        "percent_agreement": observed * 100 if observed is not None else None,
        "estimator": estimator,
        "kappa": kappa,
        "observed_agreement": observed,
        "expected_agreement": expected,
        "annotators": annotators,
    }


def binary_kappa(items: list[list[tuple[str, str]]]) -> tuple[int, float | None, float | None]:
    pairs = []
    annotators = sorted({user for item in items for user, _ in item})
    if len(annotators) != 2:
        return 0, None, None
    for item in items:
        by_user = {user: label for user, label in item}
        if set(by_user) == set(annotators):
            pairs.append((by_user[annotators[0]], by_user[annotators[1]]))
    n = len(pairs)
    if not n:
        return 0, None, None
    observed = sum(left == right for left, right in pairs) / n
    left_counts = Counter(left for left, _ in pairs)
    right_counts = Counter(right for _, right in pairs)
    expected = sum(left_counts[label] * right_counts[label] for label in ("positive", "negative")) / (n * n)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else None
    return n, observed * 100, kappa


def confusion_pairs(items: list[list[tuple[str, str]]], annotators: list[str]) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for item in items:
        valid = [(user, label) for user, label in item if label in LABELS]
        if len(annotators) == 2 and len(valid) == 2 and {user for user, _ in valid} == set(annotators):
            labels_by_user = {user: label for user, label in valid}
            counts[(labels_by_user[annotators[0]], labels_by_user[annotators[1]])] += 1
            continue
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                left, right = valid[i][1], valid[j][1]
                counts[(left, right)] += 1
                counts[(right, left)] += 1
    return counts


def answer_bundle(answer: dict[str, Any] | None) -> dict[str, str]:
    answer = answer or {}
    return {
        "annotation_status": annotation_status(answer),
        "label": label_value(answer.get("label")),
        "quality": scalar(answer.get("quality")).casefold(),
        "rationale": scalar(answer.get("rationale")),
        "subclass": scalar(answer.get("subclass")),
        "insufficient_context_reason": scalar(answer.get("insufficient_context_reason")),
    }


def subclass_values(row: dict[str, str]) -> list[str]:
    values = []
    for item in list_value(row.get("source_metadata_json")):
        if isinstance(item, str):
            item = json_value(item, {})
        if isinstance(item, dict):
            value = scalar(item.get("Subclase_Toxicidad"))
            if value and value not in values:
                values.append(value)
    return values


def read_existing_ledger(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return {
            (row.get("campaign", ""), row.get("canonical_id", "")): row
            for row in csv.DictReader(stream)
        }


def export_id_to_canonical(external_id: str, metadata: dict[str, Any]) -> str:
    canonical_id = scalar(metadata.get("canonical_id"))
    if canonical_id:
        return canonical_id
    return external_id.rsplit("-", 1)[-1]


def build_analysis(
    export_dirs: list[Path], canonical_rows: dict[str, dict[str, str]], output_dir: Path
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / "adjudication_ledger.csv"
    previous = read_existing_ledger(ledger_path)
    ledger_by_key: dict[tuple[str, str], dict[str, str]] = {}
    campaign_items: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_change_rows: list[dict[str, str]] = []
    subclass_change_rows: list[dict[str, str]] = []
    qualitative_rows: list[dict[str, str]] = []
    release_rows: list[dict[str, str]] = []
    export_summary = []

    for export_path in export_dirs:
        manifest, records = load_export(export_path)
        campaign = scalar(manifest.get("campaign")) or export_path.name
        for record in records:
            metadata = record.get("metadata", {}) or {}
            canonical_id = export_id_to_canonical(scalar(record.get("external_id")), metadata)
            source = canonical_rows.get(canonical_id, {})
            answers_by_user = submitted_answers(record)
            bundles = {user: answer_bundle(answer) for user, answer in answers_by_user.items()}
            ratings = [(user, valid_rating(answer)) for user, answer in answers_by_user.items()]
            campaign_items[campaign].append(
                {
                    "canonical_id": canonical_id,
                    "external_id": scalar(record.get("external_id")),
                    "metadata": metadata,
                    "source": source,
                    "answers": answers_by_user,
                    "bundles": bundles,
                    "ratings": ratings,
                    "campaign": campaign,
                    "fields": record.get("fields", {}) or {},
                }
            )
        export_summary.append(
            {
                "campaign": campaign,
                "export_id": manifest.get("export_id", export_path.name),
                "record_count": len(records),
                "response_count": manifest.get("response_count"),
                "path": str(export_path),
            }
        )

    agreement_by_slice: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "status": "computed_from_human_argilla_exports" if export_summary else "pending_human_exports",
        "guideline_version": TAXONOMY_VERSION,
        "operational_kappa_target": 0.75,
        "target_note": "Objetivo operativo inicial; no es garantía ni estimador de F1.",
        "campaigns": {},
        "exports": export_summary,
        "release": {"eligible_records": 0, "held_records": 0},
    }

    for campaign, items in sorted(campaign_items.items()):
        complete_items = [item["ratings"] for item in items]
        summary = nominal_metrics(complete_items)
        answer_counts = Counter()
        user_ids = set()
        for item in items:
            user_ids.update(user for user, _ in item["ratings"])
            answers = item["answers"]
            if len(answers) >= 2:
                answer_counts["records_with_two_submitted_annotators"] += 1
            if any(annotation_status(answer) == "contexto_insuficiente" for answer in answers.values()):
                answer_counts["records_flagged_insufficient_context"] += 1
            if any(answer.get("quality") in {"review", "duplicate_conflict", "insufficient_context"} for answer in answers.values()):
                answer_counts["records_flagged_quality_review"] += 1
        summary.update(
            {
                "record_count": len(items),
                "unique_annotator_count": len(user_ids),
                "unique_annotators": sorted(user_ids),
                "records_with_two_submitted_annotators": answer_counts["records_with_two_submitted_annotators"],
                "records_flagged_insufficient_context": answer_counts["records_flagged_insufficient_context"],
                "records_flagged_quality_review": answer_counts["records_flagged_quality_review"],
                "meets_operational_target": bool(summary["kappa"] is not None and summary["kappa"] >= 0.75),
            }
        )
        report["campaigns"][campaign] = summary

        external_ids = [item["external_id"] for item in items]
        if len(external_ids) != len(set(external_ids)):
            raise WorkflowError(
                f"Hay exportaciones repetidas de la campaña {campaign}; analice una sola instantánea por campaña"
            )

        def add_slice(cut_type: str, key_fn: Any) -> None:
            values: dict[str, list[list[tuple[str, str]]]] = defaultdict(list)
            for item in items:
                for value in key_fn(item):
                    values[value].append(item["ratings"])
            for value, group in sorted(values.items()):
                metrics = nominal_metrics(group)
                agreement_by_slice.append(
                    {
                        "campaign": campaign,
                        "cut_type": cut_type,
                        "cut_value": value,
                        "n": metrics["n"],
                        "percent_agreement": metrics["percent_agreement"],
                        "estimator": metrics["estimator"],
                        "kappa_or_alpha": metrics["kappa"],
                        "pairwise_comparisons": metrics["pairwise_comparisons"],
                    }
                )

        add_slice("source_file", lambda item: list_value(item["source"].get("source_files_json"))[:])
        add_slice("platform", lambda item: [scalar(item["source"].get("platform_canonical")) or "unknown"])
        add_slice("provisional_label", lambda item: [f"label_{value}" for value in list_value(item["source"].get("label_provisional_ids_json"))] or ["unlabeled"])
        add_slice("subclass", lambda item: subclass_values(item["source"]) or ["unspecified"])

        matrix = confusion_pairs(complete_items, summary["annotators"])
        for actual in LABELS:
            for second in LABELS:
                matrix_rows.append(
                    {
                        "campaign": campaign,
                        "annotator_label_row": actual,
                        "annotator_label_column": second,
                        "pair_count": matrix[(actual, second)],
                    }
                )
        for label in LABELS:
            one_vs_rest_items = []
            class_votes = 0
            total_votes = 0
            for item in complete_items:
                if len(item) < 2:
                    continue
                for user, value in item:
                    if value in LABELS:
                        total_votes += 1
                        class_votes += value == label
                # A true Cohen one-vs-rest statistic is meaningful when each item has exactly two labels.
                if len(item) == 2 and all(value in LABELS for _, value in item):
                    one_vs_rest_items.append(
                        [(user, "positive" if value == label else "negative") for user, value in item]
                    )
            both_positive = one_positive = 0
            for item in complete_items:
                valid = [value for _, value in item if value in LABELS]
                for i in range(len(valid)):
                    for j in range(i + 1, len(valid)):
                        if valid[i] == label and valid[j] == label:
                            both_positive += 1
                        elif valid[i] == label or valid[j] == label:
                            one_positive += 1
            positive_agreement = (
                2 * both_positive / (2 * both_positive + one_positive)
                if 2 * both_positive + one_positive else None
            )
            _, one_vs_rest_agreement, one_vs_rest_kappa = binary_kappa(one_vs_rest_items)
            per_class_rows.append(
                {
                    "campaign": campaign,
                    "label": label,
                    "label_name": {"0": "No Tóxico", "1": "Lenguaje Ofensivo", "2": "Discurso de Odio", "3": "Amenazas/Violencia"}[label],
                    "n_items_with_two_or_more_ratings": summary["n"],
                    "class_votes": class_votes,
                    "all_valid_votes": total_votes,
                    "vote_prevalence": class_votes / total_votes if total_votes else None,
                    "positive_agreement": positive_agreement,
                    "one_vs_rest_agreement": one_vs_rest_agreement,
                    "one_vs_rest_kappa_or_alpha": one_vs_rest_kappa,
                }
            )

        for item in items:
            canonical_id = item["canonical_id"]
            source = item["source"]
            bundles = item["bundles"]
            users = sorted(bundles)
            first_user = users[0] if users else ""
            second_user = users[1] if len(users) > 1 else ""
            answer_a = bundles.get(first_user, {})
            answer_b = bundles.get(second_user, {})
            valid_labels = [valid_rating(answer) for answer in item["answers"].values()]
            valid_labels = [label for label in valid_labels if label]
            existing = previous.get((campaign, canonical_id), {})
            source_roles = list_value(source.get("source_roles_json"))
            candidate = "candidate_unvalidated" in source_roles
            old_labels = list_value(source.get("label_provisional_ids_json"))
            raw_answers = json.dumps(
                {
                    user: {question: value for question, value in answer.items() if not question.startswith("_")}
                    for user, answer in sorted(item["answers"].items())
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            signature = raw_answers
            current_fingerprint = source_fingerprint(source) if source else scalar(item["metadata"].get("source_fingerprint"))
            same_answers = (
                existing.get("answer_signature", "") == signature
                and existing.get("source_fingerprint", "") == current_fingerprint
            )
            ledger = {
                "canonical_id": canonical_id,
                "campaign": campaign,
                "external_id": item["external_id"],
                "source_files_json": source.get("source_files_json", item["metadata"].get("source_files_json", "[]")),
                "source_roles_json": source.get("source_roles_json", "[]"),
                "platform": source.get("platform_canonical", ""),
                "provisional_labels_json": source.get("label_provisional_ids_json", "[]"),
                "provisional_subclasses_json": json.dumps(subclass_values(source), ensure_ascii=False),
                "record_disposition": source.get("record_disposition", "unknown"),
                "source_fingerprint": current_fingerprint,
                "submitted_rater_count": len(item["answers"]),
                "original_responses_json": raw_answers,
                "answer_signature": signature,
                "annotation_status_a": answer_a.get("annotation_status", ""),
                "label_a": answer_a.get("label", ""),
                "quality_a": answer_a.get("quality", ""),
                "rationale_a": answer_a.get("rationale", ""),
                "user_id_a": first_user,
                "annotation_status_b": answer_b.get("annotation_status", ""),
                "label_b": answer_b.get("label", ""),
                "quality_b": answer_b.get("quality", ""),
                "rationale_b": answer_b.get("rationale", ""),
                "user_id_b": second_user,
                "guideline_version": scalar(item["metadata"].get("guideline_version")) or TAXONOMY_VERSION,
            }
            for field in MANUAL_FIELDS:
                ledger[field] = existing.get(field, "") if same_answers else ""

            enough_responses = len(item["answers"]) >= 2
            all_consensus = enough_responses and len(valid_labels) == len(item["answers"]) and len(set(valid_labels)) == 1
            all_insufficient = enough_responses and all(
                annotation_status(answer) == "contexto_insuficiente" for answer in item["answers"].values()
            )
            if not ledger["decision_type"]:
                if all_consensus:
                    ledger.update(
                        {
                            "final_annotation_status": "anotado",
                            "final_label": valid_labels[0],
                            "final_quality": "usable" if all(bundle.get("quality") == "usable" for bundle in bundles.values()) else "",
                            "decision_type": "consensus",
                            "decision_reason": "Acuerdo de todas las respuestas humanas independientes; sin adjudicación automática de desacuerdos.",
                            "adjudicator": "consensus",
                            "provenance_review_status": "pending" if candidate else "not_required",
                        }
                    )
                elif all_insufficient:
                    ledger.update(
                        {
                            "final_annotation_status": "contexto_insuficiente",
                            "final_label": "",
                            "decision_type": "context_quarantine",
                            "decision_reason": "Todas las respuestas indican contexto insuficiente; requiere contexto recuperable o cuarentena.",
                            "adjudicator": "",
                            "provenance_review_status": "pending" if candidate else "not_required",
                        }
                    )
                elif enough_responses:
                    ledger.update(
                        {
                            "final_annotation_status": "",
                            "final_label": "",
                            "decision_type": "pending_human_adjudication",
                            "decision_reason": "",
                            "adjudicator": "",
                            "provenance_review_status": "pending" if candidate else "not_required",
                        }
                    )
                else:
                    ledger.update(
                        {
                            "final_annotation_status": "",
                            "final_label": "",
                            "decision_type": "awaiting_two_independent_submissions",
                            "decision_reason": "",
                            "adjudicator": "",
                            "provenance_review_status": "pending" if candidate else "not_required",
                        }
                    )

            insufficient = any(
                bundle.get("annotation_status") == "contexto_insuficiente" for bundle in bundles.values()
            )
            quality_flagged = any(
                bundle.get("quality") in {"review", "duplicate_conflict", "insufficient_context"}
                for bundle in bundles.values()
            )
            label_disagreement = len(set(valid_labels)) > 1
            if len(item["answers"]) >= 2 and (insufficient or quality_flagged or label_disagreement):
                issue_types = []
                if label_disagreement:
                    issue_types.append("label_disagreement")
                if insufficient:
                    issue_types.append("insufficient_context")
                if quality_flagged:
                    issue_types.append("quality_review")
                qualitative_rows.append(
                    {
                        "campaign": campaign,
                        "canonical_id": canonical_id,
                        "source_files_json": ledger["source_files_json"],
                        "platform": ledger["platform"],
                        "text_original": source.get("text_original") or item["fields"].get("original_text", ""),
                        "annotation_status_a": answer_a.get("annotation_status", ""),
                        "label_a": answer_a.get("label", ""),
                        "quality_a": answer_a.get("quality", ""),
                        "rationale_a": answer_a.get("rationale", ""),
                        "annotation_status_b": answer_b.get("annotation_status", ""),
                        "label_b": answer_b.get("label", ""),
                        "quality_b": answer_b.get("quality", ""),
                        "rationale_b": answer_b.get("rationale", ""),
                        "provisional_labels_json": ledger["provisional_labels_json"],
                        "provisional_subclasses_json": ledger["provisional_subclasses_json"],
                        "issue_types": ";".join(issue_types),
                        "decision_type": ledger["decision_type"],
                    }
                )

            blockers = []
            if len(item["answers"]) < 2:
                blockers.append("fewer_than_two_submitted_human_responses")
            if ledger.get("final_annotation_status") != "anotado":
                blockers.append("no_final_annotated_status")
            if ledger.get("final_label") not in LABELS:
                blockers.append("no_final_class_0_to_3")
            if ledger.get("final_quality") != "usable":
                blockers.append("quality_not_confirmed_usable")
            if not all(bundle.get("rationale") for bundle in bundles.values()):
                blockers.append("human_rationale_missing")
            if ledger.get("guideline_version") != TAXONOMY_VERSION:
                blockers.append("guideline_version_mismatch")
            if ledger.get("decision_type") == "pending_human_adjudication":
                blockers.append("human_adjudication_pending")
            if ledger.get("decision_type") == "adjudicated" and not ledger.get("adjudicator"):
                blockers.append("adjudicator_missing")
            if ledger.get("decision_type") == "adjudicated" and not ledger.get("decision_reason"):
                blockers.append("adjudication_reason_missing")
            if candidate and ledger.get("provenance_review_status") != "verified":
                blockers.append("candidate_provenance_not_human_verified")
            if ledger.get("decision_type") not in {"consensus", "adjudicated"}:
                blockers.append("decision_not_final")
            ledger["release_blockers"] = ";".join(dict.fromkeys(blockers))
            ledger["release_eligible"] = "yes" if not blockers else "no"
            ledger_by_key[(campaign, canonical_id)] = ledger

            final_label = ledger.get("final_label", "")
            if final_label in LABELS:
                for previous_label in old_labels:
                    previous_label = scalar(previous_label)
                    for source_name in list_value(source.get("source_files_json")) or ["unknown"]:
                        source_change_rows.append(
                            {
                                "campaign": campaign,
                                "source_file": source_name,
                                "provisional_label": previous_label,
                                "final_label": final_label,
                                "changed": str(previous_label != final_label).lower(),
                            }
                        )
                    for subclass in subclass_values(source) or ["unspecified"]:
                        subclass_change_rows.append(
                            {
                                "campaign": campaign,
                                "subclass": subclass,
                                "provisional_label": previous_label,
                                "final_label": final_label,
                                "changed": str(previous_label != final_label).lower(),
                            }
                        )
            if ledger["release_eligible"] == "yes":
                release_rows.append(
                    {
                        "canonical_id": canonical_id,
                        "campaign": campaign,
                        "text_original": source.get("text_original") or item["fields"].get("original_text", ""),
                        "label_final": final_label,
                        "annotation_status": ledger.get("final_annotation_status", ""),
                        "quality": ledger.get("final_quality", ""),
                        "decision_type": ledger.get("decision_type", ""),
                        "decision_reason": ledger.get("decision_reason", ""),
                        "adjudicator": ledger.get("adjudicator", ""),
                        "guideline_version": ledger.get("guideline_version", ""),
                        "source_files_json": ledger.get("source_files_json", "[]"),
                        "provenance_review_status": ledger.get("provenance_review_status", ""),
                        "original_responses_json": ledger.get("original_responses_json", "{}"),
                    }
                )

    for key, old in previous.items():
        if key in ledger_by_key:
            continue
        source = canonical_rows.get(old.get("canonical_id", ""), {})
        current_fingerprint = source_fingerprint(source) if source else ""
        if not current_fingerprint or old.get("source_fingerprint") != current_fingerprint:
            continue
        ledger_by_key[key] = old
        if old.get("release_eligible") == "yes":
            release_rows.append(
                {
                    "canonical_id": old.get("canonical_id", ""),
                    "campaign": old.get("campaign", ""),
                    "text_original": source.get("text_original", ""),
                    "label_final": old.get("final_label", ""),
                    "annotation_status": old.get("final_annotation_status", ""),
                    "quality": old.get("final_quality", ""),
                    "decision_type": old.get("decision_type", ""),
                    "decision_reason": old.get("decision_reason", ""),
                    "adjudicator": old.get("adjudicator", ""),
                    "guideline_version": old.get("guideline_version", ""),
                    "source_files_json": old.get("source_files_json", "[]"),
                    "provenance_review_status": old.get("provenance_review_status", ""),
                    "original_responses_json": old.get("original_responses_json", "{}"),
                }
            )
        final_label = old.get("final_label", "")
        if final_label in LABELS:
            for previous_label in list_value(old.get("provisional_labels_json", "[]")):
                for source_name in list_value(old.get("source_files_json", "[]")) or ["unknown"]:
                    source_change_rows.append(
                        {
                            "campaign": old.get("campaign", ""),
                            "source_file": source_name,
                            "provisional_label": scalar(previous_label),
                            "final_label": final_label,
                            "changed": str(scalar(previous_label) != final_label).lower(),
                        }
                    )
                for subclass in list_value(old.get("provisional_subclasses_json", "[]")) or ["unspecified"]:
                    subclass_change_rows.append(
                        {
                            "campaign": old.get("campaign", ""),
                            "subclass": scalar(subclass),
                            "provisional_label": scalar(previous_label),
                            "final_label": final_label,
                            "changed": str(scalar(previous_label) != final_label).lower(),
                        }
                    )

    ledger_rows = list(ledger_by_key.values())
    write_csv(ledger_path, LEDGER_FIELDS, ledger_rows)
    write_csv(output_dir / "adjudicated_records.csv", list(release_rows[0].keys()) if release_rows else [
        "canonical_id", "campaign", "text_original", "label_final", "annotation_status", "quality",
        "decision_type", "decision_reason", "adjudicator", "guideline_version", "source_files_json",
        "provenance_review_status", "original_responses_json",
    ], release_rows)
    write_csv(
        output_dir / "agreement_by_slice.csv",
        ["campaign", "cut_type", "cut_value", "n", "percent_agreement", "estimator", "kappa_or_alpha", "pairwise_comparisons"],
        agreement_by_slice,
    )
    write_csv(
        output_dir / "agreement_matrix.csv",
        ["campaign", "annotator_label_row", "annotator_label_column", "pair_count"],
        matrix_rows,
    )
    write_csv(
        output_dir / "agreement_by_class.csv",
        ["campaign", "label", "label_name", "n_items_with_two_or_more_ratings", "class_votes", "all_valid_votes", "vote_prevalence", "positive_agreement", "one_vs_rest_agreement", "one_vs_rest_kappa_or_alpha"],
        per_class_rows,
    )
    write_change_report(output_dir / "label_changes_by_source.csv", source_change_rows, "source_file")
    write_change_report(output_dir / "label_changes_by_subclass.csv", subclass_change_rows, "subclass")
    write_csv(
        output_dir / "qualitative_review.csv",
        [
            "campaign", "canonical_id", "source_files_json", "platform", "text_original",
            "annotation_status_a", "label_a", "quality_a", "rationale_a",
            "annotation_status_b", "label_b", "quality_b", "rationale_b",
            "provisional_labels_json", "provisional_subclasses_json", "issue_types", "decision_type",
        ],
        qualitative_rows,
    )
    qualitative_themes = Counter(
        (row.get("qualitative_theme", "").strip(), row.get("qualitative_note", "").strip())
        for row in ledger_rows
        if row.get("qualitative_theme", "").strip()
    )
    write_csv(
        output_dir / "qualitative_themes.csv",
        ["theme", "note", "cases"],
        [{"theme": theme, "note": note, "cases": count} for (theme, note), count in sorted(qualitative_themes.items())],
    )
    report["qualitative_analysis"] = {
        "cases_for_human_review": len(qualitative_rows),
        "cases_with_human_theme": sum(qualitative_themes.values()),
        "status": "coded_by_human_review" if qualitative_themes else "pending_human_coding",
        "queue_path": str(output_dir / "qualitative_review.csv"),
    }
    report["release"] = {
        "eligible_records": len(release_rows),
        "held_records": len(ledger_rows) - len(release_rows),
        "qc_gates": [
            "dos o más respuestas humanas enviadas",
            "estado final anotado y clase final 0..3",
            "calidad final usable",
            "decisión por consenso humano o adjudicación humana trazable",
            "guía 2.0.0",
            "procedencia verificada para fuentes candidatas nuevas",
        ],
        "release_path": str(output_dir / "adjudicated_records.csv"),
    }
    (output_dir / "agreement.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(output_dir, report)
    return report


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_change_report(path: Path, rows: list[dict[str, str]], grouping: str) -> None:
    counts: Counter[tuple[str, str, str, str]] = Counter()
    for row in rows:
        counts[(row["campaign"], row[grouping], row["provisional_label"], row["final_label"])] += 1
    output = [
        {
            "campaign": campaign,
            grouping: group,
            "provisional_label": provisional,
            "final_label": final,
            "record_label_memberships": count,
            "changed": provisional != final,
        }
        for (campaign, group, provisional, final), count in sorted(counts.items())
    ]
    write_csv(path, ["campaign", grouping, "provisional_label", "final_label", "record_label_memberships", "changed"], output)


def write_readme(output_dir: Path, report: dict[str, Any]) -> None:
    if report["status"] == "pending_human_exports":
        status = "Aún no hay exportaciones humanas. Las métricas tienen n=0 y no se reportan como resultados."
    else:
        status = "Métricas generadas desde las exportaciones humanas indicadas en agreement.json."
    content = f"""# Revisión y acuerdo de anotación — ISSUE-05

{status}

`agreement.json` informa acuerdo porcentual (escala 0–100), kappa de Cohen para dos anotadores o alfa nominal de Krippendorff cuando hay más de dos, con n y denominadores explícitos. El objetivo operativo inicial es kappa/alfa ≥ 0.75; no garantiza F1. `agreement_by_slice.csv` contiene cortes por fuente, plataforma, clase provisional y subclase. `agreement_matrix.csv` y `agreement_by_class.csv` muestran desacuerdos y resultados por clase.

`adjudication_ledger.csv` conserva las respuestas humanas originales y separa decisión final, motivo, adjudicador, versión de guía y revisión de procedencia. Completa manualmente los campos de adjudicación en los casos pendientes; al volver a ejecutar el script, las decisiones se conservan mientras no cambien las respuestas originales.

`qualitative_review.csv` reúne razones y desacuerdos para revisión; completa en el ledger los campos `qualitative_theme` y `qualitative_note`. `qualitative_themes.csv` resume esa codificación humana. `adjudicated_records.csv` contiene solo decisiones con clase final 0–3 y controles de calidad completos. Los casos con contexto insuficiente no se exportan como clase 0. Los registros nuevos requieren además `provenance_review_status=verified`.

`label_changes_by_source.csv` y `label_changes_by_subclass.csv` cuantifican las clases provisionales que cambian frente a decisiones finales. Los registros compartidos por varias fuentes cuentan en cada fuente a la que pertenecen.
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports", nargs="+", type=Path, required=True, help="Directorios versionados de exportación Argilla.")
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS, help="Vista canónica de ISSUE-02.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        rows = load_canonical_rows(args.records)
        canonical = {row["canonical_id"]: row for row in rows}
        result = build_analysis(args.exports, canonical, args.output_dir)
        print(json.dumps({"report": str(args.output_dir / "agreement.json"), **result}, ensure_ascii=False, indent=2))
        return 0
    except (WorkflowError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
