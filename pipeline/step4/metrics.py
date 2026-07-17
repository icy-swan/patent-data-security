"""Dependency-light binary metrics for paper-style classifier reporting."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from pipeline.step4.data import LABEL_TO_ID


def binary_metrics(
    labels: Sequence[int],
    predictions: Sequence[int],
    *,
    positive_scores: Sequence[float] | None = None,
) -> dict[str, Any]:
    if not labels or len(labels) != len(predictions):
        raise ValueError("labels and predictions must have the same non-zero length")
    if set(labels) - {0, 1} or set(predictions) - {0, 1}:
        raise ValueError("binary labels and predictions must contain only 0 and 1")
    if positive_scores is not None and len(positive_scores) != len(labels):
        raise ValueError("positive_scores must have the same length as labels")

    true_negative = sum(y == 0 and p == 0 for y, p in zip(labels, predictions, strict=True))
    false_positive = sum(y == 0 and p == 1 for y, p in zip(labels, predictions, strict=True))
    false_negative = sum(y == 1 and p == 0 for y, p in zip(labels, predictions, strict=True))
    true_positive = sum(y == 1 and p == 1 for y, p in zip(labels, predictions, strict=True))

    data_security = _class_metrics(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
    )
    other = _class_metrics(
        true_positive=true_negative,
        false_positive=false_negative,
        false_negative=false_positive,
    )
    result: dict[str, Any] = {
        "records": len(labels),
        "accuracy": (true_positive + true_negative) / len(labels),
        "balanced_accuracy": (data_security["recall"] + other["recall"]) / 2,
        "macro_precision": (data_security["precision"] + other["precision"]) / 2,
        "macro_recall": (data_security["recall"] + other["recall"]) / 2,
        "macro_f1": (data_security["f1"] + other["f1"]) / 2,
        "confusion_matrix": {
            "true_other_pred_other": true_negative,
            "true_other_pred_data_security": false_positive,
            "true_data_security_pred_other": false_negative,
            "true_data_security_pred_data_security": true_positive,
        },
        "per_class": {"OTHER": other, "DATA_SECURITY": data_security},
    }
    if positive_scores is not None:
        scores = [float(score) for score in positive_scores]
        result["roc_auc"] = _roc_auc(labels, scores)
        result["average_precision"] = _average_precision(labels, scores)
    return result


def labels_from_names(values: Sequence[str]) -> list[int]:
    try:
        return [LABEL_TO_ID[value] for value in values]
    except KeyError as exc:
        raise ValueError(f"Unknown label: {exc.args[0]}") from exc


def _class_metrics(
    *, true_positive: int, false_positive: int, false_negative: int
) -> dict[str, float | int]:
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return {
        "support": true_positive + false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ranked = sorted(zip(scores, labels, strict=True), key=lambda item: item[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2
        positive_rank_sum += average_rank * sum(label for _, label in ranked[index:end])
        index = end
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _average_precision(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = sum(labels)
    if positives == 0:
        return None
    groups: dict[float, list[int]] = defaultdict(list)
    for score, label in zip(scores, labels, strict=True):
        groups[score].append(label)
    true_positive = 0
    false_positive = 0
    previous_recall = 0.0
    area = 0.0
    for score in sorted(groups, reverse=True):
        group = groups[score]
        true_positive += sum(group)
        false_positive += len(group) - sum(group)
        recall = true_positive / positives
        precision = true_positive / (true_positive + false_positive)
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return area
