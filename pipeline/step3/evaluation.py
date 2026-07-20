"""Evaluate Step 1 routing and Step 2 labels against Step 3 result.csv."""

from __future__ import annotations

import csv
import json
import math
import sqlite3
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pipeline.common.io import atomic_json_write, sha256_file
from pipeline.step3.sampling import (
    FROZEN_RESULT_FIELDS,
    LABELS,
    Step3Paths,
    _parse_human_evaluation,
    _validate_human_result_identity,
)

Z_95 = 1.959963984540054


def evaluate_pipeline_results(
    paths: Step3Paths,
    step2_databases: Iterable[str | Path],
) -> dict[str, Any]:
    """Calculate sample and design-weighted Step 1/2 binary metrics.

    Design weights expand the 4,000 sampled records only to Step 3's eligible
    Step 2 frame: all Step 2 DATA_SECURITY records plus S-to-OTHER hard
    negatives. They do not estimate performance over excluded E-to-OTHER rows.
    """

    manifest = _read_manifest(paths.manifest)
    expected_count = int(manifest.get("target_size", 4_000))
    references = _read_reference_results(paths, expected_count=expected_count)
    step2_rows = _read_step2_rows(step2_databases)
    strata = _read_strata(manifest)

    joined: list[dict[str, Any]] = []
    sampled_strata: Counter[tuple[str, str]] = Counter()
    for reference in references:
        key = (reference["dataset_id"], reference["patent_id"])
        step2 = step2_rows.get(key)
        if step2 is None:
            raise ValueError(
                "Step 3 result has no matching succeeded Step 2 task: "
                f"dataset_id={key[0]} patent_id={key[1]}"
            )
        sampling_group = _sampling_group(step2["route"], step2["label"])
        stratum_key = (reference["application_year"], sampling_group)
        stratum = strata.get(stratum_key)
        if stratum is None:
            raise ValueError(f"No Step 3 sampling stratum for {stratum_key}")
        sampled_strata[stratum_key] += 1
        joined.append(
            {
                **reference,
                "step1_prediction": step2["route"] == "S",
                "step2_prediction": step2["label"] == "DATA_SECURITY",
                "design_weight": stratum["population"] / stratum["sample"],
            }
        )

    for key, count in sampled_strata.items():
        expected = strata[key]["sample"]
        if count != expected:
            raise ValueError(
                f"Step 3 sample count differs for {key}: result={count}, manifest={expected}"
            )

    sample_label_counts = Counter(
        "DATA_SECURITY" if row["reference_positive"] else "OTHER" for row in joined
    )
    weighted_total = sum(float(row["design_weight"]) for row in joined)
    eligible_population = sum(value["population"] for value in strata.values())
    if not math.isclose(weighted_total, eligible_population, rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError(
            "Step 3 design weights do not reproduce the eligible population: "
            f"weighted={weighted_total}, manifest={eligible_population}"
        )

    report = {
        "schema_version": "2.2.0",
        "evaluation_type": "binary_classification_against_step3_result",
        "reference": {
            "path": str(paths.results),
            "sha256": sha256_file(paths.results),
            "records": len(joined),
            "label_counts": dict(sorted(sample_label_counts.items())),
            "provenance": manifest.get(
                "result_preparation",
                {"status": "human_results_csv_provenance_not_recorded"},
            ),
        },
        "sampling_frame": {
            "name": "step3_eligible_step2_tasks",
            "definition": (
                "step2_label=DATA_SECURITY OR "
                "(step1_route=S AND step2_label=OTHER)"
            ),
            "sample_records": len(joined),
            "eligible_population": eligible_population,
            "weighting": "inverse_step3_inclusion_probability_by_year_and_sampling_group",
            "strata": [
                {
                    "application_year": year,
                    "sampling_group": group,
                    **strata[(year, group)],
                }
                for year, group in sorted(strata)
            ],
            "excluded_from_frame": "step1_route=E AND step2_label=OTHER",
        },
        "step1": {
            "prediction_mapping": {"S": "DATA_SECURITY", "E": "OTHER"},
            "sample_unweighted": _evaluate(joined, "step1_prediction", weighted=False),
            "eligible_frame_design_weighted": _evaluate(
                joined,
                "step1_prediction",
                weighted=True,
            ),
        },
        "step2": {
            "prediction_mapping": {
                "DATA_SECURITY": "DATA_SECURITY",
                "OTHER": "OTHER",
            },
            "sample_unweighted": _evaluate(joined, "step2_prediction", weighted=False),
            "eligible_frame_design_weighted": _evaluate(
                joined,
                "step2_prediction",
                weighted=True,
            ),
        },
        "reporting_guidance": {
            "recommended_primary_view": "eligible_frame_design_weighted",
            "limitations": [
                (
                    "Unweighted metrics describe only the positive-priority 4,000-record "
                    "Step 3 sample and are not full-population accuracy estimates."
                ),
                (
                    "Design-weighted metrics generalize only to the Step 3 eligible frame; "
                    "E-to-OTHER records were not sampled for human review and are excluded."
                ),
                (
                    "Step 1 metrics treat route S as a positive prediction and route E as a "
                    "negative prediction; S/E was originally designed as routing, not truth."
                ),
            ],
        },
        "evaluated_at": datetime.now(UTC).isoformat(),
    }
    manifest["evaluation"] = report
    atomic_json_write(paths.manifest, manifest)
    return report


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Step 3 manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_reference_results(
    paths: Step3Paths,
    *,
    expected_count: int,
) -> list[dict[str, Any]]:
    if not paths.results.is_file():
        raise FileNotFoundError(f"Missing Step 3 result: {paths.results}")
    required = {*FROZEN_RESULT_FIELDS, "human_evaluation"}
    with paths.results.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"Step 3 result.csv is missing fields: {missing}")
        raw_rows = [dict(row) for row in reader]
    if len(raw_rows) != expected_count:
        raise ValueError(
            f"Step 3 result.csv must contain {expected_count} rows, found {len(raw_rows)}"
        )

    references: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    for row_number, row in enumerate(raw_rows, start=2):
        evaluation = _parse_human_evaluation(
            row["human_evaluation"],
            row_number=row_number,
        )
        identity = {field: str(row.get(field, "") or "") for field in FROZEN_RESULT_FIELDS}
        identity_rows.append(identity)
        references.append({**identity, "reference_positive": evaluation})
    _validate_human_result_identity(
        paths.database,
        identity_rows,
        expected_count=expected_count,
    )
    return references


def _read_step2_rows(
    databases: Iterable[str | Path],
) -> dict[tuple[str, str], dict[str, str]]:
    paths = sorted({Path(path).resolve() for path in databases})
    if not paths:
        raise ValueError("At least one Step 2 database is required for evaluation")
    rows: dict[tuple[str, str], dict[str, str]] = {}
    for database in paths:
        if not database.is_file():
            raise FileNotFoundError(f"Missing Step 2 database: {database}")
        connection = sqlite3.connect(f"file:{database}?mode=ro&immutable=1", uri=True)
        for dataset_id, patent_id, route, result_json in connection.execute(
            "SELECT dataset_id,patent_id,route,result_json "
            "FROM tasks WHERE status='succeeded'"
        ):
            result = json.loads(result_json)
            label = str(result.get("label", ""))
            if route not in {"S", "E"} or label not in LABELS:
                connection.close()
                raise ValueError(
                    f"Invalid Step 2 evaluation row in {database}: route={route}, label={label}"
                )
            key = (str(dataset_id), str(patent_id))
            if key in rows:
                connection.close()
                raise ValueError(f"Duplicate Step 2 patent across databases: {key}")
            rows[key] = {"route": str(route), "label": label}
        connection.close()
    return rows


def _read_strata(manifest: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, int]]:
    strata: dict[tuple[str, str], dict[str, int]] = {}
    for row in manifest.get("strata", []):
        key = (str(row["application_year"]), str(row["sampling_group"]))
        population = int(row["population"])
        sample = int(row["sample"])
        if key in strata or not 0 < sample <= population:
            raise ValueError(f"Invalid Step 3 sampling stratum: {row}")
        strata[key] = {"population": population, "sample": sample}
    if not strata:
        raise ValueError("Step 3 manifest contains no sampling strata")
    return strata


def _sampling_group(route: str, label: str) -> str:
    if label == "DATA_SECURITY":
        return "positive"
    if route == "S" and label == "OTHER":
        return "hard_negative"
    raise ValueError(
        "Step 3 result contains a record outside the evaluation frame: "
        f"route={route}, step2_label={label}"
    )


def _evaluate(
    rows: list[dict[str, Any]],
    prediction_field: str,
    *,
    weighted: bool,
) -> dict[str, Any]:
    confusion = {
        "true_positive": 0.0,
        "true_negative": 0.0,
        "false_positive": 0.0,
        "false_negative": 0.0,
    }
    for row in rows:
        weight = float(row["design_weight"]) if weighted else 1.0
        prediction = bool(row[prediction_field])
        reference = bool(row["reference_positive"])
        if prediction and reference:
            confusion["true_positive"] += weight
        elif not prediction and not reference:
            confusion["true_negative"] += weight
        elif prediction:
            confusion["false_positive"] += weight
        else:
            confusion["false_negative"] += weight
    if not weighted:
        confusion = {key: int(value) for key, value in confusion.items()}
    return _metrics(confusion, include_accuracy_interval=not weighted)


def _metrics(
    confusion: Mapping[str, float | int],
    *,
    include_accuracy_interval: bool,
) -> dict[str, Any]:
    tp = float(confusion["true_positive"])
    tn = float(confusion["true_negative"])
    fp = float(confusion["false_positive"])
    fn = float(confusion["false_negative"])
    total = tp + tn + fp + fn
    accuracy = _divide(tp + tn, total)
    precision = _divide(tp, tp + fp)
    recall = _divide(tp, tp + fn)
    specificity = _divide(tn, tn + fp)
    negative_predictive_value = _divide(tn, tn + fn)
    f1 = _divide(2 * tp, 2 * tp + fp + fn)
    balanced_accuracy = (
        (recall + specificity) / 2
        if recall is not None and specificity is not None
        else None
    )
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    matthews = (tp * tn - fp * fn) / denominator if denominator else None
    expected_agreement = (
        ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (total * total)
        if total
        else None
    )
    kappa = (
        (accuracy - expected_agreement) / (1 - expected_agreement)
        if accuracy is not None
        and expected_agreement is not None
        and expected_agreement != 1
        else None
    )
    support: float | int = total
    if all(isinstance(value, int) for value in confusion.values()):
        support = int(total)
    result: dict[str, Any] = {
        "confusion_matrix": _rounded(confusion),
        "support": _round(support),
        "accuracy": _round(accuracy),
        "precision": _round(precision),
        "recall_sensitivity": _round(recall),
        "specificity": _round(specificity),
        "negative_predictive_value": _round(negative_predictive_value),
        "f1": _round(f1),
        "balanced_accuracy": _round(balanced_accuracy),
        "matthews_correlation_coefficient": _round(matthews),
        "cohen_kappa": _round(kappa),
    }
    if include_accuracy_interval and accuracy is not None:
        result["accuracy_wilson_95_ci"] = _wilson_interval(int(tp + tn), int(total))
    return result


def _divide(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _wilson_interval(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    proportion = successes / total
    z_squared = Z_95 * Z_95
    denominator = 1 + z_squared / total
    center = (proportion + z_squared / (2 * total)) / denominator
    margin = (
        Z_95
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z_squared / (4 * total * total)
        )
        / denominator
    )
    return [_round(center - margin), _round(center + margin)]


def _rounded(values: Mapping[str, float | int]) -> dict[str, float | int]:
    return {key: _round(value) for key, value in values.items()}


def _round(value: float | int | None) -> float | int | None:
    if value is None or isinstance(value, int):
        return value
    return round(value, 6)
