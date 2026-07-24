"""Reproducible cohort sampling, Gold-result merging and exact 8:1:1 splitting."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pipeline.common.io import atomic_json_write, sha256_file
from pipeline.step2.schema import PatentClassification

SAMPLING_VERSION = "step3-50000-to-10000-dual-cohort-v2.6.0"
SCHEMA_VERSION = "2.6.0"
EXPECTED_STEP2_POPULATION = 50_000
SAMPLE_SEED = "step3-50000-positive-priority-v2.6.0"
SAMPLE_ID_VERSION = "step3-50000-dual-cohort-v2.6.0"
LABELS = ("DATA_SECURITY", "OTHER")
SAMPLING_GROUPS = ("positive", "hard_negative", "easy_negative")
POSITIVE_COHORT = "positive_priority"
NEGATIVE_COHORT = "negative_priority"
NEGATIVE_SAMPLE_SEED = "step3-50000-negative-priority-v2.6.0"
NEGATIVE_PRIORITY_GROUP_TARGETS = {
    "positive": 2_000,
    "hard_negative": 1_000,
    "easy_negative": 2_000,
}
SPLITS = ("train", "validation", "test")
SPLIT_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}

SIMULATION_FIELDS = (
    "sample_id",
    "dataset_id",
    "application_year",
    "patent_id",
    "title",
    "abstract",
    "claim",
    "ipc",
    "main_ipc",
    "label",
    "confidence",
    "scope_basis",
    "processing_activities",
    "industry_sectors",
    "technical_scope",
    "legal_scope",
    "evidence",
    "reason",
    "needs_review",
    "review_reason",
    "annotation_source",
    "annotation_model",
    "annotation_prompt_version",
    "gold_status",
    "eligible_for_final_evaluation",
)

FROZEN_RESULT_FIELDS = (
    "sample_id",
    "dataset_id",
    "application_year",
    "patent_id",
    "title",
    "abstract",
    "claim",
    "ipc",
    "main_ipc",
)

MANUAL_REVIEW_FIELDS = (
    *FROZEN_RESULT_FIELDS,
    "sample_cohort",
    "step1_label",
    "step2_label",
    "step2_confidence",
    "step2_scope_basis",
    "step2_processing_activities",
    "step2_industry_sectors",
    "step2_technical_scope",
    "step2_legal_scope",
    "step2_evidence",
    "step2_reason",
    "step2_needs_review",
    "step2_review_reason",
    "human_review_label",
    "human_reason",
)

# The completed human-review file preserves the full label lineage and all
# Step 2 evidence shown to the reviewer. Only human_review_label is Gold.
RESULT_FIELDS = MANUAL_REVIEW_FIELDS


@dataclass(frozen=True)
class SamplingConfig:
    target_size: int = 5_000
    positive_size: int = 3_000
    hard_negative_size: int = 2_000
    seed: str = SAMPLE_SEED

    @property
    def group_targets(self) -> dict[str, int]:
        return {
            "positive": self.positive_size,
            "hard_negative": self.hard_negative_size,
        }


@dataclass(frozen=True)
class Step3Paths:
    root: Path
    database: Path
    manifest: Path
    progress: Path
    simulation: Path
    manual_review: Path
    manual_review_negative: Path
    result_positive: Path
    result_negative: Path
    results: Path
    train: Path
    validation: Path
    test: Path


def step3_paths(output_dir: str | Path) -> Step3Paths:
    root = Path(output_dir).resolve()
    return Step3Paths(
        root=root,
        database=root / "tasks.sqlite3",
        manifest=root / "manifest.json",
        progress=root / "progress.json",
        simulation=root / "simulation.csv",
        manual_review=root / "need_manual_review_positive.csv",
        manual_review_negative=root / "need_manual_review_negative.csv",
        result_positive=root / "result_positive.csv",
        result_negative=root / "result_negative.csv",
        results=root / "result.csv",
        train=root / "dataset" / "train.csv",
        validation=root / "dataset" / "validation.csv",
        test=root / "dataset" / "test.csv",
    )


def discover_step2_databases(step2_dir: str | Path) -> list[Path]:
    root = Path(step2_dir).resolve()
    canonical = (root / "tasks.sqlite3", *root.glob("*/tasks.sqlite3"))
    legacy = root.glob("step2_tasks_*.sqlite3")
    return sorted(
        path
        for path in {*canonical, *legacy}
        if path.is_file() and ".before_" not in path.name
    )


def prepare_sample(
    databases: Iterable[str | Path],
    output_dir: str | Path,
    *,
    config: SamplingConfig | None = None,
    rebuild: bool = False,
) -> tuple[Step3Paths, dict[str, Any]]:
    """Freeze the positive-priority year-balanced sample and simulation tasks."""

    config = config or SamplingConfig()
    _validate_config(config)
    database_paths = sorted({Path(path).resolve() for path in databases})
    if not database_paths:
        raise ValueError("At least one Step 2 database is required")
    missing = [path for path in database_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Step 2 databases: {missing}")

    paths = step3_paths(output_dir)
    _prepare_output(paths, rebuild=rebuild)
    all_records, source_summary, logical_digest = _load_population(database_paths)
    _validate_step2_population(len(all_records))
    records = [
        record
        for record in all_records
        if record["sampling_group"] in config.group_targets
    ]
    years = sorted({record["application_year"] for record in records})
    capacities = Counter(
        (record["application_year"], record["sampling_group"]) for record in records
    )

    quotas: dict[tuple[str, str], int] = {}
    for sampling_group, target in config.group_targets.items():
        group_capacities = {year: capacities[(year, sampling_group)] for year in years}
        quotas.update(
            {
                (year, sampling_group): value
                for year, value in _balanced_capacity_allocation(
                    group_capacities,
                    target,
                    seed=f"{config.seed}|{sampling_group}",
                ).items()
            }
        )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["application_year"], record["sampling_group"])].append(record)

    selected: list[dict[str, Any]] = []
    stratum_rows: list[dict[str, Any]] = []
    for stratum in sorted(grouped):
        population = grouped[stratum]
        quota = quotas[stratum]
        ranked = sorted(
            population,
            key=lambda row: _stable_score(
                config.seed,
                row["dataset_id"],
                row["patent_id"],
            ),
        )
        probability = quota / len(population)
        year, sampling_group = stratum
        stratum_rows.append(
            {
                "application_year": year,
                "sampling_group": sampling_group,
                "population": len(population),
                "sample": quota,
                "inclusion_probability": probability,
            }
        )
        for record in ranked[:quota]:
            row = dict(record)
            row.update(
                {
                    "sample_id": _sample_id(config.seed, row["dataset_id"], row["patent_id"]),
                    "sample_cohort": POSITIVE_COHORT,
                    "sampling_version": SAMPLING_VERSION,
                    "sample_seed": config.seed,
                    "sampling_stratum": f"year={year}|sampling_group={sampling_group}",
                    "stratum_population": len(population),
                    "stratum_sample_size": quota,
                    "step3_inclusion_probability": probability,
                    "step3_sample_weight": 1 / probability,
                }
            )
            combined_probability = row["step2_selection_probability"] * probability
            row["combined_inclusion_probability"] = combined_probability
            row["combined_sample_weight"] = 1 / combined_probability
            selected.append(row)

    selected.sort(key=lambda row: row["sample_id"])
    if len(selected) != config.target_size:
        raise AssertionError(f"Expected {config.target_size} selected records, got {len(selected)}")
    if len({row["patent_id"] for row in selected}) != len(selected):
        raise ValueError("Selected sample contains duplicate patent_id values")

    _initialize_task_database(paths.database, selected)
    _write_csv(
        paths.manual_review,
        MANUAL_REVIEW_FIELDS,
        (_manual_review_row(row) for row in selected),
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "sampling_version": SAMPLING_VERSION,
        "sample_id_version": SAMPLE_ID_VERSION,
        "target_size": config.target_size,
        "sampling_group_targets": config.group_targets,
        "positive_definition": "step2_label=DATA_SECURITY",
        "hard_negative_definition": "step1_route=S and step2_label=OTHER",
        "year_allocation": "equal_within_sampling_group_with_capacity_redistribution",
        "years": years,
        "seed": config.seed,
        "step2_population": len(all_records),
        "eligible_sampling_population": len(records),
        "population_by_step2_label": dict(
            sorted(Counter(r["step2_label"] for r in all_records).items())
        ),
        "eligible_population_by_sampling_group": dict(
            sorted(Counter(r["sampling_group"] for r in records).items())
        ),
        "sample_by_sampling_group": dict(
            sorted(Counter(r["sampling_group"] for r in selected).items())
        ),
        "strata": stratum_rows,
        "logical_input_sha256": logical_digest,
        "step2_sources": source_summary,
        "outputs": {
            key: str(getattr(paths, key))
            for key in ("database", "manifest", "manual_review")
        },
        "simulation_policy": {
            "simulation_is_human_gold": False,
            "eligible_for_training_splits": False,
        },
        "human_results_policy": {
            "path": str(paths.result_positive),
            "label_field": "human_review_label",
            "allowed_values": list(LABELS),
            "result_fields": list(RESULT_FIELDS),
        },
        "manual_review_policy": {
            "path": str(paths.manual_review),
            "cohort": POSITIVE_COHORT,
            "sha256": sha256_file(paths.manual_review),
            "records": config.target_size,
            "fields": list(MANUAL_REVIEW_FIELDS),
            "step2_decision_visible": True,
            "human_fields_initially_blank": ["human_review_label", "human_reason"],
        },
        "finalize_outputs": {
            "result": str(paths.results),
            "train": str(paths.train),
            "validation": str(paths.validation),
            "test": str(paths.test),
        },
        "prepared_at": _now(),
    }
    atomic_json_write(paths.manifest, manifest)
    return paths, manifest


def prepare_negative_sample(
    databases: Iterable[str | Path],
    output_dir: str | Path,
    *,
    seed: str = NEGATIVE_SAMPLE_SEED,
) -> tuple[Step3Paths, dict[str, Any]]:
    """Append a disjoint 5,000-record negative-priority review cohort.

    The requested 2:3 ratio describes frozen Step 2 predictions, not the unknown
    human Gold labels. Fixed subgroup quotas retain hard-negative emphasis while
    giving easy negatives non-zero inclusion probability in the 50,000-record frame.
    """

    paths = step3_paths(output_dir)
    required = (paths.database, paths.manifest, paths.manual_review)
    missing_outputs = [path for path in required if not path.is_file()]
    if missing_outputs:
        raise FileNotFoundError(
            "Prepare the positive-priority cohort first; missing: "
            + ", ".join(map(str, missing_outputs))
        )
    conflicting = [
        path
        for path in (paths.manual_review_negative, paths.result_negative, paths.results)
        if path.exists()
    ]
    if conflicting:
        raise FileExistsError(
            "Negative cohort or combined results already exist; refusing to overwrite: "
            + ", ".join(map(str, conflicting))
        )

    database_paths = sorted({Path(path).resolve() for path in databases})
    if not database_paths:
        raise ValueError("At least one Step 2 database is required")
    missing = [path for path in database_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Step 2 databases: {missing}")

    positive_rows = _read_csv(paths.manual_review)
    if len(positive_rows) != 5_000:
        raise ValueError(
            "Positive-priority review input must contain 5,000 rows, found "
            f"{len(positive_rows)}"
        )
    _validate_cohort_rows(
        positive_rows,
        expected_cohort=POSITIVE_COHORT,
        require_human_result=False,
        source=paths.manual_review,
    )
    if paths.result_positive.is_file():
        positive_results = _read_and_normalize_human_result(
            paths.result_positive,
            expected_count=5_000,
            expected_cohort=POSITIVE_COHORT,
        )
        _validate_human_result_identity(
            paths.database,
            positive_results,
            expected_count=5_000,
        )
    existing_patent_ids = {row["patent_id"] for row in positive_rows}
    existing_sample_ids = {row["sample_id"] for row in positive_rows}

    all_records, source_summary, logical_digest = _load_population(database_paths)
    _validate_step2_population(len(all_records))
    remaining = [row for row in all_records if row["patent_id"] not in existing_patent_ids]
    remaining_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in remaining:
        remaining_by_group[record["sampling_group"]].append(record)

    group_targets = dict(NEGATIVE_PRIORITY_GROUP_TARGETS)
    positive_target = group_targets["positive"]
    negative_target = (
        group_targets["hard_negative"] + group_targets["easy_negative"]
    )
    for group, target in group_targets.items():
        available = len(remaining_by_group[group])
        if available < target:
            raise ValueError(
                f"Negative-priority quota cannot be met for {group}: "
                f"target={target}, remaining={available}"
            )

    selected: list[dict[str, Any]] = []
    cohort_strata: list[dict[str, Any]] = []
    for sampling_group, target in group_targets.items():
        group_records = remaining_by_group[sampling_group]
        years = sorted({row["application_year"] for row in group_records})
        capacities = Counter(row["application_year"] for row in group_records)
        quotas = _balanced_capacity_allocation(
            {year: capacities[year] for year in years},
            target,
            seed=f"{seed}|{sampling_group}",
        )
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in group_records:
            grouped[record["application_year"]].append(record)
        for year in years:
            population = grouped[year]
            quota = quotas[year]
            probability = quota / len(population)
            cohort_strata.append(
                {
                    "application_year": year,
                    "sampling_group": sampling_group,
                    "remaining_population": len(population),
                    "sample": quota,
                    "conditional_inclusion_probability": probability,
                }
            )
            ranked = sorted(
                population,
                key=lambda row: _stable_score(
                    seed,
                    row["dataset_id"],
                    row["patent_id"],
                ),
            )
            for record in ranked[:quota]:
                row = dict(record)
                row.update(
                    {
                        "sample_id": _sample_id(
                            SAMPLE_SEED,
                            row["dataset_id"],
                            row["patent_id"],
                        ),
                        "sample_cohort": NEGATIVE_COHORT,
                        "sampling_version": SAMPLING_VERSION,
                        "sample_seed": seed,
                        "sampling_stratum": (
                            f"cohort={NEGATIVE_COHORT}|year={year}|"
                            f"sampling_group={sampling_group}"
                        ),
                    }
                )
                selected.append(row)

    selected.sort(key=lambda row: row["sample_id"])
    if len(selected) != 5_000:
        raise AssertionError(f"Expected 5,000 negative-priority rows, found {len(selected)}")
    if len({row["patent_id"] for row in selected}) != len(selected):
        raise ValueError("Negative-priority sample contains duplicate patent_id values")
    if existing_patent_ids & {row["patent_id"] for row in selected}:
        raise ValueError("Positive- and negative-priority cohorts overlap by patent_id")
    if existing_sample_ids & {row["sample_id"] for row in selected}:
        raise ValueError("Positive- and negative-priority cohorts overlap by sample_id")

    _append_task_database(paths.database, selected, expected_existing=5_000)
    _write_csv(
        paths.manual_review_negative,
        MANUAL_REVIEW_FIELDS,
        (_manual_review_row(row) for row in selected),
    )

    combined_group_counts = Counter()
    combined_strata_counts: Counter[tuple[str, str]] = Counter()
    for row in positive_rows:
        group = _sampling_group_from_labels(row["step1_label"], row["step2_label"])
        combined_group_counts[group] += 1
        combined_strata_counts[(row["application_year"], group)] += 1
    for row in selected:
        group = row["sampling_group"]
        combined_group_counts[group] += 1
        combined_strata_counts[(row["application_year"], group)] += 1
    population_strata = Counter(
        (row["application_year"], row["sampling_group"]) for row in all_records
    )
    combined_strata = []
    for (year, group), population in sorted(population_strata.items()):
        sample = combined_strata_counts[(year, group)]
        if not sample:
            continue
        combined_strata.append(
            {
                "application_year": year,
                "sampling_group": group,
                "population": population,
                "sample": sample,
                "inclusion_probability": sample / population,
            }
        )

    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    previous_evaluation = manifest.pop("evaluation", None)
    if previous_evaluation:
        manifest.setdefault("cohort_evaluations", {})[POSITIVE_COHORT] = previous_evaluation
    manifest.update(
        {
            "schema_version": SCHEMA_VERSION,
            "sampling_version": SAMPLING_VERSION,
            "target_size": 10_000,
            "sampling_group_targets": dict(sorted(combined_group_counts.items())),
            "negative_priority_step2_label_targets": {
                "DATA_SECURITY": positive_target,
                "OTHER": negative_target,
            },
            "positive_definition": "step2_label=DATA_SECURITY",
            "hard_negative_definition": "step1_label=DATA_SECURITY and step2_label=OTHER",
            "easy_negative_definition": "step1_label=OTHER and step2_label=OTHER",
            "step2_population": len(all_records),
            "eligible_sampling_population": len(all_records),
            "population_by_step2_label": dict(
                sorted(Counter(r["step2_label"] for r in all_records).items())
            ),
            "eligible_population_by_sampling_group": dict(
                sorted(Counter(r["sampling_group"] for r in all_records).items())
            ),
            "sample_by_sampling_group": dict(sorted(combined_group_counts.items())),
            "strata": combined_strata,
            "logical_input_sha256": logical_digest,
            "step2_sources": source_summary,
        }
    )
    manifest["outputs"] = {
        "database": str(paths.database),
        "manifest": str(paths.manifest),
        "need_manual_review_positive": str(paths.manual_review),
        "need_manual_review_negative": str(paths.manual_review_negative),
        "result_positive": str(paths.result_positive),
        "result_negative": str(paths.result_negative),
        "result": str(paths.results),
        "train": str(paths.train),
        "validation": str(paths.validation),
        "test": str(paths.test),
    }
    manifest["cohorts"] = {
        POSITIVE_COHORT: _cohort_manifest(
            paths.manual_review,
            paths.result_positive,
            positive_rows,
        ),
        NEGATIVE_COHORT: {
            **_cohort_manifest(
                paths.manual_review_negative,
                paths.result_negative,
                [_manual_review_row(row) for row in selected],
            ),
            "selection_seed": seed,
            "step2_label_targets": {
                "DATA_SECURITY": positive_target,
                "OTHER": negative_target,
            },
            "sampling_group_targets": dict(sorted(group_targets.items())),
            "conditional_strata": cohort_strata,
            "hard_negative_priority": (
                "fixed 1,000 hard-negative and 2,000 easy-negative quotas preserve "
                "boundary emphasis and complete-frame coverage"
            ),
        },
    }
    manifest["human_results_policy"] = {
        "merge_command": "python -m pipeline.step3 merge",
        "inputs": [str(paths.result_positive), str(paths.result_negative)],
        "path": str(paths.results),
        "label_field": "human_review_label",
        "allowed_values": list(LABELS),
        "expected_records": 10_000,
        "result_fields": list(RESULT_FIELDS),
    }
    manifest["annotation_state"] = {
        "positive_review_complete": paths.result_positive.is_file(),
        "negative_review_complete": False,
        "combined_result_complete": False,
        "stale_split_files_must_not_be_used": True,
    }
    manifest["dual_cohort_expansion"] = {
        "positive_records_preserved": 5_000,
        "negative_records_added": 5_000,
        "cohort_overlap_records": 0,
        "negative_step2_predicted_ratio": "2:3",
        "negative_subgroup_ratio": "hard_negative:easy_negative=1:2",
        "combined_step2_predicted_ratio": "1:1",
        "gold_ratio_status": "unknown_until_negative_human_review",
        "expanded_at": _now(),
    }
    manifest.pop("human_results", None)
    atomic_json_write(paths.manifest, manifest)
    return paths, manifest


def merge_review_results(
    paths: Step3Paths,
    *,
    expected_per_cohort: int = 5_000,
) -> dict[str, Any]:
    """Validate and merge the two independently reviewed cohorts into result.csv."""

    positive = _read_and_normalize_human_result(
        paths.result_positive,
        expected_count=expected_per_cohort,
        expected_cohort=POSITIVE_COHORT,
    )
    negative = _read_and_normalize_human_result(
        paths.result_negative,
        expected_count=expected_per_cohort,
        expected_cohort=NEGATIVE_COHORT,
    )
    combined = sorted([*positive, *negative], key=lambda row: row["sample_id"])
    if len({row["sample_id"] for row in combined}) != len(combined):
        raise ValueError("Review cohorts overlap by sample_id")
    if len({row["patent_id"] for row in combined}) != len(combined):
        raise ValueError("Review cohorts overlap by patent_id")
    _validate_human_result_identity(
        paths.database,
        combined,
        expected_count=expected_per_cohort * 2,
    )
    _write_csv(paths.results, RESULT_FIELDS, (_result_row(row) for row in combined))

    report = {
        "strategy": "validated_disjoint_union_by_sample_id",
        "inputs": {
            POSITIVE_COHORT: {
                "path": str(paths.result_positive),
                "sha256": sha256_file(paths.result_positive),
                "records": len(positive),
                "label_counts": dict(
                    sorted(Counter(row["human_review_label"] for row in positive).items())
                ),
            },
            NEGATIVE_COHORT: {
                "path": str(paths.result_negative),
                "sha256": sha256_file(paths.result_negative),
                "records": len(negative),
                "label_counts": dict(
                    sorted(Counter(row["human_review_label"] for row in negative).items())
                ),
            },
        },
        "output": {
            "path": str(paths.results),
            "sha256": sha256_file(paths.results),
            "records": len(combined),
            "label_counts": dict(
                sorted(Counter(row["human_review_label"] for row in combined).items())
            ),
        },
        "duplicate_sample_ids": 0,
        "duplicate_patent_ids": 0,
        "merged_at": _now(),
    }
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    manifest["result_preparation"] = report
    manifest["annotation_state"] = {
        "positive_review_complete": True,
        "negative_review_complete": True,
        "combined_result_complete": True,
        "stale_split_files_must_not_be_used": True,
    }
    manifest.pop("human_results", None)
    manifest.pop("evaluation", None)
    atomic_json_write(paths.manifest, manifest)
    return report


def assign_exact_splits(
    rows: list[dict[str, Any]],
    *,
    seed: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assign exact 80/10/10 totals, approximately stratified by year and final label."""

    total = len(rows)
    if total < 10:
        raise ValueError("At least 10 annotated rows are required for an 8:1:1 split")
    validation_target = round(total * SPLIT_RATIOS["validation"])
    test_target = round(total * SPLIT_RATIOS["test"])
    train_target = total - validation_target - test_target

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["application_year"]), str(row["label"]))].append(row)
    sizes = {stratum: len(values) for stratum, values in grouped.items()}
    validation_quota = _proportional_quota(sizes, validation_target)
    remaining = {key: sizes[key] - validation_quota[key] for key in sizes}
    test_quota = _proportional_quota(sizes, test_target, capacities=remaining)

    output: list[dict[str, Any]] = []
    report_strata: list[dict[str, Any]] = []
    for stratum in sorted(grouped):
        ranked = sorted(
            grouped[stratum],
            key=lambda row: _stable_score(seed, row["sample_id"], "split"),
        )
        n_validation = validation_quota[stratum]
        n_test = test_quota[stratum]
        for index, row in enumerate(ranked):
            item = dict(row)
            item["split_group_id"] = _text_group_id(item)
            if index < n_validation:
                item["data_split"] = "validation"
            elif index < n_validation + n_test:
                item["data_split"] = "test"
            else:
                item["data_split"] = "train"
            output.append(item)
        report_strata.append(
            {
                "application_year": stratum[0],
                "label": stratum[1],
                "total": len(ranked),
                "train": len(ranked) - n_validation - n_test,
                "validation": n_validation,
                "test": n_test,
            }
        )

    _keep_text_groups_together(output, seed=seed)
    output.sort(key=lambda row: row["sample_id"])
    counts = Counter(row["data_split"] for row in output)
    expected = {"train": train_target, "validation": validation_target, "test": test_target}
    if dict(counts) != expected:
        raise AssertionError(f"Split totals differ: actual={dict(counts)}, expected={expected}")
    return output, {"ratios": SPLIT_RATIOS, "counts": expected, "strata": report_strata}


def write_simulation_dataset(
    paths: Step3Paths,
    rows: list[dict[str, Any]],
    *,
    annotation_model: str,
    annotation_prompt_version: str,
) -> dict[str, Any]:
    """Write the rich model-simulation audit file without creating training splits."""

    normalized: list[dict[str, Any]] = []
    for row in rows:
        annotation = PatentClassification.model_validate(row["annotation"])
        item = {
            **{field: row.get(field, "") for field in FROZEN_RESULT_FIELDS},
            **annotation.model_dump(),
            "annotation_source": "codex_model_simulation",
            "annotation_model": annotation_model,
            "annotation_prompt_version": annotation_prompt_version,
            "gold_status": "provisional_not_human_gold",
            "eligible_for_final_evaluation": False,
        }
        normalized.append(item)
    _write_csv(
        paths.simulation,
        SIMULATION_FIELDS,
        (_simulation_row(row) for row in normalized),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "annotation_source": "codex_model_simulation",
        "annotation_model": annotation_model,
        "annotation_prompt_version": annotation_prompt_version,
        "gold_status": "provisional_not_human_gold",
        "eligible_for_training_splits": False,
        "records": len(normalized),
        "label_counts": dict(sorted(Counter(row["label"] for row in normalized).items())),
        "output": str(paths.simulation),
        "written_at": _now(),
    }


def finalize_human_results(
    paths: Step3Paths,
    *,
    split_seed: str = "step3-human-split-v2.6.0",
    expected_count: int | None = None,
) -> dict[str, Any]:
    """Validate explicit human Gold labels and create exact 8:1:1 splits."""

    if expected_count is None:
        manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
        expected_count = int(manifest.get("target_size", 10_000))
    if not paths.results.is_file():
        raise FileNotFoundError(
            f"Missing combined human annotation file: {paths.results}; run merge first"
        )
    with paths.results.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        input_fields = tuple(reader.fieldnames or ())
        missing = sorted(set(RESULT_FIELDS) - set(input_fields))
        if missing:
            raise ValueError(f"Human result.csv is missing fields: {missing}")
        del reader
    normalized = _read_and_normalize_human_result(
        paths.results,
        expected_count=expected_count,
        expected_cohort=None,
    )

    _validate_human_result_identity(paths.database, normalized, expected_count=expected_count)
    assigned, report = assign_exact_splits(normalized, seed=split_seed)
    by_split = {
        split: [row for row in assigned if row["data_split"] == split] for split in SPLITS
    }

    for path in (paths.results, paths.train, paths.validation, paths.test):
        path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(paths.results, RESULT_FIELDS, (_result_row(row) for row in assigned))
    _write_csv(paths.train, RESULT_FIELDS, (_result_row(row) for row in by_split["train"]))
    _write_csv(
        paths.validation,
        RESULT_FIELDS,
        (_result_row(row) for row in by_split["validation"]),
    )
    _write_csv(paths.test, RESULT_FIELDS, (_result_row(row) for row in by_split["test"]))
    report.update(
        {
            "schema_version": SCHEMA_VERSION,
            "source": "human_results_csv",
            "source_path": str(paths.results),
            "source_sha256": sha256_file(paths.results),
            "split_seed": split_seed,
            "result_fields": list(RESULT_FIELDS),
            "removed_input_fields": sorted(set(input_fields) - set(RESULT_FIELDS)),
            "gold_label_field": "human_review_label",
            "human_review_label_counts": dict(
                sorted(Counter(row["human_review_label"] for row in assigned).items())
            ),
            "written_at": _now(),
        }
    )
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    manifest["human_results"] = report
    manifest["outputs"].update(
        {
            "result": str(paths.results),
            "train": str(paths.train),
            "validation": str(paths.validation),
            "test": str(paths.test),
        }
    )
    atomic_json_write(paths.manifest, manifest)
    return report


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _read_and_normalize_human_result(
    path: Path,
    *,
    expected_count: int,
    expected_cohort: str | None,
) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing reviewed cohort: {path}")
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = sorted(set(RESULT_FIELDS) - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"{path.name} is missing fields: {missing}")
        raw_rows = [dict(row) for row in reader]
    if len(raw_rows) != expected_count:
        raise ValueError(
            f"{path.name} must contain {expected_count} rows, found {len(raw_rows)}"
        )
    _validate_cohort_rows(
        raw_rows,
        expected_cohort=expected_cohort,
        require_human_result=True,
        source=path,
    )

    normalized: list[dict[str, Any]] = []
    for row_number, row in enumerate(raw_rows, start=2):
        step1_label = _parse_review_label(
            row["step1_label"], field="step1_label", row_number=row_number
        )
        step2_label = _parse_review_label(
            row["step2_label"], field="step2_label", row_number=row_number
        )
        human_review_label = _parse_review_label(
            row["human_review_label"], field="human_review_label", row_number=row_number
        )
        human_reason = str(row["human_reason"] or "").strip()
        item = {field: str(row.get(field, "") or "") for field in RESULT_FIELDS}
        item.update(
            {
                "step1_label": step1_label,
                "step2_label": step2_label,
                "step2_needs_review": str(
                    _parse_boolean(
                        row["step2_needs_review"],
                        field="step2_needs_review",
                        row_number=row_number,
                    )
                ).lower(),
                "human_review_label": human_review_label,
                "human_reason": human_reason,
                "label": human_review_label,
            }
        )
        normalized.append(item)
    return normalized


def _validate_cohort_rows(
    rows: list[dict[str, Any]],
    *,
    expected_cohort: str | None,
    require_human_result: bool,
    source: Path,
) -> None:
    sample_ids: set[str] = set()
    patent_ids: set[str] = set()
    allowed_cohorts = {POSITIVE_COHORT, NEGATIVE_COHORT}
    for row_number, row in enumerate(rows, start=2):
        sample_id = str(row.get("sample_id", "")).strip()
        patent_id = str(row.get("patent_id", "")).strip()
        cohort = str(row.get("sample_cohort", "")).strip()
        if not sample_id or sample_id in sample_ids:
            raise ValueError(f"{source.name} row {row_number}: duplicate or empty sample_id")
        if not patent_id or patent_id in patent_ids:
            raise ValueError(f"{source.name} row {row_number}: duplicate or empty patent_id")
        if cohort not in allowed_cohorts:
            raise ValueError(
                f"{source.name} row {row_number}: sample_cohort must be one of "
                f"{sorted(allowed_cohorts)}, got {cohort!r}"
            )
        if expected_cohort is not None and cohort != expected_cohort:
            raise ValueError(
                f"{source.name} row {row_number}: expected sample_cohort="
                f"{expected_cohort}, got {cohort}"
            )
        if require_human_result:
            _parse_review_label(
                row.get("human_review_label"),
                field="human_review_label",
                row_number=row_number,
            )
            if not str(row.get("human_reason", "")).strip():
                raise ValueError(f"{source.name} row {row_number}: human_reason is required")
        sample_ids.add(sample_id)
        patent_ids.add(patent_id)


def _cohort_manifest(
    review_path: Path,
    result_path: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    cohort = str(rows[0].get("sample_cohort", "")) if rows else ""
    step2_counts = Counter(str(row.get("step2_label", "")) for row in rows)
    group_counts = Counter(
        _sampling_group_from_labels(
            str(row.get("step1_label", "")),
            str(row.get("step2_label", "")),
        )
        for row in rows
    )
    result_complete = result_path.is_file()
    output = {
        "cohort": cohort,
        "records": len(rows),
        "review_input": str(review_path),
        "review_input_sha256": sha256_file(review_path),
        "result": str(result_path),
        "result_status": "complete" if result_complete else "awaiting_human_review",
        "step2_label_counts": dict(sorted(step2_counts.items())),
        "sampling_group_counts": dict(sorted(group_counts.items())),
    }
    if result_complete:
        result_rows = _read_csv(result_path)
        output["result_sha256"] = sha256_file(result_path)
        output["human_review_label_counts"] = dict(
            sorted(Counter(row.get("human_review_label", "") for row in result_rows).items())
        )
    return output


def _append_task_database(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    expected_existing: int,
) -> None:
    with _exclusive_database_update(path):
        connection = sqlite3.connect(path)
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            connection.close()
            raise ValueError(f"SQLite integrity check failed for {path}")
        existing = connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        if existing != expected_existing:
            connection.close()
            raise ValueError(
                f"Frozen task database must contain {expected_existing} rows before append, "
                f"found {existing}"
            )
        now = _now()
        try:
            connection.executemany(
                """
                INSERT INTO tasks (
                  sample_id,dataset_id,application_year,patent_id,payload_json,
                  status,created_at,updated_at
                ) VALUES (?,?,?,?,?,'pending',?,?)
                """,
                [
                    (
                        row["sample_id"],
                        row["dataset_id"],
                        row["application_year"],
                        row["patent_id"],
                        json.dumps(
                            {
                                "patent_id": row["patent_id"],
                                "title": row["title"],
                                "abstract": row["abstract"],
                                "claim": row["claim"],
                                "ipc": row["ipc"],
                                "main_ipc": row["main_ipc"],
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        now,
                        now,
                    )
                    for row in rows
                ],
            )
            final_count = connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
            if final_count != expected_existing + len(rows):
                raise AssertionError(
                    f"Expected {expected_existing + len(rows)} tasks, found {final_count}"
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


def _load_population(
    databases: list[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    records: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    seen_patents: set[str] = set()
    for database in databases:
        connection = sqlite3.connect(f"file:{database}?mode=ro&immutable=1", uri=True)
        connection.row_factory = sqlite3.Row
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            connection.close()
            raise ValueError(f"SQLite integrity check failed for {database}: {integrity}")
        status_counts = dict(
            connection.execute("SELECT status, COUNT(*) FROM tasks GROUP BY status")
        )
        if set(status_counts) != {"succeeded"}:
            connection.close()
            raise ValueError(
                f"Step 2 must be completely succeeded before sampling: {database} {status_counts}"
            )
        source_counts: Counter[str] = Counter()
        for row in connection.execute("SELECT * FROM tasks ORDER BY dataset_id, patent_id"):
            patent_id = str(row["patent_id"])
            if patent_id in seen_patents:
                connection.close()
                raise ValueError(f"Duplicate patent_id across Step 2 databases: {patent_id}")
            seen_patents.add(patent_id)
            result = PatentClassification.model_validate_json(row["result_json"])
            payload = json.loads(row["payload_json"])
            dataset_id = str(row["dataset_id"])
            source_counts[result.label] += 1
            record = {
                "dataset_id": dataset_id,
                "application_year": dataset_id,
                "patent_id": patent_id,
                "source_row_number": row["source_row_number"],
                "step2_route": row["route"],
                "step1_label": "DATA_SECURITY" if row["route"] == "S" else "OTHER",
                "step2_selection_group": row["selection_group"],
                "step2_selection_probability": float(row["selection_probability"]),
                "step2_sample_weight": float(row["sample_weight"]),
                "step2_label": result.label,
                "sampling_group": _sampling_group(row["route"], result.label),
                "step2_confidence": result.confidence,
                "step2_scope_basis": result.scope_basis,
                "step2_processing_activities": result.processing_activities,
                "step2_industry_sectors": result.industry_sectors,
                "step2_technical_scope": result.technical_scope,
                "step2_legal_scope": result.legal_scope,
                "step2_evidence": [item.model_dump() for item in result.evidence],
                "step2_reason": result.reason,
                "step2_needs_review": result.needs_review,
                "step2_review_reason": result.review_reason,
                "step2_requested_model": row["requested_model"] or "",
                "step2_actual_model": row["actual_model"] or "",
                "step2_prompt_version": row["prompt_version"] or "",
                "step2_response_id": row["response_id"] or "",
                **{
                    field: str(payload.get(field, "") or "")
                    for field in ("title", "abstract", "claim", "ipc", "main_ipc")
                },
            }
            records.append(record)
            digest.update(
                json.dumps(
                    [
                        dataset_id,
                        patent_id,
                        row["route"],
                        result.label,
                        payload,
                    ],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            )
        sources.append(
            {
                "database": str(database),
                "records": sum(source_counts.values()),
                "by_step2_label": dict(sorted(source_counts.items())),
            }
        )
        connection.close()
    return records, sources, digest.hexdigest()


def _sampling_group(route: str, label: str) -> str:
    if label == "DATA_SECURITY":
        return "positive"
    if route == "S" and label == "OTHER":
        return "hard_negative"
    if route == "E" and label == "OTHER":
        return "easy_negative"
    raise ValueError(f"Unsupported Step 1/2 label combination: route={route!r}, label={label!r}")


def _sampling_group_from_labels(step1_label: str, step2_label: str) -> str:
    route = "S" if step1_label == "DATA_SECURITY" else "E"
    if step1_label not in LABELS or step2_label not in LABELS:
        raise ValueError(
            f"Unsupported Step 1/2 labels: step1={step1_label!r}, step2={step2_label!r}"
        )
    return _sampling_group(route, step2_label)


def _validate_step2_population(population_size: int) -> None:
    if population_size != EXPECTED_STEP2_POPULATION:
        raise ValueError(
            "Step 3 requires the frozen 50,000-record Step 2 frame; "
            f"found {population_size} records"
        )


def _balanced_capacity_allocation(
    capacities: Mapping[str, int],
    target: int,
    *,
    seed: str,
) -> dict[str, int]:
    if sum(capacities.values()) < target:
        available = sum(capacities.values())
        raise ValueError(
            f"Stable label quota {target} cannot be met; total available={available}, "
            f"capacities={dict(capacities)}"
        )
    quotas = {key: 0 for key in capacities}
    tie_order = {key: _stable_score(seed, key, "year-capacity-allocation") for key in capacities}
    for _ in range(target):
        eligible = [key for key, capacity in capacities.items() if quotas[key] < capacity]
        chosen = min(eligible, key=lambda key: (quotas[key], tie_order[key]))
        quotas[chosen] += 1
    return quotas


def _proportional_quota(
    sizes: Mapping[tuple[str, str], int],
    target: int,
    *,
    capacities: Mapping[tuple[str, str], int] | None = None,
) -> dict[tuple[str, str], int]:
    total = sum(sizes.values())
    if target > sum((capacities or sizes).values()):
        raise ValueError("Split target exceeds available capacity")
    raw = {key: sizes[key] * target / total for key in sizes}
    limits = capacities or sizes
    quota = {key: min(int(raw[key]), limits[key]) for key in sizes}
    remaining = target - sum(quota.values())
    while remaining:
        eligible = [key for key in sizes if quota[key] < limits[key]]
        chosen = max(
            eligible,
            key=lambda key: (raw[key] - quota[key], _stable_score("split-quota", *key)),
        )
        quota[chosen] += 1
        remaining -= 1
    return quota


def _validate_config(config: SamplingConfig) -> None:
    if config.target_size < 10:
        raise ValueError("target_size must be at least 10")
    if config.positive_size + config.hard_negative_size != config.target_size:
        raise ValueError("positive_size + hard_negative_size must equal target_size")
    if (
        config.target_size != 5_000
        or config.positive_size != 3_000
        or config.hard_negative_size != 2_000
    ):
        raise ValueError(
            "The positive-priority cohort freezes 3,000 Step 2 positives and "
            "2,000 S-to-OTHER hard negatives"
        )


def _prepare_output(paths: Step3Paths, *, rebuild: bool) -> None:
    managed = [value for key, value in asdict(paths).items() if key != "root"]
    existing = [Path(path) for path in managed if Path(path).exists()]
    if existing and not rebuild:
        raise FileExistsError(
            "Step 3 outputs already exist; use --rebuild to replace the frozen sample: "
            + ", ".join(map(str, existing))
        )
    if rebuild:
        for path in managed:
            Path(path).unlink(missing_ok=True)
        for suffix in ("-wal", "-shm", ".run.lock"):
            paths.database.with_name(paths.database.name + suffix).unlink(missing_ok=True)
        try:
            paths.train.parent.rmdir()
        except OSError:
            pass
    paths.root.mkdir(parents=True, exist_ok=True)


def _initialize_task_database(path: Path, rows: list[dict[str, Any]]) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=FULL;
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE tasks (
            sample_id TEXT PRIMARY KEY,
            dataset_id TEXT NOT NULL,
            application_year TEXT NOT NULL,
            patent_id TEXT NOT NULL UNIQUE,
            payload_json TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('pending','running','succeeded','failed')),
            attempts INTEGER NOT NULL DEFAULT 0,
            requested_model TEXT,
            actual_model TEXT,
            prompt_version TEXT,
            prompt_sha256 TEXT,
            schema_sha256 TEXT,
            response_id TEXT,
            annotation_json TEXT,
            raw_response TEXT,
            usage_json TEXT,
            error TEXT,
            elapsed_seconds REAL NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed_at TEXT
        );
        CREATE INDEX idx_step3_tasks_status ON tasks(status, sample_id);
        """
    )
    now = _now()
    connection.executemany(
        """
        INSERT INTO tasks (
          sample_id,dataset_id,application_year,patent_id,payload_json,status,created_at,updated_at
        ) VALUES (?,?,?,?,?,'pending',?,?)
        """,
        [
            (
                row["sample_id"],
                row["dataset_id"],
                row["application_year"],
                row["patent_id"],
                json.dumps(
                    {
                        "patent_id": row["patent_id"],
                        "title": row["title"],
                        "abstract": row["abstract"],
                        "claim": row["claim"],
                        "ipc": row["ipc"],
                        "main_ipc": row["main_ipc"],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                now,
                now,
            )
            for row in rows
        ],
    )
    connection.commit()
    connection.close()
    for suffix in ("-wal", "-shm"):
        path.with_name(path.name + suffix).unlink(missing_ok=True)


def _simulation_row(row: Mapping[str, Any]) -> dict[str, Any]:
    json_fields = {"scope_basis", "processing_activities", "industry_sectors", "evidence"}
    return {
        field: (
            json.dumps(row.get(field, []), ensure_ascii=False)
            if field in json_fields
            else row.get(field, "")
        )
        for field in SIMULATION_FIELDS
    }


def _manual_review_row(row: Mapping[str, Any]) -> dict[str, Any]:
    json_fields = {
        "step2_scope_basis",
        "step2_processing_activities",
        "step2_industry_sectors",
        "step2_evidence",
    }
    return {
        field: (
            json.dumps(row.get(field, []), ensure_ascii=False)
            if field in json_fields
            else "true"
            if field == "step2_needs_review" and bool(row.get(field))
            else "false"
            if field == "step2_needs_review"
            else ""
            if field in {"human_review_label", "human_reason"}
            else row.get(field, "")
        )
        for field in MANUAL_REVIEW_FIELDS
    }


def _result_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row.get(field, "") for field in RESULT_FIELDS}


def _parse_review_label(value: Any, *, field: str, row_number: int) -> str:
    normalized = str(value or "").strip().upper()
    if normalized in LABELS:
        return normalized
    raise ValueError(
        f"Row {row_number}: {field} must be DATA_SECURITY or OTHER, got {value!r}"
    )


def _parse_boolean(value: Any, *, field: str, row_number: int) -> bool:
    normalized = str(value or "").strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Row {row_number}: {field} must be exactly true or false, got {value!r}")


def _parse_confidence(value: Any, *, row_number: int) -> float:
    try:
        confidence = float(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"Row {row_number}: confidence must be a number") from exc
    if not 0 <= confidence <= 1:
        raise ValueError(f"Row {row_number}: confidence must be between 0 and 1")
    return confidence


def _parse_json_value(value: Any, *, field: str, row_number: int) -> Any:
    try:
        return json.loads(str(value or ""))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Row {row_number}: {field} must be valid JSON") from exc


def _parse_controlled_list(
    value: Any,
    *,
    field: str,
    allowed: set[str],
    maximum: int,
    row_number: int,
) -> list[str]:
    try:
        parsed = json.loads(str(value or ""))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Row {row_number}: {field} must be a JSON array") from exc
    if not isinstance(parsed, list) or not parsed or len(parsed) > maximum:
        raise ValueError(f"Row {row_number}: {field} must contain 1 to {maximum} values")
    if any(not isinstance(item, str) or item not in allowed for item in parsed):
        raise ValueError(f"Row {row_number}: {field} contains an invalid value: {parsed}")
    if len(parsed) != len(set(parsed)):
        raise ValueError(f"Row {row_number}: {field} contains duplicate values")
    if "other" in parsed and parsed != ["other"]:
        raise ValueError(f"Row {row_number}: {field} cannot mix 'other' with specific values")
    return parsed


def _validate_human_result_identity(
    database_path: Path,
    rows: list[dict[str, Any]],
    *,
    expected_count: int,
) -> None:
    if not database_path.is_file():
        raise FileNotFoundError(f"Missing frozen Step 3 task database: {database_path}")
    connection = sqlite3.connect(
        f"file:{database_path}?mode=ro&immutable=1",
        uri=True,
    )
    connection.row_factory = sqlite3.Row
    frozen_rows = []
    for task in connection.execute(
        "SELECT sample_id,dataset_id,application_year,patent_id,payload_json "
        "FROM tasks ORDER BY sample_id"
    ):
        payload = json.loads(task["payload_json"])
        frozen_rows.append(
            {
                "sample_id": str(task["sample_id"]),
                "dataset_id": str(task["dataset_id"]),
                "application_year": str(task["application_year"]),
                "patent_id": str(task["patent_id"]),
                **{
                    field: str(payload.get(field, "") or "")
                    for field in ("title", "abstract", "claim", "ipc", "main_ipc")
                },
            }
        )
    connection.close()
    if len(frozen_rows) != expected_count:
        raise ValueError(
            f"Frozen task database must contain {expected_count} rows, found {len(frozen_rows)}"
        )
    frozen_by_id = {row["sample_id"]: row for row in frozen_rows}
    if len(frozen_by_id) != len(frozen_rows):
        raise ValueError("Frozen task database contains duplicate sample_id values")

    result_ids = [str(row["sample_id"]) for row in rows]
    if len(set(result_ids)) != len(result_ids):
        raise ValueError("Human result.csv contains duplicate sample_id values")
    if set(result_ids) != set(frozen_by_id):
        missing = sorted(set(frozen_by_id) - set(result_ids))[:5]
        unexpected = sorted(set(result_ids) - set(frozen_by_id))[:5]
        raise ValueError(
            f"Human result.csv sample_id set differs: missing={missing}, unexpected={unexpected}"
        )
    patent_ids = [str(row["patent_id"]) for row in rows]
    if len(set(patent_ids)) != len(patent_ids):
        raise ValueError("Human result.csv contains duplicate patent_id values")

    for row in rows:
        frozen = frozen_by_id[str(row["sample_id"])]
        for field in FROZEN_RESULT_FIELDS[1:]:
            if str(row[field]) != str(frozen.get(field, "")):
                raise ValueError(
                    f"Human result.csv changed frozen field {field} for {row['sample_id']}"
                )
        if not any(str(row[field]).strip() for field in ("title", "abstract", "claim")):
            raise ValueError(f"Human result.csv has no patent text for {row['sample_id']}")


def _keep_text_groups_together(rows: list[dict[str, Any]], *, seed: str) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["split_group_id"]].append(row)
    singleton_ids = {group_id for group_id, values in groups.items() if len(values) == 1}
    for group_id in sorted(groups):
        members = groups[group_id]
        member_splits = Counter(row["data_split"] for row in members)
        if len(member_splits) == 1:
            continue
        if len({row["label"] for row in members}) != 1:
            raise ValueError(f"Exact-text duplicates have conflicting final labels: {group_id}")
        target = min(
            member_splits,
            key=lambda split: (
                -member_splits[split],
                _stable_score(seed, group_id, split),
            ),
        )
        for member in members:
            origin = member["data_split"]
            if origin == target:
                continue
            candidates = [
                row
                for row in rows
                if row["data_split"] == target
                and row["split_group_id"] in singleton_ids
                and row["application_year"] == member["application_year"]
                and row["label"] == member["label"]
            ]
            if not candidates:
                candidates = [
                    row
                    for row in rows
                    if row["data_split"] == target
                    and row["split_group_id"] in singleton_ids
                    and row["label"] == member["label"]
                ]
            if not candidates:
                candidates = [
                    row
                    for row in rows
                    if row["data_split"] == target and row["split_group_id"] in singleton_ids
                ]
            if not candidates:
                raise ValueError(f"Cannot keep exact-text group together: {group_id}")
            replacement = min(
                candidates,
                key=lambda row: _stable_score(seed, group_id, row["sample_id"]),
            )
            replacement["data_split"] = origin
            member["data_split"] = target
    for values in groups.values():
        if len({row["data_split"] for row in values}) != 1:
            raise AssertionError("An exact-text group crosses dataset splits")


def _text_group_id(row: Mapping[str, Any]) -> str:
    text = "\u241f".join(
        " ".join(str(row.get(field, "") or "").split()) for field in ("title", "abstract", "claim")
    )
    if not text.strip("\u241f"):
        return f"patent:{row['sample_id']}"
    return "text:" + hashlib.sha256(text.encode()).hexdigest()


@contextmanager
def _exclusive_database_update(database: Path) -> Iterator[None]:
    lock_path = database.with_name(database.name + ".run.lock")
    owns_lock = False
    try:
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"A Step 3 runner is active for {database}") from None
            owns_lock = True
            yield
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    finally:
        if owns_lock:
            lock_path.unlink(missing_ok=True)


def _csv_record_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8-sig", newline="") as file:
        return sum(1 for _ in csv.DictReader(file))


def _write_csv(path: Path, fields: tuple[str, ...], rows: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _stable_score(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _sample_id(seed: str, dataset_id: str, patent_id: str) -> str:
    digest = hashlib.blake2b(
        f"{SAMPLE_ID_VERSION}|{seed}|{dataset_id}|{patent_id}".encode(),
        digest_size=12,
    ).hexdigest()
    return f"step3-{digest}"


def _now() -> str:
    return datetime.now(UTC).isoformat()
