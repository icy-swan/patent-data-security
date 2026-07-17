"""Reproducible 4,000-record sampling and exact 8:1:1 splitting."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, get_args

from pipeline.common.io import atomic_json_write, sha256_file
from pipeline.step2.schema import IndustrySector, PatentClassification, ScopeBasis

SAMPLING_VERSION = "step3-positive-priority-hard-negative-v2.2.0"
LABELS = ("DATA_SECURITY", "OTHER")
SAMPLING_GROUPS = ("positive", "hard_negative")
SPLITS = ("train", "validation", "test")
SPLIT_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}

AUDIT_FIELDS = (
    "sample_id",
    "sampling_version",
    "sample_seed",
    "dataset_id",
    "application_year",
    "patent_id",
    "source_row_number",
    "sampling_stratum",
    "sampling_group",
    "stratum_population",
    "stratum_sample_size",
    "step3_inclusion_probability",
    "step3_sample_weight",
    "step2_route",
    "step2_selection_group",
    "step2_selection_probability",
    "step2_sample_weight",
    "combined_inclusion_probability",
    "combined_sample_weight",
    "step2_label",
    "step2_confidence",
    "step2_review_flag",
    "step2_requested_model",
    "step2_actual_model",
    "step2_prompt_version",
    "step2_response_id",
    "title",
    "abstract",
    "claim",
    "ipc",
    "main_ipc",
)

BLINDED_FIELDS = (
    "sample_id",
    "dataset_id",
    "application_year",
    "patent_id",
    "title",
    "abstract",
    "claim",
    "ipc",
    "main_ipc",
    "annotator_id",
    "annotation_status",
    "annotation_json",
    "submitted_at",
)

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
    "review_flag",
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

RESULT_FIELDS = (
    *FROZEN_RESULT_FIELDS,
    "human_evaluation",
    "scope_basis",
    "industry_sectors",
)


@dataclass(frozen=True)
class SamplingConfig:
    target_size: int = 4_000
    positive_size: int = 3_000
    hard_negative_size: int = 1_000
    seed: str = "step3-positive-priority-hard-negative-v2.2.0"

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
    audit: Path
    blinded: Path
    manifest: Path
    progress: Path
    simulation: Path
    results: Path
    train: Path
    validation: Path
    test: Path
    split_report: Path


def step3_paths(output_dir: str | Path) -> Step3Paths:
    root = Path(output_dir).resolve()
    return Step3Paths(
        root=root,
        database=root / "state" / "tasks.sqlite3",
        audit=root / "sample" / "audit.csv",
        blinded=root / "sample" / "annotation_input.csv",
        manifest=root / "sample" / "manifest.json",
        progress=root / "state" / "progress.json",
        simulation=root / "dataset" / "simulation.csv",
        results=root / "dataset" / "results.csv",
        train=root / "dataset" / "splits" / "train.csv",
        validation=root / "dataset" / "splits" / "validation.csv",
        test=root / "dataset" / "splits" / "test.csv",
        split_report=root / "dataset" / "split_report.json",
    )


def discover_step2_databases(step2_dir: str | Path) -> list[Path]:
    root = Path(step2_dir).resolve()
    canonical = root.glob("*/tasks.sqlite3")
    legacy = root.glob("step2_tasks_*.sqlite3")
    return sorted(
        path
        for path in {*canonical, *legacy}
        if ".before_" not in path.name
    )


def prepare_sample(
    databases: Iterable[str | Path],
    output_dir: str | Path,
    *,
    config: SamplingConfig | None = None,
    rebuild: bool = False,
) -> tuple[Step3Paths, dict[str, Any]]:
    """Freeze the positive-priority year-balanced sample and blinded tasks."""

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
    records = [record for record in all_records if record["sampling_group"]]
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

    _write_csv(paths.audit, AUDIT_FIELDS, (_audit_row(row) for row in selected))
    _write_csv(paths.blinded, BLINDED_FIELDS, (_blinded_row(row) for row in selected))
    _initialize_task_database(paths.database, selected)
    manifest = {
        "schema_version": "2.2.0",
        "sampling_version": SAMPLING_VERSION,
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
        "outputs": {key: str(value) for key, value in asdict(paths).items() if key != "root"},
        "provisional_annotation_policy": {
            "simulation_is_human_gold": False,
            "eligible_for_final_evaluation": False,
        },
        "prepared_at": _now(),
    }
    atomic_json_write(paths.manifest, manifest)
    return paths, manifest


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
            **{field: row.get(field, "") for field in BLINDED_FIELDS[:9]},
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
        "schema_version": "2.2.0",
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
    split_seed: str = "step3-human-split-v2.2.0",
    expected_count: int = 4_000,
) -> dict[str, Any]:
    """Validate and sanitize human results.csv, then create minimal 8:1:1 splits."""

    if not paths.results.is_file():
        raise FileNotFoundError(f"Missing human annotation file: {paths.results}")
    with paths.results.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        input_fields = tuple(reader.fieldnames or ())
        missing = sorted(set(RESULT_FIELDS) - set(input_fields))
        if missing:
            raise ValueError(f"Human results.csv is missing fields: {missing}")
        raw_rows = [dict(row) for row in reader]
    if len(raw_rows) != expected_count:
        raise ValueError(
            f"Human results.csv must contain {expected_count} rows, found {len(raw_rows)}"
        )

    normalized: list[dict[str, Any]] = []
    for row_number, row in enumerate(raw_rows, start=2):
        evaluation = _parse_human_evaluation(row["human_evaluation"], row_number=row_number)
        scope_basis = _parse_controlled_list(
            row["scope_basis"],
            field="scope_basis",
            allowed=set(get_args(ScopeBasis)),
            maximum=3,
            row_number=row_number,
        )
        industry_sectors = _parse_controlled_list(
            row["industry_sectors"],
            field="industry_sectors",
            allowed=set(get_args(IndustrySector)),
            maximum=9,
            row_number=row_number,
        )
        if not evaluation and scope_basis != ["other"]:
            raise ValueError(f"Row {row_number}: false requires scope_basis=['other']")
        if not evaluation and industry_sectors != ["other"]:
            raise ValueError(f"Row {row_number}: false requires industry_sectors=['other']")
        if evaluation and "other" in scope_basis:
            raise ValueError(f"Row {row_number}: true cannot use scope_basis='other'")
        item = {field: str(row.get(field, "") or "") for field in FROZEN_RESULT_FIELDS}
        item.update(
            {
                "human_evaluation": evaluation,
                "scope_basis": scope_basis,
                "industry_sectors": industry_sectors,
                "label": "DATA_SECURITY" if evaluation else "OTHER",
            }
        )
        normalized.append(item)

    _validate_human_result_identity(paths.blinded, normalized, expected_count=expected_count)
    assigned, report = assign_exact_splits(normalized, seed=split_seed)
    by_split = {
        split: [row for row in assigned if row["data_split"] == split] for split in SPLITS
    }

    for path in (paths.results, paths.train, paths.validation, paths.test, paths.split_report):
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
            "schema_version": "2.2.0",
            "source": "human_results_csv",
            "source_path": str(paths.results),
            "source_sha256": sha256_file(paths.results),
            "split_seed": split_seed,
            "result_fields": list(RESULT_FIELDS),
            "removed_input_fields": sorted(set(input_fields) - set(RESULT_FIELDS)),
            "human_evaluation_mapping": {"true": "DATA_SECURITY", "false": "OTHER"},
            "human_evaluation_counts": dict(
                sorted(
                    Counter(
                        "true" if row["human_evaluation"] else "false" for row in assigned
                    ).items()
                )
            ),
            "written_at": _now(),
        }
    )
    atomic_json_write(paths.split_report, report)
    return report


def _load_population(
    databases: list[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    records: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    seen_patents: set[str] = set()
    for database in databases:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
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
                "step2_selection_group": row["selection_group"],
                "step2_selection_probability": float(row["selection_probability"]),
                "step2_sample_weight": float(row["sample_weight"]),
                "step2_label": result.label,
                "sampling_group": _sampling_group(row["route"], result.label),
                "step2_confidence": result.confidence,
                "step2_review_flag": result.review_flag,
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
    return ""


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
        config.target_size != 4_000
        or config.positive_size != 3_000
        or config.hard_negative_size != 1_000
    ):
        raise ValueError(
            "Step 3 v2.2.0 freezes 3,000 positives and 1,000 S-to-OTHER hard negatives"
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
        paths.database.with_name(paths.database.name + ".run.lock").unlink(missing_ok=True)
    for path in managed:
        Path(path).parent.mkdir(parents=True, exist_ok=True)


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


def _audit_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row.get(field, "") for field in AUDIT_FIELDS}


def _blinded_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **{field: row.get(field, "") for field in BLINDED_FIELDS[:9]},
        "annotator_id": "",
        "annotation_status": "pending",
        "annotation_json": "",
        "submitted_at": "",
    }


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


def _result_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: (
            "true"
            if field == "human_evaluation" and bool(row.get(field))
            else "false"
            if field == "human_evaluation"
            else json.dumps(row.get(field, []), ensure_ascii=False)
            if field in {"scope_basis", "industry_sectors"}
            else row.get(field, "")
        )
        for field in RESULT_FIELDS
    }


def _parse_human_evaluation(value: Any, *, row_number: int) -> bool:
    normalized = str(value or "").strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(
        f"Row {row_number}: human_evaluation must be exactly true or false, got {value!r}"
    )


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
    blinded_path: Path,
    rows: list[dict[str, Any]],
    *,
    expected_count: int,
) -> None:
    if not blinded_path.is_file():
        raise FileNotFoundError(f"Missing frozen Step 3 annotation input: {blinded_path}")
    with blinded_path.open(encoding="utf-8-sig", newline="") as file:
        frozen_rows = list(csv.DictReader(file))
    if len(frozen_rows) != expected_count:
        raise ValueError(
            f"Frozen annotation input must contain {expected_count} rows, found {len(frozen_rows)}"
        )
    frozen_by_id = {row["sample_id"]: row for row in frozen_rows}
    if len(frozen_by_id) != len(frozen_rows):
        raise ValueError("Frozen annotation input contains duplicate sample_id values")

    result_ids = [str(row["sample_id"]) for row in rows]
    if len(set(result_ids)) != len(result_ids):
        raise ValueError("Human results.csv contains duplicate sample_id values")
    if set(result_ids) != set(frozen_by_id):
        missing = sorted(set(frozen_by_id) - set(result_ids))[:5]
        unexpected = sorted(set(result_ids) - set(frozen_by_id))[:5]
        raise ValueError(
            f"Human results.csv sample_id set differs: missing={missing}, unexpected={unexpected}"
        )
    patent_ids = [str(row["patent_id"]) for row in rows]
    if len(set(patent_ids)) != len(patent_ids):
        raise ValueError("Human results.csv contains duplicate patent_id values")

    for row in rows:
        frozen = frozen_by_id[str(row["sample_id"])]
        for field in FROZEN_RESULT_FIELDS[1:]:
            if str(row[field]) != str(frozen.get(field, "")):
                raise ValueError(
                    f"Human results.csv changed frozen field {field} for {row['sample_id']}"
                )
        if not any(str(row[field]).strip() for field in ("title", "abstract", "claim")):
            raise ValueError(f"Human results.csv has no patent text for {row['sample_id']}")


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
        f"{SAMPLING_VERSION}|{seed}|{dataset_id}|{patent_id}".encode(),
        digest_size=12,
    ).hexdigest()
    return f"step3-{digest}"


def _now() -> str:
    return datetime.now(UTC).isoformat()
