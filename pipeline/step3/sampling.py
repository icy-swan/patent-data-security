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
from typing import Any

from pipeline.common.io import atomic_json_write
from pipeline.step2.schema import PatentClassification

SAMPLING_VERSION = "step3-balanced-year-label-v2.2.0"
LABELS = ("DATA_SECURITY", "OTHER")
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

DATASET_FIELDS = (
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
    "data_split",
)


@dataclass(frozen=True)
class SamplingConfig:
    target_size: int = 4_000
    positive_share: float = 0.5
    seed: str = "step3-balanced-year-label-v2.2.0"

    @property
    def label_targets(self) -> dict[str, int]:
        positive = round(self.target_size * self.positive_share)
        return {"DATA_SECURITY": positive, "OTHER": self.target_size - positive}


@dataclass(frozen=True)
class Step3Paths:
    root: Path
    database: Path
    audit: Path
    blinded: Path
    manifest: Path
    progress: Path
    annotations: Path
    dataset: Path
    train: Path
    validation: Path
    test: Path
    split_report: Path


def step3_paths(output_dir: str | Path) -> Step3Paths:
    root = Path(output_dir).resolve()
    return Step3Paths(
        root=root,
        database=root / "step3_tasks.sqlite3",
        audit=root / "step3_sample_audit.csv",
        blinded=root / "step3_annotation_input_blinded.csv",
        manifest=root / "step3_sample_manifest.json",
        progress=root / "step3_simulation_progress.json",
        annotations=root / "step3_simulated_annotations.csv",
        dataset=root / "step3_dataset_provisional.csv",
        train=root / "step3_train_provisional.csv",
        validation=root / "step3_validation_provisional.csv",
        test=root / "step3_test_provisional.csv",
        split_report=root / "step3_split_report.json",
    )


def discover_step2_databases(step2_dir: str | Path) -> list[Path]:
    return sorted(Path(step2_dir).resolve().glob("step2_tasks_*.sqlite3"))


def prepare_sample(
    databases: Iterable[str | Path],
    output_dir: str | Path,
    *,
    config: SamplingConfig | None = None,
    rebuild: bool = False,
) -> tuple[Step3Paths, dict[str, Any]]:
    """Freeze the balanced year-by-Step-2-label sample and blinded tasks."""

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
    records, source_summary, logical_digest = _load_population(database_paths)
    years = sorted({record["application_year"] for record in records})
    capacities = Counter((record["application_year"], record["step2_label"]) for record in records)

    quotas: dict[tuple[str, str], int] = {}
    for label, target in config.label_targets.items():
        label_capacities = {year: capacities[(year, label)] for year in years}
        quotas.update(
            {
                (year, label): value
                for year, value in _balanced_capacity_allocation(
                    label_capacities,
                    target,
                    seed=f"{config.seed}|{label}",
                ).items()
            }
        )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["application_year"], record["step2_label"])].append(record)

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
        year, label = stratum
        stratum_rows.append(
            {
                "application_year": year,
                "step2_label": label,
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
                    "sampling_stratum": f"year={year}|step2_label={label}",
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
        "label_targets": config.label_targets,
        "year_allocation": "equal_within_step2_label_with_capacity_redistribution",
        "years": years,
        "seed": config.seed,
        "population": len(records),
        "population_by_step2_label": dict(
            sorted(Counter(r["step2_label"] for r in records).items())
        ),
        "sample_by_step2_label": dict(sorted(Counter(r["step2_label"] for r in selected).items())),
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

    output.sort(key=lambda row: row["sample_id"])
    counts = Counter(row["data_split"] for row in output)
    expected = {"train": train_target, "validation": validation_target, "test": test_target}
    if dict(counts) != expected:
        raise AssertionError(f"Split totals differ: actual={dict(counts)}, expected={expected}")
    return output, {"ratios": SPLIT_RATIOS, "counts": expected, "strata": report_strata}


def write_provisional_dataset(
    paths: Step3Paths,
    rows: list[dict[str, Any]],
    *,
    split_seed: str,
    annotation_model: str,
    annotation_prompt_version: str,
) -> dict[str, Any]:
    """Write model-simulated annotations and clearly non-gold split files."""

    normalized: list[dict[str, Any]] = []
    for row in rows:
        annotation = PatentClassification.model_validate(row["annotation"])
        item = {
            **{field: row.get(field, "") for field in BLINDED_FIELDS[:9]},
            **annotation.model_dump(),
            "annotation_source": "openai_model_simulation",
            "annotation_model": annotation_model,
            "annotation_prompt_version": annotation_prompt_version,
            "gold_status": "provisional_not_human_gold",
            "eligible_for_final_evaluation": False,
        }
        normalized.append(item)
    assigned, report = assign_exact_splits(normalized, seed=split_seed)

    _write_csv(paths.annotations, DATASET_FIELDS, (_dataset_row(row) for row in assigned))
    _write_csv(paths.dataset, DATASET_FIELDS, (_dataset_row(row) for row in assigned))
    by_split = {split: [row for row in assigned if row["data_split"] == split] for split in SPLITS}
    _write_csv(paths.train, DATASET_FIELDS, (_dataset_row(row) for row in by_split["train"]))
    _write_csv(
        paths.validation,
        DATASET_FIELDS,
        (_dataset_row(row) for row in by_split["validation"]),
    )
    _write_csv(paths.test, DATASET_FIELDS, (_dataset_row(row) for row in by_split["test"]))
    report.update(
        {
            "schema_version": "2.2.0",
            "split_seed": split_seed,
            "annotation_source": "openai_model_simulation",
            "annotation_model": annotation_model,
            "gold_status": "provisional_not_human_gold",
            "eligible_for_final_evaluation": False,
            "label_counts": dict(sorted(Counter(row["label"] for row in assigned).items())),
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
    if not 0 < config.positive_share < 1:
        raise ValueError("positive_share must be strictly between 0 and 1")
    if config.target_size != 4_000 or config.positive_share != 0.5:
        raise ValueError("Step 3 v2.2.0 freezes target_size=4000 and positive_share=0.5")


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


def _dataset_row(row: Mapping[str, Any]) -> dict[str, Any]:
    json_fields = {"scope_basis", "processing_activities", "industry_sectors", "evidence"}
    return {
        field: (
            json.dumps(row.get(field, []), ensure_ascii=False)
            if field in json_fields
            else row.get(field, "")
        )
        for field in DATASET_FIELDS
    }


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
