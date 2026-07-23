"""Prepare Step 2 tasks while keeping model evidence separate from routing metadata."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pipeline.common.datasets import dataset_id
from pipeline.common.io import atomic_json_write, sha256_file
from pipeline.common.records import PatentRecord, iter_patent_records
from pipeline.step1.matcher import normalize_text
from pipeline.step2.prompt import PromptBundle, build_dynamic_payload, load_prompt_bundle

DEFAULT_POOL_ID = "pool-50000"
DEFAULT_POOL_SIZE = 50_000
DEFAULT_POOL_SEED = "step2-global-pool-v1"


@dataclass(frozen=True)
class Step2TaskPaths:
    database: Path
    manifest: Path
    requests: Path
    results: Path
    progress: Path


def task_paths(output_dir: str | Path, dataset: str) -> Step2TaskPaths:
    root = Path(output_dir).resolve() / dataset
    return Step2TaskPaths(
        database=root / "tasks.sqlite3",
        manifest=root / "manifest.json",
        requests=root / "requests.jsonl",
        results=root / "result.csv",
        progress=root / "progress.json",
    )


def prepare_tasks(
    raw_path: str | Path,
    step1_results_path: str | Path,
    output_dir: str | Path,
    *,
    prompt_bundle: PromptBundle | None = None,
    encoding: str = "utf-8-sig",
    rebuild: bool = False,
) -> tuple[Step2TaskPaths, dict[str, Any]]:
    """Materialize Step 1's frozen task pool and bind each response to a local patent ID."""

    raw = Path(raw_path).resolve()
    step1 = Path(step1_results_path).resolve()
    dataset = dataset_id(raw)
    paths = task_paths(output_dir, dataset)
    bundle = prompt_bundle or load_prompt_bundle()
    selections = _read_selected_step1_rows(step1)
    normalized = {
        patent_id: {
            **selection,
            "dataset_id": dataset,
            "upstream_selection_probability": selection["selection_probability"],
            "pool_selection_probability": "1",
            "pool_sample_seed": "",
            "pool_sample_score": "",
        }
        for patent_id, selection in selections.items()
    }
    manifest_context = {
        "step": "step2_task_preparation",
        "dataset_id": dataset,
        "raw_input_path": str(raw),
        "raw_input_size_bytes": raw.stat().st_size,
        "raw_input_sha256": sha256_file(raw),
        "step1_results_path": str(step1),
        "step1_results_sha256": sha256_file(step1),
        "prompt_version": bundle.prompt_version,
        "law_resource_version": bundle.law_resource_version,
        "resource_sha256": bundle.resource_hashes,
    }
    return _materialize_tasks(
        sources=((dataset, raw),),
        selections=normalized,
        paths=paths,
        task_namespace=dataset,
        manifest_context=manifest_context,
        prompt_bundle=bundle,
        encoding=encoding,
        rebuild=rebuild,
    )


def prepare_task_pool(
    raw_paths: Sequence[str | Path],
    step1_results_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    pool_size: int = DEFAULT_POOL_SIZE,
    pool_seed: str = DEFAULT_POOL_SEED,
    pool_id: str = DEFAULT_POOL_ID,
    prompt_bundle: PromptBundle | None = None,
    encoding: str = "utf-8-sig",
    rebuild: bool = False,
) -> tuple[Step2TaskPaths, dict[str, Any]]:
    """Select and materialize one fixed-size, cross-year Step 2 request pool."""

    raw_by_dataset = _paths_by_dataset(raw_paths, source="raw input")
    step1_by_dataset = _step1_paths_by_dataset(step1_results_paths)
    if set(raw_by_dataset) != set(step1_by_dataset):
        raise ValueError(
            "Raw and Step 1 dataset IDs differ: "
            f"raw_only={sorted(set(raw_by_dataset) - set(step1_by_dataset))}, "
            f"step1_only={sorted(set(step1_by_dataset) - set(raw_by_dataset))}"
        )
    candidates, frame = _read_pool_candidates(step1_by_dataset)
    selections, sampling = _fixed_size_pool_sample(
        candidates,
        pool_size=pool_size,
        pool_seed=pool_seed,
    )
    bundle = prompt_bundle or load_prompt_bundle()
    source_manifests = [
        _source_manifest(
            dataset=dataset,
            raw=raw_by_dataset[dataset],
            step1=step1_by_dataset[dataset],
            selected_count=sampling["selected_by_dataset"].get(dataset, 0),
        )
        for dataset in sorted(raw_by_dataset)
    ]
    manifest_context = {
        "step": "step2_fixed_size_pool_preparation",
        "dataset_id": pool_id,
        "candidate_frame": frame,
        "pool_sampling": sampling,
        "sources": source_manifests,
        "prompt_version": bundle.prompt_version,
        "law_resource_version": bundle.law_resource_version,
        "resource_sha256": bundle.resource_hashes,
    }
    sources = tuple(
        (dataset, raw_by_dataset[dataset]) for dataset in sorted(raw_by_dataset)
    )
    return _materialize_tasks(
        sources=sources,
        selections=selections,
        paths=task_paths(output_dir, pool_id),
        task_namespace=pool_id,
        manifest_context=manifest_context,
        prompt_bundle=bundle,
        encoding=encoding,
        rebuild=rebuild,
    )


def _read_selected_step1_rows(path: Path) -> dict[str, dict[str, str]]:
    selections: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if not _truthy(row.get("selected_for_step2", "")):
                continue
            patent_id = row.get("patent_id", "").strip()
            if not patent_id:
                raise ValueError("Selected Step 1 row has no patent_id")
            if patent_id in selections:
                raise ValueError(f"Duplicate selected patent_id in Step 1 results: {patent_id}")
            selections[patent_id] = {
                "source_row_number": row["source_row_number"],
                "route": row["route"],
                "selection_group": row["selection_group"],
                "selection_probability": row["selection_probability"],
                "sample_weight": row["sample_weight"],
            }
    if not selections:
        raise ValueError("Step 1 results contain no selected Step 2 tasks")
    return selections


def _paths_by_dataset(
    paths: Sequence[str | Path],
    *,
    source: str,
) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for value in paths:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing {source}: {path}")
        token = dataset_id(path)
        if token in resolved:
            raise ValueError(
                f"Duplicate {source} dataset ID {token}: {resolved[token]} and {path}"
            )
        resolved[token] = path
    if not resolved:
        raise ValueError(f"No {source} files were supplied")
    return resolved


def _step1_paths_by_dataset(
    paths: Sequence[str | Path],
) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for value in paths:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing Step 1 result: {path}")
        token = path.parent.name
        if token in resolved:
            raise ValueError(
                f"Duplicate Step 1 dataset ID {token}: {resolved[token]} and {path}"
            )
        resolved[token] = path
    if not resolved:
        raise ValueError("No Step 1 result files were supplied")
    return resolved


def _read_pool_candidates(
    step1_by_dataset: Mapping[str, Path],
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    candidates: dict[str, dict[str, str]] = {}
    rows_by_dataset: Counter[str] = Counter()
    unique_by_dataset: Counter[str] = Counter()
    rows_by_route: Counter[str] = Counter()
    rows_by_group: Counter[str] = Counter()
    duplicate_rows: list[dict[str, str]] = []

    for dataset in sorted(step1_by_dataset):
        selections = _read_selected_step1_rows(step1_by_dataset[dataset])
        for patent_id, selection in selections.items():
            rows_by_dataset[dataset] += 1
            rows_by_route[selection["route"]] += 1
            rows_by_group[selection["selection_group"]] += 1
            candidate = {**selection, "dataset_id": dataset}
            canonical = candidates.setdefault(patent_id, candidate)
            if canonical is candidate:
                unique_by_dataset[dataset] += 1
                continue
            duplicate_rows.append(
                {
                    "patent_id": patent_id,
                    "kept_dataset_id": canonical["dataset_id"],
                    "discarded_dataset_id": dataset,
                }
            )

    if not candidates:
        raise ValueError("The cross-year Step 1 candidate frame is empty")
    frame = {
        "unit": "unique_patent",
        "candidate_rows": sum(rows_by_dataset.values()),
        "unique_patents": len(candidates),
        "duplicate_cross_dataset_rows": len(duplicate_rows),
        "candidate_rows_by_dataset": dict(sorted(rows_by_dataset.items())),
        "canonical_unique_patents_by_dataset": dict(sorted(unique_by_dataset.items())),
        "candidate_rows_by_route": dict(sorted(rows_by_route.items())),
        "candidate_rows_by_selection_group": dict(sorted(rows_by_group.items())),
        "cross_dataset_duplicates": duplicate_rows,
        "deduplication_rule": (
            "Keep the lexicographically earliest dataset_id; within a dataset Step 1 "
            "already retains its preferred canonical association."
        ),
    }
    return candidates, frame


def _fixed_size_pool_sample(
    candidates: Mapping[str, Mapping[str, str]],
    *,
    pool_size: int,
    pool_seed: str,
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    if not pool_seed:
        raise ValueError("pool_seed must not be empty")
    if not 1 <= pool_size <= len(candidates):
        raise ValueError(
            f"pool_size must be in [1, {len(candidates)}], got {pool_size}"
        )

    ranked = sorted(
        (
            hashlib.sha256(f"{pool_seed}|{patent_id}".encode()).hexdigest(),
            patent_id,
        )
        for patent_id in candidates
    )
    probability = pool_size / len(candidates)
    selected: dict[str, dict[str, str]] = {}
    by_dataset: Counter[str] = Counter()
    by_route: Counter[str] = Counter()
    by_group: Counter[str] = Counter()
    for score, patent_id in ranked[:pool_size]:
        candidate = dict(candidates[patent_id])
        upstream_probability = float(candidate["selection_probability"])
        combined_probability = upstream_probability * probability
        candidate.update(
            {
                "upstream_selection_probability": f"{upstream_probability:.17g}",
                "pool_selection_probability": f"{probability:.17g}",
                "selection_probability": f"{combined_probability:.17g}",
                "sample_weight": f"{1 / combined_probability:.17g}",
                "pool_sample_seed": pool_seed,
                "pool_sample_score": score,
            }
        )
        selected[patent_id] = candidate
        by_dataset[candidate["dataset_id"]] += 1
        by_route[candidate["route"]] += 1
        by_group[candidate["selection_group"]] += 1

    sampling = {
        "method": "fixed_size_sha256_order_without_replacement",
        "unit": "unique_patent",
        "hash_input": "pool_seed + '|' + patent_id",
        "pool_seed": pool_seed,
        "candidate_unique_patents": len(candidates),
        "target_size": pool_size,
        "pool_selection_probability": probability,
        "combined_probability_formula": (
            "step1_selection_probability * pool_selection_probability"
        ),
        "sample_weight_formula": "1 / combined_selection_probability",
        "selected_by_dataset": dict(sorted(by_dataset.items())),
        "selected_by_route": dict(sorted(by_route.items())),
        "selected_by_selection_group": dict(sorted(by_group.items())),
    }
    return selected, sampling


def _source_manifest(
    *,
    dataset: str,
    raw: Path,
    step1: Path,
    selected_count: int,
) -> dict[str, Any]:
    step1_manifest_path = step1.with_name("manifest.json")
    if not step1_manifest_path.is_file():
        raise FileNotFoundError(f"Missing Step 1 manifest: {step1_manifest_path}")
    step1_manifest = json.loads(step1_manifest_path.read_text(encoding="utf-8"))
    if step1_manifest.get("dataset_id") != dataset:
        raise ValueError(
            f"Step 1 manifest dataset mismatch for {dataset}: "
            f"{step1_manifest.get('dataset_id')!r}"
        )
    expected_size = int(step1_manifest.get("input_size_bytes", -1))
    actual_size = raw.stat().st_size
    if expected_size != actual_size:
        raise ValueError(
            f"Raw size differs from the Step 1 manifest for {dataset}: "
            f"expected {expected_size}, got {actual_size}"
        )
    candidate_count = sum(
        int(value)
        for value in step1_manifest["stats"]["selected_for_step2"].values()
    )
    return {
        "dataset_id": dataset,
        "raw_input_path": str(raw),
        "raw_input_size_bytes": actual_size,
        "raw_input_sha256": sha256_file(raw),
        "step1_results_path": str(step1),
        "step1_results_sha256": sha256_file(step1),
        "step1_manifest_path": str(step1_manifest_path),
        "step1_manifest_sha256": sha256_file(step1_manifest_path),
        "step1_candidate_rows": candidate_count,
        "selected_for_pool": selected_count,
    }


def _materialize_tasks(
    *,
    sources: Sequence[tuple[str, Path]],
    selections: Mapping[str, Mapping[str, str]],
    paths: Step2TaskPaths,
    task_namespace: str,
    manifest_context: Mapping[str, Any],
    prompt_bundle: PromptBundle,
    encoding: str,
    rebuild: bool,
) -> tuple[Step2TaskPaths, dict[str, Any]]:
    _prepare_output(paths, rebuild=rebuild)
    selected_by_dataset: dict[str, dict[int, tuple[str, Mapping[str, str]]]] = (
        defaultdict(dict)
    )
    for patent_id, selection in selections.items():
        dataset = selection["dataset_id"]
        source_row_number = int(selection["source_row_number"])
        if source_row_number in selected_by_dataset[dataset]:
            other = selected_by_dataset[dataset][source_row_number][0]
            raise ValueError(
                f"Two selected patents use {dataset} source row {source_row_number}: "
                f"{other} and {patent_id}"
            )
        selected_by_dataset[dataset][source_row_number] = (patent_id, selection)

    source_ids = {dataset for dataset, _ in sources}
    missing_sources = sorted(set(selected_by_dataset) - source_ids)
    if missing_sources:
        raise ValueError(f"Selections have no raw source: {missing_sources}")

    connection = _connect(paths.database)
    inserted = 0
    mismatched: list[dict[str, Any]] = []
    created_at = _now()
    try:
        _initialize_database(connection)
        for dataset, raw in sources:
            targets = selected_by_dataset.get(dataset, {})
            if not targets:
                continue
            max_row = max(targets)
            found: set[int] = set()
            batch: list[tuple[Any, ...]] = []
            for record in iter_patent_records(raw, encoding=encoding):
                if record.row_number > max_row:
                    break
                target = targets.get(record.row_number)
                if target is None:
                    continue
                expected_patent_id, selection = target
                actual_patent_id = _record_patent_id(record)
                if not actual_patent_id:
                    actual_patent_id = _synthetic_patent_id(record)
                if actual_patent_id != expected_patent_id:
                    mismatched.append(
                        {
                            "dataset_id": dataset,
                            "source_row_number": record.row_number,
                            "expected_patent_id": expected_patent_id,
                            "actual_patent_id": actual_patent_id,
                        }
                    )
                    continue
                payload = build_dynamic_payload(
                    {
                        "patent_id": expected_patent_id,
                        **record.values,
                    }
                )
                batch.append(
                    (
                        _task_id(task_namespace, expected_patent_id),
                        dataset,
                        expected_patent_id,
                        record.row_number,
                        selection["route"],
                        selection["selection_group"],
                        float(selection["upstream_selection_probability"]),
                        float(selection["pool_selection_probability"]),
                        float(selection["selection_probability"]),
                        float(selection["sample_weight"]),
                        selection["pool_sample_seed"],
                        selection["pool_sample_score"],
                        json.dumps(
                            payload,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        created_at,
                        created_at,
                    )
                )
                found.add(record.row_number)
                if len(batch) >= 1_000:
                    _insert_task_batch(connection, batch)
                    inserted += len(batch)
                    batch.clear()
            if batch:
                _insert_task_batch(connection, batch)
                inserted += len(batch)
            missing_rows = sorted(set(targets) - found)
            if missing_rows:
                preview = ", ".join(map(str, missing_rows[:10]))
                raise ValueError(
                    f"{dataset} is missing {len(missing_rows)} selected source rows "
                    f"(first: {preview})"
                )
        if mismatched:
            first = mismatched[0]
            raise ValueError(
                f"{len(mismatched)} selected source rows have a different patent ID; "
                f"first mismatch: {first}"
            )
        if inserted != len(selections):
            raise ValueError(
                f"Materialized {inserted} tasks, expected {len(selections)}"
            )

        _write_requests(paths.requests, connection)
        counts = _task_counts(connection)
        manifest = {
            "schema_version": "2.2.0",
            **manifest_context,
            "task_counts": counts,
            "statistics_binding": {
                "selected_patent_ids": len(selections),
                "materialized_tasks": inserted,
                "duplicate_task_patent_ids": 0,
                "missing_source_rows": 0,
                "mismatched_source_patent_ids": 0,
            },
            "outputs": {
                "database": str(paths.database),
                "manifest": str(paths.manifest),
                "requests": str(paths.requests),
                "results": str(paths.results),
                "progress": str(paths.progress),
            },
            "requests_sha256": sha256_file(paths.requests),
            "llm_requests_executed": 0,
            "prepared_at": _now(),
        }
        connection.execute(
            "INSERT INTO meta(key, value) VALUES ('task_manifest', ?)",
            (json.dumps(manifest, ensure_ascii=False, separators=(",", ":")),),
        )
        connection.commit()
        atomic_json_write(paths.manifest, manifest)
        return paths, manifest
    except BaseException:
        connection.close()
        _remove_outputs(paths)
        raise
    finally:
        if connection:
            connection.close()


def _prepare_output(paths: Step2TaskPaths, *, rebuild: bool) -> None:
    paths.database.parent.mkdir(parents=True, exist_ok=True)
    final_paths = (
        paths.database,
        paths.manifest,
        paths.requests,
        paths.results,
        paths.progress,
    )
    if not rebuild and any(path.exists() for path in final_paths):
        raise FileExistsError("Step 2 outputs already exist; pass --rebuild to replace them")
    if rebuild:
        _remove_outputs(paths)


def _remove_outputs(paths: Step2TaskPaths) -> None:
    for path in (
        paths.database,
        paths.manifest,
        paths.requests,
        paths.results,
        paths.progress,
        paths.database.with_name(paths.database.name + "-wal"),
        paths.database.with_name(paths.database.name + "-shm"),
        paths.database.with_name(paths.database.name + ".run.lock"),
        paths.manifest.with_suffix(paths.manifest.suffix + ".partial"),
        paths.requests.with_suffix(paths.requests.suffix + ".partial"),
    ):
        path.unlink(missing_ok=True)


def _insert_task_batch(
    connection: sqlite3.Connection,
    values: Sequence[tuple[Any, ...]],
) -> None:
    connection.executemany(
        """
        INSERT INTO tasks (
          task_id, dataset_id, patent_id, source_row_number, route, selection_group,
          upstream_selection_probability, pool_selection_probability,
          selection_probability, sample_weight, pool_sample_seed, pool_sample_score,
          payload_json, status, attempts, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?)
        """,
        values,
    )
    connection.commit()


def _write_requests(path: Path, connection: sqlite3.Connection) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    with temporary.open("w", encoding="utf-8", newline="") as file:
        for row in connection.execute(
            "SELECT payload_json FROM tasks ORDER BY dataset_id, source_row_number, patent_id"
        ):
            file.write(row[0])
            file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def _task_counts(connection: sqlite3.Connection) -> dict[str, Any]:
    def grouped(column: str) -> dict[str, int]:
        return {
            str(key): int(count)
            for key, count in connection.execute(
                f"SELECT {column}, COUNT(*) FROM tasks GROUP BY {column} ORDER BY {column}"
            )
        }

    return {
        "total": int(connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]),
        "by_dataset": grouped("dataset_id"),
        "by_route": grouped("route"),
        "by_selection_group": grouped("selection_group"),
    }


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    return connection


def _initialize_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE tasks (
            task_id TEXT PRIMARY KEY,
            dataset_id TEXT NOT NULL,
            patent_id TEXT NOT NULL UNIQUE,
            source_row_number INTEGER NOT NULL,
            route TEXT NOT NULL CHECK(route IN ('S','E')),
            selection_group TEXT NOT NULL,
            upstream_selection_probability REAL NOT NULL,
            pool_selection_probability REAL NOT NULL,
            selection_probability REAL NOT NULL,
            sample_weight REAL NOT NULL,
            pool_sample_seed TEXT NOT NULL,
            pool_sample_score TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('pending','running','succeeded','failed')),
            attempts INTEGER NOT NULL,
            requested_model TEXT,
            actual_model TEXT,
            prompt_version TEXT,
            prefix_sha256 TEXT,
            law_sha256 TEXT,
            schema_sha256 TEXT,
            response_id TEXT,
            result_json TEXT,
            raw_response TEXT,
            usage_json TEXT,
            normalization_json TEXT,
            cache_mode TEXT,
            prompt_tokens INTEGER,
            cached_tokens INTEGER,
            cache_write_tokens INTEGER,
            cache_hit_ratio REAL,
            error TEXT,
            elapsed_seconds REAL NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed_at TEXT,
            UNIQUE(dataset_id, source_row_number)
        );
        CREATE INDEX idx_tasks_status
          ON tasks(status, route, dataset_id, source_row_number);
        CREATE TABLE task_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            attempt_number INTEGER NOT NULL,
            outcome TEXT NOT NULL CHECK(outcome IN ('succeeded','failed')),
            response_id TEXT,
            actual_model TEXT,
            raw_response TEXT,
            usage_json TEXT,
            normalization_json TEXT,
            error TEXT,
            elapsed_seconds REAL NOT NULL,
            completed_at TEXT NOT NULL,
            FOREIGN KEY(task_id) REFERENCES tasks(task_id)
        );
        CREATE INDEX idx_task_attempts_task ON task_attempts(task_id, attempt_number);
        """
    )
    connection.commit()


def _record_patent_id(record: PatentRecord) -> str:
    for field in ("application_number", "publication_number", "grant_number"):
        if value := record.get(field).strip():
            return value
    return ""


def _synthetic_patent_id(record: PatentRecord) -> str:
    parts = [
        normalize_text(record.get("title")),
        normalize_text(record.get("applicant")),
        normalize_text(record.get("application_date")),
    ]
    if not any(parts):
        parts.append(f"source-row-{record.row_number}")
    digest = hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:32]
    return f"synthetic-{digest}"


def _task_id(dataset: str, patent_id: str) -> str:
    digest = hashlib.blake2b(f"{dataset}|{patent_id}".encode(), digest_size=12).hexdigest()
    return f"patent-{dataset}-{digest}"


def _truthy(value: str) -> bool:
    return value.strip().casefold() in {"1", "true", "yes", "y"}


def _now() -> str:
    return datetime.now(UTC).isoformat()
