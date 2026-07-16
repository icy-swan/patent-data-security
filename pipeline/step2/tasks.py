"""Prepare Step 2 tasks while keeping model evidence separate from routing metadata."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pipeline.common.datasets import dataset_id
from pipeline.common.io import atomic_json_write, sha256_file
from pipeline.common.records import PatentRecord, iter_patent_records
from pipeline.step2.prompt import PromptBundle, build_dynamic_payload, load_prompt_bundle


@dataclass(frozen=True)
class Step2TaskPaths:
    database: Path
    manifest: Path
    results: Path
    progress: Path


def task_paths(output_dir: str | Path, dataset: str) -> Step2TaskPaths:
    root = Path(output_dir).resolve()
    return Step2TaskPaths(
        database=root / f"step2_tasks_{dataset}.sqlite3",
        manifest=root / f"step2_task_manifest_{dataset}.json",
        results=root / f"step2_results_{dataset}.csv",
        progress=root / f"step2_progress_{dataset}.json",
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
    paths.database.parent.mkdir(parents=True, exist_ok=True)
    all_paths = (paths.database, paths.manifest, paths.results, paths.progress)
    if not rebuild and any(path.exists() for path in all_paths):
        raise FileExistsError("Step 2 task state exists; pass rebuild=True to replace it")
    if rebuild:
        for path in all_paths:
            path.unlink(missing_ok=True)
        for suffix in ("-wal", "-shm"):
            paths.database.with_name(paths.database.name + suffix).unlink(missing_ok=True)

    bundle = prompt_bundle or load_prompt_bundle()
    selections = _read_selected_step1_rows(step1)
    by_source_row: dict[int, dict[str, str]] = {}
    for patent_id, selection in selections.items():
        row_number = int(selection["source_row_number"])
        if row_number in by_source_row:
            raise ValueError(f"Multiple patents point to source row {row_number}")
        by_source_row[row_number] = {"patent_id": patent_id, **selection}

    connection = _connect(paths.database)
    _initialize_database(connection)
    found: set[int] = set()
    binding_mismatches = 0
    route_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    for record in iter_patent_records(raw, encoding=encoding):
        selection = by_source_row.get(record.row_number)
        if selection is None:
            continue
        patent_id = selection["patent_id"]
        actual_id = _record_patent_id(record)
        if not patent_id.startswith("synthetic-") and actual_id != patent_id:
            binding_mismatches += 1
            continue
        payload = build_dynamic_payload(
            {
                "patent_id": patent_id,
                "title": record.get("title"),
                "abstract": record.get("abstract"),
                "claim": record.get("claim"),
                "ipc": record.get("ipc"),
                "main_ipc": record.get("main_ipc"),
            }
        )
        now = _now()
        connection.execute(
            """
            INSERT INTO tasks (
                task_id, dataset_id, patent_id, source_row_number, route,
                selection_group, selection_probability, sample_weight,
                payload_json, status, attempts, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?)
            """,
            (
                _task_id(dataset, patent_id),
                dataset,
                patent_id,
                record.row_number,
                selection["route"],
                selection["selection_group"],
                float(selection["selection_probability"]),
                float(selection["sample_weight"]),
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                now,
                now,
            ),
        )
        found.add(record.row_number)
        route_counts[selection["route"]] += 1
        group_counts[selection["selection_group"]] += 1
    missing_rows = sorted(set(by_source_row) - found)
    if binding_mismatches or missing_rows:
        connection.close()
        paths.database.unlink(missing_ok=True)
        raise ValueError(
            "Step 2 local binding failed: "
            f"patent_id_mismatches={binding_mismatches}, missing_source_rows={missing_rows[:10]}"
        )

    task_count = connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    unique_patents = connection.execute(
        "SELECT COUNT(DISTINCT patent_id) FROM tasks"
    ).fetchone()[0]
    if task_count != len(selections) or unique_patents != task_count:
        connection.close()
        paths.database.unlink(missing_ok=True)
        raise ValueError("Step 2 task count or patent uniqueness check failed")

    manifest = {
        "schema_version": "2.1.0",
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
        "statistics_binding": {
            "association_key": "local task_id -> local patent_id",
            "model_echo_required": False,
            "patent_id_in_dynamic_payload": True,
            "patent_id_semantic_use_allowed": False,
            "patent_id_mismatches": binding_mismatches,
            "missing_source_rows": len(missing_rows),
            "duplicate_task_patent_ids": task_count - unique_patents,
        },
        "task_counts": {
            "total": task_count,
            "by_route": dict(sorted(route_counts.items())),
            "by_selection_group": dict(sorted(group_counts.items())),
        },
        "database": str(paths.database),
        "prepared_at": _now(),
    }
    connection.execute(
        "INSERT INTO meta(key,value) VALUES('task_manifest',?)",
        (json.dumps(manifest, ensure_ascii=False, separators=(",", ":")),),
    )
    connection.commit()
    connection.close()
    atomic_json_write(paths.manifest, manifest)
    return paths, manifest


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
            source_row_number INTEGER NOT NULL UNIQUE,
            route TEXT NOT NULL CHECK(route IN ('S','E')),
            selection_group TEXT NOT NULL,
            selection_probability REAL NOT NULL,
            sample_weight REAL NOT NULL,
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
            completed_at TEXT
        );
        CREATE INDEX idx_tasks_status ON tasks(status, source_row_number);
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


def _task_id(dataset: str, patent_id: str) -> str:
    digest = hashlib.blake2b(f"{dataset}|{patent_id}".encode(), digest_size=12).hexdigest()
    return f"patent-{dataset}-{digest}"


def _truthy(value: str) -> bool:
    return value.strip().casefold() in {"1", "true", "yes", "y"}


def _now() -> str:
    return datetime.now(UTC).isoformat()
