"""Resumable one-by-one Step 2 classification runner backed by SQLite."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from patent_data_security.datasets import dataset_id
from patent_data_security.records import PatentRecord, iter_patent_records
from patent_data_security.step2_prompt import ArkClassificationResponse, VolcengineArkClient

RESULT_FIELDS = (
    "task_id",
    "dataset_id",
    "patent_id",
    "source_row_number",
    "keyword_level",
    "selection_group",
    "selection_probability",
    "sample_weight",
    "status",
    "attempts",
    "requested_model",
    "actual_model",
    "response_id",
    "cat",
    "confidence",
    "subtype",
    "evidence",
    "reason",
    "review_flag",
    "review_reason",
    "elapsed_seconds",
    "usage",
    "error",
    "completed_at",
)


@dataclass(frozen=True)
class Step2Paths:
    database: Path
    results: Path
    progress: Path


def step2_paths(output_dir: str | Path, dataset: str) -> Step2Paths:
    root = Path(output_dir).resolve()
    return Step2Paths(
        database=root / f"classification_state_{dataset}.sqlite3",
        results=root / f"classification_results_{dataset}.csv",
        progress=root / f"classification_progress_{dataset}.json",
    )


def prepare_classification_tasks(
    raw_path: str | Path,
    step1_dir: str | Path,
    output_dir: str | Path,
    *,
    e_sample_rate: float = 0.02,
    e_sample_seed: str = "step2-e-sample-v1",
    encoding: str = "utf-8-sig",
    rebuild: bool = False,
    stop_requested: Callable[[], bool] | None = None,
) -> tuple[Step2Paths, dict[str, Any]]:
    """Create resumable tasks: all unique S/W/R patents plus a stable 2% E sample."""

    if not 0 < e_sample_rate <= 1:
        raise ValueError("e_sample_rate must be in (0, 1]")
    stop_requested = stop_requested or (lambda: False)
    source = Path(raw_path).resolve()
    dataset = dataset_id(source)
    step1 = Path(step1_dir).resolve()
    paths = step2_paths(output_dir, dataset)
    paths.database.parent.mkdir(parents=True, exist_ok=True)
    if rebuild:
        for path in asdict(paths).values():
            Path(path).unlink(missing_ok=True)
        paths.database.with_name(paths.database.name + "-wal").unlink(missing_ok=True)
        paths.database.with_name(paths.database.name + "-shm").unlink(missing_ok=True)

    connection = _connect(paths.database)
    _initialize_database(connection)
    existing = _read_meta(connection, "prepared_summary")
    if existing and not rebuild:
        summary = json.loads(existing)
        expected = {
            "input_path": str(source),
            "input_size_bytes": source.stat().st_size,
            "e_sample_rate": e_sample_rate,
            "e_sample_seed": e_sample_seed,
        }
        mismatches = {
            key: (summary.get(key), value)
            for key, value in expected.items()
            if summary.get(key) != value
        }
        if mismatches:
            connection.close()
            raise ValueError(
                f"Prepared Step 2 state has different inputs; use --rebuild: {mismatches}"
            )
        connection.close()
        if not paths.progress.is_file():
            _write_progress(paths, _progress_from_database(paths, model="not_started"))
        return paths, summary

    tier_files = {tier: step1 / f"keyword_{tier}_{dataset}.csv" for tier in "SWRE"}
    missing = [path for path in tier_files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Step 1 files: {', '.join(map(str, missing))}")

    # Iterate S -> W -> R so a patent duplicated across company rows keeps its strongest tier.
    selected: dict[str, dict[str, Any]] = {}
    unique_by_tier = {tier: 0 for tier in "SWR"}
    for tier in "SWR":
        with tier_files[tier].open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                if stop_requested():
                    connection.close()
                    raise InterruptedError("Step 2 preparation stopped by request")
                patent_id = row["patent_id"]
                if patent_id in selected:
                    continue
                selected[patent_id] = _selection_from_step1_row(row, tier, 1.0)
                unique_by_tier[tier] += 1

    # E is sampled by unique patent_id rather than company-association row.
    seen_e: set[str] = set()
    e_population = 0
    e_selected = 0
    with tier_files["E"].open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if stop_requested():
                connection.close()
                raise InterruptedError("Step 2 preparation stopped by request")
            patent_id = row["patent_id"]
            if patent_id in selected or patent_id in seen_e:
                continue
            seen_e.add(patent_id)
            e_population += 1
            if not _stable_sample(
                f"{dataset}|{patent_id}", e_sample_rate, e_sample_seed
            ):
                continue
            selected[patent_id] = _selection_from_step1_row(row, "E", e_sample_rate)
            e_selected += 1

    selections_by_row = {
        int(selection["source_row_number"]): (patent_id, selection)
        for patent_id, selection in selected.items()
    }
    inserted = 0
    connection.execute("DELETE FROM tasks")
    for record in iter_patent_records(source, encoding=encoding, include_raw=False):
        if stop_requested():
            connection.close()
            raise InterruptedError("Step 2 preparation stopped by request")
        selected_item = selections_by_row.get(record.row_number)
        if selected_item is None:
            continue
        patent_id, selection = selected_item
        payload = _payload_from_record(record, dataset, patent_id, selection)
        task_id = _task_id(dataset, patent_id)
        connection.execute(
            """
            INSERT INTO tasks (
                task_id, dataset_id, patent_id, source_row_number, keyword_level,
                selection_group, selection_probability, sample_weight, payload_json,
                status, attempts, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?)
            """,
            (
                task_id,
                dataset,
                patent_id,
                record.row_number,
                selection["keyword_level"],
                selection["selection_group"],
                selection["selection_probability"],
                selection["sample_weight"],
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                _now(),
                _now(),
            ),
        )
        inserted += 1

    if inserted != len(selected):
        raise ValueError(
            f"Only materialized {inserted} of {len(selected)} selected patents from raw CSV"
        )
    summary = {
        "dataset_id": dataset,
        "input_path": str(source),
        "input_size_bytes": source.stat().st_size,
        "step1_dir": str(step1),
        "unique_swr": unique_by_tier,
        "e_unique_population": e_population,
        "e_sample_rate": e_sample_rate,
        "e_sample_seed": e_sample_seed,
        "e_selected": e_selected,
        "total_tasks": inserted,
        "prepared_at": _now(),
    }
    _write_meta(connection, "prepared_summary", json.dumps(summary, ensure_ascii=False))
    connection.commit()
    connection.close()
    export_results(paths)
    _write_progress(paths, _progress_from_database(paths, model="not_started"))
    return paths, summary


def run_classification_tasks(
    paths: Step2Paths,
    client: VolcengineArkClient,
    *,
    max_attempts: int = 3,
    retry_delay_seconds: float = 2,
    stop_requested: Callable[[], bool] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Continuously request one patent at a time; safe to stop and resume."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    stop_requested = stop_requested or (lambda: False)
    connection = _connect(paths.database)
    _initialize_database(connection)
    connection.execute("UPDATE tasks SET status='pending' WHERE status='running'")
    _write_meta_if_absent(connection, "run_started_at", _now())
    connection.commit()
    export_results(paths, connection=connection)

    while not stop_requested():
        task = connection.execute(
            """
            SELECT * FROM tasks
            WHERE status='pending' AND attempts < ?
            ORDER BY CASE keyword_level WHEN 'S' THEN 0 WHEN 'W' THEN 1
                     WHEN 'R' THEN 2 ELSE 3 END, source_row_number
            LIMIT 1
            """,
            (max_attempts,),
        ).fetchone()
        if task is None:
            break

        attempts = int(task["attempts"]) + 1
        connection.execute(
            "UPDATE tasks SET status='running', attempts=?, requested_model=?, updated_at=? "
            "WHERE task_id=?",
            (attempts, client.model, _now(), task["task_id"]),
        )
        connection.commit()
        started = time.monotonic()
        final_status = "pending"
        try:
            response = client.classify(json.loads(task["payload_json"]))
            elapsed = response.elapsed_seconds
            _save_success(connection, task["task_id"], response, elapsed)
            final_status = "succeeded"
        except Exception as error:  # noqa: BLE001 - failures are persisted for later review
            elapsed = time.monotonic() - started
            final_status = "failed" if attempts >= max_attempts else "pending"
            _save_failure(connection, task["task_id"], error, elapsed, final_status)
        connection.commit()

        if final_status in {"succeeded", "failed"}:
            _append_result(paths, connection, task["task_id"])
        progress = _progress_from_database(paths, model=client.model, connection=connection)
        _write_progress(paths, progress)
        if progress_callback:
            progress_callback(progress)
        if final_status == "pending" and retry_delay_seconds:
            time.sleep(retry_delay_seconds)

    progress = _progress_from_database(paths, model=client.model, connection=connection)
    progress["stopped_by_request"] = bool(stop_requested())
    _write_progress(paths, progress)
    export_results(paths, connection=connection)
    connection.close()
    return progress


def read_progress(paths: Step2Paths) -> dict[str, Any]:
    if paths.progress.is_file():
        return json.loads(paths.progress.read_text(encoding="utf-8"))
    return _progress_from_database(paths, model="unknown")


def export_results(paths: Step2Paths, *, connection: sqlite3.Connection | None = None) -> None:
    own_connection = connection is None
    connection = connection or _connect(paths.database)
    temporary = paths.results.with_suffix(paths.results.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in connection.execute(
            "SELECT * FROM tasks WHERE status IN ('succeeded','failed') "
            "ORDER BY source_row_number"
        ):
            writer.writerow(_result_row(row))
    os.replace(temporary, paths.results)
    if own_connection:
        connection.close()


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    return connection


def _initialize_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            dataset_id TEXT NOT NULL,
            patent_id TEXT NOT NULL,
            source_row_number INTEGER NOT NULL,
            keyword_level TEXT NOT NULL,
            selection_group TEXT NOT NULL,
            selection_probability REAL NOT NULL,
            sample_weight REAL NOT NULL,
            payload_json TEXT NOT NULL,
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL,
            requested_model TEXT,
            actual_model TEXT,
            response_id TEXT,
            result_json TEXT,
            raw_response TEXT,
            usage_json TEXT,
            error TEXT,
            elapsed_seconds REAL NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status, keyword_level);
        """
    )
    connection.commit()


def _selection_from_step1_row(
    row: dict[str, str], tier: str, probability: float
) -> dict[str, Any]:
    return {
        "source_row_number": int(row["source_row_number"]),
        "keyword_level": tier,
        "keyword_hits": json.loads(row["keyword_hits"]),
        "diagnostic_hits": json.loads(row["diagnostic_hits"]),
        "selection_group": tier if tier != "E" else "E_sample",
        "selection_probability": probability,
        "sample_weight": 1 / probability,
    }


def _payload_from_record(
    record: PatentRecord,
    dataset: str,
    patent_id: str,
    selection: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset_id": dataset,
        "patent_id": patent_id,
        "source_row_number": record.row_number,
        "application_year": record.get("application_year"),
        "title": record.get("title"),
        "abstract": record.get("abstract"),
        "claim": record.get("claim"),
        "ipc": record.get("ipc"),
        "main_ipc": record.get("main_ipc"),
        "keyword_level": selection["keyword_level"],
        "keyword_hits": selection["keyword_hits"],
        "diagnostic_hits": selection["diagnostic_hits"],
        "selection_group": selection["selection_group"],
        "selection_probability": selection["selection_probability"],
        "sample_weight": selection["sample_weight"],
    }


def _save_success(
    connection: sqlite3.Connection,
    task_id: str,
    response: ArkClassificationResponse,
    elapsed: float,
) -> None:
    connection.execute(
        """
        UPDATE tasks SET status='succeeded', actual_model=?, response_id=?, result_json=?,
          raw_response=?, usage_json=?, error=NULL,
          elapsed_seconds=elapsed_seconds+?, updated_at=?, completed_at=?
        WHERE task_id=?
        """,
        (
            response.actual_model,
            response.response_id,
            response.classification.model_dump_json(),
            response.raw_text,
            json.dumps(response.usage, ensure_ascii=False),
            elapsed,
            _now(),
            _now(),
            task_id,
        ),
    )


def _save_failure(
    connection: sqlite3.Connection,
    task_id: str,
    error: Exception,
    elapsed: float,
    status: str,
) -> None:
    connection.execute(
        """
        UPDATE tasks SET status=?, error=?, elapsed_seconds=elapsed_seconds+?,
          updated_at=?, completed_at=? WHERE task_id=?
        """,
        (
            status,
            f"{type(error).__name__}: {error}"[:2000],
            elapsed,
            _now(),
            _now() if status == "failed" else None,
            task_id,
        ),
    )


def _progress_from_database(
    paths: Step2Paths,
    *,
    model: str,
    connection: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    own_connection = connection is None
    connection = connection or _connect(paths.database)
    counts = {
        row["status"]: row["count"]
        for row in connection.execute(
            "SELECT status, COUNT(*) AS count FROM tasks GROUP BY status"
        )
    }
    total = sum(counts.values())
    completed = counts.get("succeeded", 0) + counts.get("failed", 0)
    pending = counts.get("pending", 0) + counts.get("running", 0)
    dataset_row = connection.execute("SELECT dataset_id FROM tasks LIMIT 1").fetchone()
    timing = connection.execute(
        "SELECT COALESCE(SUM(elapsed_seconds),0) AS elapsed, "
        "COALESCE(SUM(attempts),0) AS attempts, "
        "COALESCE(AVG(CASE WHEN status IN ('succeeded','failed') "
        "THEN elapsed_seconds END),0) AS avg_task FROM tasks"
    ).fetchone()
    average_request = timing["elapsed"] / timing["attempts"] if timing["attempts"] else 0
    average_task = float(timing["avg_task"] or average_request)
    progress = {
        "dataset_id": dataset_row[0] if dataset_row else "",
        "database": str(paths.database),
        "results": str(paths.results),
        "model": model,
        "actual_models": [
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT actual_model FROM tasks WHERE actual_model IS NOT NULL"
            )
        ],
        "total": total,
        "completed": completed,
        "succeeded": counts.get("succeeded", 0),
        "failed": counts.get("failed", 0),
        "pending": pending,
        "progress_percent": round(completed / total * 100, 4) if total else 0,
        "average_request_seconds": round(average_request, 3),
        "average_completed_task_seconds": round(average_task, 3),
        "eta_seconds": round(pending * average_task, 3) if average_task else None,
        "updated_at": _now(),
    }
    if own_connection:
        connection.close()
    return progress


def _append_result(paths: Step2Paths, connection: sqlite3.Connection, task_id: str) -> None:
    row = connection.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
    write_header = not paths.results.exists() or paths.results.stat().st_size == 0
    with paths.results.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(_result_row(row))


def _result_row(row: sqlite3.Row) -> dict[str, Any]:
    result = json.loads(row["result_json"]) if row["result_json"] else {}
    return {
        "task_id": row["task_id"],
        "dataset_id": row["dataset_id"],
        "patent_id": row["patent_id"],
        "source_row_number": row["source_row_number"],
        "keyword_level": row["keyword_level"],
        "selection_group": row["selection_group"],
        "selection_probability": row["selection_probability"],
        "sample_weight": row["sample_weight"],
        "status": row["status"],
        "attempts": row["attempts"],
        "requested_model": row["requested_model"] or "",
        "actual_model": row["actual_model"] or "",
        "response_id": row["response_id"] or "",
        "cat": result.get("cat", ""),
        "confidence": result.get("confidence", ""),
        "subtype": result.get("subtype", ""),
        "evidence": json.dumps(result.get("evidence", []), ensure_ascii=False),
        "reason": result.get("reason", ""),
        "review_flag": result.get("review_flag", ""),
        "review_reason": result.get("review_reason", ""),
        "elapsed_seconds": round(row["elapsed_seconds"], 3),
        "usage": row["usage_json"] or "{}",
        "error": row["error"] or "",
        "completed_at": row["completed_at"] or "",
    }


def _write_progress(paths: Step2Paths, progress: dict[str, Any]) -> None:
    temporary = paths.progress.with_suffix(paths.progress.suffix + ".tmp")
    temporary.write_text(
        json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, paths.progress)


def _task_id(dataset: str, patent_id: str) -> str:
    digest = hashlib.blake2b(
        f"{dataset}|{patent_id}".encode(), digest_size=12
    ).hexdigest()
    return f"patent-{dataset}-{digest}"


def _stable_sample(key: str, probability: float, seed: str) -> bool:
    digest = hashlib.blake2b(f"{seed}|{key}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64 < probability


def _read_meta(connection: sqlite3.Connection, key: str) -> str | None:
    row = connection.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else None


def _write_meta(connection: sqlite3.Connection, key: str, value: str) -> None:
    connection.execute(
        "INSERT INTO meta(key,value) VALUES(?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


def _write_meta_if_absent(connection: sqlite3.Connection, key: str, value: str) -> None:
    connection.execute("INSERT OR IGNORE INTO meta(key,value) VALUES(?,?)", (key, value))


def _patent_id(record: PatentRecord) -> str:
    return (
        record.get("application_number")
        or record.get("publication_number")
        or record.get("grant_number")
        or f"source-row-{record.row_number}"
    )


def _now() -> str:
    return datetime.now(UTC).isoformat()
