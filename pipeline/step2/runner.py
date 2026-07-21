"""Resumable concurrent Step 2 request runner backed by local SQLite state."""

from __future__ import annotations

import csv
import fcntl
import json
import os
import sqlite3
import time
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from pipeline.common.io import atomic_json_write
from pipeline.step2.client import (
    ClassificationOutputError,
    ClassificationResponse,
    VolcengineArkClient,
)
from pipeline.step2.tasks import Step2TaskPaths

RESULT_FIELDS = (
    "task_id",
    "dataset_id",
    "patent_id",
    "source_row_number",
    "route",
    "selection_group",
    "selection_probability",
    "sample_weight",
    "status",
    "attempts",
    "requested_model",
    "actual_model",
    "prompt_version",
    "prefix_sha256",
    "law_sha256",
    "schema_sha256",
    "response_id",
    "step2_label",
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
    "normalization_events",
    "cache_mode",
    "prompt_tokens",
    "cached_tokens",
    "cache_write_tokens",
    "cache_hit_ratio",
    "elapsed_seconds",
    "usage",
    "error",
    "completed_at",
)


def run_tasks(
    paths: Step2TaskPaths,
    client: VolcengineArkClient,
    *,
    max_attempts: int = 3,
    retry_delay_seconds: float = 2,
    concurrency: int = 10,
    stop_requested: Callable[[], bool] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run pending tasks; every response is persisted under its pre-existing task row."""

    if max_attempts < 1 or concurrency < 1 or retry_delay_seconds < 0:
        raise ValueError("Invalid retry or concurrency settings")
    stop_requested = stop_requested or (lambda: False)
    run_started_monotonic = time.monotonic()
    run_started_at = _now()
    with _exclusive_lock(paths.database):
        connection = _connect(paths.database)
        try:
            _ensure_runtime_schema(connection)
            _validate_prompt_identity(connection, client)
            connection.execute("UPDATE tasks SET status='pending' WHERE status='running'")
            connection.commit()
            export_results(paths, connection=connection)
            initial_progress = _progress(
                connection,
                paths,
                client,
                concurrency,
                run_started_monotonic=run_started_monotonic,
                run_started_at=run_started_at,
            )
            atomic_json_write(paths.progress, initial_progress)
            if progress_callback:
                progress_callback(initial_progress)
            _run_loop(
                paths,
                connection,
                client,
                max_attempts=max_attempts,
                retry_delay_seconds=retry_delay_seconds,
                concurrency=concurrency,
                stop_requested=stop_requested,
                progress_callback=progress_callback,
                run_started_monotonic=run_started_monotonic,
                run_started_at=run_started_at,
            )
            progress = _progress(
                connection,
                paths,
                client,
                concurrency,
                run_started_monotonic=run_started_monotonic,
                run_started_at=run_started_at,
            )
            progress["stopped_by_request"] = bool(stop_requested())
            atomic_json_write(paths.progress, progress)
            export_results(paths, connection=connection)
            return progress
        finally:
            connection.close()


def export_results(
    paths: Step2TaskPaths,
    *,
    connection: sqlite3.Connection | None = None,
) -> None:
    own_connection = connection is None
    connection = connection or _connect(paths.database)
    _ensure_runtime_schema(connection)
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


def read_progress(paths: Step2TaskPaths) -> dict[str, Any]:
    if paths.progress.is_file():
        return json.loads(paths.progress.read_text(encoding="utf-8"))
    connection = sqlite3.connect(
        f"file:{paths.database}?mode=ro&immutable=1",
        uri=True,
    )
    connection.row_factory = sqlite3.Row
    try:
        manifest = json.loads(
            connection.execute("SELECT value FROM meta WHERE key='task_manifest'").fetchone()[0]
        )
        counts = {
            row["status"]: row["count"]
            for row in connection.execute(
                "SELECT status, COUNT(*) count FROM tasks GROUP BY status"
            )
        }
        return {"dataset_id": manifest["dataset_id"], "counts": counts}
    finally:
        connection.close()


def _run_loop(
    paths: Step2TaskPaths,
    connection: sqlite3.Connection,
    client: VolcengineArkClient,
    *,
    max_attempts: int,
    retry_delay_seconds: float,
    concurrency: int,
    stop_requested: Callable[[], bool],
    progress_callback: Callable[[dict[str, Any]], None] | None,
    run_started_monotonic: float,
    run_started_at: str,
) -> None:
    def classify(task: sqlite3.Row) -> ClassificationResponse | Exception:
        try:
            return client.classify(json.loads(task["payload_json"]))
        except Exception as error:  # noqa: BLE001 - persisted for retry/audit
            return error

    in_flight: dict[Future[Any], tuple[sqlite3.Row, int, float]] = {}
    retry_not_before: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        while True:
            now = time.monotonic()
            retry_not_before = {
                task_id: deadline
                for task_id, deadline in retry_not_before.items()
                if deadline > now
            }
            while not stop_requested() and len(in_flight) < concurrency:
                claimed = _claim_next(
                    connection,
                    client.model,
                    max_attempts,
                    excluded_task_ids=set(retry_not_before),
                )
                if claimed is None:
                    break
                task, attempts = claimed
                started = time.monotonic()
                in_flight[pool.submit(classify, task)] = (task, attempts, started)
            if not in_flight:
                if stop_requested() or not retry_not_before:
                    break
                time.sleep(min(max(0.0, min(retry_not_before.values()) - time.monotonic()), 0.1))
                continue
            timeout = None
            if retry_not_before:
                timeout = max(0.0, min(retry_not_before.values()) - time.monotonic())
            completed, _ = wait(in_flight, timeout=timeout, return_when=FIRST_COMPLETED)
            if not completed:
                continue
            for future in completed:
                task, attempts, started = in_flight.pop(future)
                outcome = future.result()
                elapsed = (
                    outcome.elapsed_seconds
                    if isinstance(outcome, ClassificationResponse)
                    else time.monotonic() - started
                )
                final_status = _persist_outcome(
                    connection,
                    task,
                    attempts,
                    max_attempts,
                    outcome,
                    elapsed,
                )
                if final_status == "pending" and retry_delay_seconds:
                    retry_not_before[task["task_id"]] = (
                        time.monotonic() + retry_delay_seconds
                    )
            progress = _progress(
                connection,
                paths,
                client,
                concurrency,
                run_started_monotonic=run_started_monotonic,
                run_started_at=run_started_at,
            )
            atomic_json_write(paths.progress, progress)
            if progress_callback:
                progress_callback(progress)


def _claim_next(
    connection: sqlite3.Connection,
    model: str,
    max_attempts: int,
    *,
    excluded_task_ids: set[str],
) -> tuple[sqlite3.Row, int] | None:
    parameters: list[Any] = [max_attempts]
    exclusion = ""
    if excluded_task_ids:
        placeholders = ",".join("?" for _ in excluded_task_ids)
        exclusion = f" AND task_id NOT IN ({placeholders})"
        parameters.extend(sorted(excluded_task_ids))
    task = connection.execute(
        f"""
        SELECT * FROM tasks
        WHERE status='pending' AND attempts < ? {exclusion}
        ORDER BY CASE route WHEN 'S' THEN 0 ELSE 1 END, source_row_number
        LIMIT 1
        """,
        parameters,
    ).fetchone()
    if task is None:
        return None
    attempts = int(task["attempts"]) + 1
    connection.execute(
        "UPDATE tasks SET status='running', attempts=?, requested_model=?, updated_at=? "
        "WHERE task_id=?",
        (attempts, model, _now(), task["task_id"]),
    )
    connection.commit()
    return task, attempts


def _persist_outcome(
    connection: sqlite3.Connection,
    task: sqlite3.Row,
    attempts: int,
    max_attempts: int,
    outcome: ClassificationResponse | Exception,
    elapsed: float,
) -> str:
    if isinstance(outcome, Exception):
        status = "failed" if attempts >= max_attempts else "pending"
        raw_text = ""
        response_id = ""
        actual_model = ""
        usage: dict[str, Any] = {}
        normalization_events: tuple[str, ...] = ()
        if isinstance(outcome, ClassificationOutputError):
            raw_text = outcome.raw_text
            response_id = outcome.response_id
            actual_model = outcome.actual_model
            usage = outcome.usage
            normalization_events = outcome.normalization_events
        error_text = f"{type(outcome).__name__}: {outcome}"[:2000]
        usage_json = json.dumps(usage, ensure_ascii=False, separators=(",", ":"))
        normalization_json = json.dumps(
            normalization_events, ensure_ascii=False, separators=(",", ":")
        )
        _insert_attempt(
            connection,
            task_id=task["task_id"],
            attempt_number=attempts,
            outcome="failed",
            response_id=response_id,
            actual_model=actual_model,
            raw_response=raw_text,
            usage_json=usage_json,
            normalization_json=normalization_json,
            error=error_text,
            elapsed_seconds=elapsed,
        )
        connection.execute(
            """
            UPDATE tasks SET status=?, actual_model=COALESCE(NULLIF(?,''),actual_model),
              response_id=COALESCE(NULLIF(?,''),response_id), raw_response=?, usage_json=?,
              normalization_json=?, error=?, elapsed_seconds=elapsed_seconds+?,
              updated_at=?, completed_at=? WHERE task_id=?
            """,
            (
                status,
                actual_model,
                response_id,
                raw_text,
                usage_json,
                normalization_json,
                error_text,
                elapsed,
                _now(),
                _now() if status == "failed" else None,
                task["task_id"],
            ),
        )
    else:
        usage_json = json.dumps(outcome.usage, ensure_ascii=False, separators=(",", ":"))
        normalization_json = json.dumps(
            outcome.normalization_events, ensure_ascii=False, separators=(",", ":")
        )
        _insert_attempt(
            connection,
            task_id=task["task_id"],
            attempt_number=attempts,
            outcome="succeeded",
            response_id=outcome.response_id,
            actual_model=outcome.actual_model,
            raw_response=outcome.raw_text,
            usage_json=usage_json,
            normalization_json=normalization_json,
            error=None,
            elapsed_seconds=elapsed,
        )
        connection.execute(
            """
            UPDATE tasks SET status='succeeded', actual_model=?, prompt_version=?,
              prefix_sha256=?, law_sha256=?, schema_sha256=?, response_id=?,
              result_json=?, raw_response=?, usage_json=?, normalization_json=?,
              cache_mode=?, prompt_tokens=?, cached_tokens=?, cache_write_tokens=?,
              cache_hit_ratio=?, error=NULL, elapsed_seconds=elapsed_seconds+?,
              updated_at=?, completed_at=?
            WHERE task_id=?
            """,
            (
                outcome.actual_model,
                outcome.prompt_version,
                outcome.prefix_sha256,
                outcome.law_sha256,
                outcome.schema_sha256,
                outcome.response_id,
                outcome.classification.model_dump_json(),
                outcome.raw_text,
                usage_json,
                normalization_json,
                outcome.cache_mode,
                outcome.prompt_tokens,
                outcome.cached_tokens,
                outcome.cache_write_tokens,
                outcome.cache_hit_ratio,
                elapsed,
                _now(),
                _now(),
                task["task_id"],
            ),
        )
        status = "succeeded"
    connection.commit()
    return status


def _insert_attempt(
    connection: sqlite3.Connection,
    *,
    task_id: str,
    attempt_number: int,
    outcome: str,
    response_id: str,
    actual_model: str,
    raw_response: str,
    usage_json: str,
    normalization_json: str,
    error: str | None,
    elapsed_seconds: float,
) -> None:
    connection.execute(
        """
        INSERT INTO task_attempts (
          task_id, attempt_number, outcome, response_id, actual_model, raw_response,
          usage_json, normalization_json, error, elapsed_seconds, completed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            task_id,
            attempt_number,
            outcome,
            response_id,
            actual_model,
            raw_response,
            usage_json,
            normalization_json,
            error,
            elapsed_seconds,
            _now(),
        ),
    )


def _progress(
    connection: sqlite3.Connection,
    paths: Step2TaskPaths,
    client: VolcengineArkClient,
    concurrency: int,
    *,
    run_started_monotonic: float,
    run_started_at: str,
) -> dict[str, Any]:
    counts = {
        row["status"]: row["count"]
        for row in connection.execute(
            "SELECT status, COUNT(*) count FROM tasks GROUP BY status"
        )
    }
    total = sum(counts.values())
    completed = counts.get("succeeded", 0) + counts.get("failed", 0)
    pending = counts.get("pending", 0) + counts.get("running", 0)
    usage = connection.execute(
        """
        SELECT SUM(prompt_tokens) prompt_tokens, SUM(cached_tokens) cached_tokens,
          SUM(cache_write_tokens) cache_write_tokens,
          COUNT(cached_tokens) cached_observations,
          COALESCE(SUM(elapsed_seconds),0) cumulative_request_seconds,
          COALESCE(SUM(attempts - CASE WHEN status='running' THEN 1 ELSE 0 END),0)
            completed_attempts,
          COALESCE(AVG(CASE WHEN status IN ('succeeded','failed')
            THEN elapsed_seconds END),0) average_completed_task_seconds
        FROM tasks
        """
    ).fetchone()
    prompt_tokens = usage["prompt_tokens"]
    cached_tokens = usage["cached_tokens"]
    ratio = (
        cached_tokens / prompt_tokens
        if usage["cached_observations"] and prompt_tokens
        else None
    )
    cumulative_request_seconds = float(usage["cumulative_request_seconds"] or 0)
    completed_attempts = int(usage["completed_attempts"] or 0)
    average_request_seconds = (
        cumulative_request_seconds / completed_attempts if completed_attempts else 0
    )
    average_completed_task_seconds = float(
        usage["average_completed_task_seconds"] or average_request_seconds
    )
    eta_seconds = (
        round(pending * average_completed_task_seconds / concurrency, 3)
        if average_completed_task_seconds
        else None
    )
    run_elapsed_seconds = round(time.monotonic() - run_started_monotonic, 3)
    estimated_finish_at = (
        (datetime.now(UTC) + timedelta(seconds=eta_seconds)).isoformat()
        if eta_seconds is not None
        else None
    )
    manifest = json.loads(
        connection.execute("SELECT value FROM meta WHERE key='task_manifest'").fetchone()[0]
    )
    return {
        "schema_version": "2.1.0",
        "dataset_id": manifest["dataset_id"],
        "database": str(paths.database),
        "result": str(paths.results),
        "model": client.model,
        "prompt_version": client.prompt_bundle.prompt_version,
        "prefix_sha256": client.prompt_bundle.prefix_sha256,
        "law_sha256": client.prompt_bundle.law_sha256,
        "total": total,
        "completed": completed,
        "succeeded": counts.get("succeeded", 0),
        "failed": counts.get("failed", 0),
        "pending": pending,
        "concurrency": concurrency,
        "progress_percent": round(completed / total * 100, 4) if total else 0,
        "run_started_at": run_started_at,
        "run_elapsed_seconds": run_elapsed_seconds,
        "cumulative_request_seconds": round(cumulative_request_seconds, 3),
        "average_request_seconds": round(average_request_seconds, 3),
        "average_completed_task_seconds": round(average_completed_task_seconds, 3),
        "eta_seconds": eta_seconds,
        "estimated_finish_at": estimated_finish_at,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens if usage["cached_observations"] else None,
            "cache_write_tokens": usage["cache_write_tokens"],
            "cache_hit_ratio": ratio,
        },
        "updated_at": _now(),
    }


def _validate_prompt_identity(
    connection: sqlite3.Connection,
    client: VolcengineArkClient,
) -> None:
    row = connection.execute("SELECT value FROM meta WHERE key='task_manifest'").fetchone()
    if row is None:
        raise ValueError("Step 2 task manifest is missing from the database")
    manifest = json.loads(row[0])
    expected = manifest["resource_sha256"]
    actual = client.prompt_bundle.resource_hashes
    if expected != actual or manifest["prompt_version"] != client.prompt_bundle.prompt_version:
        raise ValueError("Prepared tasks and client use different Prompt resources")


def _result_row(row: sqlite3.Row) -> dict[str, Any]:
    result = json.loads(row["result_json"]) if row["result_json"] else {}
    return {
        "task_id": row["task_id"],
        "dataset_id": row["dataset_id"],
        "patent_id": row["patent_id"],
        "source_row_number": row["source_row_number"],
        "route": row["route"],
        "selection_group": row["selection_group"],
        "selection_probability": row["selection_probability"],
        "sample_weight": row["sample_weight"],
        "status": row["status"],
        "attempts": row["attempts"],
        "requested_model": row["requested_model"] or "",
        "actual_model": row["actual_model"] or "",
        "prompt_version": row["prompt_version"] or "",
        "prefix_sha256": row["prefix_sha256"] or "",
        "law_sha256": row["law_sha256"] or "",
        "schema_sha256": row["schema_sha256"] or "",
        "response_id": row["response_id"] or "",
        "step2_label": result.get("label", ""),
        "confidence": result.get("confidence", ""),
        "scope_basis": json.dumps(result.get("scope_basis", []), ensure_ascii=False),
        "processing_activities": json.dumps(
            result.get("processing_activities", []), ensure_ascii=False
        ),
        "industry_sectors": json.dumps(result.get("industry_sectors", []), ensure_ascii=False),
        "technical_scope": result.get("technical_scope", ""),
        "legal_scope": result.get("legal_scope", ""),
        "evidence": json.dumps(result.get("evidence", []), ensure_ascii=False),
        "reason": result.get("reason", ""),
        "needs_review": result.get("needs_review", result.get("review_flag", "")),
        "review_reason": result.get("review_reason", ""),
        "normalization_events": row["normalization_json"] or "[]",
        "cache_mode": row["cache_mode"] or "",
        "prompt_tokens": row["prompt_tokens"] if row["prompt_tokens"] is not None else "",
        "cached_tokens": row["cached_tokens"] if row["cached_tokens"] is not None else "",
        "cache_write_tokens": (
            row["cache_write_tokens"]
            if row["cache_write_tokens"] is not None
            else ""
        ),
        "cache_hit_ratio": row["cache_hit_ratio"] if row["cache_hit_ratio"] is not None else "",
        "elapsed_seconds": round(row["elapsed_seconds"], 3),
        "usage": row["usage_json"] or "{}",
        "error": row["error"] or "",
        "completed_at": row["completed_at"] or "",
    }


@contextmanager
def _exclusive_lock(database: Path) -> Iterator[None]:
    lock_path = database.with_name(database.name + ".run.lock")
    owns_lock = False
    try:
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"A Step 2 runner is already active for {database}") from None
            owns_lock = True
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    finally:
        if owns_lock:
            lock_path.unlink(missing_ok=True)


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    return connection


def _ensure_runtime_schema(connection: sqlite3.Connection) -> None:
    columns = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(tasks)")
    }
    if "normalization_json" not in columns:
        connection.execute("ALTER TABLE tasks ADD COLUMN normalization_json TEXT")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS task_attempts (
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
        CREATE INDEX IF NOT EXISTS idx_task_attempts_task
          ON task_attempts(task_id, attempt_number);
        """
    )
    connection.commit()


def _now() -> str:
    return datetime.now(UTC).isoformat()
