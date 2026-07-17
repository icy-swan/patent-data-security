"""Resumable local-Codex runner and exporter for Step 3 model simulation."""

from __future__ import annotations

import fcntl
import json
import sqlite3
import time
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pipeline.common.io import atomic_json_write
from pipeline.step3.client import BatchAnnotationResponse, CodexAnnotationClient
from pipeline.step3.sampling import Step3Paths, write_simulation_dataset
from pipeline.step3.schema import IndependentAnnotation


def run_simulation(
    paths: Step3Paths,
    client: CodexAnnotationClient,
    *,
    batch_size: int = 20,
    max_attempts: int = 3,
    retry_delay_seconds: float = 2,
    stop_requested: Callable[[], bool] = lambda: False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if not 1 <= batch_size <= 25:
        raise ValueError("batch_size must be between 1 and 25")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    started = time.monotonic()
    started_at = _now()
    with _exclusive_lock(paths.database):
        connection = _connect(paths.database)
        connection.execute("UPDATE tasks SET status='pending' WHERE status='running'")
        connection.commit()
        _validate_identity(connection, client)
        while not stop_requested():
            pending = connection.execute(
                "SELECT * FROM tasks WHERE status IN ('pending','failed') AND attempts < ? "
                "ORDER BY sample_id LIMIT ?",
                (max_attempts, batch_size),
            ).fetchall()
            if not pending:
                break
            ids = [row["sample_id"] for row in pending]
            connection.executemany(
                "UPDATE tasks SET status='running', attempts=attempts+1, updated_at=? "
                "WHERE sample_id=?",
                [(_now(), sample_id) for sample_id in ids],
            )
            connection.commit()
            payloads = []
            for row in pending:
                payload = json.loads(row["payload_json"])
                payload["sample_id"] = row["sample_id"]
                payloads.append(payload)
            try:
                response = client.annotate_batch(payloads)
            except Exception as error:
                for row in pending:
                    _record_failure(connection, row["sample_id"], error)
            else:
                per_item_elapsed = response.elapsed_seconds / len(pending)
                for index, row in enumerate(pending):
                    sample_id = row["sample_id"]
                    _record_success(
                        connection,
                        sample_id,
                        response.annotations[sample_id],
                        response,
                        elapsed_seconds=per_item_elapsed,
                        usage=response.usage if index == 0 else {},
                        raw_response=response.raw_response if index == 0 else "",
                    )
            connection.commit()
            progress = _progress(
                connection,
                client,
                batch_size=batch_size,
                started_at=started_at,
                elapsed=time.monotonic() - started,
            )
            atomic_json_write(paths.progress, progress)
            if progress_callback:
                progress_callback(progress)
            if retry_delay_seconds and _has_retryable(connection, max_attempts):
                time.sleep(retry_delay_seconds)

        progress = _progress(
            connection,
            client,
            batch_size=batch_size,
            started_at=started_at,
            elapsed=time.monotonic() - started,
        )
        atomic_json_write(paths.progress, progress)
        if progress["succeeded"] == progress["total"]:
            rows = _completed_rows(connection)
            progress.pop("split_report", None)
            progress["simulation_summary"] = write_simulation_dataset(
                paths,
                rows,
                annotation_model=client.model,
                annotation_prompt_version=client.prompt.version,
            )
            atomic_json_write(paths.progress, progress)
        connection.close()
    return progress


def read_progress(paths: Step3Paths) -> dict[str, Any]:
    progress: dict[str, Any] = {}
    if paths.progress.is_file():
        progress = json.loads(paths.progress.read_text(encoding="utf-8"))
    if not paths.database.is_file():
        return {"status": "not_prepared", "database": str(paths.database)}
    connection = _connect(paths.database)
    counts = Counter(dict(connection.execute("SELECT status,COUNT(*) FROM tasks GROUP BY status")))
    running_row = connection.execute(
        "SELECT MIN(updated_at) FROM tasks WHERE status='running'"
    ).fetchone()
    connection.close()
    total = sum(counts.values())
    progress.update(
        {
            "total": total,
            "completed": counts["succeeded"],
            "succeeded": counts["succeeded"],
            "failed": counts["failed"],
            "pending": total - counts["succeeded"],
            "queued": counts["pending"],
            "running": counts["running"],
            "runner_active": _runner_active(paths.database),
            "current_batch_started_at": running_row[0] if running_row else None,
            "progress_percent": (
                round(counts["succeeded"] / total * 100, 4) if total else 0
            ),
        }
    )
    return progress


def _validate_identity(connection: sqlite3.Connection, client: CodexAnnotationClient) -> None:
    row = connection.execute("SELECT value FROM meta WHERE key='annotation_identity'").fetchone()
    identity = {
        "model": client.model,
        "reasoning_effort": client.reasoning_effort,
        "prompt_version": client.prompt.version,
        "prompt_sha256": client.prompt.prompt_sha256,
        "schema_sha256": client.prompt.schema_sha256,
    }
    if row is None:
        connection.execute(
            "INSERT INTO meta(key,value) VALUES('annotation_identity',?)",
            (json.dumps(identity, ensure_ascii=False, sort_keys=True),),
        )
        connection.commit()
        return
    if json.loads(row[0]) != identity:
        raise ValueError("Prepared Step 3 run already uses a different model or prompt identity")


def _record_success(
    connection: sqlite3.Connection,
    sample_id: str,
    annotation: IndependentAnnotation,
    response: BatchAnnotationResponse,
    *,
    elapsed_seconds: float,
    usage: dict[str, Any],
    raw_response: str,
) -> None:
    connection.execute(
        """
        UPDATE tasks SET
          status='succeeded',requested_model=?,actual_model=?,prompt_version=?,
          prompt_sha256=?,schema_sha256=?,response_id=?,annotation_json=?,raw_response=?,
          usage_json=?,error=NULL,elapsed_seconds=elapsed_seconds+?,updated_at=?,completed_at=?
        WHERE sample_id=?
        """,
        (
            response.requested_model,
            response.actual_model,
            response.prompt_version,
            response.prompt_sha256,
            response.schema_sha256,
            response.response_id,
            annotation.model_dump_json(),
            raw_response,
            json.dumps(usage, ensure_ascii=False, separators=(",", ":")),
            elapsed_seconds,
            _now(),
            _now(),
            sample_id,
        ),
    )


def _record_failure(connection: sqlite3.Connection, sample_id: str, error: Exception) -> None:
    connection.execute(
        "UPDATE tasks SET status='failed',error=?,updated_at=? WHERE sample_id=?",
        (f"{type(error).__name__}: {error}", _now(), sample_id),
    )


def _progress(
    connection: sqlite3.Connection,
    client: CodexAnnotationClient,
    *,
    batch_size: int,
    started_at: str,
    elapsed: float,
) -> dict[str, Any]:
    counts = Counter(dict(connection.execute("SELECT status,COUNT(*) FROM tasks GROUP BY status")))
    total = sum(counts.values())
    completed = counts["succeeded"]
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    request_seconds = 0.0
    for row in connection.execute(
        "SELECT usage_json,elapsed_seconds FROM tasks WHERE status='succeeded'"
    ):
        request_seconds += float(row["elapsed_seconds"])
        values = json.loads(row["usage_json"] or "{}")
        for key in usage:
            usage[key] += int(values.get(key, 0) or 0)
    average = request_seconds / counts["succeeded"] if counts["succeeded"] else 0
    pending = total - completed
    eta = pending * average if average else None
    return {
        "schema_version": "2.2.0",
        "database": str(connection.execute("PRAGMA database_list").fetchone()[2]),
        "model": client.model,
        "reasoning_effort": client.reasoning_effort,
        "prompt_version": client.prompt.version,
        "total": total,
        "completed": completed,
        "succeeded": counts["succeeded"],
        "failed": counts["failed"],
        "pending": pending,
        "running": counts["running"],
        "batch_size": batch_size,
        "progress_percent": round(completed / total * 100, 4) if total else 0,
        "run_started_at": started_at,
        "run_elapsed_seconds": round(elapsed, 3),
        "cumulative_request_seconds": round(request_seconds, 3),
        "average_success_seconds": round(average, 3),
        "eta_seconds": round(eta, 3) if eta is not None else None,
        "usage": usage,
        "updated_at": _now(),
    }


def _completed_rows(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in connection.execute("SELECT * FROM tasks ORDER BY sample_id"):
        payload = json.loads(row["payload_json"])
        rows.append(
            {
                "sample_id": row["sample_id"],
                "dataset_id": row["dataset_id"],
                "application_year": row["application_year"],
                "patent_id": row["patent_id"],
                **{
                    key: payload.get(key, "")
                    for key in ("title", "abstract", "claim", "ipc", "main_ipc")
                },
                "annotation": json.loads(row["annotation_json"]),
            }
        )
    return rows


def _has_retryable(connection: sqlite3.Connection, max_attempts: int) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM tasks WHERE status='failed' AND attempts < ? LIMIT 1",
            (max_attempts,),
        ).fetchone()
        is not None
    )


def _runner_active(database: Path) -> bool:
    lock_path = database.with_name(database.name + ".run.lock")
    if not lock_path.is_file():
        return False
    with lock_path.open("r+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    lock_path.unlink(missing_ok=True)
    return False


@contextmanager
def _exclusive_lock(database: Path) -> Iterator[None]:
    lock_path = database.with_name(database.name + ".run.lock")
    owns_lock = False
    try:
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"A Step 3 runner is already active for {database}") from None
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
    return connection


def _now() -> str:
    return datetime.now(UTC).isoformat()
