"""Resumable concurrent runner and exporter for Step 3 model simulation."""

from __future__ import annotations

import fcntl
import json
import sqlite3
import time
from collections import Counter
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pipeline.common.io import atomic_json_write
from pipeline.step3.client import AnnotationResponse, OpenAIAnnotationClient
from pipeline.step3.sampling import Step3Paths, write_provisional_dataset


def run_simulation(
    paths: Step3Paths,
    client: OpenAIAnnotationClient,
    *,
    concurrency: int = 5,
    max_attempts: int = 3,
    retry_delay_seconds: float = 2,
    stop_requested: Callable[[], bool] = lambda: False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
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
                (max_attempts, concurrency),
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
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures: dict[Future[AnnotationResponse], sqlite3.Row] = {
                    executor.submit(client.annotate, json.loads(row["payload_json"])): row
                    for row in pending
                }
                for future in as_completed(futures):
                    row = futures[future]
                    try:
                        response = future.result()
                    except Exception as error:
                        _record_failure(connection, row["sample_id"], error)
                    else:
                        _record_success(connection, row["sample_id"], response)
                    connection.commit()
                    progress = _progress(
                        connection,
                        client,
                        concurrency=concurrency,
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
            concurrency=concurrency,
            started_at=started_at,
            elapsed=time.monotonic() - started,
        )
        atomic_json_write(paths.progress, progress)
        if progress["succeeded"] == progress["total"]:
            rows = _completed_rows(connection)
            progress["split_report"] = write_provisional_dataset(
                paths,
                rows,
                split_seed="step3-split-v2.2.0",
                annotation_model=client.model,
                annotation_prompt_version=client.prompt.version,
            )
            atomic_json_write(paths.progress, progress)
        connection.close()
    return progress


def read_progress(paths: Step3Paths) -> dict[str, Any]:
    if paths.progress.is_file():
        return json.loads(paths.progress.read_text(encoding="utf-8"))
    if not paths.database.is_file():
        return {"status": "not_prepared", "database": str(paths.database)}
    connection = _connect(paths.database)
    counts = dict(connection.execute("SELECT status,COUNT(*) FROM tasks GROUP BY status"))
    connection.close()
    total = sum(counts.values())
    return {
        "total": total,
        **counts,
        "progress_percent": 0 if not total else counts.get("succeeded", 0) / total * 100,
    }


def _validate_identity(connection: sqlite3.Connection, client: OpenAIAnnotationClient) -> None:
    row = connection.execute("SELECT value FROM meta WHERE key='annotation_identity'").fetchone()
    identity = {
        "model": client.model,
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
    response: AnnotationResponse,
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
            response.annotation.model_dump_json(),
            response.raw_response,
            json.dumps(response.usage, ensure_ascii=False, separators=(",", ":")),
            response.elapsed_seconds,
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
    client: OpenAIAnnotationClient,
    *,
    concurrency: int,
    started_at: str,
    elapsed: float,
) -> dict[str, Any]:
    counts = Counter(dict(connection.execute("SELECT status,COUNT(*) FROM tasks GROUP BY status")))
    total = sum(counts.values())
    completed = counts["succeeded"] + counts["failed"]
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
    eta = pending * average / concurrency if average else None
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
        "concurrency": concurrency,
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


@contextmanager
def _exclusive_lock(database: Path) -> Iterator[None]:
    lock_path = database.with_name(database.name + ".run.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"A Step 3 runner is already active for {database}") from None
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def _now() -> str:
    return datetime.now(UTC).isoformat()
