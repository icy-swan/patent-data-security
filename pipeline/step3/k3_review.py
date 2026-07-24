"""Resumable one-patent-per-request Kimi K3 review through Ark Agent Plan."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import sqlite3
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from pipeline.common.io import atomic_json_write, sha256_file
from pipeline.step2.client import (
    _extract_output_text,
    _parse_json_object,
    _response_diagnostic,
    _usage_dict,
)
from pipeline.step2.prompt import DEFAULT_RESOURCE_DIR, load_prompt_bundle
from pipeline.step3.sampling import (
    MANUAL_REVIEW_FIELDS,
    NEGATIVE_COHORT,
    POSITIVE_COHORT,
)

DEFAULT_K3_MODEL = "kimi-k3"
AGENT_PLAN_BASE_URL = "https://ark.cn-beijing.volces.com/api/plan/v3"
K3_REVIEW_VERSION = "step3-kimi-k3-agent-plan-review-v1.2.0"
K3_PROMPT_VERSION = "step3-k3-review-with-data-security-law-v1.2.0"
K3_REVIEW_FIELDS = ("k3_review_label", "k3_reason")
K3_RESULT_FIELDS = MANUAL_REVIEW_FIELDS + K3_REVIEW_FIELDS
K3_PROMPT_PATH = Path(__file__).resolve().parent / "resources" / "k3_review_prompt.txt"


class K3Review(BaseModel):
    """The only two review fields written by Kimi K3."""

    model_config = ConfigDict(extra="forbid")

    k3_review_label: Literal["DATA_SECURITY", "OTHER"]
    k3_reason: str = Field(min_length=5, max_length=2_000)


@dataclass(frozen=True)
class K3ReviewResponse:
    review: K3Review
    response_id: str
    requested_model: str
    actual_model: str
    prompt_version: str
    prompt_sha256: str
    law_sha256: str
    schema_sha256: str
    elapsed_seconds: float
    usage: dict[str, Any]
    raw_text: str


@dataclass(frozen=True)
class K3ReviewPaths:
    root: Path
    database: Path
    manifest: Path
    progress: Path
    result: Path
    pid: Path
    log: Path


class K3ReviewOutputError(ValueError):
    """A structured-output failure retaining response metadata for audit."""

    def __init__(
        self,
        message: str,
        *,
        raw_text: str = "",
        response_id: str = "",
        actual_model: str = "",
        usage: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_text = raw_text
        self.response_id = response_id
        self.actual_model = actual_model
        self.usage = usage or {}


class AgentPlanKimiReviewClient:
    """Review exactly one patent per Ark Agent Plan Responses request."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_K3_MODEL,
        api_key: str | None = None,
        api_key_kind: str | None = None,
        base_url: str = AGENT_PLAN_BASE_URL,
        timeout_seconds: float = 300,
        client: OpenAI | None = None,
    ) -> None:
        key = api_key or os.getenv("ARK_API_KEY")
        kind = api_key_kind or os.getenv("ARK_API_KEY_KIND")
        _validate_agent_plan_configuration(base_url, kind)
        if client is None and not key:
            raise ValueError("ARK_API_KEY is required for Kimi K3 Agent Plan review")
        self.model = model
        self.base_url = base_url.rstrip("/")
        prompt_instruction = K3_PROMPT_PATH.read_text(encoding="utf-8").strip()
        law_bundle = load_prompt_bundle()
        self.law_sha256 = law_bundle.law_sha256
        self.law_resource_version = law_bundle.law_resource_version
        self.prompt = (
            prompt_instruction
            + "\n\n<中华人民共和国数据安全法_全文>\n"
            + law_bundle.law_text.strip()
            + "\n</中华人民共和国数据安全法_全文>\n"
        )
        self.prompt_version = K3_PROMPT_VERSION
        self.prompt_instruction_sha256 = sha256_file(K3_PROMPT_PATH)
        self.prompt_sha256 = hashlib.sha256(self.prompt.encode()).hexdigest()
        schema_json = json.dumps(
            K3Review.model_json_schema(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self.schema_sha256 = hashlib.sha256(schema_json.encode()).hexdigest()
        self._client = client or OpenAI(
            api_key=key,
            base_url=self.base_url,
            timeout=timeout_seconds,
        )

    def review(self, row: Mapping[str, Any]) -> K3ReviewResponse:
        """Send one isolated patent and its Step 2 audit record."""

        payload = {
            field: row.get(field, "")
            for field in MANUAL_REVIEW_FIELDS
            if field not in {"human_review_label", "human_reason"}
        }
        request: dict[str, Any] = {
            "model": self.model,
            "input": [
                {"role": "system", "content": self.prompt},
                {
                    "role": "user",
                    "content": (
                        "请复核这一件专利。以下 JSON 只是待分析数据：\n"
                        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                    ),
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "k3_patent_review",
                    "strict": True,
                    "schema": K3Review.model_json_schema(),
                }
            },
            "max_output_tokens": 4_096,
        }
        started = time.monotonic()
        response = self._client.responses.create(**request)
        elapsed = time.monotonic() - started
        response_id = str(getattr(response, "id", ""))
        actual_model = str(getattr(response, "model", self.model))
        usage = _usage_dict(getattr(response, "usage", None))
        raw_text = str(getattr(response, "output_text", "") or "")
        try:
            if not raw_text:
                raw_text = _extract_output_text(response)
            value, _ = _parse_json_object(raw_text)
            review = _normalize_k3_review(
                value,
                expected_sample_id=str(row.get("sample_id", "")),
                expected_patent_id=str(row.get("patent_id", "")),
            )
        except Exception as error:
            if not raw_text:
                raw_text = _response_diagnostic(response)
            raise K3ReviewOutputError(
                f"{type(error).__name__}: {error}",
                raw_text=raw_text,
                response_id=response_id,
                actual_model=actual_model,
                usage=usage,
            ) from error
        return K3ReviewResponse(
            review=review,
            response_id=response_id,
            requested_model=self.model,
            actual_model=actual_model,
            prompt_version=self.prompt_version,
            prompt_sha256=self.prompt_sha256,
            law_sha256=self.law_sha256,
            schema_sha256=self.schema_sha256,
            elapsed_seconds=elapsed,
            usage=usage,
            raw_text=raw_text,
        )


def k3_review_paths(output_dir: str | Path) -> K3ReviewPaths:
    root = Path(output_dir).resolve()
    return K3ReviewPaths(
        root=root,
        database=root / "k3_tasks.sqlite3",
        manifest=root / "k3_manifest.json",
        progress=root / "k3_progress.json",
        result=root / "k3_result.csv",
        pid=root / "k3_runner.pid",
        log=root / "k3_runner.log",
    )


def prepare_k3_review(
    output_dir: str | Path,
    *,
    rebuild: bool = False,
) -> tuple[K3ReviewPaths, dict[str, Any]]:
    """Freeze both Step 3 review queues into a K3-only task database."""

    paths = k3_review_paths(output_dir)
    source_paths = (
        paths.root / "need_manual_review_positive.csv",
        paths.root / "need_manual_review_negative.csv",
    )
    missing = [path for path in source_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Step 3 review inputs: {missing}")
    if paths.database.exists() and not rebuild:
        raise FileExistsError(
            f"K3 review is already prepared at {paths.database}; use --rebuild to replace it"
        )
    if _runner_active(paths.database):
        raise RuntimeError("Cannot rebuild K3 review while its runner is active")

    manifest_path = paths.root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing Step 3 manifest: {manifest_path}")
    step3_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_total = int(step3_manifest.get("target_size", 0))
    if expected_total <= 0:
        raise ValueError("Step 3 manifest has no positive target_size")

    frozen_rows: list[dict[str, str]] = []
    source_records: list[dict[str, Any]] = []
    for source_path, expected_cohort in zip(
        source_paths,
        (POSITIVE_COHORT, NEGATIVE_COHORT),
        strict=True,
    ):
        fields, rows = _read_review_csv(source_path)
        if tuple(fields) != MANUAL_REVIEW_FIELDS:
            raise ValueError(
                f"{source_path} fields differ from the frozen Step 3 review contract"
            )
        wrong_cohort = [
            row["sample_id"]
            for row in rows
            if row["sample_cohort"] != expected_cohort
        ]
        if wrong_cohort:
            raise ValueError(
                f"{source_path} contains rows outside {expected_cohort}: {wrong_cohort[:5]}"
            )
        if any(row["human_review_label"] or row["human_reason"] for row in rows):
            raise ValueError(
                f"{source_path} already contains human review values; K3 input must be frozen"
            )
        frozen_rows.extend(rows)
        source_records.append(
            {
                "path": str(source_path),
                "sha256": sha256_file(source_path),
                "records": len(rows),
                "sample_cohort": expected_cohort,
            }
        )
    if len(frozen_rows) != expected_total:
        raise ValueError(
            f"K3 review requires all {expected_total} Step 3 rows, found {len(frozen_rows)}"
        )
    sample_ids = [row["sample_id"] for row in frozen_rows]
    patent_ids = [row["patent_id"] for row in frozen_rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("K3 review input contains duplicate sample_id values")
    if len(patent_ids) != len(set(patent_ids)):
        raise ValueError("K3 review input contains duplicate patent_id values")

    paths.root.mkdir(parents=True, exist_ok=True)
    temporary = paths.database.with_suffix(paths.database.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=FULL;
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE tasks (
              sample_id TEXT PRIMARY KEY,
              patent_id TEXT NOT NULL UNIQUE,
              source_order INTEGER NOT NULL UNIQUE,
              input_sha256 TEXT NOT NULL,
              payload_json TEXT NOT NULL,
              status TEXT NOT NULL CHECK(status IN ('pending','running','succeeded','failed')),
              attempts INTEGER NOT NULL DEFAULT 0,
              requested_model TEXT,
              actual_model TEXT,
              prompt_version TEXT,
              prompt_sha256 TEXT,
              law_sha256 TEXT,
              schema_sha256 TEXT,
              response_id TEXT,
              review_json TEXT,
              raw_response TEXT,
              usage_json TEXT,
              error TEXT,
              elapsed_seconds REAL NOT NULL DEFAULT 0,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              completed_at TEXT
            );
            CREATE INDEX idx_k3_tasks_status ON tasks(status, source_order);
            """
        )
        now = _now()
        task_rows = []
        for source_order, row in enumerate(frozen_rows):
            payload_json = json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            task_rows.append(
                (
                    row["sample_id"],
                    row["patent_id"],
                    source_order,
                    hashlib.sha256(payload_json.encode()).hexdigest(),
                    payload_json,
                    now,
                    now,
                )
            )
        connection.executemany(
            """
            INSERT INTO tasks (
              sample_id,patent_id,source_order,input_sha256,payload_json,
              status,created_at,updated_at
            ) VALUES (?,?,?,?,?,'pending',?,?)
            """,
            task_rows,
        )
        connection.commit()
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("K3 SQLite integrity check failed during preparation")
    finally:
        connection.close()
    for suffix in ("-wal", "-shm"):
        temporary.with_name(temporary.name + suffix).unlink(missing_ok=True)
        paths.database.with_name(paths.database.name + suffix).unlink(missing_ok=True)
    os.replace(temporary, paths.database)

    for path in (paths.progress, paths.result, paths.pid, paths.log):
        path.unlink(missing_ok=True)
    law_bundle = load_prompt_bundle()
    manifest = {
        "review_version": K3_REVIEW_VERSION,
        "model": DEFAULT_K3_MODEL,
        "request_granularity": "one_patent_per_request",
        "agent_plan_base_url": AGENT_PLAN_BASE_URL,
        "agent_plan_key_kind": "agent-plan",
        "fixed_prompt": {
            "prompt_version": K3_PROMPT_VERSION,
            "instruction_path": str(K3_PROMPT_PATH),
            "instruction_sha256": sha256_file(K3_PROMPT_PATH),
            "law_path": str(DEFAULT_RESOURCE_DIR / "data_security_law.txt"),
            "law_sha256": law_bundle.law_sha256,
            "law_resource_version": law_bundle.law_resource_version,
            "composition": "review_instruction + full_data_security_law",
        },
        "records": len(frozen_rows),
        "sources": source_records,
        "source_step3_manifest": str(manifest_path),
        "source_step3_manifest_sha256": sha256_file(manifest_path),
        "database": str(paths.database),
        "output": str(paths.result),
        "output_fields": list(K3_RESULT_FIELDS),
        "gold_status": "provisional_model_review_not_human_gold",
        "eligible_for_training_or_final_evaluation": False,
        "prepared_at": _now(),
    }
    atomic_json_write(paths.manifest, manifest)
    return paths, manifest


def run_k3_reviews(
    paths: K3ReviewPaths,
    client: AgentPlanKimiReviewClient,
    *,
    max_attempts: int = 3,
    retry_delay_seconds: float = 2,
    concurrency: int = 5,
    stop_requested: Callable[[], bool] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run isolated K3 requests with resumable SQLite persistence."""

    if max_attempts < 1 or concurrency < 1 or retry_delay_seconds < 0:
        raise ValueError("Invalid K3 retry or concurrency settings")
    if not paths.database.is_file() or not paths.manifest.is_file():
        raise FileNotFoundError("Run prepare-k3 before run-k3")
    _validate_frozen_sources(paths)
    stop_requested = stop_requested or (lambda: False)
    started = time.monotonic()
    started_at = _now()
    with _exclusive_lock(paths.database):
        connection = _connect(paths.database)
        try:
            _validate_runtime_identity(connection, client)
            connection.execute("UPDATE tasks SET status='pending' WHERE status='running'")
            connection.commit()
            progress = _progress(
                connection,
                paths,
                client,
                concurrency=concurrency,
                started=started,
                started_at=started_at,
            )
            atomic_json_write(paths.progress, progress)
            if progress_callback:
                progress_callback(progress)
            _run_loop(
                connection,
                paths,
                client,
                max_attempts=max_attempts,
                retry_delay_seconds=retry_delay_seconds,
                concurrency=concurrency,
                stop_requested=stop_requested,
                progress_callback=progress_callback,
                started=started,
                started_at=started_at,
            )
            progress = _progress(
                connection,
                paths,
                client,
                concurrency=concurrency,
                started=started,
                started_at=started_at,
            )
            progress["stopped_by_request"] = bool(stop_requested())
            if progress["succeeded"] == progress["total"]:
                progress["output"] = export_k3_results(paths, connection=connection)
            atomic_json_write(paths.progress, progress)
            return progress
        finally:
            connection.close()


def export_k3_results(
    paths: K3ReviewPaths,
    *,
    connection: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """Atomically write the complete 10,000-row K3 review CSV."""

    own_connection = connection is None
    connection = connection or _connect(paths.database)
    try:
        counts = dict(
            connection.execute("SELECT status,COUNT(*) FROM tasks GROUP BY status")
        )
        total = sum(counts.values())
        if counts.get("succeeded", 0) != total:
            raise ValueError(
                "k3_result.csv is written only after every task succeeds; "
                f"status_counts={counts}"
            )
        temporary = paths.result.with_suffix(paths.result.suffix + ".partial")
        label_counts: Counter[str] = Counter()
        with temporary.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=K3_RESULT_FIELDS)
            writer.writeheader()
            for task in connection.execute("SELECT * FROM tasks ORDER BY source_order"):
                payload = json.loads(task["payload_json"])
                review = K3Review.model_validate_json(task["review_json"])
                label_counts[review.k3_review_label] += 1
                writer.writerow(
                    {
                        **{field: payload.get(field, "") for field in MANUAL_REVIEW_FIELDS},
                        **review.model_dump(),
                    }
                )
        os.replace(temporary, paths.result)
        return {
            "path": str(paths.result),
            "records": total,
            "fields": list(K3_RESULT_FIELDS),
            "label_counts": dict(sorted(label_counts.items())),
            "sha256": sha256_file(paths.result),
            "gold_status": "provisional_model_review_not_human_gold",
            "written_at": _now(),
        }
    finally:
        if own_connection:
            connection.close()


def read_k3_progress(paths: K3ReviewPaths) -> dict[str, Any]:
    if not paths.database.is_file():
        return {
            "status": "not_prepared",
            "database": str(paths.database),
            "result": str(paths.result),
        }
    runner_active = _runner_active(paths.database)
    immutable = "" if runner_active else "&immutable=1"
    connection = sqlite3.connect(
        f"file:{paths.database}?mode=ro{immutable}",
        uri=True,
    )
    try:
        counts = dict(connection.execute("SELECT status,COUNT(*) FROM tasks GROUP BY status"))
    finally:
        connection.close()
    total = sum(counts.values())
    succeeded = counts.get("succeeded", 0)
    progress = {}
    if paths.progress.is_file():
        progress = json.loads(paths.progress.read_text(encoding="utf-8"))
    progress.update(
        {
            "total": total,
            "succeeded": succeeded,
            "failed": counts.get("failed", 0),
            "queued": counts.get("pending", 0),
            "running": counts.get("running", 0),
            "pending": total - succeeded,
            "progress_percent": round(succeeded / total * 100, 4) if total else 0,
            "runner_active": runner_active,
            "result_exists": paths.result.is_file(),
            "result": str(paths.result),
        }
    )
    return progress


def reset_failed_k3_reviews(paths: K3ReviewPaths) -> int:
    """Reset only terminal failed rows; never touch succeeded responses."""

    if not paths.database.is_file():
        raise FileNotFoundError(f"K3 review is not prepared: {paths.database}")
    if _runner_active(paths.database):
        raise RuntimeError("Cannot reset failed K3 tasks while the runner is active")
    connection = _connect(paths.database)
    try:
        count = int(
            connection.execute(
                "SELECT COUNT(*) FROM tasks WHERE status='failed'"
            ).fetchone()[0]
        )
        connection.execute(
            """
            UPDATE tasks SET status='pending',attempts=0,error=NULL,updated_at=?,
              completed_at=NULL WHERE status='failed'
            """,
            (_now(),),
        )
        connection.commit()
    finally:
        connection.close()
    paths.result.unlink(missing_ok=True)
    return count


def _run_loop(
    connection: sqlite3.Connection,
    paths: K3ReviewPaths,
    client: AgentPlanKimiReviewClient,
    *,
    max_attempts: int,
    retry_delay_seconds: float,
    concurrency: int,
    stop_requested: Callable[[], bool],
    progress_callback: Callable[[dict[str, Any]], None] | None,
    started: float,
    started_at: str,
) -> None:
    def review(task: sqlite3.Row) -> K3ReviewResponse | Exception:
        try:
            return client.review(json.loads(task["payload_json"]))
        except Exception as error:  # noqa: BLE001 - persisted for retry and audit
            return error

    in_flight: dict[Future[Any], tuple[sqlite3.Row, int, float]] = {}
    retry_not_before: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        while True:
            now = time.monotonic()
            retry_not_before = {
                sample_id: deadline
                for sample_id, deadline in retry_not_before.items()
                if deadline > now
            }
            while not stop_requested() and len(in_flight) < concurrency:
                claimed = _claim_next(
                    connection,
                    client.model,
                    max_attempts,
                    excluded_sample_ids=set(retry_not_before),
                )
                if claimed is None:
                    break
                task, attempts = claimed
                in_flight[pool.submit(review, task)] = (
                    task,
                    attempts,
                    time.monotonic(),
                )
            if not in_flight:
                if stop_requested() or not retry_not_before:
                    break
                time.sleep(
                    min(
                        max(0.0, min(retry_not_before.values()) - time.monotonic()),
                        0.1,
                    )
                )
                continue
            timeout = None
            if retry_not_before:
                timeout = max(
                    0.0,
                    min(retry_not_before.values()) - time.monotonic(),
                )
            completed, _ = wait(
                in_flight,
                timeout=timeout,
                return_when=FIRST_COMPLETED,
            )
            if not completed:
                continue
            for future in completed:
                task, attempts, request_started = in_flight.pop(future)
                outcome = future.result()
                elapsed = (
                    outcome.elapsed_seconds
                    if isinstance(outcome, K3ReviewResponse)
                    else time.monotonic() - request_started
                )
                status = _persist_outcome(
                    connection,
                    task["sample_id"],
                    attempts,
                    max_attempts,
                    outcome,
                    elapsed,
                )
                if status == "pending" and retry_delay_seconds:
                    retry_not_before[task["sample_id"]] = (
                        time.monotonic() + retry_delay_seconds
                    )
            progress = _progress(
                connection,
                paths,
                client,
                concurrency=concurrency,
                started=started,
                started_at=started_at,
            )
            atomic_json_write(paths.progress, progress)
            if progress_callback:
                progress_callback(progress)


def _claim_next(
    connection: sqlite3.Connection,
    model: str,
    max_attempts: int,
    *,
    excluded_sample_ids: set[str],
) -> tuple[sqlite3.Row, int] | None:
    parameters: list[Any] = [max_attempts]
    exclusion = ""
    if excluded_sample_ids:
        placeholders = ",".join("?" for _ in excluded_sample_ids)
        exclusion = f" AND sample_id NOT IN ({placeholders})"
        parameters.extend(sorted(excluded_sample_ids))
    task = connection.execute(
        f"""
        SELECT * FROM tasks
        WHERE status='pending' AND attempts < ? {exclusion}
        ORDER BY source_order
        LIMIT 1
        """,
        parameters,
    ).fetchone()
    if task is None:
        return None
    attempts = int(task["attempts"]) + 1
    connection.execute(
        """
        UPDATE tasks SET status='running',attempts=?,requested_model=?,updated_at=?
        WHERE sample_id=?
        """,
        (attempts, model, _now(), task["sample_id"]),
    )
    connection.commit()
    return task, attempts


def _persist_outcome(
    connection: sqlite3.Connection,
    sample_id: str,
    attempts: int,
    max_attempts: int,
    outcome: K3ReviewResponse | Exception,
    elapsed: float,
) -> str:
    if isinstance(outcome, Exception):
        status = "failed" if attempts >= max_attempts else "pending"
        response_id = ""
        actual_model = ""
        raw_text = ""
        usage: dict[str, Any] = {}
        if isinstance(outcome, K3ReviewOutputError):
            response_id = outcome.response_id
            actual_model = outcome.actual_model
            raw_text = outcome.raw_text
            usage = outcome.usage
        connection.execute(
            """
            UPDATE tasks SET status=?,actual_model=COALESCE(NULLIF(?,''),actual_model),
              response_id=COALESCE(NULLIF(?,''),response_id),raw_response=?,usage_json=?,
              error=?,elapsed_seconds=elapsed_seconds+?,updated_at=?,completed_at=?
            WHERE sample_id=?
            """,
            (
                status,
                actual_model,
                response_id,
                raw_text,
                json.dumps(usage, ensure_ascii=False, separators=(",", ":")),
                f"{type(outcome).__name__}: {outcome}"[:2_000],
                elapsed,
                _now(),
                _now() if status == "failed" else None,
                sample_id,
            ),
        )
    else:
        connection.execute(
            """
            UPDATE tasks SET status='succeeded',actual_model=?,prompt_version=?,
              prompt_sha256=?,law_sha256=?,schema_sha256=?,response_id=?,review_json=?,
              raw_response=?,usage_json=?,error=NULL,
              elapsed_seconds=elapsed_seconds+?,updated_at=?,completed_at=?
            WHERE sample_id=?
            """,
            (
                outcome.actual_model,
                outcome.prompt_version,
                outcome.prompt_sha256,
                outcome.law_sha256,
                outcome.schema_sha256,
                outcome.response_id,
                outcome.review.model_dump_json(),
                outcome.raw_text,
                json.dumps(outcome.usage, ensure_ascii=False, separators=(",", ":")),
                elapsed,
                _now(),
                _now(),
                sample_id,
            ),
        )
        status = "succeeded"
    connection.commit()
    return status


def _progress(
    connection: sqlite3.Connection,
    paths: K3ReviewPaths,
    client: AgentPlanKimiReviewClient,
    *,
    concurrency: int,
    started: float,
    started_at: str,
) -> dict[str, Any]:
    counts = dict(connection.execute("SELECT status,COUNT(*) FROM tasks GROUP BY status"))
    total = sum(counts.values())
    succeeded = counts.get("succeeded", 0)
    failed = counts.get("failed", 0)
    request_seconds = float(
        connection.execute(
            "SELECT COALESCE(SUM(elapsed_seconds),0) FROM tasks WHERE status='succeeded'"
        ).fetchone()[0]
    )
    average = request_seconds / succeeded if succeeded else 0
    unfinished = total - succeeded - failed
    eta = unfinished * average / concurrency if average else None
    usage: Counter[str] = Counter()
    for (usage_json,) in connection.execute(
        "SELECT usage_json FROM tasks WHERE status='succeeded'"
    ):
        for key, value in json.loads(usage_json or "{}").items():
            if isinstance(value, int | float):
                usage[key] += value
    return {
        "review_version": K3_REVIEW_VERSION,
        "database": str(paths.database),
        "result": str(paths.result),
        "model": client.model,
        "base_url": client.base_url,
        "request_granularity": "one_patent_per_request",
        "total": total,
        "succeeded": succeeded,
        "failed": failed,
        "queued": counts.get("pending", 0),
        "running": counts.get("running", 0),
        "pending": total - succeeded,
        "concurrency": concurrency,
        "progress_percent": round(succeeded / total * 100, 4) if total else 0,
        "run_started_at": started_at,
        "run_elapsed_seconds": round(time.monotonic() - started, 3),
        "cumulative_request_seconds": round(request_seconds, 3),
        "average_success_seconds": round(average, 3),
        "eta_seconds": round(eta, 3) if eta is not None else None,
        "usage": dict(sorted(usage.items())),
        "updated_at": _now(),
    }


def _validate_runtime_identity(
    connection: sqlite3.Connection,
    client: AgentPlanKimiReviewClient,
) -> None:
    identity = {
        "model": client.model,
        "base_url": client.base_url,
        "prompt_version": client.prompt_version,
        "prompt_sha256": client.prompt_sha256,
        "law_sha256": client.law_sha256,
        "schema_sha256": client.schema_sha256,
        "request_granularity": "one_patent_per_request",
    }
    row = connection.execute(
        "SELECT value FROM meta WHERE key='review_identity'"
    ).fetchone()
    if row is None:
        connection.execute(
            "INSERT INTO meta(key,value) VALUES('review_identity',?)",
            (json.dumps(identity, ensure_ascii=False, sort_keys=True),),
        )
        connection.commit()
        return
    if json.loads(row[0]) != identity:
        raise ValueError("Prepared K3 review already uses a different model or prompt identity")


def _validate_frozen_sources(paths: K3ReviewPaths) -> None:
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    for source in manifest["sources"]:
        path = Path(source["path"])
        if not path.is_file() or sha256_file(path) != source["sha256"]:
            raise ValueError(f"Frozen K3 input changed after preparation: {path}")


def _validate_agent_plan_configuration(base_url: str, api_key_kind: str | None) -> None:
    normalized = base_url.rstrip("/")
    if normalized != AGENT_PLAN_BASE_URL:
        raise ValueError(
            "Kimi K3 review must use the Ark Agent Plan base URL "
            f"{AGENT_PLAN_BASE_URL}, got {normalized}"
        )
    if api_key_kind != "agent-plan":
        raise ValueError(
            "Kimi K3 review requires ARK_API_KEY_KIND=agent-plan; "
            f"got {api_key_kind!r}"
        )


def _normalize_k3_review(
    value: Mapping[str, Any],
    *,
    expected_sample_id: str,
    expected_patent_id: str,
) -> K3Review:
    """Normalize Kimi's observed aliases while preserving the two-column contract."""

    returned_sample_id = str(value.get("sample_id", "") or "")
    returned_patent_id = str(value.get("patent_id", "") or "")
    if returned_sample_id and returned_sample_id != expected_sample_id:
        raise ValueError(
            "K3 returned a different sample_id: "
            f"expected={expected_sample_id!r}, actual={returned_sample_id!r}"
        )
    if returned_patent_id and returned_patent_id != expected_patent_id:
        raise ValueError(
            "K3 returned a different patent_id: "
            f"expected={expected_patent_id!r}, actual={returned_patent_id!r}"
        )

    label_keys = (
        "k3_review_label",
        "k3_final_label",
        "k3_label",
        "step3_label",
        "final_label",
        "label",
    )
    labels = {
        str(value[key]).strip().upper()
        for key in label_keys
        if value.get(key) not in (None, "")
    }
    invalid_labels = sorted(labels - {"DATA_SECURITY", "OTHER"})
    if invalid_labels:
        raise ValueError(f"K3 returned invalid label values: {invalid_labels}")
    if len(labels) != 1:
        raise ValueError(f"K3 must return one unambiguous review label, got {sorted(labels)}")

    reason = next(
        (
            str(value[key]).strip()
            for key in ("k3_reason", "review_reason", "reason")
            if str(value.get(key, "") or "").strip()
        ),
        "",
    )
    return K3Review(
        k3_review_label=next(iter(labels)),
        k3_reason=reason,
    )


def _read_review_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or ()), list(reader)


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


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
                raise RuntimeError(
                    f"A K3 review runner is already active for {database}"
                ) from None
            owns_lock = True
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    finally:
        if owns_lock:
            lock_path.unlink(missing_ok=True)


def _now() -> str:
    return datetime.now(UTC).isoformat()
