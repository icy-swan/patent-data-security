"""Stream, match, deduplicate and export Step 1 results."""

from __future__ import annotations

import csv
import hashlib
import json
import multiprocessing as mp
import os
import sqlite3
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pipeline.common.datasets import dataset_id
from pipeline.common.io import atomic_json_write
from pipeline.common.records import PatentRecord, iter_patent_records
from pipeline.step1.matcher import KeywordMatcher, normalize_text
from pipeline.step1.taxonomy import KeywordBundle, load_keyword_bundle

OUTPUT_FIELDS = (
    "dataset_id",
    "patent_id",
    "synthetic_id",
    "source_row_number",
    "association_count",
    "application_year",
    "stock_code",
    "company_name",
    "industry",
    "market",
    "title",
    "ipc",
    "main_ipc",
    "route",
    "selected_for_step2",
    "selection_group",
    "selection_probability",
    "sample_weight",
    "sample_seed",
    "valid_hit_count",
    "descriptive_hit_count",
    "technical_hit_count",
    "matched_concepts",
    "keyword_hits",
    "context_hits",
    "diagnostic_hits",
    "ipc_audit_hits",
    "keyword_version",
    "source_manifest_version",
    "methodology_version",
    "processed_at",
)


@dataclass(frozen=True)
class Step1Outputs:
    result: Path
    manifest: Path


@dataclass(frozen=True)
class _ProcessedRecord:
    patent_id: str
    route: str
    route_rank: int
    quality_score: int
    source_row_number: int
    payload_json: str


_WORKER_MATCHER: KeywordMatcher | None = None
_WORKER_BUNDLE: KeywordBundle | None = None


def run_step1(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    bundle: KeywordBundle | None = None,
    encoding: str = "utf-8-sig",
    workers: int = 1,
    worker_chunksize: int = 100,
    sqlite_batch_size: int = 5_000,
    progress_every: int = 100_000,
    limit: int | None = None,
    e_sample_rate: float = 0.02,
    e_sample_seed: str = "step1-e-random-v2",
    overwrite: bool = False,
) -> Step1Outputs:
    """Run local Step 1 and write one row per unique patent.

    Input is streamed. A compact SQLite table resolves duplicate company-patent
    associations without retaining the 2GB-scale source CSV in memory.
    """

    if workers < 1:
        raise ValueError("workers must be at least 1")
    if worker_chunksize < 1 or sqlite_batch_size < 1:
        raise ValueError("batch sizes must be at least 1")
    if not 0 < e_sample_rate <= 1:
        raise ValueError("e_sample_rate must be in (0, 1]")

    source = Path(input_path).resolve()
    dataset = dataset_id(source)
    destination = Path(output_dir).resolve() / dataset
    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination / "result.csv"
    manifest_path = destination / "manifest.json"
    staging_path = destination / ".tasks.partial.sqlite3"
    lock_path = destination / ".step1.lock"

    final_paths = (result_path, manifest_path)
    if not overwrite and any(path.exists() for path in final_paths):
        raise FileExistsError("Step 1 outputs already exist; pass --overwrite to replace them")
    if overwrite:
        for path in final_paths:
            path.unlink(missing_ok=True)
    staging_path.unlink(missing_ok=True)

    resources = bundle or load_keyword_bundle()
    started_at = datetime.now(UTC)
    started = time.monotonic()
    raw_rows = 0

    with _exclusive_lock(lock_path):
        connection = _open_staging_database(staging_path)
        pool: Any = None
        try:
            records = iter_patent_records(source, encoding=encoding, limit=limit)
            if workers > 1:
                method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
                pool = mp.get_context(method).Pool(
                    processes=workers,
                    initializer=_initialize_worker,
                    initargs=(resources,),
                )
                processed_records = pool.imap(
                    _process_record_worker,
                    records,
                    chunksize=worker_chunksize,
                )
            else:
                matcher = KeywordMatcher(resources)
                processed_records = (
                    _process_record(record, matcher, resources) for record in records
                )

            batch: list[tuple[Any, ...]] = []
            for processed in processed_records:
                raw_rows += 1
                batch.append(
                    (
                        processed.patent_id,
                        processed.route,
                        processed.route_rank,
                        processed.quality_score,
                        processed.source_row_number,
                        processed.payload_json,
                    )
                )
                if len(batch) >= sqlite_batch_size:
                    _upsert_batch(connection, batch)
                    batch.clear()
                if progress_every and raw_rows % progress_every == 0:
                    elapsed = max(time.monotonic() - started, 0.001)
                    unique_count = connection.execute("SELECT COUNT(*) FROM results").fetchone()[0]
                    print(
                        f"step1_rows={raw_rows:,} unique={unique_count:,} "
                        f"rate={raw_rows / elapsed:,.0f}/s",
                        file=sys.stderr,
                        flush=True,
                    )
            if batch:
                _upsert_batch(connection, batch)
            if pool is not None:
                pool.close()
                pool.join()
                pool = None

            export_stats = _export_results(
                connection,
                result_path,
                dataset=dataset,
                processed_at=started_at.isoformat(),
                e_sample_rate=e_sample_rate,
                e_sample_seed=e_sample_seed,
            )
            elapsed = time.monotonic() - started
            summary = {
                "schema_version": "2.0.0",
                "step": "step1_keyword_context_routing",
                "dataset_id": dataset,
                "input_path": str(source),
                "input_size_bytes": source.stat().st_size,
                "keyword_version": resources.keyword_version,
                "methodology_version": resources.methodology_version,
                "source_manifest_version": resources.source_manifest_version,
                "resource_sha256": resources.hashes,
                "validation_status": resources.validation_protocol["status"],
                "matching": {
                    "fields": resources.keywords["matching"]["fields"],
                    "context_window_chars": resources.keywords["matching"][
                        "context_window_chars"
                    ],
                    "ipc_changes_route": False,
                },
                "sampling": {
                    "e_sample_rate": e_sample_rate,
                    "e_sample_seed": e_sample_seed,
                    "unit": "unique_patent",
                },
                "workers": workers,
                "stats": {
                    "input_rows": raw_rows,
                    **export_stats,
                },
                "elapsed_seconds": round(elapsed, 3),
                "rows_per_second": round(raw_rows / max(elapsed, 0.001), 3),
                "outputs": {"result": str(result_path)},
                "llm_requests_executed": 0,
            }
            atomic_json_write(manifest_path, summary)
        except BaseException:
            if pool is not None:
                pool.terminate()
                pool.join()
            result_path.unlink(missing_ok=True)
            raise
        finally:
            connection.close()
            for path in (
                staging_path,
                staging_path.with_name(staging_path.name + "-wal"),
                staging_path.with_name(staging_path.name + "-shm"),
                result_path.with_suffix(result_path.suffix + ".partial"),
            ):
                path.unlink(missing_ok=True)

        return Step1Outputs(
            result=result_path,
            manifest=manifest_path,
        )


def _initialize_worker(bundle: KeywordBundle) -> None:
    global _WORKER_MATCHER, _WORKER_BUNDLE
    _WORKER_BUNDLE = bundle
    _WORKER_MATCHER = KeywordMatcher(bundle)


def _process_record_worker(record: PatentRecord) -> _ProcessedRecord:
    if _WORKER_MATCHER is None or _WORKER_BUNDLE is None:
        raise RuntimeError("Step 1 worker was not initialized")
    return _process_record(record, _WORKER_MATCHER, _WORKER_BUNDLE)


def _process_record(
    record: PatentRecord,
    matcher: KeywordMatcher,
    bundle: KeywordBundle,
) -> _ProcessedRecord:
    patent_id, synthetic_id = _patent_identity(record)
    result = matcher.match(record)
    compact = lambda value: json.dumps(  # noqa: E731
        value, ensure_ascii=False, separators=(",", ":")
    )
    payload = {
        "patent_id": patent_id,
        "synthetic_id": synthetic_id,
        "source_row_number": record.row_number,
        "application_year": record.get("application_year"),
        "stock_code": record.get("stock_code"),
        "company_name": record.get("company_name"),
        "industry": record.get("industry"),
        "market": record.get("market"),
        "title": record.get("title"),
        "ipc": record.get("ipc"),
        "main_ipc": record.get("main_ipc"),
        "route": result.route,
        "valid_hit_count": result.valid_hit_count,
        "descriptive_hit_count": result.descriptive_hit_count,
        "technical_hit_count": result.technical_hit_count,
        "matched_concepts": compact(result.matched_concepts),
        "keyword_hits": compact(result.keyword_hits_jsonable()),
        "context_hits": compact(result.context_hits_jsonable()),
        "diagnostic_hits": compact(result.diagnostics_jsonable()),
        "ipc_audit_hits": compact(result.ipc_audit_jsonable()),
        "keyword_version": bundle.keyword_version,
        "source_manifest_version": bundle.source_manifest_version,
        "methodology_version": bundle.methodology_version,
    }
    quality_score = (
        result.valid_hit_count * 1_000_000
        + bool(record.get("claim")) * 100_000
        + bool(record.get("abstract")) * 50_000
        + bool(record.get("title")) * 10_000
        + min(len(record.get("claim")), 9_999)
        + min(len(record.get("abstract")), 9_999)
    )
    return _ProcessedRecord(
        patent_id=patent_id,
        route=result.route,
        route_rank=1 if result.route == "S" else 0,
        quality_score=quality_score,
        source_row_number=record.row_number,
        payload_json=compact(payload),
    )


def _patent_identity(record: PatentRecord) -> tuple[str, bool]:
    for field in ("application_number", "publication_number", "grant_number"):
        value = record.get(field).strip()
        if value:
            return value, False
    parts = [
        normalize_text(record.get("title")),
        normalize_text(record.get("applicant")),
        normalize_text(record.get("application_date")),
    ]
    if not any(parts):
        parts.append(f"source-row-{record.row_number}")
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:32]
    return f"synthetic-{digest}", True


def _open_staging_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute(
        """
        CREATE TABLE results (
            patent_id TEXT PRIMARY KEY,
            route TEXT NOT NULL,
            route_rank INTEGER NOT NULL,
            quality_score INTEGER NOT NULL,
            source_row_number INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            association_count INTEGER NOT NULL DEFAULT 1
        ) WITHOUT ROWID
        """
    )
    return connection


def _upsert_batch(connection: sqlite3.Connection, values: list[tuple[Any, ...]]) -> None:
    prefer_new = (
        "excluded.route_rank > results.route_rank OR "
        "(excluded.route_rank = results.route_rank AND "
        "excluded.quality_score > results.quality_score)"
    )
    connection.executemany(
        f"""
        INSERT INTO results (
            patent_id, route, route_rank, quality_score,
            source_row_number, payload_json
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(patent_id) DO UPDATE SET
            association_count = results.association_count + 1,
            route = CASE WHEN {prefer_new} THEN excluded.route ELSE results.route END,
            route_rank = CASE WHEN {prefer_new}
                THEN excluded.route_rank ELSE results.route_rank END,
            quality_score = CASE WHEN {prefer_new}
                THEN excluded.quality_score ELSE results.quality_score END,
            source_row_number = CASE WHEN {prefer_new}
                THEN excluded.source_row_number ELSE results.source_row_number END,
            payload_json = CASE WHEN {prefer_new}
                THEN excluded.payload_json ELSE results.payload_json END
        """,
        values,
    )
    connection.commit()


def _export_results(
    connection: sqlite3.Connection,
    path: Path,
    *,
    dataset: str,
    processed_at: str,
    e_sample_rate: float,
    e_sample_seed: str,
) -> dict[str, Any]:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.unlink(missing_ok=True)
    route_counts = {"S": 0, "E": 0}
    selected_counts = {"S_all": 0, "E_random": 0}
    unique_patents = 0
    total_associations = 0
    with partial.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        cursor = connection.execute(
            """
            SELECT payload_json, association_count
            FROM results
            ORDER BY source_row_number, patent_id
            """
        )
        for payload_json, association_count in cursor:
            row = json.loads(payload_json)
            route = row["route"]
            if route == "S":
                selected = True
                selection_group = "S_all"
                probability = 1.0
                sample_weight: float | str = 1.0
                sample_seed = ""
            else:
                selected = _stable_sample(
                    f"{dataset}|{row['patent_id']}",
                    e_sample_rate,
                    e_sample_seed,
                )
                selection_group = "E_random" if selected else "E_not_selected"
                probability = e_sample_rate
                sample_weight = 1 / e_sample_rate if selected else ""
                sample_seed = e_sample_seed
            output = {
                "dataset_id": dataset,
                **row,
                "association_count": association_count,
                "selected_for_step2": str(selected).lower(),
                "selection_group": selection_group,
                "selection_probability": f"{probability:.12g}",
                "sample_weight": (
                    f"{sample_weight:.12g}" if isinstance(sample_weight, float) else ""
                ),
                "sample_seed": sample_seed,
                "processed_at": processed_at,
            }
            writer.writerow({field: output.get(field, "") for field in OUTPUT_FIELDS})
            unique_patents += 1
            total_associations += int(association_count)
            route_counts[route] += 1
            if selected:
                selected_counts[selection_group] += 1
    os.replace(partial, path)
    return {
        "unique_patents": unique_patents,
        "duplicate_association_rows": total_associations - unique_patents,
        "route_counts": route_counts,
        "selected_for_step2": selected_counts,
    }


def _stable_sample(identity: str, probability: float, seed: str) -> bool:
    digest = hashlib.sha256(f"{seed}|{identity}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return value < probability


@contextmanager
def _exclusive_lock(path: Path) -> Any:
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"Another Step 1 run appears active: {path}") from error
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode())
        yield
    finally:
        os.close(descriptor)
        path.unlink(missing_ok=True)
