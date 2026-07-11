"""Streaming annual routing pipeline with deterministic E sampling and checkpoints."""

from __future__ import annotations

import csv
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TextIO

from patent_data_security.records import PatentRecord, iter_patent_records
from patent_data_security.routing import PatentRouter
from patent_data_security.taxonomy import TaxonomyBundle, load_taxonomies

ROUTE_FIELDS = (
    "patent_id",
    "source_row_number",
    "application_year",
    "stock_code",
    "company_name",
    "industry",
    "market",
    "title",
    "normalized_ipc",
    "keyword_level",
    "ipc_level",
    "route_level",
    "keyword_hits",
    "ipc_hits",
    "diagnostic_hits",
    "e_stratum",
    "is_e_sample",
    "selection_probability",
    "sample_weight",
    "classification_key",
    "taxonomy_version",
    "process_status",
)


@dataclass(frozen=True)
class SamplingConfig:
    seed: str = "data-security-e-sample-v1"
    both_docs_rate: float = 0.002
    one_doc_rate: float = 0.004
    no_docs_rate: float = 0.0005

    def rate_for(self, record: PatentRecord) -> float:
        present = sum(bool(record.get(field)) for field in ("abstract", "claim"))
        if present == 2:
            return self.both_docs_rate
        if present == 1:
            return self.one_doc_rate
        return self.no_docs_rate


@dataclass
class RouteStats:
    records: int = 0
    candidate_rows: int = 0
    candidates: int = 0
    e_samples: int = 0
    route_levels: Counter[str] = field(default_factory=Counter)
    keyword_levels: Counter[str] = field(default_factory=Counter)
    ipc_levels: Counter[str] = field(default_factory=Counter)
    missing_abstract: int = 0
    missing_claim: int = 0
    missing_both_docs: int = 0
    diagnostics: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": self.records,
            "candidate_rows": self.candidate_rows,
            "candidates": self.candidates,
            "e_samples": self.e_samples,
            "route_levels": dict(self.route_levels),
            "keyword_levels": dict(self.keyword_levels),
            "ipc_levels": dict(self.ipc_levels),
            "missing_abstract": self.missing_abstract,
            "missing_claim": self.missing_claim,
            "missing_both_docs": self.missing_both_docs,
            "diagnostics": dict(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RouteStats:
        stats = cls()
        for name in (
            "records",
            "candidate_rows",
            "candidates",
            "e_samples",
            "missing_abstract",
            "missing_claim",
            "missing_both_docs",
        ):
            setattr(stats, name, int(value.get(name, 0)))
        for name in ("route_levels", "keyword_levels", "ipc_levels", "diagnostics"):
            setattr(stats, name, Counter(value.get(name, {})))
        return stats


@dataclass(frozen=True)
class RouteOutputs:
    routes: Path
    candidates: Path
    summary: Path


@dataclass(frozen=True)
class ProcessedPatent:
    source_row_number: int
    route_row: dict[str, Any]
    candidate: dict[str, Any] | None
    is_candidate_row: bool
    missing_abstract: bool
    missing_claim: bool
    sampled: bool
    diagnostic_ids: tuple[str, ...]


_WORKER_ROUTER: PatentRouter | None = None
_WORKER_TAXONOMY_VERSION = ""
_WORKER_SAMPLING = SamplingConfig()


def route_csv(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    taxonomy: TaxonomyBundle | None = None,
    sampling: SamplingConfig | None = None,
    encoding: str = "utf-8-sig",
    checkpoint_every: int = 50_000,
    progress_every: int = 100_000,
    workers: int = 1,
    worker_chunksize: int = 100,
    limit: int | None = None,
    resume: bool = False,
    overwrite: bool = False,
) -> RouteOutputs:
    """Route a large CSV while keeping memory use bounded."""

    source = Path(input_path).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    year_token = _year_token(source)
    outputs = RouteOutputs(
        routes=destination / f"patent_routes_{year_token}.csv",
        candidates=destination / f"patent_llm_candidates_{year_token}.jsonl",
        summary=destination / f"route_summary_{year_token}.json",
    )
    partial_routes = outputs.routes.with_suffix(outputs.routes.suffix + ".partial")
    partial_candidates = outputs.candidates.with_suffix(outputs.candidates.suffix + ".partial")
    checkpoint_path = destination / f"route_checkpoint_{year_token}.json"

    if not resume and not overwrite and any(path.exists() for path in outputs.__dict__.values()):
        raise FileExistsError("Route outputs already exist; pass --overwrite to replace them")
    if overwrite:
        cleanup_paths = (
            *outputs.__dict__.values(),
            partial_routes,
            partial_candidates,
            checkpoint_path,
        )
        for path in cleanup_paths:
            path.unlink(missing_ok=True)

    bundle = taxonomy or load_taxonomies()
    sampling = sampling or SamplingConfig()
    router = PatentRouter(bundle)
    checkpoint = _load_checkpoint(checkpoint_path) if resume else None
    stats = RouteStats.from_dict(checkpoint["stats"]) if checkpoint else RouteStats()
    last_row = int(checkpoint["last_source_row"]) if checkpoint else 1
    if checkpoint:
        _truncate_to(partial_routes, int(checkpoint["routes_offset"]))
        _truncate_to(partial_candidates, int(checkpoint["candidates_offset"]))
    candidate_keys = _load_candidate_keys(partial_candidates) if checkpoint else {}

    route_mode = "a" if checkpoint else "w"
    candidate_mode = "a" if checkpoint else "w"
    started = time.monotonic()
    with (
        partial_routes.open(route_mode, encoding="utf-8", newline="") as route_file,
        partial_candidates.open(candidate_mode, encoding="utf-8", newline="") as candidate_file,
    ):
        route_writer = csv.DictWriter(route_file, fieldnames=ROUTE_FIELDS)
        if not checkpoint:
            route_writer.writeheader()
        records = (
            record
            for record in iter_patent_records(
                source, encoding=encoding, limit=limit, include_raw=False
            )
            if record.row_number > last_row
        )
        if workers > 1:
            context = mp.get_context("fork")
            pool: Any = context.Pool(
                processes=workers,
                initializer=_initialize_worker,
                initargs=(bundle, sampling),
            )
            processed_records = pool.imap(
                _process_record_worker, records, chunksize=worker_chunksize
            )
        else:
            pool = None
            processed_records = (
                _process_record(record, router, bundle.version, sampling)
                for record in records
            )
        try:
            for processed in processed_records:
                candidate = processed.candidate
                if candidate is not None:
                    patent_id = candidate["patent_id"]
                    existing_key = candidate_keys.get(patent_id)
                    if existing_key:
                        processed.route_row["classification_key"] = existing_key
                        processed.route_row["process_status"] = "pending_llm_shared"
                        candidate = None
                    else:
                        candidate_keys[patent_id] = candidate["custom_id"]
                        processed.route_row["classification_key"] = candidate["custom_id"]
                route_writer.writerow(processed.route_row)
                if candidate is not None:
                    candidate_file.write(
                        json.dumps(
                            candidate,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                _update_processed_stats(stats, processed, wrote_candidate=candidate is not None)
                last_row = processed.source_row_number

                if stats.records % checkpoint_every == 0:
                    _save_checkpoint(
                        checkpoint_path,
                        source,
                        last_row,
                        route_file,
                        candidate_file,
                        stats,
                        bundle.version,
                    )
                if progress_every and stats.records % progress_every == 0:
                    elapsed = max(time.monotonic() - started, 0.001)
                    print(
                        f"routed={stats.records:,} candidates={stats.candidates:,} "
                        f"rate={stats.records / elapsed:,.0f}/s",
                        file=sys.stderr,
                        flush=True,
                    )
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        _save_checkpoint(
            checkpoint_path,
            source,
            last_row,
            route_file,
            candidate_file,
            stats,
            bundle.version,
        )

    os.replace(partial_routes, outputs.routes)
    os.replace(partial_candidates, outputs.candidates)
    elapsed_seconds = time.monotonic() - started
    summary = {
        "schema_version": "1.0.0",
        "input_path": str(source),
        "input_size_bytes": source.stat().st_size,
        "taxonomy_version": bundle.version,
        "sampling": asdict(sampling),
        "workers": workers,
        "stats": stats.to_dict(),
        "last_source_row": last_row,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "records_per_second": round(stats.records / max(elapsed_seconds, 0.001), 3),
        "outputs": {name: str(path) for name, path in outputs.__dict__.items()},
        "llm_status": "pending_llm",
    }
    _atomic_json_write(outputs.summary, summary)
    checkpoint_path.unlink(missing_ok=True)
    return outputs


def stable_selection(key: str, probability: float, seed: str) -> bool:
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    digest = hashlib.blake2b(f"{seed}|{key}".encode(), digest_size=8).digest()
    value = int.from_bytes(digest, "big") / 2**64
    return value < probability


def _initialize_worker(bundle: TaxonomyBundle, sampling: SamplingConfig) -> None:
    global _WORKER_ROUTER, _WORKER_SAMPLING, _WORKER_TAXONOMY_VERSION
    _WORKER_ROUTER = PatentRouter(bundle)
    _WORKER_TAXONOMY_VERSION = bundle.version
    _WORKER_SAMPLING = sampling


def _process_record_worker(record: PatentRecord) -> ProcessedPatent:
    if _WORKER_ROUTER is None:
        raise RuntimeError("Routing worker was not initialized")
    return _process_record(
        record,
        _WORKER_ROUTER,
        _WORKER_TAXONOMY_VERSION,
        _WORKER_SAMPLING,
    )


def _process_record(
    record: PatentRecord,
    router: PatentRouter,
    taxonomy_version: str,
    sampling: SamplingConfig,
) -> ProcessedPatent:
    routing = router.route(record)
    sampled, probability, stratum = _sample_e(record, routing.route_level, sampling)
    is_candidate = routing.route_level != "E" or sampled
    patent_id = _patent_id(record)
    row = _route_row(
        record,
        patent_id,
        routing,
        taxonomy_version,
        sampled,
        probability,
        stratum,
        is_candidate,
    )
    return ProcessedPatent(
        source_row_number=record.row_number,
        route_row=row,
        candidate=_candidate_record(record, patent_id, row) if is_candidate else None,
        is_candidate_row=is_candidate,
        missing_abstract=not record.get("abstract"),
        missing_claim=not record.get("claim"),
        sampled=sampled,
        diagnostic_ids=tuple(hit.pattern_id for hit in routing.diagnostic_hits),
    )


def _sample_e(
    record: PatentRecord, route_level: str, config: SamplingConfig
) -> tuple[bool, float, str]:
    completeness = "both" if record.get("abstract") and record.get("claim") else "one"
    if not record.get("abstract") and not record.get("claim"):
        completeness = "none"
    ipc_section = next(
        (
            character.upper()
            for character in record.get("main_ipc")
            if character.upper() in "ABCDEFGH"
        ),
        "NONE",
    )
    stratum = "|".join(
        (
            record.get("application_year") or "unknown_year",
            record.get("industry") or "unknown_industry",
            ipc_section,
            completeness,
        )
    )
    if route_level != "E":
        return False, 1.0, stratum
    probability = config.rate_for(record)
    key = f"{stratum}|{_patent_id(record)}"
    return stable_selection(key, probability, config.seed), probability, stratum


def _route_row(
    record: PatentRecord,
    patent_id: str,
    routing: Any,
    taxonomy_version: str,
    sampled: bool,
    probability: float,
    stratum: str,
    is_candidate: bool,
) -> dict[str, Any]:
    def compact_json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    return {
        "patent_id": patent_id,
        "source_row_number": record.row_number,
        "application_year": record.get("application_year"),
        "stock_code": record.get("stock_code"),
        "company_name": record.get("company_name"),
        "industry": record.get("industry"),
        "market": record.get("market"),
        "title": record.get("title"),
        "normalized_ipc": ";".join(routing.normalized_ipc),
        "keyword_level": routing.keyword_level,
        "ipc_level": routing.ipc_level,
        "route_level": routing.route_level,
        "keyword_hits": compact_json(routing.keyword_hits_jsonable()),
        "ipc_hits": compact_json(routing.ipc_hits_jsonable()),
        "diagnostic_hits": compact_json(routing.diagnostics_jsonable()),
        "e_stratum": stratum,
        "is_e_sample": str(sampled).lower(),
        "selection_probability": f"{probability:.6f}",
        "sample_weight": f"{1 / probability:.6f}" if is_candidate else "",
        "classification_key": f"patent-{record.row_number}" if is_candidate else "",
        "taxonomy_version": taxonomy_version,
        "process_status": "pending_llm" if is_candidate else "e_not_sampled",
    }


def _candidate_record(
    record: PatentRecord, patent_id: str, route_row: dict[str, Any]
) -> dict[str, Any]:
    abstract, abstract_truncated = _truncate(record.get("abstract"), 6_000)
    claim, claim_truncated = _truncate(record.get("claim"), 10_000)
    return {
        "custom_id": f"patent-{record.row_number}",
        "patent_id": patent_id,
        "source_row_number": record.row_number,
        "application_year": record.get("application_year"),
        "title": record.get("title"),
        "abstract": abstract,
        "claim": claim,
        "ipc": record.get("ipc"),
        "main_ipc": record.get("main_ipc"),
        "keyword_level": route_row["keyword_level"],
        "ipc_level": route_row["ipc_level"],
        "route_level": route_row["route_level"],
        "keyword_hits": json.loads(route_row["keyword_hits"]),
        "ipc_hits": json.loads(route_row["ipc_hits"]),
        "diagnostic_hits": json.loads(route_row["diagnostic_hits"]),
        "is_e_sample": route_row["is_e_sample"] == "true",
        "selection_probability": float(route_row["selection_probability"]),
        "sample_weight": float(route_row["sample_weight"]),
        "taxonomy_version": route_row["taxonomy_version"],
        "text_truncated": abstract_truncated or claim_truncated,
    }


def _patent_id(record: PatentRecord) -> str:
    return (
        record.get("application_number")
        or record.get("publication_number")
        or record.get("grant_number")
        or f"source-row-{record.row_number}"
    )


def _truncate(value: str, limit: int) -> tuple[str, bool]:
    if len(value) <= limit:
        return value, False
    return value[:limit], True


def _update_processed_stats(
    stats: RouteStats, processed: ProcessedPatent, *, wrote_candidate: bool
) -> None:
    row = processed.route_row
    stats.records += 1
    stats.candidate_rows += int(processed.is_candidate_row)
    stats.candidates += int(wrote_candidate)
    stats.e_samples += int(processed.sampled)
    stats.route_levels[row["route_level"]] += 1
    stats.keyword_levels[row["keyword_level"]] += 1
    stats.ipc_levels[row["ipc_level"]] += 1
    stats.missing_abstract += int(processed.missing_abstract)
    stats.missing_claim += int(processed.missing_claim)
    stats.missing_both_docs += int(processed.missing_abstract and processed.missing_claim)
    for pattern_id in processed.diagnostic_ids:
        stats.diagnostics[pattern_id] += 1


def _save_checkpoint(
    path: Path,
    source: Path,
    last_row: int,
    route_file: TextIO,
    candidate_file: TextIO,
    stats: RouteStats,
    taxonomy_version: str,
) -> None:
    route_file.flush()
    candidate_file.flush()
    payload = {
        "input_path": str(source),
        "input_size_bytes": source.stat().st_size,
        "taxonomy_version": taxonomy_version,
        "last_source_row": last_row,
        "routes_offset": route_file.buffer.tell(),
        "candidates_offset": candidate_file.buffer.tell(),
        "stats": stats.to_dict(),
    }
    _atomic_json_write(path, payload)


def _load_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _load_candidate_keys(path: Path) -> dict[str, str]:
    keys: dict[str, str] = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            candidate = json.loads(line)
            keys[candidate["patent_id"]] = candidate["custom_id"]
    return keys


def _truncate_to(path: Path, offset: int) -> None:
    with path.open("r+b") as file:
        file.truncate(offset)


def _atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _year_token(path: Path) -> str:
    digits = "".join(character for character in path.stem if character.isdigit())
    return digits[:4] if len(digits) >= 4 else "dataset"
