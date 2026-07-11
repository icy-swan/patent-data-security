"""Step 1: deterministic keyword extraction with auditable context retrieval."""

from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from patent_data_security.records import PatentRecord, iter_patent_records
from patent_data_security.routing import PatentRouter
from patent_data_security.taxonomy import TaxonomyBundle, load_taxonomies

KEYWORD_ROUTE_FIELDS = (
    "patent_id",
    "source_row_number",
    "application_year",
    "stock_code",
    "company_name",
    "industry",
    "market",
    "title",
    "keyword_level",
    "keyword_hit_count",
    "context_hit_count",
    "context_scope_modes",
    "keyword_hits",
    "diagnostic_hits",
    "taxonomy_version",
)
TIERS = ("S", "W", "R", "E")


@dataclass(frozen=True)
class KeywordExtractionOutputs:
    s: Path
    w: Path
    r: Path
    e: Path
    summary: Path

    def by_tier(self) -> dict[str, Path]:
        return {"S": self.s, "W": self.w, "R": self.r, "E": self.e}


@dataclass
class KeywordExtractionStats:
    records: int = 0
    levels: Counter[str] = field(default_factory=Counter)
    keyword_hits: int = 0
    context_hits: int = 0
    context_scopes: Counter[str] = field(default_factory=Counter)
    diagnostics: int = 0
    missing_abstract: int = 0
    missing_claim: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": self.records,
            "levels": {tier: self.levels[tier] for tier in TIERS},
            "keyword_hits": self.keyword_hits,
            "context_hits": self.context_hits,
            "context_scopes": dict(self.context_scopes),
            "diagnostics": self.diagnostics,
            "missing_abstract": self.missing_abstract,
            "missing_claim": self.missing_claim,
        }


@dataclass(frozen=True)
class _ProcessedKeywordRecord:
    level: str
    row: dict[str, Any]
    keyword_hits: int
    context_hits: int
    context_scopes: tuple[str, ...]
    diagnostics: int
    missing_abstract: bool
    missing_claim: bool


_WORKER_ROUTER: PatentRouter | None = None
_WORKER_TAXONOMY_VERSION = ""


def extract_keywords_csv(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    taxonomy: TaxonomyBundle | None = None,
    encoding: str = "utf-8-sig",
    workers: int = 1,
    worker_chunksize: int = 100,
    limit: int | None = None,
    progress_every: int = 100_000,
    overwrite: bool = False,
) -> KeywordExtractionOutputs:
    """Scan DOCS fields once and write separate S/W/R/E keyword files.

    This function is fully local and deliberately has no dependency on the LLM module.
    """

    if workers < 1:
        raise ValueError("workers must be at least 1")
    source = Path(input_path).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    year = _year_token(source)
    outputs = KeywordExtractionOutputs(
        s=destination / f"keyword_S_{year}.csv",
        w=destination / f"keyword_W_{year}.csv",
        r=destination / f"keyword_R_{year}.csv",
        e=destination / f"keyword_E_{year}.csv",
        summary=destination / f"keyword_summary_{year}.json",
    )
    output_paths = (*outputs.by_tier().values(), outputs.summary)
    if not overwrite and any(path.exists() for path in output_paths):
        raise FileExistsError("Step 1 outputs already exist; pass --overwrite to replace them")
    if overwrite:
        for path in output_paths:
            path.unlink(missing_ok=True)

    bundle = taxonomy or load_taxonomies()
    router = PatentRouter(bundle)
    partials = {
        tier: path.with_suffix(path.suffix + ".partial")
        for tier, path in outputs.by_tier().items()
    }
    for path in partials.values():
        path.unlink(missing_ok=True)

    stats = KeywordExtractionStats()
    started = time.monotonic()
    files = {
        tier: path.open("w", encoding="utf-8", newline="")
        for tier, path in partials.items()
    }
    pool: Any = None
    try:
        writers = {
            tier: csv.DictWriter(files[tier], fieldnames=KEYWORD_ROUTE_FIELDS)
            for tier in TIERS
        }
        for writer in writers.values():
            writer.writeheader()

        records = iter_patent_records(
            source,
            encoding=encoding,
            limit=limit,
            include_raw=False,
        )
        if workers > 1:
            pool = mp.get_context("fork").Pool(
                processes=workers,
                initializer=_initialize_worker,
                initargs=(bundle,),
            )
            processed_records = pool.imap(
                _process_record_worker,
                records,
                chunksize=worker_chunksize,
            )
        else:
            processed_records = (
                _process_record(record, router, bundle.docs["taxonomy_version"])
                for record in records
            )

        for processed in processed_records:
            writers[processed.level].writerow(processed.row)
            _update_stats(stats, processed)
            if progress_every and stats.records % progress_every == 0:
                elapsed = max(time.monotonic() - started, 0.001)
                print(
                    f"step1_records={stats.records:,} "
                    f"S={stats.levels['S']:,} W={stats.levels['W']:,} "
                    f"R={stats.levels['R']:,} E={stats.levels['E']:,} "
                    f"rate={stats.records / elapsed:,.0f}/s",
                    file=sys.stderr,
                    flush=True,
                )
    except BaseException:
        if pool is not None:
            pool.terminate()
        raise
    else:
        if pool is not None:
            pool.close()
    finally:
        if pool is not None:
            pool.join()
        for file in files.values():
            file.close()

    for tier, partial in partials.items():
        os.replace(partial, outputs.by_tier()[tier])

    elapsed = time.monotonic() - started
    summary = {
        "schema_version": "1.0.0",
        "step": "step1_keyword_extraction",
        "input_path": str(source),
        "input_size_bytes": source.stat().st_size,
        "docs_taxonomy_version": bundle.docs["taxonomy_version"],
        "docs_fields": list(bundle.docs["docs_fields"]),
        "context_policy": {
            "preferred_scope": "sentence",
            "fallback_scope": "window",
            "fallback_window_chars_each_side": router.context_window,
        },
        "workers": workers,
        "stats": stats.to_dict(),
        "elapsed_seconds": round(elapsed, 3),
        "records_per_second": round(stats.records / max(elapsed, 0.001), 3),
        "outputs": {tier: str(path) for tier, path in outputs.by_tier().items()},
        "llm_requests_executed": 0,
    }
    _atomic_json_write(outputs.summary, summary)
    return outputs


def _initialize_worker(bundle: TaxonomyBundle) -> None:
    global _WORKER_ROUTER, _WORKER_TAXONOMY_VERSION
    _WORKER_ROUTER = PatentRouter(bundle)
    _WORKER_TAXONOMY_VERSION = bundle.docs["taxonomy_version"]


def _process_record_worker(record: PatentRecord) -> _ProcessedKeywordRecord:
    if _WORKER_ROUTER is None:
        raise RuntimeError("Keyword extraction worker was not initialized")
    return _process_record(record, _WORKER_ROUTER, _WORKER_TAXONOMY_VERSION)


def _process_record(
    record: PatentRecord,
    router: PatentRouter,
    taxonomy_version: str,
) -> _ProcessedKeywordRecord:
    routing = router.route_keywords(record)
    hits = routing.keyword_hits_jsonable()
    diagnostics = routing.diagnostics_jsonable()
    context_hit_count = sum(len(hit["context_hits"]) for hit in hits)
    context_scopes = tuple(hit["context_scope"] for hit in hits)
    context_scope_modes = sorted(set(context_scopes))
    compact = lambda value: json.dumps(  # noqa: E731
        value, ensure_ascii=False, separators=(",", ":")
    )
    row = {
        "patent_id": _patent_id(record),
        "source_row_number": record.row_number,
        "application_year": record.get("application_year"),
        "stock_code": record.get("stock_code"),
        "company_name": record.get("company_name"),
        "industry": record.get("industry"),
        "market": record.get("market"),
        "title": record.get("title"),
        "keyword_level": routing.keyword_level,
        "keyword_hit_count": len(hits),
        "context_hit_count": context_hit_count,
        "context_scope_modes": ";".join(context_scope_modes),
        "keyword_hits": compact(hits),
        "diagnostic_hits": compact(diagnostics),
        "taxonomy_version": taxonomy_version,
    }
    return _ProcessedKeywordRecord(
        level=routing.keyword_level,
        row=row,
        keyword_hits=len(hits),
        context_hits=context_hit_count,
        context_scopes=context_scopes,
        diagnostics=len(diagnostics),
        missing_abstract=not record.get("abstract"),
        missing_claim=not record.get("claim"),
    )


def _update_stats(stats: KeywordExtractionStats, processed: _ProcessedKeywordRecord) -> None:
    stats.records += 1
    stats.levels[processed.level] += 1
    stats.keyword_hits += processed.keyword_hits
    stats.context_hits += processed.context_hits
    stats.context_scopes.update(processed.context_scopes)
    stats.diagnostics += processed.diagnostics
    stats.missing_abstract += int(processed.missing_abstract)
    stats.missing_claim += int(processed.missing_claim)


def _patent_id(record: PatentRecord) -> str:
    return (
        record.get("application_number")
        or record.get("publication_number")
        or record.get("grant_number")
        or f"source-row-{record.row_number}"
    )


def _atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _year_token(path: Path) -> str:
    digits = "".join(character for character in path.stem if character.isdigit())
    return digits[:4] if len(digits) >= 4 else "dataset"
