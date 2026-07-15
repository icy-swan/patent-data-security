"""Streaming reader for Chinese patent CSV records."""

from __future__ import annotations

import csv
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

COLUMN_ALIASES = {
    "stock_code": ("关联股票代码", "code"),
    "company_name": ("企业名称",),
    "industry": ("上市公司行业",),
    "market": ("上市板块",),
    "title": ("专利名称",),
    "applicant": ("申请人",),
    "application_number": ("申请号",),
    "application_date": ("申请日",),
    "application_year": ("申请年份",),
    "publication_number": ("公开公告号",),
    "grant_number": ("授权公告号",),
    "ipc": ("IPC分类号",),
    "main_ipc": ("IPC主分类号",),
    "abstract": ("摘要文本",),
    "claim": ("主权项内容",),
}


@dataclass(frozen=True)
class PatentRecord:
    """Normalized fields needed by Step 1."""

    row_number: int
    values: dict[str, str]

    def get(self, field: str, default: str = "") -> str:
        return self.values.get(field, default)


def iter_patent_records(
    csv_path: str | Path,
    *,
    encoding: str = "utf-8-sig",
    limit: int | None = None,
) -> Iterator[PatentRecord]:
    """Yield records without loading the source CSV into memory."""

    _maximize_csv_field_size_limit()
    path = Path(csv_path)
    with path.open("r", encoding=encoding, newline="") as file:
        reader = csv.reader(file)
        try:
            header = next(reader)
        except StopIteration:
            return
        fieldnames = [_clean_header(value) for value in header]
        indexes = {name: index for index, name in enumerate(fieldnames)}
        canonical_indexes = {
            canonical: tuple(indexes[alias] for alias in aliases if alias in indexes)
            for canonical, aliases in COLUMN_ALIASES.items()
        }
        for offset, row in enumerate(reader, start=2):
            if limit is not None and offset > limit + 1:
                break
            cells = [_normalize_cell(value) for value in row]
            values = {
                canonical: _first_indexed(cells, candidates)
                for canonical, candidates in canonical_indexes.items()
            }
            yield PatentRecord(row_number=offset, values=values)


def _first_indexed(row: list[str], indexes: tuple[int, ...]) -> str:
    for index in indexes:
        if index < len(row) and row[index]:
            return row[index]
    return ""


def _clean_header(value: str) -> str:
    return value.removeprefix("\ufeff").strip()


def _normalize_cell(value: str | None) -> str:
    cleaned = (value or "").strip()
    if cleaned.startswith('="') and cleaned.endswith('"'):
        return cleaned[2:-1].strip()
    return cleaned


def _maximize_csv_field_size_limit() -> None:
    value = sys.maxsize
    while True:
        try:
            csv.field_size_limit(value)
            return
        except OverflowError:
            value //= 10

