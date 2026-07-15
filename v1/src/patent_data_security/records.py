"""Utilities for reading patent CSV records."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

COLUMN_ALIASES = {
    "stock_code": ("关联股票代码", "code"),
    "company_name": ("企业名称",),
    "company_location": ("企业注册地",),
    "listed_company_relation": ("与上市公司关系",),
    "stock_name": ("股票简称",),
    "industry": ("上市公司行业",),
    "market": ("上市板块",),
    "title": ("专利名称",),
    "patent_type": ("专利类型",),
    "applicant": ("申请人",),
    "applicant_type": ("申请人类型",),
    "applicant_address": ("申请人地址",),
    "applicant_region": ("申请人地区",),
    "applicant_city": ("申请人城市",),
    "applicant_district": ("申请人区县",),
    "application_number": ("申请号",),
    "application_date": ("申请日",),
    "application_year": ("申请年份",),
    "publication_number": ("公开公告号",),
    "publication_date": ("公开公告日",),
    "publication_year": ("公开公告年份",),
    "grant_number": ("授权公告号",),
    "grant_date": ("授权公告日",),
    "grant_year": ("授权公告年份",),
    "ipc": ("IPC分类号",),
    "main_ipc": ("IPC主分类号",),
    "inventor": ("发明人",),
    "abstract": ("摘要文本",),
    "claim": ("主权项内容",),
}

CLASSIFICATION_FIELDS = ("title", "abstract", "claim", "ipc", "main_ipc")


@dataclass(frozen=True)
class PatentRecord:
    """A normalized patent record ready for classification."""

    row_number: int
    values: dict[str, str]
    raw: dict[str, str]

    def get(self, field: str, default: str = "") -> str:
        return self.values.get(field, default)

    @property
    def classification_text(self) -> str:
        parts = []
        labels = {
            "title": "专利名称",
            "abstract": "摘要",
            "claim": "主权项",
            "ipc": "IPC分类号",
            "main_ipc": "IPC主分类号",
        }
        for field in CLASSIFICATION_FIELDS:
            value = self.get(field)
            if value:
                parts.append(f"{labels[field]}：{value}")
        return "\n".join(parts)


def iter_patent_records(
    csv_path: str | Path,
    *,
    encoding: str = "utf-8-sig",
    limit: int | None = None,
    include_raw: bool = True,
) -> Iterator[PatentRecord]:
    """Yield normalized patent records from a CSV file without loading it all."""

    path = Path(csv_path)
    with path.open("r", encoding=encoding, newline="") as file:
        reader = csv.reader(file)
        try:
            header_row = next(reader)
        except StopIteration:
            return
        fieldnames = [_clean_header(name) for name in header_row]
        field_indexes = {name: index for index, name in enumerate(fieldnames)}
        canonical_indexes = {
            canonical_name: tuple(
                field_indexes[alias] for alias in aliases if alias in field_indexes
            )
            for canonical_name, aliases in COLUMN_ALIASES.items()
        }

        for index, row in enumerate(reader, start=2):
            if limit is not None and index > limit + 1:
                break
            normalized_cells = [_normalize_cell(value) for value in row]
            values = {
                canonical_name: _first_indexed(normalized_cells, indexes)
                for canonical_name, indexes in canonical_indexes.items()
            }
            yield PatentRecord(
                row_number=index,
                values=values,
                raw=(
                    dict(zip(fieldnames, normalized_cells, strict=False)) if include_raw else {}
                ),
            )


def read_patent_records(
    csv_path: str | Path,
    *,
    encoding: str = "utf-8-sig",
    limit: int | None = None,
) -> list[PatentRecord]:
    """Read normalized patent records into a list.

    Prefer ``iter_patent_records`` for the full source CSV because it is large.
    """

    return list(iter_patent_records(csv_path, encoding=encoding, limit=limit))


def _normalize_record(raw: dict[str, str]) -> dict[str, str]:
    normalized = {}
    for canonical_name, aliases in COLUMN_ALIASES.items():
        normalized[canonical_name] = _first_present(raw, aliases)
    return normalized


def _first_present(row: dict[str, str], names: Iterable[str]) -> str:
    for name in names:
        value = row.get(name, "")
        if value:
            return value
    return ""


def _first_indexed(row: list[str], indexes: tuple[int, ...]) -> str:
    for index in indexes:
        if index < len(row) and row[index]:
            return row[index]
    return ""


def _clean_header(value: str) -> str:
    return value.removeprefix("\ufeff").strip()


def _normalize_cell(value: str | None) -> str:
    if value is None:
        return ""

    cleaned = value.strip()
    if cleaned.startswith('="') and cleaned.endswith('"'):
        return cleaned[2:-1].strip()
    return cleaned
