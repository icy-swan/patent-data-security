"""Migrate existing Step 1-3 CSVs to explicit label-lineage fields.

The migration is intentionally lossless: Step 2 SQLite response JSON remains
unchanged as immutable request provenance. CSV interfaces are rewritten to use
step1_label, step2_label and explicit human/codex review labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pipeline.common.io import atomic_json_write, sha256_file
from pipeline.step3.sampling import LABELS, MANUAL_REVIEW_FIELDS, SCHEMA_VERSION


def migrate(root: Path) -> dict[str, Any]:
    root = root.resolve()
    step1_files = sorted((root / "data" / "step1").glob("*/result.csv"))
    step2_databases = sorted((root / "data" / "step2").glob("*/tasks.sqlite3"))
    if not step1_files or not step2_databases:
        raise FileNotFoundError("Expected Step 1 CSVs and Step 2 task databases")

    for path in step1_files:
        _migrate_step1_csv(path)
        _update_step1_manifest(path.with_name("manifest.json"))

    step2_index: dict[tuple[str, str], dict[str, str]] = {}
    for database in step2_databases:
        index = _read_step2_index(database)
        step2_index.update(index)
        result_path = database.with_name("result.csv")
        output_fields = _migrate_step2_csv(result_path, index)
        _update_step2_manifest(database.with_name("manifest.json"), output_fields)

    step3_root = root / "data" / "step3"
    migrated_step3: list[str] = []
    source_hashes: dict[str, str] = {}
    for name in ("need_manual_review.csv", "result.csv", "codex_result.csv"):
        path = step3_root / name
        if path.is_file():
            source_hashes[name] = sha256_file(path)
            _migrate_step3_csv(path, step2_index)
            migrated_step3.append(name)
    _update_step3_manifest(step3_root / "manifest.json", migrated_step3, source_hashes)
    return {
        "step1_csvs": [str(path) for path in step1_files],
        "step2_csvs": [str(path.parent / "result.csv") for path in step2_databases],
        "step3_csvs": migrated_step3,
    }


def _migrate_step1_csv(path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".label-migration.tmp")
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = list(reader.fieldnames or ())
        if "step1_label" not in fields:
            fields.insert(fields.index("route") + 1, "step1_label")
        with temporary.open("w", encoding="utf-8-sig", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row_number, row in enumerate(reader, start=2):
                route = row.get("route", "")
                if route not in {"S", "E"}:
                    raise ValueError(f"{path}:{row_number}: invalid route {route!r}")
                expected = "DATA_SECURITY" if route == "S" else "OTHER"
                current = row.get("step1_label", "").strip()
                if current and current != expected:
                    raise ValueError(f"{path}:{row_number}: step1_label conflicts with route")
                row["step1_label"] = expected
                writer.writerow(row)
    os.replace(temporary, path)


def _read_step2_index(database: Path) -> dict[tuple[str, str], dict[str, str]]:
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro&immutable=1", uri=True)
    index: dict[tuple[str, str], dict[str, str]] = {}
    try:
        for dataset_id, patent_id, route, result_json in connection.execute(
            "SELECT dataset_id,patent_id,route,result_json FROM tasks "
            "WHERE status='succeeded'"
        ):
            result = json.loads(result_json)
            label = str(result["label"])
            if label not in LABELS or route not in {"S", "E"}:
                raise ValueError(f"Invalid Step 2 label lineage for {patent_id}")
            index[(str(dataset_id), str(patent_id))] = {
                "step1_label": "DATA_SECURITY" if route == "S" else "OTHER",
                "step2_label": label,
            }
    finally:
        connection.close()
    return index


def _migrate_step2_csv(
    path: Path,
    index: dict[tuple[str, str], dict[str, str]],
) -> list[str]:
    fields, rows = _read_csv(path)
    fields = _replace_field(fields, "label", "step2_label")
    fields = _replace_field(fields, "review_flag", "needs_review")
    for row_number, row in enumerate(rows, start=2):
        key = (str(row.get("dataset_id", "")), str(row.get("patent_id", "")))
        lineage = index.get(key)
        if lineage is None:
            raise ValueError(f"{path}:{row_number}: no matching Step 2 task for {key}")
        if "label" in row:
            row["step2_label"] = row.pop("label")
        if row.get("step2_label") != lineage["step2_label"]:
            raise ValueError(f"{path}:{row_number}: step2_label changed from task database")
        if "review_flag" in row:
            row["needs_review"] = _canonical_boolean(
                row.pop("review_flag"), path=path, row_number=row_number
            )
    _write_csv(path, fields, rows)
    return fields


def _migrate_step3_csv(
    path: Path,
    step2_index: dict[tuple[str, str], dict[str, str]],
) -> None:
    fields, rows = _read_csv(path)
    fields = _replace_field(fields, "step2_review_flag", "step2_needs_review")
    fields = _replace_field(fields, "human_evaluation", "human_review_label")
    fields = _replace_field(fields, "codex_review_review_flag", "codex_review_label")
    if "step1_label" not in fields:
        fields.insert(fields.index("step2_label"), "step1_label")

    for row_number, row in enumerate(rows, start=2):
        key = (str(row.get("dataset_id", "")), str(row.get("patent_id", "")))
        lineage = step2_index.get(key)
        if lineage is None:
            raise ValueError(f"{path}:{row_number}: no matching Step 2 task for {key}")
        if row.get("step2_label") != lineage["step2_label"]:
            raise ValueError(f"{path}:{row_number}: step2_label changed from task database")
        row["step1_label"] = lineage["step1_label"]
        if "step2_review_flag" in row:
            row["step2_needs_review"] = _canonical_boolean(
                row.pop("step2_review_flag"), path=path, row_number=row_number
            )
        if "human_evaluation" in row:
            row["human_review_label"] = _legacy_flag_to_label(
                row.pop("human_evaluation"),
                step2_label=lineage["step2_label"],
                path=path,
                row_number=row_number,
            )
        if "codex_review_review_flag" in row:
            row["codex_review_label"] = _legacy_flag_to_label(
                row.pop("codex_review_review_flag"),
                step2_label=lineage["step2_label"],
                path=path,
                row_number=row_number,
            )
        for field in ("human_review_label", "codex_review_label"):
            value = str(row.get(field, "") or "").strip().upper()
            if value and value not in LABELS:
                raise ValueError(f"{path}:{row_number}: invalid {field}={value!r}")
            if field in fields:
                row[field] = value
    _write_csv(path, fields, rows)


def _legacy_flag_to_label(
    value: Any,
    *,
    step2_label: str,
    path: Path,
    row_number: int,
) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return ""
    if normalized == "false":
        return step2_label
    if normalized == "true":
        return "OTHER" if step2_label == "DATA_SECURITY" else "DATA_SECURITY"
    raise ValueError(f"{path}:{row_number}: legacy review flag is not true/false")


def _canonical_boolean(value: Any, *, path: Path, row_number: int) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"{path}:{row_number}: needs_review must be true/false")
    return normalized


def _replace_field(fields: list[str], old: str, new: str) -> list[str]:
    fields = list(fields)
    if old in fields:
        index = fields.index(old)
        fields[index] = new
    return fields


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or ()), [dict(row) for row in reader]


def _write_csv(path: Path, fields: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    fields = list(fields)
    temporary = path.with_suffix(path.suffix + ".label-migration.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _update_step1_manifest(path: Path) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "2.1.0"
    manifest["label_mapping"] = {"S": "DATA_SECURITY", "E": "OTHER"}
    manifest["output_label_field"] = "step1_label"
    atomic_json_write(path, manifest)


def _update_step2_manifest(path: Path, output_fields: list[str]) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["csv_output_contract"] = {
        "schema_version": "2.2.0",
        "label_field": "step2_label",
        "uncertainty_field": "needs_review",
        "fields": output_fields,
        "legacy_result_json_preserved": True,
    }
    atomic_json_write(path, manifest)


def _update_step3_manifest(
    path: Path,
    migrated_csvs: list[str],
    source_hashes: dict[str, str],
) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    previous_migration = manifest.get("label_field_migration", {})
    already_migrated = previous_migration.get("version") == "explicit-review-label-v1"
    manifest["schema_version"] = SCHEMA_VERSION
    manifest["human_results_policy"] = {
        "path": str(path.parent / "result.csv"),
        "label_field": "human_review_label",
        "allowed_values": list(LABELS),
        "result_fields": list(MANUAL_REVIEW_FIELDS),
    }
    manifest["manual_review_policy"]["fields"] = list(MANUAL_REVIEW_FIELDS)
    manifest["manual_review_policy"]["sha256"] = sha256_file(
        path.parent / "need_manual_review.csv"
    )
    manifest["manual_review_policy"]["human_fields_initially_blank"] = [
        "human_review_label",
        "human_reason",
    ]
    manifest["result_preparation"] = {
        "status": "human_review_completed_gold",
        "review_mode": "step2_assisted_human_review",
        "gold_label_field": "human_review_label",
        "allowed_labels": list(LABELS),
        "records": int(manifest["target_size"]),
    }
    if not already_migrated:
        manifest.pop("evaluation", None)
    manifest["label_field_migration"] = {
        "version": "explicit-review-label-v1",
        "migrated_csvs": migrated_csvs,
        "source_sha256": (
            previous_migration.get("source_sha256", source_hashes)
            if already_migrated
            else source_hashes
        ),
        "legacy_boolean_review_conclusions_removed": True,
        "migrated_at": (
            previous_migration.get("migrated_at", datetime.now(UTC).isoformat())
            if already_migrated
            else datetime.now(UTC).isoformat()
        ),
    }
    atomic_json_write(path, manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(migrate(args.project_root), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
