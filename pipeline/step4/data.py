"""Build frozen RoBERTa and MaaS SFT datasets from Step 3 splits."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pipeline.common.io import atomic_json_write, sha256_file
from pipeline.step2.prompt import (
    PROMPT_VERSION,
    PromptBundle,
    build_dynamic_message,
    load_prompt_bundle,
)
from pipeline.step2.schema import PatentClassification

DATASET_VERSION = "data-security-binary-v1.2.0"
SCHEMA_VERSION = "1.1.0"
SFT_PROMPT_VERSION = PROMPT_VERSION
LABEL_TO_ID = {"OTHER": 0, "DATA_SECURITY": 1}
HUMAN_EVALUATION_TO_LABEL = {"false": "OTHER", "true": "DATA_SECURITY"}
SPLITS = ("train", "validation", "test")
EXPECTED_SPLIT_COUNTS = {"train": 8_000, "validation": 1_000, "test": 1_000}

REQUIRED_FIELDS = (
    "sample_id",
    "dataset_id",
    "application_year",
    "patent_id",
    "title",
    "abstract",
    "claim",
    "ipc",
    "main_ipc",
    "human_evaluation",
    "confidence",
    "scope_basis",
    "processing_activities",
    "industry_sectors",
    "technical_scope",
    "legal_scope",
    "evidence",
    "reason",
    "review_flag",
    "review_reason",
)

CLASSIFIER_FIELDS = (
    "sample_id",
    "patent_id",
    "application_year",
    "title",
    "abstract",
    "claim",
    "ipc",
    "main_ipc",
    "human_evaluation",
    "scope_basis",
    "industry_sectors",
    "label",
    "label_id",
    "data_split",
)

SFT_INDEX_FIELDS = (
    "data_split",
    "line_number",
    "sample_id",
    "patent_id",
    "label",
    "message_sha256",
)


@dataclass(frozen=True)
class Step4Paths:
    root: Path
    dataset_root: Path
    classifier_train: Path
    classifier_validation: Path
    classifier_test: Path
    sft_train: Path
    sft_validation: Path
    sft_index: Path
    manifest: Path
    model: Path
    state: Path
    reports: Path


def step4_paths(output_dir: str | Path) -> Step4Paths:
    root = Path(output_dir).resolve()
    dataset_root = root / "dataset"
    classifier = dataset_root / "classifier"
    sft = dataset_root / "sft"
    return Step4Paths(
        root=root,
        dataset_root=dataset_root,
        classifier_train=classifier / "train.jsonl",
        classifier_validation=classifier / "validation.jsonl",
        classifier_test=classifier / "test.jsonl",
        sft_train=sft / "train.jsonl",
        sft_validation=sft / "validation.jsonl",
        sft_index=sft / "index.csv",
        manifest=dataset_root / "manifest.json",
        model=root / "model" / "roberta",
        state=root / "state" / "roberta",
        reports=root / "reports" / "roberta",
    )


def prepare_datasets(
    step3_dir: str | Path,
    output_dir: str | Path,
    *,
    rebuild: bool = False,
    expected_counts: Mapping[str, int] = EXPECTED_SPLIT_COUNTS,
    prompt_bundle: PromptBundle | None = None,
) -> tuple[Step4Paths, dict[str, Any]]:
    """Validate frozen splits and export classifier plus MaaS conversational JSONL."""

    step3_root = Path(step3_dir).resolve()
    results_path = step3_root / "result.csv"
    step3_manifest_path = step3_root / "manifest.json"
    source_paths = {
        split: step3_root / "dataset" / f"{split}.csv" for split in SPLITS
    }
    missing = [
        path
        for path in (step3_manifest_path, results_path, *source_paths.values())
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing Step 3 split files: {missing}")

    production_prompt = prompt_bundle or load_prompt_bundle()

    expected = {split: int(expected_counts[split]) for split in SPLITS}
    step3_manifest = json.loads(step3_manifest_path.read_text(encoding="utf-8"))
    _validate_step3_manifest(
        step3_manifest,
        results_path=results_path,
        expected_count=sum(expected.values()),
        expected_splits=expected,
    )
    paths = step4_paths(output_dir)
    _prepare_output(paths, rebuild=rebuild)

    rows_by_split = {
        split: _read_and_validate_split(path, split=split, expected_count=expected[split])
        for split, path in source_paths.items()
    }
    all_rows = [row for split in SPLITS for row in rows_by_split[split]]
    result_rows = _read_and_validate_results(
        results_path,
        expected_count=sum(expected.values()),
    )
    _validate_splits_match_results(all_rows, result_rows)
    provenance = _validate_collection(all_rows)

    classifier_outputs = {
        "train": paths.classifier_train,
        "validation": paths.classifier_validation,
        "test": paths.classifier_test,
    }
    for split, destination in classifier_outputs.items():
        _write_jsonl(destination, (_classifier_row(row) for row in rows_by_split[split]))

    sft_outputs = {"train": paths.sft_train, "validation": paths.sft_validation}
    index_rows: list[dict[str, Any]] = []
    for split, destination in sft_outputs.items():
        messages = [
            _sft_row(row, prompt_bundle=production_prompt)
            for row in rows_by_split[split]
        ]
        _write_jsonl(destination, messages)
        for line_number, (source, message) in enumerate(
            zip(rows_by_split[split], messages, strict=True), start=1
        ):
            encoded = json.dumps(
                message, ensure_ascii=False, separators=(",", ":")
            ).encode()
            index_rows.append(
                {
                    "data_split": split,
                    "line_number": line_number,
                    "sample_id": source["sample_id"],
                    "patent_id": source["patent_id"],
                    "label": source["label"],
                    "message_sha256": hashlib.sha256(encoded).hexdigest(),
                }
            )
    _write_csv(paths.sft_index, SFT_INDEX_FIELDS, index_rows)

    output_files = {
        "classifier_train": paths.classifier_train,
        "classifier_validation": paths.classifier_validation,
        "classifier_test": paths.classifier_test,
        "sft_train": paths.sft_train,
        "sft_validation": paths.sft_validation,
        "sft_index": paths.sft_index,
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        "source_step3_root": str(step3_root),
        "source_step3_manifest": {
            "path": str(step3_manifest_path),
            "sha256": sha256_file(step3_manifest_path),
        },
        "source_results": {
            "path": str(results_path),
            "sha256": sha256_file(results_path),
        },
        "source_files": {
            split: {"path": str(path), "sha256": sha256_file(path)}
            for split, path in source_paths.items()
        },
        "split_counts": expected,
        "label_counts": {
            split: dict(sorted(Counter(row["label"] for row in rows).items()))
            for split, rows in rows_by_split.items()
        },
        "classifier": {
            "task": "single_label_binary_classification",
            "label_to_id": LABEL_TO_ID,
            "paper_default_text_fields": ["abstract"],
            "training_loss": "unweighted_cross_entropy",
            "validation_selection_metric": "accuracy",
            "prediction_rule": "softmax_argmax",
            "test_is_frozen": True,
        },
        "sft": {
            "format": "messages_jsonl",
            "top_level_fields": ["messages"],
            "splits": ["train", "validation"],
            "test_exported": False,
            "message_roles": ["system", "user", "assistant"],
            "system_prompt_source": "step2_production_static_prefix",
            "user_message_source": "step2_production_dynamic_message",
            "assistant_target": "step2_compatible_structured_classification",
            "assistant_target_fields": list(PatentClassification.model_fields),
            "prompt_version": SFT_PROMPT_VERSION,
            "system_prompt_sha256": production_prompt.prefix_sha256,
            "schema_sha256": production_prompt.schema_sha256,
            "resource_hashes": production_prompt.resource_hashes,
        },
        "annotation_provenance": provenance,
        "outputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in output_files.items()
        },
        "prepared_at": datetime.now(UTC).isoformat(),
    }
    atomic_json_write(paths.manifest, manifest)
    return paths, manifest


def _read_and_validate_split(
    path: Path, *, split: str, expected_count: int
) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fields = set(reader.fieldnames or ())
        missing_fields = sorted(set(REQUIRED_FIELDS) - fields)
        if missing_fields:
            raise ValueError(f"{path} is missing fields: {missing_fields}")
        unexpected_fields = sorted(fields - set(REQUIRED_FIELDS))
        if unexpected_fields:
            raise ValueError(f"{path} contains forbidden fields: {unexpected_fields}")
        rows = [dict(row) for row in reader]
    if len(rows) != expected_count:
        raise ValueError(f"{split} must contain {expected_count} rows, found {len(rows)}")
    for row in rows:
        for field in ("sample_id", "patent_id"):
            if not row[field].strip():
                raise ValueError(f"{field} is empty in {split}")
        evaluation = row["human_evaluation"].strip().lower()
        if evaluation not in HUMAN_EVALUATION_TO_LABEL:
            raise ValueError(
                f"Invalid human_evaluation for {row['sample_id']}: {row['human_evaluation']}"
            )
        row["human_evaluation"] = evaluation
        row["label"] = HUMAN_EVALUATION_TO_LABEL[evaluation]
        row["assistant_target"] = _annotation_target(row)
        row["data_split"] = split
        if not row["abstract"].strip():
            raise ValueError(f"Paper-style abstract input is empty for {row['sample_id']}")
        if not any(row[field].strip() for field in ("title", "abstract", "claim")):
            raise ValueError(f"Patent text is empty for {row['sample_id']}")
    if set(row["label"] for row in rows) != set(LABEL_TO_ID):
        raise ValueError(f"{split} must contain both labels")
    return rows


def _validate_step3_manifest(
    manifest: Mapping[str, Any],
    *,
    results_path: Path,
    expected_count: int,
    expected_splits: Mapping[str, int],
) -> None:
    if int(manifest.get("target_size", 0)) != expected_count:
        raise ValueError(
            "Step 3 manifest target_size does not match the Step 4 dataset contract"
        )
    human_results = manifest.get("human_results")
    if not isinstance(human_results, dict):
        raise ValueError("Step 3 manifest has no finalized human_results report")
    if human_results.get("counts") != dict(expected_splits):
        raise ValueError("Step 3 finalized split counts differ from Step 4 expectations")
    if human_results.get("source_sha256") != sha256_file(results_path):
        raise ValueError("Step 3 result.csv changed after finalize")


def _read_and_validate_results(path: Path, *, expected_count: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fields = set(reader.fieldnames or ())
        if fields != set(REQUIRED_FIELDS):
            raise ValueError(
                f"{path} fields differ: missing={sorted(set(REQUIRED_FIELDS) - fields)}, "
                f"forbidden={sorted(fields - set(REQUIRED_FIELDS))}"
            )
        rows = [dict(row) for row in reader]
    if len(rows) != expected_count:
        raise ValueError(f"result.csv must contain {expected_count} rows, found {len(rows)}")
    for row in rows:
        evaluation = row["human_evaluation"].strip().lower()
        if evaluation not in HUMAN_EVALUATION_TO_LABEL:
            raise ValueError(
                f"Invalid human_evaluation for {row['sample_id']}: {row['human_evaluation']}"
            )
        row["human_evaluation"] = evaluation
        row["label"] = HUMAN_EVALUATION_TO_LABEL[evaluation]
        row["assistant_target"] = _annotation_target(row)
    return rows


def _validate_splits_match_results(
    split_rows: list[dict[str, str]], result_rows: list[dict[str, str]]
) -> None:
    results_by_id = {row["sample_id"]: row for row in result_rows}
    if len(results_by_id) != len(result_rows):
        raise ValueError("Step 3 result.csv contains duplicate sample_id values")
    split_ids = [row["sample_id"] for row in split_rows]
    if len(set(split_ids)) != len(split_ids) or set(split_ids) != set(results_by_id):
        raise ValueError("Step 3 splits do not contain exactly the result.csv sample_id set")
    for row in split_rows:
        source = results_by_id[row["sample_id"]]
        for field in REQUIRED_FIELDS:
            if row[field] != source[field]:
                raise ValueError(
                    f"Step 3 split changed field {field} for {row['sample_id']}"
                )


def _validate_collection(rows: list[dict[str, str]]) -> dict[str, Any]:
    for field in ("sample_id", "patent_id"):
        counts = Counter(row[field] for row in rows)
        duplicates = sorted(value for value, count in counts.items() if count > 1)
        if duplicates:
            raise ValueError(f"Duplicate {field} values: {duplicates[:5]}")

    text_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        normalized = "\u241f".join(
            " ".join(row[field].split()) for field in ("title", "abstract", "claim")
        )
        text_splits[hashlib.sha256(normalized.encode()).hexdigest()].add(row["data_split"])
    crossed_texts = [digest for digest, splits in text_splits.items() if len(splits) > 1]
    if crossed_texts:
        raise ValueError(f"Exact patent text crosses splits: {crossed_texts[:5]}")

    return {
        "source": "human_results_csv",
        "human_evaluation_mapping": HUMAN_EVALUATION_TO_LABEL,
        "human_evaluation_counts": dict(
            sorted(Counter(row["human_evaluation"] for row in rows).items())
        ),
    }


def _classifier_row(row: Mapping[str, str]) -> dict[str, Any]:
    return {
        field: LABEL_TO_ID[row["label"]] if field == "label_id" else row[field]
        for field in CLASSIFIER_FIELDS
    }


def _sft_row(row: Mapping[str, Any], *, prompt_bundle: PromptBundle) -> dict[str, Any]:
    assistant = json.dumps(
        row["assistant_target"], ensure_ascii=False, separators=(",", ":")
    )
    return {
        "messages": [
            {"role": "system", "content": prompt_bundle.static_prefix},
            {"role": "user", "content": build_dynamic_message(row)},
            {"role": "assistant", "content": assistant},
        ]
    }


def _annotation_target(row: Mapping[str, str]) -> dict[str, Any]:
    sample_id = row.get("sample_id", "<unknown>")
    try:
        annotation = PatentClassification.model_validate(
            {
                "label": row["label"],
                "confidence": float(row["confidence"]),
                "scope_basis": json.loads(row["scope_basis"]),
                "processing_activities": json.loads(row["processing_activities"]),
                "industry_sectors": json.loads(row["industry_sectors"]),
                "technical_scope": row["technical_scope"],
                "legal_scope": row["legal_scope"],
                "evidence": json.loads(row["evidence"]),
                "reason": row["reason"],
                "review_flag": _exact_boolean(row["review_flag"]),
                "review_reason": row["review_reason"],
            }
        )
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid structured annotation for {sample_id}: {exc}") from exc
    for evidence in annotation.evidence:
        if evidence.quote not in row[evidence.field]:
            raise ValueError(
                f"Invalid structured annotation for {sample_id}: "
                f"evidence is not verbatim in {evidence.field}"
            )
    return annotation.model_dump(mode="json")


def _exact_boolean(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"expected true or false, got {value!r}")


def _prepare_output(paths: Step4Paths, *, rebuild: bool) -> None:
    managed = (
        paths.classifier_train,
        paths.classifier_validation,
        paths.classifier_test,
        paths.sft_train,
        paths.sft_validation,
        paths.sft_index,
        paths.manifest,
    )
    existing = [path for path in managed if path.exists()]
    if existing and not rebuild:
        raise FileExistsError(f"Step 4 dataset outputs already exist: {existing}")
    if rebuild:
        for path in managed:
            path.unlink(missing_ok=True)
    for path in managed:
        path.parent.mkdir(parents=True, exist_ok=True)


def _write_jsonl(path: Path, rows: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def paths_as_json(paths: Step4Paths) -> dict[str, str]:
    return {key: str(value) for key, value in asdict(paths).items()}
