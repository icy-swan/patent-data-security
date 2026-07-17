from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from pipeline.step3.runner import _exclusive_lock, _runner_active
from pipeline.step3.sampling import (
    RESULT_FIELDS,
    _balanced_capacity_allocation,
    _initialize_task_database,
    _sampling_group,
    assign_exact_splits,
    discover_step2_databases,
    finalize_human_results,
    step3_paths,
)


def test_sampling_groups_prioritize_positives_and_s_to_other_hard_negatives() -> None:
    assert _sampling_group("S", "DATA_SECURITY") == "positive"
    assert _sampling_group("E", "DATA_SECURITY") == "positive"
    assert _sampling_group("S", "OTHER") == "hard_negative"
    assert _sampling_group("E", "OTHER") == ""


def test_balanced_allocation_redistributes_only_after_a_year_hits_capacity() -> None:
    allocation = _balanced_capacity_allocation(
        {"2019": 2, "2020": 10, "2021": 10},
        12,
        seed="test",
    )

    assert allocation["2019"] == 2
    assert sum(allocation.values()) == 12
    assert abs(allocation["2020"] - allocation["2021"]) <= 1


def test_balanced_allocation_rejects_a_label_shortage() -> None:
    with pytest.raises(ValueError, match="cannot be met"):
        _balanced_capacity_allocation({"2020": 5, "2021": 4}, 10, seed="test")


def test_exact_split_is_stable_and_stratified() -> None:
    rows = [
        {
            "sample_id": f"{year}-{label}-{index}",
            "application_year": str(year),
            "label": label,
        }
        for year in range(2012, 2022)
        for label in ("DATA_SECURITY", "OTHER")
        for index in range(200)
    ]

    first, report = assign_exact_splits(rows, seed="stable-split")
    second, _ = assign_exact_splits(rows, seed="stable-split")

    assert Counter(row["data_split"] for row in first) == {
        "train": 3_200,
        "validation": 400,
        "test": 400,
    }
    assert report["counts"] == {"train": 3_200, "validation": 400, "test": 400}
    assert [(row["sample_id"], row["data_split"]) for row in first] == [
        (row["sample_id"], row["data_split"]) for row in second
    ]
    per_stratum = Counter(
        (row["application_year"], row["label"], row["data_split"]) for row in first
    )
    assert set(per_stratum.values()) == {20, 160}


def test_exact_text_duplicates_never_cross_splits() -> None:
    rows = [
        {
            "sample_id": f"sample-{index}",
            "application_year": "2021",
            "label": "DATA_SECURITY" if index < 50 else "OTHER",
            "title": "重复文本" if index in {0, 1} else f"标题{index}",
            "abstract": "相同摘要" if index in {0, 1} else f"摘要{index}",
            "claim": "相同权利要求" if index in {0, 1} else f"权利要求{index}",
        }
        for index in range(100)
    ]

    assigned, _ = assign_exact_splits(rows, seed="grouped-split")
    duplicates = [row for row in assigned if row["sample_id"] in {"sample-0", "sample-1"}]

    assert len({row["split_group_id"] for row in duplicates}) == 1
    assert len({row["data_split"] for row in duplicates}) == 1
    assert Counter(row["data_split"] for row in assigned) == {
        "train": 80,
        "validation": 10,
        "test": 10,
    }


def test_database_discovery_excludes_retry_backups(tmp_path: Path) -> None:
    current = tmp_path / "2021" / "tasks.sqlite3"
    current.parent.mkdir()
    backup = tmp_path / "step2_tasks_2021.before_retry_20260716.sqlite3"
    current.touch()
    backup.touch()

    assert discover_step2_databases(tmp_path) == [current]


def test_step3_paths_use_flat_runtime_and_dataset_directories(tmp_path: Path) -> None:
    paths = step3_paths(tmp_path / "step3")

    assert paths.database == paths.root / "tasks.sqlite3"
    assert paths.manifest == paths.root / "manifest.json"
    assert paths.progress == paths.root / "progress.json"
    assert paths.simulation == paths.root / "simulation.csv"
    assert paths.results == paths.root / "result.csv"
    assert paths.train == paths.root / "dataset" / "train.csv"


def test_status_check_does_not_create_a_stale_lock(tmp_path: Path) -> None:
    database = tmp_path / "state" / "tasks.sqlite3"
    database.parent.mkdir()
    database.touch()

    assert _runner_active(database) is False
    assert not database.with_name("tasks.sqlite3.run.lock").exists()


def test_runner_removes_lock_after_exit(tmp_path: Path) -> None:
    database = tmp_path / "state" / "tasks.sqlite3"
    database.parent.mkdir()
    database.touch()
    lock = database.with_name("tasks.sqlite3.run.lock")

    with _exclusive_lock(database):
        assert lock.is_file()

    assert not lock.exists()


def test_finalize_human_results_strips_metadata_and_creates_clean_splits(
    tmp_path: Path,
) -> None:
    paths = step3_paths(tmp_path / "step3")
    _write_human_result_fixture(paths, extra_fields=True)

    report = finalize_human_results(paths, expected_count=20, split_seed="test-human-split")

    result_fields, results = _read_csv(paths.results)
    assert result_fields == list(RESULT_FIELDS)
    assert len(results) == 20
    assert {row["human_evaluation"] for row in results} == {"true", "false"}
    assert report["removed_input_fields"] == ["confidence", "review_flag", "step2_label"]
    assert report["counts"] == {"train": 16, "validation": 2, "test": 2}
    for split, path in {
        "train": paths.train,
        "validation": paths.validation,
        "test": paths.test,
    }.items():
        fields, rows = _read_csv(path)
        assert fields == list(RESULT_FIELDS)
        assert len(rows) == report["counts"][split]
        assert "label" not in fields
        assert "data_split" not in fields
        assert "annotation_model" not in fields


def test_finalize_human_results_rejects_non_boolean_human_label(tmp_path: Path) -> None:
    paths = step3_paths(tmp_path / "step3")
    _write_human_result_fixture(paths, first_evaluation="yes")

    with pytest.raises(ValueError, match="must be exactly true or false"):
        finalize_human_results(paths, expected_count=20)


def test_finalize_human_results_rejects_changed_frozen_text(tmp_path: Path) -> None:
    paths = step3_paths(tmp_path / "step3")
    _write_human_result_fixture(paths, changed_title=True)

    with pytest.raises(ValueError, match="changed frozen field title"):
        finalize_human_results(paths, expected_count=20)


def _write_human_result_fixture(
    paths,
    *,
    extra_fields: bool = False,
    first_evaluation: str = "true",
    changed_title: bool = False,
) -> None:
    paths.results.parent.mkdir(parents=True, exist_ok=True)
    frozen_rows = []
    result_rows = []
    for index in range(20):
        frozen = {
            "sample_id": f"sample-{index}",
            "dataset_id": "2021",
            "application_year": "2021",
            "patent_id": f"CN{index}",
            "title": f"专利名称{index}",
            "abstract": f"专利摘要{index}",
            "claim": f"主权项{index}",
            "ipc": "G06F21/00",
            "main_ipc": "G06F21/00",
        }
        evaluation = first_evaluation if index == 0 else "true" if index % 2 == 0 else "false"
        positive = evaluation.lower() == "true"
        result = {
            **frozen,
            "human_evaluation": evaluation,
            "scope_basis": json.dumps(
                ["cryptography"] if positive else ["other"], ensure_ascii=False
            ),
            "industry_sectors": json.dumps(
                ["telecommunications"] if positive else ["other"], ensure_ascii=False
            ),
            "confidence": "0.99",
            "review_flag": "false",
            "step2_label": "DATA_SECURITY",
        }
        if changed_title and index == 0:
            result["title"] = "被修改的名称"
        frozen_rows.append(frozen)
        result_rows.append(result)
    _initialize_task_database(paths.database, frozen_rows)
    paths.manifest.write_text(
        json.dumps({"outputs": {}}, ensure_ascii=False),
        encoding="utf-8",
    )
    fields = list(RESULT_FIELDS)
    if extra_fields:
        fields.extend(("confidence", "review_flag", "step2_label"))
    with paths.results.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result_rows)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or ()), list(reader)
