from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from pipeline.step3.sampling import (
    _balanced_capacity_allocation,
    _sampling_group,
    assign_exact_splits,
    discover_step2_databases,
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
    current = tmp_path / "step2_tasks_2021.sqlite3"
    backup = tmp_path / "step2_tasks_2021.before_retry_20260716.sqlite3"
    current.touch()
    backup.touch()

    assert discover_step2_databases(tmp_path) == [current]
