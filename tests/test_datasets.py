from pathlib import Path

import pytest

from patent_data_security.datasets import dataset_id, discover_files


def test_dataset_id_prefers_year_and_discovery_rejects_collisions(tmp_path: Path) -> None:
    first = tmp_path / "patents_2020.csv"
    second = tmp_path / "上市公司专利_2021.csv"
    first.touch()
    second.touch()

    assert dataset_id(first) == "2020"
    assert [path.name for path in discover_files(None, tmp_path, "*.csv")] == [
        "patents_2020.csv",
        "上市公司专利_2021.csv",
    ]

    collision = tmp_path / "other_2020.csv"
    collision.touch()
    with pytest.raises(ValueError, match="collide"):
        discover_files(None, tmp_path, "*.csv")
