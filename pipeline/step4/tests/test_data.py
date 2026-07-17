from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from pipeline.step4.data import REQUIRED_FIELDS, prepare_datasets


def test_prepare_exports_classifier_and_blind_test_free_sft(tmp_path: Path) -> None:
    step3 = _write_step3_fixture(tmp_path / "step3")
    paths, manifest = prepare_datasets(
        step3,
        tmp_path / "step4",
        expected_counts={"train": 16, "validation": 2, "test": 2},
    )

    assert _read_jsonl(paths.classifier_train)[0]["label_id"] in {0, 1}
    assert len(_read_jsonl(paths.classifier_train)) == 16
    assert len(_read_jsonl(paths.classifier_validation)) == 2
    assert len(_read_jsonl(paths.classifier_test)) == 2

    sft_train = _read_jsonl(paths.sft_train)
    sft_validation = _read_jsonl(paths.sft_validation)
    assert len(sft_train) == 16
    assert len(sft_validation) == 2
    assert all(set(row) == {"messages"} for row in sft_train + sft_validation)
    assert all(
        [message["role"] for message in row["messages"]] == ["user", "assistant"]
        for row in sft_train + sft_validation
    )
    assert all(
        row["messages"][-1]["content"] in {"DATA_SECURITY", "OTHER"}
        for row in sft_train + sft_validation
    )
    assert not hasattr(paths, "sft_test")
    assert manifest["classifier"]["training_loss"] == "unweighted_cross_entropy"
    assert manifest["classifier"]["validation_selection_metric"] == "accuracy"
    assert manifest["classifier"]["prediction_rule"] == "softmax_argmax"
    assert manifest["sft"]["test_exported"] is False
    assert manifest["annotation_provenance"]["source"] == "human_results_csv"
    assert manifest["annotation_provenance"]["human_evaluation_counts"] == {
        "false": 10,
        "true": 10,
    }


def test_prepare_rejects_exact_text_crossing_datasets(tmp_path: Path) -> None:
    step3 = _write_step3_fixture(tmp_path / "step3", cross_split_text=True)

    with pytest.raises(ValueError, match="Exact patent text crosses splits"):
        prepare_datasets(
            step3,
            tmp_path / "step4",
            expected_counts={"train": 16, "validation": 2, "test": 2},
        )


def _write_step3_fixture(root: Path, *, cross_split_text: bool = False) -> Path:
    split_counts = {"train": 16, "validation": 2, "test": 2}
    global_index = 0
    all_rows: list[dict[str, str]] = []
    for split, count in split_counts.items():
        path = root / "dataset" / f"{split}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=REQUIRED_FIELDS)
            writer.writeheader()
            for index in range(count):
                positive = index % 2 == 0
                text_index = 0 if cross_split_text and index == 0 else global_index
                row = {
                    "sample_id": f"sample-{global_index}",
                    "dataset_id": "2021",
                    "application_year": "2021",
                    "patent_id": f"CN{global_index}",
                    "title": f"专利名称{text_index}",
                    "abstract": f"专利摘要{text_index}",
                    "claim": f"主权项{text_index}",
                    "ipc": "G06F21/00",
                    "main_ipc": "G06F21/00",
                    "human_evaluation": "true" if positive else "false",
                    "scope_basis": json.dumps(
                        ["cryptography"] if positive else ["other"], ensure_ascii=False
                    ),
                    "industry_sectors": json.dumps(
                        ["telecommunications"] if positive else ["other"], ensure_ascii=False
                    ),
                }
                writer.writerow(row)
                all_rows.append(row)
                global_index += 1
    results = root / "result.csv"
    with results.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=REQUIRED_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    return root


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
