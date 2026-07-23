from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from pipeline.step2.prompt import build_dynamic_message, load_prompt_bundle
from pipeline.step4.data import EXPECTED_SPLIT_COUNTS, REQUIRED_FIELDS, prepare_datasets


def test_default_split_contract_is_frozen_gold_8_1_1() -> None:
    assert EXPECTED_SPLIT_COUNTS == {
        "train": 8_000,
        "validation": 1_000,
        "test": 1_000,
    }


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
        [message["role"] for message in row["messages"]]
        == ["system", "user", "assistant"]
        for row in sft_train + sft_validation
    )
    assert all(
        json.loads(row["messages"][-1]["content"])["label"]
        in {"DATA_SECURITY", "OTHER"}
        for row in sft_train + sft_validation
    )
    assert all(
        json.loads(row["messages"][-1]["content"])["reason"]
        for row in sft_train + sft_validation
    )
    production_prompt = load_prompt_bundle()
    first_source = _read_jsonl(paths.classifier_train)[0]
    assert sft_train[0]["messages"][0]["content"] == production_prompt.static_prefix
    assert sft_train[0]["messages"][1]["content"] == build_dynamic_message(first_source)
    assert not hasattr(paths, "sft_test")
    assert manifest["classifier"]["training_loss"] == "unweighted_cross_entropy"
    assert manifest["classifier"]["validation_selection_metric"] == "accuracy"
    assert manifest["classifier"]["prediction_rule"] == "softmax_argmax"
    assert manifest["sft"]["test_exported"] is False
    assert manifest["sft"]["prompt_version"] == production_prompt.prompt_version
    assert manifest["sft"]["assistant_target"] == (
        "step2_compatible_structured_classification"
    )
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
                    "confidence": "0.99",
                    "scope_basis": json.dumps(
                        ["cryptography"] if positive else ["other"], ensure_ascii=False
                    ),
                    "processing_activities": json.dumps(
                        ["storage"] if positive else ["other"], ensure_ascii=False
                    ),
                    "industry_sectors": json.dumps(
                        ["telecommunications"] if positive else ["other"], ensure_ascii=False
                    ),
                    "technical_scope": f"分析专利摘要{text_index}披露的技术方案",
                    "legal_scope": (
                        "属于数据安全范围" if positive else "未跨过数据安全领域边界"
                    ),
                    "evidence": json.dumps(
                        [{"field": "abstract", "quote": f"专利摘要{text_index}"}],
                        ensure_ascii=False,
                    ),
                    "reason": "存在实质保护机制" if positive else "只有普通数据处理",
                    "review_flag": "false",
                    "review_reason": "",
                }
                writer.writerow(row)
                all_rows.append(row)
                global_index += 1
    results = root / "result.csv"
    with results.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=REQUIRED_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "target_size": 20,
                "human_results": {
                    "counts": split_counts,
                    "source_sha256": hashlib.sha256(results.read_bytes()).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    return root


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
