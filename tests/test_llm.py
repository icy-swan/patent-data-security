import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from patent_data_security.classification import PatentClassification
from patent_data_security.llm import merge_batch_outputs, prepare_batch_files


def valid_label() -> dict:
    return {
        "cat": 1,
        "confidence": 0.91,
        "subtype": "privacy_computing",
        "evidence": ["采用联邦学习保护模型参数"],
        "reason": "核心方案保护联合建模中的模型参数。",
        "review_flag": False,
        "review_reason": "",
    }


def test_classification_schema_has_only_three_classes_and_consistent_subtypes() -> None:
    assert PatentClassification.model_validate(valid_label()).cat == 1

    invalid = {**valid_label(), "cat": 3}
    with pytest.raises(ValidationError):
        PatentClassification.model_validate(invalid)


def test_prepare_and_merge_batch_files(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.jsonl"
    candidate = {
        "custom_id": "patent-2",
        "title": "联合建模方法",
        "abstract": "采用联邦学习保护模型参数",
        "claim": "一种联合建模方法",
        "ipc": "G06F21/62",
        "main_ipc": "G06F21/62",
        "keyword_level": "S",
        "ipc_level": "S",
        "route_level": "S",
    }
    candidates.write_text(json.dumps(candidate, ensure_ascii=False) + "\n", encoding="utf-8")

    prepared = prepare_batch_files(candidates, tmp_path / "batches", model="test-model")
    request = json.loads(prepared.files[0].read_text(encoding="utf-8"))

    assert prepared.requests == 1
    assert request["custom_id"] == "patent-2"
    assert request["url"] == "/v1/chat/completions"
    assert request["body"]["response_format"]["type"] == "json_schema"

    response = {
        "custom_id": "patent-2",
        "response": {"body": {"choices": [{"message": {"content": json.dumps(valid_label())}}]}},
    }
    batch_output = tmp_path / "batch-output.jsonl"
    batch_output.write_text(json.dumps(response) + "\n", encoding="utf-8")
    destination = tmp_path / "classifications.csv"

    counts = merge_batch_outputs([batch_output], destination, model_name="test-model")

    assert counts == {"validated": 1, "failed": 0}
    assert "privacy_computing" in destination.read_text(encoding="utf-8")
