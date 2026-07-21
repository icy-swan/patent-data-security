from __future__ import annotations

import json

from pipeline.step3.prompt import build_annotation_message, load_annotation_prompt


def test_dynamic_message_contains_patent_only_and_no_step2_decision() -> None:
    message = build_annotation_message(
        {
            "patent_id": "CN-test",
            "title": "测试名称",
            "abstract": "测试摘要",
            "claim": "测试权利要求",
            "ipc": "G06F",
            "main_ipc": "G06F21/00",
            "step2_label": "DATA_SECURITY",
            "step2_confidence": 1.0,
            "route": "S",
        }
    )
    payload = json.loads(message.split("\n", 1)[1])

    assert payload == {
        "patent_id": "CN-test",
        "title": "测试名称",
        "abstract": "测试摘要",
        "claim": "测试权利要求",
        "ipc": "G06F",
        "main_ipc": "G06F21/00",
    }


def test_prompt_and_schema_have_frozen_hashes() -> None:
    prompt = load_annotation_prompt()

    assert prompt.version == "step3-independent-binary-v2.3.0"
    assert len(prompt.prompt_sha256) == 64
    assert len(prompt.schema_sha256) == 64
