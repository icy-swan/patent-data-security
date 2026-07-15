import json
from types import SimpleNamespace

import pytest

from pipeline.step2.client import OpenAICompatibleClient
from pipeline.step2.prompt import (
    ROUTING_FIELDS,
    build_dynamic_message,
    build_dynamic_payload,
    load_prompt_bundle,
)
from pipeline.step2.schema import PatentClassification


def valid_result() -> dict:
    return {
        "label": "DATA_SECURITY",
        "confidence": 0.94,
        "scope_basis": ["cryptography"],
        "technical_scope": "专利披露密钥协商和加密传输。",
        "legal_scope": "该机制使传输数据处于有效保护状态。",
        "evidence": [{"field": "claim", "quote": "通过密钥协商加密传输数据"}],
        "reason": "密码协议属于数据安全基础技术。",
        "review_flag": False,
        "review_reason": "",
    }


def test_law_and_scope_resources_are_complete_and_aligned() -> None:
    bundle = load_prompt_bundle()

    assert bundle.law_text.count("第五十五条") == 1
    assert "中华人民共和国数据安全法" in bundle.law_text
    assert "网络攻击、网络侵入、系统漏洞、恶意程序" in bundle.static_prefix
    assert "密码技术本身即属于范围" in bundle.static_prefix
    assert bundle.resource_hashes["law_text"] == bundle.law_sha256
    assert bundle.resource_hashes["static_prefix"] == bundle.prefix_sha256


def test_dynamic_payload_keeps_patent_id_but_excludes_step1_routing() -> None:
    patent = {
        "patent_id": "CN202610000001.0",
        "title": "密码协议",
        "abstract": "使用密钥协商",
        "claim": "通过密钥协商加密传输数据",
        "ipc": "H04L 9/00",
        "main_ipc": "H04L 9/00",
        "route": "S",
        "selection_group": "S_all",
        "keyword_hits": [{"matched_text": "加密"}],
    }

    payload = build_dynamic_payload(patent)
    message = build_dynamic_message(patent)

    assert payload["patent_id"] == "CN202610000001.0"
    assert set(payload).isdisjoint(ROUTING_FIELDS)
    assert "CN202610000001.0" in message
    assert "S_all" not in message
    assert "keyword_hits" not in message


def test_client_uses_stable_prefix_and_observes_cache_usage() -> None:
    bundle = load_prompt_bundle()

    class FakeResponses:
        def __init__(self) -> None:
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                id="resp-test",
                model="actual-model",
                output_text=json.dumps(valid_result(), ensure_ascii=False),
                usage=SimpleNamespace(
                    model_dump=lambda: {
                        "input_tokens": 8000,
                        "input_tokens_details": {
                            "cached_tokens": 7000,
                            "cache_write_tokens": 0,
                        },
                        "output_tokens": 300,
                    }
                ),
            )

    responses = FakeResponses()
    client = OpenAICompatibleClient(
        model="requested-model",
        prompt_bundle=bundle,
        prompt_cache_key="pds-v2-cache",
        client=SimpleNamespace(responses=responses),
    )
    result = client.classify(
        {
            "patent_id": "CN1",
            "claim": "通过密钥协商加密传输数据",
        }
    )

    call = responses.calls[0]
    assert call["input"][0]["content"] == bundle.static_prefix
    assert "CN1" in call["input"][1]["content"]
    assert call["prompt_cache_key"] == "pds-v2-cache"
    assert result.classification.label == "DATA_SECURITY"
    assert result.cached_tokens == 7000
    assert result.cache_hit_ratio == 0.875


def test_other_schema_cannot_claim_positive_scope() -> None:
    value = valid_result()
    value.update({"label": "OTHER", "scope_basis": ["cryptography"]})
    with pytest.raises(ValueError, match="OTHER requires"):
        PatentClassification.model_validate(value)
