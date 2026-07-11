import json
from types import SimpleNamespace

from patent_data_security.step2_prompt import (
    VolcengineArkClient,
    build_classification_prompt,
)


def valid_label() -> dict:
    return {
        "cat": 1,
        "confidence": 0.91,
        "subtype": "privacy_computing",
        "core_invention": "通过联邦建模机制处理并保护模型参数。",
        "evidence_chain": {
            "protected_object_or_activity": "联合建模中的模型参数处理",
            "security_goal_or_risk": "防止模型参数泄露",
            "technical_mechanism": "联邦学习",
            "causal_centrality": "保护参数是核心技术效果",
            "missing_or_ambiguous_link": "",
        },
        "evidence": ["采用联邦学习保护模型参数"],
        "reason": "核心方案保护联合建模中的模型参数。",
        "review_flag": False,
        "review_reason": "",
    }


class FakeResponses:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            id="resp-test",
            model="ark-test-model",
            output_text=json.dumps(valid_label(), ensure_ascii=False),
            usage=SimpleNamespace(model_dump=lambda: {"input_tokens": 10, "output_tokens": 20}),
        )


def test_prompt_excludes_routing_hints_and_base_client_makes_one_request() -> None:
    patent = {
        "patent_id": "CN1",
        "title": "联合建模方法",
        "abstract": "采用联邦学习保护模型参数",
        "claim": "一种联合建模方法",
        "keyword_level": "S",
        "keyword_hits": [
            {
                "keyword": "联邦学习",
                "context_hits": [{"context_id": "CTX-DATA-OBJECT", "keyword": "模型参数"}],
            }
        ],
    }
    prompt = build_classification_prompt(patent)
    assert "CTX-DATA-OBJECT" not in prompt
    assert "联邦学习保护模型参数" in prompt
    assert "关键词层级" not in prompt
    assert "只返回一个 JSON 对象" not in prompt

    responses = FakeResponses()
    client = VolcengineArkClient(
        model="requested-model",
        client=SimpleNamespace(responses=responses),
    )
    result = client.classify(patent)

    assert len(responses.calls) == 1
    assert responses.calls[0]["model"] == "requested-model"
    assert result.classification.cat == 1
    assert result.actual_model == "ark-test-model"


def test_category_two_requires_uncertain_subtype_and_review() -> None:
    label = valid_label()
    label.update(
        {
            "cat": 2,
            "subtype": "potential_data_security",
            "review_flag": True,
            "review_reason": "保护机制是否直接作用于数据仍需核验。",
        }
    )
    label["evidence_chain"]["missing_or_ambiguous_link"] = (
        "主权项未明确保护机制是否直接作用于数据。"
    )
    responses = FakeResponses()
    responses.create = lambda **_kwargs: SimpleNamespace(
        id="resp-2",
        model="ark-test-model",
        output_text=json.dumps(label, ensure_ascii=False),
        usage=None,
    )
    client = VolcengineArkClient(
        model="requested-model", client=SimpleNamespace(responses=responses)
    )
    result = client.classify({"title": "边界专利"})
    assert result.classification.cat == 2
    assert result.classification.review_flag is True
