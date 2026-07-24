from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from pipeline.step3.k3_review import (
    AGENT_PLAN_BASE_URL,
    DEFAULT_K3_MODEL,
    K3_RESULT_FIELDS,
    AgentPlanKimiReviewClient,
    K3Review,
    K3ReviewResponse,
    _normalize_k3_review,
    prepare_k3_review,
    run_k3_reviews,
)
from pipeline.step3.sampling import (
    MANUAL_REVIEW_FIELDS,
    NEGATIVE_COHORT,
    POSITIVE_COHORT,
)


def test_k3_client_sends_one_patent_and_requests_only_two_review_fields() -> None:
    class FakeResponses:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def create(self, **request: Any) -> Any:
            self.calls.append(request)
            return SimpleNamespace(
                id="resp-k3",
                model=DEFAULT_K3_MODEL,
                usage=SimpleNamespace(input_tokens=100, output_tokens=20, total_tokens=120),
                output_text=json.dumps(
                    {
                        "k3_review_label": "DATA_SECURITY",
                        "k3_reason": "主权项明确披露对隐私数据进行加密存储。",
                    },
                    ensure_ascii=False,
                ),
            )

    responses = FakeResponses()
    client = AgentPlanKimiReviewClient(
        api_key_kind="agent-plan",
        client=SimpleNamespace(responses=responses),
    )
    result = client.review(
        {
            "sample_id": "sample-1",
            "patent_id": "CN1",
            "title": "隐私数据加密方法",
            "abstract": "对隐私数据进行加密。",
            "claim": "将隐私数据加密后存储。",
            "step2_label": "OTHER",
            "step2_reason": "待复核",
            "human_review_label": "",
            "human_reason": "",
        }
    )

    assert result.review.k3_review_label == "DATA_SECURITY"
    assert len(responses.calls) == 1
    request = responses.calls[0]
    assert request["model"] == "kimi-k3"
    assert request["max_output_tokens"] == 4_096
    assert set(request["text"]["format"]["schema"]["properties"]) == {
        "k3_review_label",
        "k3_reason",
    }
    system_prompt = request["input"][0]["content"]
    assert "<中华人民共和国数据安全法_全文>" in system_prompt
    assert "第五十五条" in system_prompt
    assert len(system_prompt) > 6_000
    assert result.law_sha256 == client.law_sha256
    dynamic_message = request["input"][1]["content"]
    assert dynamic_message.count('"sample_id"') == 1
    assert "human_review_label" not in dynamic_message
    assert "human_reason" not in dynamic_message


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            {
                "k3_label": "OTHER",
                "k3_final_label": "OTHER",
                "k3_reason": "专利只涉及机械结构，不含数据安全技术措施。",
                "k3_confidence": 0.98,
            },
            "OTHER",
        ),
        (
            {
                "sample_id": "sample-1",
                "patent_id": "CN1",
                "step3_label": "DATA_SECURITY",
                "k3_reason": "主权项披露加密与身份鉴别技术。",
                "evidence": [],
            },
            "DATA_SECURITY",
        ),
    ],
)
def test_normalizes_observed_kimi_aliases(
    value: dict[str, Any],
    expected: str,
) -> None:
    review = _normalize_k3_review(
        value,
        expected_sample_id="sample-1",
        expected_patent_id="CN1",
    )

    assert review.k3_review_label == expected
    assert review.k3_reason


def test_k3_client_rejects_non_agent_plan_configuration() -> None:
    with pytest.raises(ValueError, match="Agent Plan base URL"):
        AgentPlanKimiReviewClient(
            api_key_kind="agent-plan",
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            client=SimpleNamespace(),
        )
    with pytest.raises(ValueError, match="ARK_API_KEY_KIND=agent-plan"):
        AgentPlanKimiReviewClient(
            api_key_kind="platform",
            client=SimpleNamespace(),
        )


def test_k3_runner_keeps_separate_state_and_appends_exact_review_columns(
    tmp_path: Path,
) -> None:
    root = tmp_path / "step3"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps({"target_size": 2}, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_review_input(
        root / "need_manual_review_positive.csv",
        sample_id="sample-positive",
        patent_id="CN-P",
        cohort=POSITIVE_COHORT,
        step2_label="DATA_SECURITY",
    )
    _write_review_input(
        root / "need_manual_review_negative.csv",
        sample_id="sample-negative",
        patent_id="CN-N",
        cohort=NEGATIVE_COHORT,
        step2_label="OTHER",
    )

    paths, manifest = prepare_k3_review(root)

    assert paths.database.name == "k3_tasks.sqlite3"
    assert paths.result.name == "k3_result.csv"
    assert manifest["request_granularity"] == "one_patent_per_request"
    assert not paths.result.exists()
    connection = sqlite3.connect(paths.database)
    assert connection.execute("SELECT status,COUNT(*) FROM tasks GROUP BY status").fetchall() == [
        ("pending", 2)
    ]
    connection.close()

    class FakeClient:
        model = DEFAULT_K3_MODEL
        base_url = AGENT_PLAN_BASE_URL
        prompt_version = "test-prompt"
        prompt_sha256 = "prompt-sha"
        law_sha256 = "law-sha"
        schema_sha256 = "schema-sha"

        def __init__(self) -> None:
            self.seen: list[str] = []

        def review(self, row: dict[str, str]) -> K3ReviewResponse:
            self.seen.append(row["sample_id"])
            label = row["step2_label"]
            return K3ReviewResponse(
                review=K3Review(
                    k3_review_label=label,
                    k3_reason=f"根据专利正文复核，维持 {label} 标签。",
                ),
                response_id=f"resp-{row['sample_id']}",
                requested_model=self.model,
                actual_model=self.model,
                prompt_version=self.prompt_version,
                prompt_sha256=self.prompt_sha256,
                law_sha256=self.law_sha256,
                schema_sha256=self.schema_sha256,
                elapsed_seconds=0.01,
                usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                raw_text="{}",
            )

    client = FakeClient()
    progress = run_k3_reviews(
        paths,
        client,  # type: ignore[arg-type]
        concurrency=2,
        max_attempts=1,
        retry_delay_seconds=0,
    )

    assert progress["succeeded"] == 2
    assert sorted(client.seen) == ["sample-negative", "sample-positive"]
    with paths.result.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    assert tuple(reader.fieldnames or ()) == K3_RESULT_FIELDS
    assert [row["sample_id"] for row in rows] == [
        "sample-positive",
        "sample-negative",
    ]
    assert [row["k3_review_label"] for row in rows] == [
        "DATA_SECURITY",
        "OTHER",
    ]
    assert all(row["k3_reason"] for row in rows)
    assert all(not row["human_review_label"] and not row["human_reason"] for row in rows)
    assert not (root / "tasks.sqlite3").exists()


def _write_review_input(
    path: Path,
    *,
    sample_id: str,
    patent_id: str,
    cohort: str,
    step2_label: str,
) -> None:
    row = {field: "" for field in MANUAL_REVIEW_FIELDS}
    row.update(
        {
            "sample_id": sample_id,
            "dataset_id": "2024",
            "application_year": "2024",
            "patent_id": patent_id,
            "title": "测试专利",
            "abstract": "测试摘要",
            "claim": "测试主权项",
            "ipc": "G06F21/00",
            "main_ipc": "G06F21/00",
            "sample_cohort": cohort,
            "step1_label": "DATA_SECURITY",
            "step2_label": step2_label,
            "step2_confidence": "0.9",
            "step2_scope_basis": json.dumps(["other"]),
            "step2_processing_activities": json.dumps(["other"]),
            "step2_industry_sectors": json.dumps(["other"]),
            "step2_technical_scope": "测试",
            "step2_legal_scope": "测试",
            "step2_evidence": "[]",
            "step2_reason": "测试理由",
            "step2_needs_review": "false",
        }
    )
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANUAL_REVIEW_FIELDS)
        writer.writeheader()
        writer.writerow(row)
