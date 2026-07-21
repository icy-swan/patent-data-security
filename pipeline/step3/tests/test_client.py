from __future__ import annotations

import json

from pipeline.step3.client import _normalized_batch
from pipeline.step3.schema import CodexAnnotationBatch


def test_normalizes_other_conflicts_before_contract_validation() -> None:
    raw = json.dumps(
        {
            "annotations": [
                {
                    "sample_id": "sample-1",
                    "label": "DATA_SECURITY",
                    "confidence": 0.9,
                    "scope_basis": ["data_confidentiality", "other"],
                    "processing_activities": ["storage", "other", "storage"],
                    "industry_sectors": ["other"],
                    "technical_scope": "加密存储",
                    "legal_scope": "保护数据保密性",
                    "evidence": [{"field": "claim", "quote": "对数据进行加密存储"}],
                    "reason": "权利要求披露加密存储。",
                    "needs_review": False,
                    "review_reason": "应被清空",
                }
            ]
        },
        ensure_ascii=False,
    )

    normalized = _normalized_batch(raw)
    parsed = CodexAnnotationBatch.model_validate(normalized)
    annotation = parsed.annotations[0]

    assert annotation.scope_basis == ["data_confidentiality"]
    assert annotation.processing_activities == ["storage"]
    assert annotation.industry_sectors == ["other"]
    assert annotation.review_reason == ""
