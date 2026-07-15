"""One-request Volcengine Ark Responses client reused from v1."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from pipeline.step2.prompt import PromptBundle, build_dynamic_message, load_prompt_bundle
from pipeline.step2.schema import PatentClassification

ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"


@dataclass(frozen=True)
class ClassificationResponse:
    classification: PatentClassification
    response_id: str
    requested_model: str
    actual_model: str
    elapsed_seconds: float
    usage: dict[str, Any]
    prompt_tokens: int | None
    cached_tokens: int | None
    cache_write_tokens: int | None
    cache_hit_ratio: float | None
    cache_mode: str
    prompt_version: str
    prefix_sha256: str
    law_sha256: str
    schema_sha256: str
    raw_text: str


class VolcengineArkClient:
    """Send exactly one independent request for each patent.

    Requests use Ark's official ``/api/v3/responses`` endpoint through its documented
    OpenAI-SDK compatibility. Cache observations come only from Ark response usage.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str = ARK_BASE_URL,
        timeout_seconds: float = 180,
        prompt_bundle: PromptBundle | None = None,
        client: OpenAI | None = None,
    ) -> None:
        key = api_key or os.getenv("ARK_API_KEY")
        if client is None and not key:
            raise ValueError("ARK_API_KEY is required for Volcengine Ark requests")
        self.model = model
        self.base_url = base_url
        self.prompt_bundle = prompt_bundle or load_prompt_bundle()
        self.cache_mode = "ark_responses_stable_prefix"
        self._client = client or OpenAI(
            api_key=key,
            base_url=base_url,
            timeout=timeout_seconds,
        )

    def classify(self, patent: Mapping[str, Any]) -> ClassificationResponse:
        """Classify one patent; response association remains local to the runner task."""

        request: dict[str, Any] = {
            "model": self.model,
            "input": [
                {"role": "system", "content": self.prompt_bundle.static_prefix},
                {"role": "user", "content": build_dynamic_message(patent)},
            ],
        }
        started = time.monotonic()
        response = self._client.responses.create(**request)
        elapsed = time.monotonic() - started
        raw_text = getattr(response, "output_text", "") or _extract_output_text(response)
        classification = PatentClassification.model_validate(_parse_json_object(raw_text))
        usage = _usage_dict(getattr(response, "usage", None))
        prompt_tokens, cached_tokens, cache_write_tokens = _cache_usage(usage)
        cache_hit_ratio = (
            cached_tokens / prompt_tokens
            if cached_tokens is not None and prompt_tokens not in (None, 0)
            else None
        )
        return ClassificationResponse(
            classification=classification,
            response_id=str(getattr(response, "id", "")),
            requested_model=self.model,
            actual_model=str(getattr(response, "model", self.model)),
            elapsed_seconds=elapsed,
            usage=usage,
            prompt_tokens=prompt_tokens,
            cached_tokens=cached_tokens,
            cache_write_tokens=cache_write_tokens,
            cache_hit_ratio=cache_hit_ratio,
            cache_mode=self.cache_mode,
            prompt_version=self.prompt_bundle.prompt_version,
            prefix_sha256=self.prompt_bundle.prefix_sha256,
            law_sha256=self.prompt_bundle.law_sha256,
            schema_sha256=self.prompt_bundle.schema_sha256,
            raw_text=raw_text,
        )


def _parse_json_object(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Model response does not contain a JSON object")
    value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Model response JSON must be an object")
    return value


def _extract_output_text(response: Any) -> str:
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", "") != "message":
            continue
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", "")
            if text:
                return str(text)
    raise ValueError("Response did not contain output text")


def _usage_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        value = usage.model_dump()
        return value if isinstance(value, dict) else {}
    if isinstance(usage, dict):
        return usage
    return {}


def _cache_usage(usage: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    prompt_tokens = _optional_int(usage.get("input_tokens", usage.get("prompt_tokens")))
    details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    if hasattr(details, "model_dump"):
        details = details.model_dump()
    if not isinstance(details, dict):
        details = {}
    cached = _optional_int(details.get("cached_tokens"))
    written = _optional_int(details.get("cache_write_tokens"))
    return prompt_tokens, cached, written


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
