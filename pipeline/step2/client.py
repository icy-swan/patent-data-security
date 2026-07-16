"""One-request Volcengine Ark Responses client reused from v1."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from json_repair import repair_json
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
    normalization_events: tuple[str, ...] = ()


class ClassificationOutputError(ValueError):
    """A model-output failure that retains the response for later diagnosis."""

    def __init__(
        self,
        message: str,
        *,
        raw_text: str = "",
        response_id: str = "",
        actual_model: str = "",
        usage: dict[str, Any] | None = None,
        normalization_events: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.raw_text = raw_text
        self.response_id = response_id
        self.actual_model = actual_model
        self.usage = usage or {}
        self.normalization_events = normalization_events


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
        self.cache_mode = "ark_responses_structured_stable_prefix"
        self._client = client or OpenAI(
            api_key=key,
            base_url=base_url,
            timeout=timeout_seconds,
        )

    def classify(self, patent: Mapping[str, Any]) -> ClassificationResponse:
        """Classify one patent; response association remains local to the runner task."""

        retry_instruction = str(patent.get("_retry_output_instruction", "") or "").strip()
        retry_input_mode = str(patent.get("_retry_input_mode", "") or "").strip()
        input_messages: list[dict[str, str]] = [
            {"role": "system", "content": self.prompt_bundle.static_prefix}
        ]
        if retry_instruction:
            input_messages.append({"role": "system", "content": retry_instruction})
        input_messages.append(
            {"role": "user", "content": build_dynamic_message(patent)}
        )
        request: dict[str, Any] = {
            "model": self.model,
            "input": input_messages,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "patent_classification",
                    "strict": True,
                    "schema": PatentClassification.model_json_schema(),
                }
            },
            "max_output_tokens": 4096,
        }
        started = time.monotonic()
        response = self._client.responses.create(**request)
        elapsed = time.monotonic() - started
        response_id = str(getattr(response, "id", ""))
        actual_model = str(getattr(response, "model", self.model))
        usage = _usage_dict(getattr(response, "usage", None))
        raw_text = str(getattr(response, "output_text", "") or "")
        initial_events: list[str] = []
        if retry_instruction:
            initial_events.append("retry_output_instruction")
        if retry_input_mode:
            initial_events.append(f"retry_input_mode:{retry_input_mode}")
        normalization_events = tuple(initial_events)
        try:
            if not raw_text:
                try:
                    raw_text = _extract_output_text(response)
                except ValueError as error:
                    raw_text = _response_diagnostic(response)
                    raise ClassificationOutputError(
                        f"{error}; response_status={getattr(response, 'status', None)!r}; "
                        f"response_error={getattr(response, 'error', None)!r}",
                        raw_text=raw_text,
                        response_id=response_id,
                        actual_model=actual_model,
                        usage=usage,
                    ) from error
            value, parse_events = _parse_json_object(raw_text)
            value, normalization_events = _normalize_contract(
                value, normalization_events + parse_events
            )
            classification = PatentClassification.model_validate(value)
        except Exception as error:
            if isinstance(error, ClassificationOutputError):
                raise
            raise ClassificationOutputError(
                f"{type(error).__name__}: {error}; "
                f"response_status={getattr(response, 'status', None)!r}; "
                f"incomplete_details={getattr(response, 'incomplete_details', None)!r}; "
                f"response_error={getattr(response, 'error', None)!r}",
                raw_text=raw_text,
                response_id=response_id,
                actual_model=actual_model,
                usage=usage,
                normalization_events=normalization_events,
            ) from error
        prompt_tokens, cached_tokens, cache_write_tokens = _cache_usage(usage)
        cache_hit_ratio = (
            cached_tokens / prompt_tokens
            if cached_tokens is not None and prompt_tokens not in (None, 0)
            else None
        )
        return ClassificationResponse(
            classification=classification,
            response_id=response_id,
            requested_model=self.model,
            actual_model=actual_model,
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
            normalization_events=normalization_events,
        )


def _parse_json_object(raw_text: str) -> tuple[dict[str, Any], tuple[str, ...]]:
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    fragment = cleaned[start : end + 1] if start >= 0 and end >= start else cleaned
    events: list[str] = []
    try:
        value = json.loads(fragment)
    except json.JSONDecodeError:
        value = repair_json(fragment, return_objects=True)
        events.append("json_repair")
    if not isinstance(value, dict):
        if start < 0:
            raise ValueError("Model response does not contain a JSON object")
        raise ValueError("Model response JSON must be an object")
    return value, tuple(events)


def _normalize_contract(
    value: dict[str, Any],
    initial_events: tuple[str, ...] = (),
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Apply only deterministic schema-contract fixes and record each one."""

    normalized = dict(value)
    events = list(initial_events)
    label = normalized.get("label")
    dimension_fields = ("scope_basis", "processing_activities", "industry_sectors")

    for field_name in dimension_fields:
        field_value = normalized.get(field_name)
        if isinstance(field_value, list):
            deduplicated = list(dict.fromkeys(field_value))
            if deduplicated != field_value:
                normalized[field_name] = deduplicated
                events.append(f"deduplicate:{field_name}")

    if label == "OTHER":
        for field_name in dimension_fields:
            if normalized.get(field_name) != ["other"]:
                normalized[field_name] = ["other"]
                events.append(f"other_contract:{field_name}")
    elif label == "DATA_SECURITY":
        for field_name in dimension_fields:
            field_value = normalized.get(field_name)
            if isinstance(field_value, list) and "other" in field_value and len(field_value) > 1:
                normalized[field_name] = [item for item in field_value if item != "other"]
                events.append(f"drop_other:{field_name}")

    if normalized.get("review_flag") is False and normalized.get("review_reason"):
        normalized["review_reason"] = ""
        events.append("clear_review_reason")

    return normalized, tuple(events)


def _response_diagnostic(response: Any) -> str:
    if hasattr(response, "model_dump_json"):
        return str(response.model_dump_json())
    if hasattr(response, "model_dump"):
        return json.dumps(response.model_dump(), ensure_ascii=False, default=str)
    return repr(response)


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
