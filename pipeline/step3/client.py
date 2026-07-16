"""OpenAI Responses client for provisional independent Step 3 annotation."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from pipeline.step3.prompt import (
    AnnotationPrompt,
    build_annotation_message,
    load_annotation_prompt,
)
from pipeline.step3.schema import IndependentAnnotation

DEFAULT_MODEL = "gpt-5.6-sol"


@dataclass(frozen=True)
class AnnotationResponse:
    annotation: IndependentAnnotation
    response_id: str
    requested_model: str
    actual_model: str
    prompt_version: str
    prompt_sha256: str
    schema_sha256: str
    elapsed_seconds: float
    usage: dict[str, Any]
    raw_response: str


class OpenAIAnnotationClient:
    """Make one stored-disabled structured request for each blinded patent."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        reasoning_effort: str = "high",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 300,
        prompt: AnnotationPrompt | None = None,
        client: OpenAI | None = None,
    ) -> None:
        key = api_key or os.getenv("OPENAI_API_KEY")
        if client is None and not key:
            raise ValueError("OPENAI_API_KEY is required for GPT-5.6 simulation")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.prompt = prompt or load_annotation_prompt()
        self._client = client or OpenAI(
            api_key=key,
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
            timeout=timeout_seconds,
        )

    def annotate(self, patent: Mapping[str, Any]) -> AnnotationResponse:
        started = time.monotonic()
        response = self._client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": self.prompt.text},
                {"role": "user", "content": build_annotation_message(patent)},
            ],
            text_format=IndependentAnnotation,
            reasoning={"effort": self.reasoning_effort},
            max_output_tokens=3_000,
            store=False,
        )
        elapsed = time.monotonic() - started
        annotation = response.output_parsed
        if annotation is None:
            refusal = _refusal_text(response)
            raise ValueError(f"Response has no parsed annotation; refusal={refusal!r}")
        usage = _model_dump(getattr(response, "usage", None))
        return AnnotationResponse(
            annotation=annotation,
            response_id=str(getattr(response, "id", "")),
            requested_model=self.model,
            actual_model=str(getattr(response, "model", self.model)),
            prompt_version=self.prompt.version,
            prompt_sha256=self.prompt.prompt_sha256,
            schema_sha256=self.prompt.schema_sha256,
            elapsed_seconds=elapsed,
            usage=usage,
            raw_response=_response_json(response),
        )


def _model_dump(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    return value if isinstance(value, dict) else {}


def _response_json(response: Any) -> str:
    if hasattr(response, "model_dump_json"):
        return str(response.model_dump_json())
    return json.dumps(_model_dump(response), ensure_ascii=False, default=str)


def _refusal_text(response: Any) -> str:
    for output in getattr(response, "output", []) or []:
        for item in getattr(output, "content", []) or []:
            if getattr(item, "type", "") == "refusal":
                return str(getattr(item, "refusal", ""))
    return ""
