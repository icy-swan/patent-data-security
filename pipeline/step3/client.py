"""Local authenticated Codex CLI client for provisional Step 3 annotation."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline.step3.prompt import AnnotationPrompt, load_annotation_prompt
from pipeline.step3.schema import CodexAnnotationBatch, IndependentAnnotation

DEFAULT_MODEL = "gpt-5.6-sol"


@dataclass(frozen=True)
class BatchAnnotationResponse:
    annotations: dict[str, IndependentAnnotation]
    response_id: str
    requested_model: str
    actual_model: str
    prompt_version: str
    prompt_sha256: str
    schema_sha256: str
    elapsed_seconds: float
    usage: dict[str, Any]
    raw_response: str
    raw_events: str


class CodexAnnotationClient:
    """Run schema-constrained batches through the user's existing Codex login."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        reasoning_effort: str = "high",
        workspace: str | Path,
        prompt: AnnotationPrompt | None = None,
        codex_binary: str | None = None,
        timeout_seconds: float = 1_800,
    ) -> None:
        binary = codex_binary or shutil.which("codex")
        if not binary:
            raise ValueError("codex CLI is required for keyless Step 3 simulation")
        self.codex_binary = binary
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.workspace = Path(workspace).resolve()
        self.prompt = prompt or load_annotation_prompt()
        self.timeout_seconds = timeout_seconds

    def annotate_batch(
        self,
        patents: Sequence[Mapping[str, Any]],
    ) -> BatchAnnotationResponse:
        if not patents:
            raise ValueError("A Codex annotation batch cannot be empty")
        if len(patents) > 25:
            raise ValueError("A Codex annotation batch cannot exceed 25 patents")
        sample_ids = [str(patent["sample_id"]) for patent in patents]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("A Codex annotation batch contains duplicate sample_id values")

        schema = CodexAnnotationBatch.model_json_schema()
        input_rows = [
            {
                "sample_id": patent["sample_id"],
                "title": patent.get("title", ""),
                "abstract": patent.get("abstract", ""),
                "claim": patent.get("claim", ""),
                "ipc": patent.get("ipc", ""),
                "main_ipc": patent.get("main_ipc", ""),
            }
            for patent in patents
        ]
        instruction = (
            self.prompt.text
            + "\n这是一个封闭的批量标注任务。不得调用任何工具，不得读取工作区文件，"
            "不得搜索网络，也不得修改文件。逐条独立判断；不要让上一条专利影响下一条。"
            "返回 annotations 数组，数量、sample_id 集合必须与输入完全一致。\n"
            "<PATENTS>\n"
            + json.dumps(input_rows, ensure_ascii=False, separators=(",", ":"))
            + "\n</PATENTS>\n"
        )

        with tempfile.TemporaryDirectory(prefix="step3-codex-") as temporary:
            temp = Path(temporary)
            schema_path = temp / "schema.json"
            output_path = temp / "answer.json"
            schema_path.write_text(
                json.dumps(schema, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            command = [
                self.codex_binary,
                "exec",
                "--ephemeral",
                "--ignore-user-config",
                "--model",
                self.model,
                "--config",
                f'model_reasoning_effort="{self.reasoning_effort}"',
                "--sandbox",
                "read-only",
                "--cd",
                str(self.workspace),
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(output_path),
                "--json",
                "-",
            ]
            started = time.monotonic()
            completed = subprocess.run(  # noqa: S603 - fixed authenticated local CLI
                command,
                input=instruction,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            elapsed = time.monotonic() - started
            if completed.returncode != 0:
                diagnostic = completed.stderr.strip() or completed.stdout.strip()
                raise RuntimeError(
                    f"codex exec failed with exit code {completed.returncode}: {diagnostic[-2000:]}"
                )
            if not output_path.is_file():
                raise RuntimeError("codex exec completed without an output message")
            raw_response = output_path.read_text(encoding="utf-8")

        parsed = CodexAnnotationBatch.model_validate(_normalized_batch(raw_response))
        returned_ids = [item.sample_id for item in parsed.annotations]
        if Counter(returned_ids) != Counter(sample_ids):
            raise ValueError(
                "Codex batch returned a different sample_id multiset: "
                f"expected={sample_ids}, actual={returned_ids}"
            )
        annotations = {
            item.sample_id: IndependentAnnotation.model_validate(
                item.model_dump(exclude={"sample_id"})
            )
            for item in parsed.annotations
        }
        response_id, actual_model, usage = _event_metadata(completed.stdout, self.model)
        return BatchAnnotationResponse(
            annotations=annotations,
            response_id=response_id,
            requested_model=self.model,
            actual_model=actual_model,
            prompt_version=self.prompt.version,
            prompt_sha256=self.prompt.prompt_sha256,
            schema_sha256=self.prompt.schema_sha256,
            elapsed_seconds=elapsed,
            usage=usage,
            raw_response=raw_response,
            raw_events=completed.stdout,
        )


def _event_metadata(raw_events: str, requested_model: str) -> tuple[str, str, dict[str, Any]]:
    response_id = ""
    actual_model = requested_model
    usage: dict[str, Any] = {}
    for line in raw_events.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "thread.started":
            response_id = str(event.get("thread_id", ""))
        if event.get("type") == "turn.completed":
            value = event.get("usage")
            if isinstance(value, dict):
                usage = value
        model = event.get("model")
        if isinstance(model, str) and model:
            actual_model = model
    return response_id, actual_model, usage


def _normalized_batch(raw_response: str) -> dict[str, Any]:
    """Apply only deterministic Step 2 contract fixes before Pydantic validation."""

    value = json.loads(raw_response)
    annotations = value.get("annotations") if isinstance(value, dict) else None
    if not isinstance(annotations, list):
        return value
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        dimensions = ("scope_basis", "processing_activities", "industry_sectors")
        for field in dimensions:
            field_value = annotation.get(field)
            if isinstance(field_value, list):
                annotation[field] = list(dict.fromkeys(field_value))
        if annotation.get("label") == "OTHER":
            for field in dimensions:
                annotation[field] = ["other"]
        elif annotation.get("label") == "DATA_SECURITY":
            for field in dimensions:
                field_value = annotation.get(field)
                if isinstance(field_value, list) and len(field_value) > 1:
                    annotation[field] = [item for item in field_value if item != "other"]
        if "review_flag" in annotation and "needs_review" not in annotation:
            annotation["needs_review"] = annotation.pop("review_flag")
        if annotation.get("needs_review") is False and annotation.get("review_reason"):
            annotation["review_reason"] = ""
    return value
