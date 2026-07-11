"""Prompt, structured output, Batch preparation, and Batch result parsing."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from patent_data_security.classification import PatentClassification

PROMPT_VERSION = "data-security-three-class-v1.0.0"
SYSTEM_PROMPT = """你是中国专利技术分类研究员。
请依据专利的核心技术对象与核心改进，判断其与数据安全的关系。

类别定义：
1 = 数据安全相关。核心技术方案直接保护数据、个人信息、敏感数据、数据库、文件、数据流或
数据资产，或保护数据收集、存储、使用、加工、传输、共享、公开、流通、删除等生命周期环节。
2 = 安全相关但非数据安全。属于网络、系统、应用、设备、通信、交易或物理安全，但核心
保护对象不是数据或数据处理活动。
3 = 不相关。与安全无关，或只涉及生产、食品、交通、消防等一般安全。

边界规则：
- 关键词和 IPC 只是候选线索，不能替代对技术方案的判断。
- 裸加密、认证、密钥、哈希、签名、防火墙或网络安全不自动属于类别 1；必须看其直接保护对象。
- 数据只是输入、测量结果或普通处理对象，而核心改进不在数据保护时，不属于类别 1。
- 联邦学习、同态加密、安全多方计算等也要核实其在本专利中的实际核心用途。
- 只能引用给定标题、摘要、主权项或 IPC 中的证据，不得补充不存在的技术事实。
- 证据不足或 1/2 边界模糊时，仍选择 1、2、3 中最合理的一类，并设置 review_flag=true。
"""


@dataclass(frozen=True)
class BatchPreparation:
    files: tuple[Path, ...]
    requests: int
    manifest: Path


def classify_candidate(
    candidate: dict[str, Any],
    *,
    model: str,
    client: OpenAI | None = None,
) -> PatentClassification:
    """Classify one candidate through the Responses API with Pydantic parsing."""

    api = client or OpenAI()
    response = api.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(candidate)},
        ],
        text_format=PatentClassification,
    )
    if response.output_parsed is None:
        raise ValueError("Model response did not contain a parsed classification")
    return response.output_parsed


def prepare_batch_files(
    candidates_path: str | Path,
    output_dir: str | Path,
    *,
    model: str,
    max_requests: int = 20_000,
    max_bytes: int = 180_000_000,
) -> BatchPreparation:
    """Convert internal candidates to upload-ready Chat Completions Batch JSONL files."""

    source = Path(candidates_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    schema = PatentClassification.model_json_schema()
    files: list[Path] = []
    request_count = 0
    part_count = 0
    current_file = None
    current_path: Path | None = None
    current_requests = 0
    current_bytes = 0

    try:
        with source.open(encoding="utf-8") as candidates:
            for raw_line in candidates:
                candidate = json.loads(raw_line)
                request = _batch_request(candidate, model, schema)
                encoded = (
                    json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n"
                ).encode("utf-8")
                if len(encoded) > max_bytes:
                    raise ValueError(
                        f"One request ({candidate['custom_id']}) exceeds max_bytes={max_bytes}"
                    )
                rotate = current_file is None or current_requests >= max_requests
                rotate = rotate or (
                    current_requests > 0 and current_bytes + len(encoded) > max_bytes
                )
                if rotate:
                    if current_file is not None:
                        current_file.close()
                    part_count += 1
                    current_path = destination / f"batch_{part_count:04d}.jsonl"
                    current_file = current_path.open("wb")
                    files.append(current_path)
                    current_requests = 0
                    current_bytes = 0
                current_file.write(encoded)
                current_requests += 1
                current_bytes += len(encoded)
                request_count += 1
    finally:
        if current_file is not None:
            current_file.close()

    manifest = destination / "batch_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "prompt_version": PROMPT_VERSION,
                "model": model,
                "source_candidates": str(source.resolve()),
                "requests": request_count,
                "files": [
                    {"path": str(path.resolve()), "size_bytes": path.stat().st_size}
                    for path in files
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return BatchPreparation(tuple(files), request_count, manifest)


def submit_batch_file(path: str | Path, *, client: OpenAI | None = None) -> str:
    """Upload and submit one prepared file; return the OpenAI Batch ID."""

    api = client or OpenAI()
    with Path(path).open("rb") as file:
        upload = api.files.create(file=file, purpose="batch")
    batch = api.batches.create(
        input_file_id=upload.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"project": "patent-data-security", "prompt_version": PROMPT_VERSION},
    )
    return batch.id


def merge_batch_outputs(
    output_paths: list[str | Path],
    destination: str | Path,
    *,
    model_name: str,
) -> dict[str, int]:
    """Validate Batch responses and write a compact classification CSV."""

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "custom_id",
        "source_row_number",
        "cat",
        "confidence",
        "subtype",
        "evidence",
        "reason",
        "review_flag",
        "review_reason",
        "model_name",
        "prompt_version",
        "process_status",
        "error",
    )
    counts = {"validated": 0, "failed": 0}
    with target.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for path_value in output_paths:
            with Path(path_value).open(encoding="utf-8") as responses:
                for raw_line in responses:
                    response = json.loads(raw_line)
                    custom_id = response.get("custom_id", "")
                    base = {
                        "custom_id": custom_id,
                        "source_row_number": _row_from_custom_id(custom_id),
                        "model_name": model_name,
                        "prompt_version": PROMPT_VERSION,
                    }
                    try:
                        content = response["response"]["body"]["choices"][0]["message"][
                            "content"
                        ]
                        label = PatentClassification.model_validate_json(content)
                        writer.writerow(
                            {
                                **base,
                                **label.model_dump(),
                                "evidence": json.dumps(label.evidence, ensure_ascii=False),
                                "review_flag": str(label.review_flag).lower(),
                                "process_status": "classified",
                                "error": "",
                            }
                        )
                        counts["validated"] += 1
                    except (KeyError, IndexError, TypeError, ValueError) as error:
                        writer.writerow(
                            {
                                **base,
                                "process_status": "failed",
                                "error": str(error)[:1000],
                            }
                        )
                        counts["failed"] += 1
    return counts


def build_user_prompt(candidate: dict[str, Any]) -> str:
    routed_evidence = {
        "keyword_level": candidate.get("keyword_level"),
        "ipc_level": candidate.get("ipc_level"),
        "route_level": candidate.get("route_level"),
        "keyword_hits": candidate.get("keyword_hits", []),
        "ipc_hits": candidate.get("ipc_hits", []),
        "diagnostic_hits": candidate.get("diagnostic_hits", []),
    }
    return "\n".join(
        (
            f"专利名称：{candidate.get('title', '')}",
            f"IPC分类号：{candidate.get('ipc', '')}",
            f"IPC主分类号：{candidate.get('main_ipc', '')}",
            f"摘要：{candidate.get('abstract', '')}",
            f"主权项：{candidate.get('claim', '')}",
            "路由线索（只作提示，不是标签）："
            + json.dumps(routed_evidence, ensure_ascii=False, separators=(",", ":")),
            "请输出符合给定 JSON Schema 的判断。",
        )
    )


def openai_client_from_env() -> OpenAI:
    kwargs: dict[str, str] = {}
    if base_url := os.getenv("OPENAI_BASE_URL"):
        kwargs["base_url"] = base_url
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"), **kwargs)


def _batch_request(
    candidate: dict[str, Any], model: str, schema: dict[str, Any]
) -> dict[str, Any]:
    return {
        "custom_id": candidate["custom_id"],
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(candidate)},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "patent_classification",
                    "strict": True,
                    "schema": schema,
                },
            },
        },
    }


def _row_from_custom_id(custom_id: str) -> str:
    prefix = "patent-"
    return custom_id[len(prefix) :] if custom_id.startswith(prefix) else ""
