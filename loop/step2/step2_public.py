"""供应商无关的公开版 Step 2：固定抽样、单件识别与精简结果导出。

公开代码不包含任何模型名称、服务地址、鉴权方式或厂商 SDK。调用者通过 ModelAdapter
注入自己的实现；本模块只定义论文方法需要的输入、输出和可恢复任务状态。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import re
import sqlite3
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

ALLOWED_LABELS = {"DATA_SECURITY", "OTHER"}
ALLOWED_EVIDENCE_FIELDS = {"title", "abstract", "claim"}
MODEL_INPUT_FIELDS = ("title", "abstract", "claim", "ipc")
REQUEST_FIELDS = (
    "task_id",
    "dataset_id",
    "patent_id",
    "application_date",
    "application_year",
    "title",
    "abstract",
    "claim",
    "ipc",
    "step1_selection_probability",
    "step2_pool_inclusion_probability",
    "combined_step2_inclusion_probability",
)
RESULT_FIELDS = REQUEST_FIELDS + (
    "step1_route",
    "step2_label",
    "step2_reason",
    "step2_evidence",
)


class ModelAdapter(Protocol):
    """由使用者在公开代码之外实现的通用单件模型适配器。"""

    def classify(
        self,
        *,
        system_prompt: str,
        patent: Mapping[str, str],
    ) -> Mapping[str, Any]:
        """每次只接收一件专利，并返回最小 JSON 决策。"""


@dataclass(frozen=True)
class Step2Decision:
    label: str
    reason: str
    evidence: tuple[dict[str, str], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "reason": self.reason,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class PublicStep2Paths:
    root: Path
    database: Path
    requests: Path
    result: Path
    manifest: Path
    progress: Path
    prompt: Path


def public_step2_paths(output_dir: str | Path) -> PublicStep2Paths:
    root = Path(output_dir).resolve()
    return PublicStep2Paths(
        root=root,
        database=root / "tasks.sqlite3",
        requests=root / "requests.jsonl",
        result=root / "result.csv",
        manifest=root / "manifest.json",
        progress=root / "progress.json",
        prompt=root / "prompt.txt",
    )


def prepare_public_step2(
    config_path: str | Path,
    *,
    step1_overrides: Sequence[str | Path] | None = None,
    output_override: str | Path | None = None,
    overwrite: bool = False,
) -> tuple[PublicStep2Paths, dict[str, Any]]:
    """从公开版 Step 1 结果构造固定规模的 Step 2 任务池，不调用模型。"""

    config_file = Path(config_path).resolve()
    config = read_json(config_file)
    configured_sources = config.get("step1_results", [])
    source_values = list(step1_overrides or configured_sources)
    if not source_values:
        raise ValueError(
            "step1_results 在公开模板中故意留空；请传入 --step1-result "
            "或填写本地配置"
        )
    source_paths = [
        resolve_from(config_file.parent, str(value))
        for value in source_values
    ]
    missing = [path for path in source_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Step 1 结果不存在：{missing}")

    output_value = str(output_override or config.get("output_dir", "output"))
    paths = public_step2_paths(resolve_from(config_file.parent, output_value))
    prompt_path = resolve_from(
        config_file.parent,
        str(config.get("prompt_file", "prompt.template.txt")),
    )
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt 文件不存在：{prompt_path}")
    pool_size = int(config.get("pool_size", 50_000))
    pool_seed = str(config.get("pool_seed", "public-step2-pool-v1"))
    encoding = str(config.get("encoding", "utf-8-sig"))
    if pool_size < 1:
        raise ValueError("pool_size 必须为正整数")

    candidates: dict[str, dict[str, str]] = {}
    source_records = []
    for source_path in source_paths:
        rows = read_step1_rows(source_path, encoding=encoding)
        selected_rows = [
            row
            for row in rows
            if str(row.get("selected_for_step2", "")).strip().lower() == "true"
        ]
        for row in selected_rows:
            patent_id = str(row.get("patent_id", "")).strip()
            if not patent_id:
                raise ValueError(f"{source_path} 存在空 patent_id")
            application_date = str(row.get("application_date", "") or "").strip()
            candidate = {
                "dataset_id": str(row.get("dataset_id", "")).strip(),
                "patent_id": patent_id,
                "application_date": application_date,
                "application_year": extract_application_year(
                    application_date,
                    fallback=str(row.get("dataset_id", "")).strip(),
                ),
                "title": str(row.get("title", "") or ""),
                "abstract": str(row.get("abstract", "") or ""),
                "claim": str(row.get("claim", "") or ""),
                "ipc": str(row.get("ipc", "") or ""),
                "step1_route": str(row.get("route", "") or ""),
                "step1_selection_probability": format_probability(
                    parse_probability(
                        row.get("selection_probability", "1"),
                        field="selection_probability",
                    )
                ),
            }
            existing = candidates.get(patent_id)
            if existing is None or _candidate_quality(candidate) > _candidate_quality(existing):
                candidates[patent_id] = candidate
        source_records.append(
            {
                "file_name": source_path.name,
                "sha256": sha256_file(source_path),
                "records": len(rows),
                "selected_records": len(selected_rows),
            }
        )
    if len(candidates) < pool_size:
        raise ValueError(
            f"候选专利不足：pool_size={pool_size}，唯一候选={len(candidates)}"
        )

    ranked = sorted(
        candidates.values(),
        key=lambda row: (
            stable_score(pool_seed, row["patent_id"]),
            row["patent_id"],
        ),
    )
    selected = ranked[:pool_size]
    pool_probability = pool_size / len(candidates)
    prepared_rows = []
    for source_order, row in enumerate(selected):
        task_id = "public-step2-" + hashlib.sha256(
            f"public-step2-task-v1|{row['patent_id']}".encode()
        ).hexdigest()[:24]
        step1_probability = float(row["step1_selection_probability"])
        prepared_rows.append(
            {
                "task_id": task_id,
                "source_order": source_order,
                **row,
                "step2_pool_inclusion_probability": format_probability(
                    pool_probability
                ),
                "combined_step2_inclusion_probability": format_probability(
                    step1_probability * pool_probability
                ),
            }
        )

    paths.root.mkdir(parents=True, exist_ok=True)
    managed = (
        paths.database,
        paths.requests,
        paths.result,
        paths.manifest,
        paths.progress,
        paths.prompt,
    )
    if not overwrite and any(path.exists() for path in managed):
        raise FileExistsError("Step 2 输出已存在；如需替换请传入 --overwrite")
    if overwrite:
        for path in managed:
            path.unlink(missing_ok=True)
        for suffix in ("-wal", "-shm"):
            paths.database.with_name(paths.database.name + suffix).unlink(missing_ok=True)

    create_task_database(paths.database, prepared_rows)
    write_requests(paths.requests, prepared_rows)
    atomic_write_text(paths.prompt, prompt_path.read_text(encoding="utf-8"))
    manifest = {
        "public_method_version": "public-step2-generic-adapter-v1",
        "input_sources": source_records,
        "input_source_addresses": [],
        "pool": {
            "candidate_patents": len(candidates),
            "selected_patents": len(prepared_rows),
            "inclusion_probability": pool_probability,
            "seed": pool_seed,
            "selection": "SHA256 fixed-size sampling without replacement",
            "step1_route_counts": dict(
                sorted(Counter(row["step1_route"] for row in prepared_rows).items())
            ),
        },
        "prompt": {
            "file_name": paths.prompt.name,
            "sha256": sha256_file(paths.prompt),
            "response_fields": ["label", "reason", "evidence"],
        },
        "model_adapter": {
            "implementation_included": False,
            "provider": "",
            "model": "",
            "endpoint": "",
            "credential_field": "",
        },
        "outputs": {
            "requests_file": paths.requests.name,
            "database_file": paths.database.name,
            "result_file": paths.result.name,
            "progress_file": paths.progress.name,
            "prompt_file": paths.prompt.name,
        },
        "excluded_fields": [
            "confidence",
            "token_usage",
            "cache_metrics",
            "request_latency",
            "response_id",
            "provider",
            "model",
            "endpoint",
        ],
        "llm_requests_executed_during_prepare": 0,
        "created_at": now(),
    }
    atomic_write_json(paths.manifest, manifest)
    return paths, manifest


def run_public_step2(
    paths: PublicStep2Paths,
    adapter: ModelAdapter,
    *,
    max_attempts: int = 3,
    concurrency: int = 1,
) -> dict[str, Any]:
    """逐件调用通用适配器；结果仅保存论文所需的三个字段。"""

    if max_attempts < 1 or concurrency < 1:
        raise ValueError("max_attempts 和 concurrency 必须为正整数")
    if not paths.database.is_file() or not paths.manifest.is_file():
        raise FileNotFoundError("请先执行 prepare")
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    prompt_file = paths.root / manifest["prompt"]["file_name"]
    if not prompt_file.is_file():
        raise FileNotFoundError(f"Prompt 文件不存在：{prompt_file}")
    if sha256_file(prompt_file) != manifest["prompt"]["sha256"]:
        raise ValueError("Prompt 文件哈希发生变化，拒绝混用旧任务")
    system_prompt = prompt_file.read_text(encoding="utf-8").strip() + "\n"

    connection = connect(paths.database)
    try:
        connection.execute("UPDATE tasks SET status='pending' WHERE status='running'")
        connection.commit()
        _run_loop(
            connection,
            adapter,
            system_prompt=system_prompt,
            max_attempts=max_attempts,
            concurrency=concurrency,
            progress_path=paths.progress,
        )
        progress = task_progress(connection)
        if progress["succeeded"] == progress["total"]:
            progress["output"] = export_results(paths, connection=connection)
        atomic_write_json(paths.progress, progress)
        return progress
    finally:
        connection.close()


def read_progress(paths: PublicStep2Paths) -> dict[str, Any]:
    if not paths.database.is_file():
        return {"status": "not_prepared"}
    connection = connect(paths.database, read_only=True)
    try:
        return task_progress(connection)
    finally:
        connection.close()


def validate_decision(
    value: Mapping[str, Any],
    patent: Mapping[str, str],
) -> Step2Decision:
    """严格拒绝置信度、token 等额外字段，并核对逐字证据。"""

    expected_fields = {"label", "reason", "evidence"}
    actual_fields = set(value)
    if actual_fields != expected_fields:
        raise ValueError(
            "模型输出字段必须严格为 label/reason/evidence；"
            f"缺少={sorted(expected_fields - actual_fields)}，"
            f"多余={sorted(actual_fields - expected_fields)}"
        )
    label = str(value["label"]).strip().upper()
    if label not in ALLOWED_LABELS:
        raise ValueError(f"非法 label：{label!r}")
    reason = str(value["reason"]).strip()
    if not 5 <= len(reason) <= 2_000:
        raise ValueError("reason 长度必须在 5 到 2000 字符之间")
    evidence_value = value["evidence"]
    if not isinstance(evidence_value, list) or not 1 <= len(evidence_value) <= 3:
        raise ValueError("evidence 必须包含 1 至 3 条记录")
    evidence = []
    for index, item in enumerate(evidence_value, start=1):
        if not isinstance(item, Mapping) or set(item) != {"field", "quote"}:
            raise ValueError(f"evidence[{index}] 只能包含 field 和 quote")
        field = str(item["field"]).strip()
        quote = str(item["quote"]).strip()
        if field not in ALLOWED_EVIDENCE_FIELDS:
            raise ValueError(f"evidence[{index}] 的 field 非法：{field!r}")
        if not quote or quote not in str(patent.get(field, "")):
            raise ValueError(f"evidence[{index}] 不是 {field} 的逐字引文")
        evidence.append({"field": field, "quote": quote})
    return Step2Decision(
        label=label,
        reason=reason,
        evidence=tuple(evidence),
    )


def _run_loop(
    connection: sqlite3.Connection,
    adapter: ModelAdapter,
    *,
    system_prompt: str,
    max_attempts: int,
    concurrency: int,
    progress_path: Path,
) -> None:
    def classify(task: sqlite3.Row) -> Step2Decision | Exception:
        patent = json.loads(task["payload_json"])
        public_patent = {
            field: str(patent.get(field, "")) for field in MODEL_INPUT_FIELDS
        }
        try:
            response = adapter.classify(
                system_prompt=system_prompt,
                patent=public_patent,
            )
            return validate_decision(response, public_patent)
        except Exception as error:  # noqa: BLE001 - 错误需要落库供重试
            return error

    in_flight: dict[Future[Any], tuple[str, int]] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        while True:
            while len(in_flight) < concurrency:
                claimed = claim_next(connection, max_attempts)
                if claimed is None:
                    break
                task, attempts = claimed
                in_flight[pool.submit(classify, task)] = (task["task_id"], attempts)
            if not in_flight:
                break
            completed, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in completed:
                task_id, attempts = in_flight.pop(future)
                persist_outcome(
                    connection,
                    task_id,
                    attempts,
                    max_attempts,
                    future.result(),
                )
            atomic_write_json(progress_path, task_progress(connection))


def claim_next(
    connection: sqlite3.Connection,
    max_attempts: int,
) -> tuple[sqlite3.Row, int] | None:
    task = connection.execute(
        """
        SELECT * FROM tasks
        WHERE status='pending' AND attempts < ?
        ORDER BY source_order
        LIMIT 1
        """,
        (max_attempts,),
    ).fetchone()
    if task is None:
        return None
    attempts = int(task["attempts"]) + 1
    connection.execute(
        "UPDATE tasks SET status='running',attempts=?,updated_at=? WHERE task_id=?",
        (attempts, now(), task["task_id"]),
    )
    connection.commit()
    return task, attempts


def persist_outcome(
    connection: sqlite3.Connection,
    task_id: str,
    attempts: int,
    max_attempts: int,
    outcome: Step2Decision | Exception,
) -> None:
    if isinstance(outcome, Exception):
        status = "failed" if attempts >= max_attempts else "pending"
        connection.execute(
            """
            UPDATE tasks SET status=?,error_code=?,updated_at=?,completed_at=?
            WHERE task_id=?
            """,
            (
                status,
                generic_error_code(outcome),
                now(),
                now() if status == "failed" else None,
                task_id,
            ),
        )
    else:
        connection.execute(
            """
            UPDATE tasks SET status='succeeded',decision_json=?,error_code=NULL,
              updated_at=?,completed_at=? WHERE task_id=?
            """,
            (
                json.dumps(outcome.as_dict(), ensure_ascii=False, separators=(",", ":")),
                now(),
                now(),
                task_id,
            ),
        )
    connection.commit()


def export_results(
    paths: PublicStep2Paths,
    *,
    connection: sqlite3.Connection,
) -> dict[str, Any]:
    counts = dict(connection.execute("SELECT status,COUNT(*) FROM tasks GROUP BY status"))
    total = sum(counts.values())
    if counts.get("succeeded", 0) != total:
        raise ValueError(f"仍有未成功任务：{counts}")
    partial = paths.result.with_suffix(paths.result.suffix + ".partial")
    label_counts: Counter[str] = Counter()
    with partial.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for task in connection.execute("SELECT * FROM tasks ORDER BY source_order"):
            payload = json.loads(task["payload_json"])
            decision = json.loads(task["decision_json"])
            label_counts[decision["label"]] += 1
            writer.writerow(
                {
                    **{field: payload.get(field, "") for field in REQUEST_FIELDS},
                    "step1_route": payload["step1_route"],
                    "step2_label": decision["label"],
                    "step2_reason": decision["reason"],
                    "step2_evidence": json.dumps(
                        decision["evidence"],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            )
    os.replace(partial, paths.result)
    return {
        "file_name": paths.result.name,
        "sha256": sha256_file(paths.result),
        "records": total,
        "label_counts": dict(sorted(label_counts.items())),
        "fields": list(RESULT_FIELDS),
    }


def task_progress(connection: sqlite3.Connection) -> dict[str, Any]:
    counts = dict(connection.execute("SELECT status,COUNT(*) FROM tasks GROUP BY status"))
    total = sum(counts.values())
    succeeded = counts.get("succeeded", 0)
    return {
        "total": total,
        "succeeded": succeeded,
        "failed": counts.get("failed", 0),
        "pending": total - succeeded,
        "queued": counts.get("pending", 0),
        "running": counts.get("running", 0),
        "progress_percent": round(succeeded / total * 100, 4) if total else 0,
        "updated_at": now(),
    }


def create_task_database(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=FULL;
            CREATE TABLE tasks (
              task_id TEXT PRIMARY KEY,
              patent_id TEXT NOT NULL UNIQUE,
              source_order INTEGER NOT NULL UNIQUE,
              payload_json TEXT NOT NULL,
              status TEXT NOT NULL CHECK(status IN ('pending','running','succeeded','failed')),
              attempts INTEGER NOT NULL DEFAULT 0,
              decision_json TEXT,
              error_code TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              completed_at TEXT
            );
            CREATE INDEX idx_public_step2_status ON tasks(status,source_order);
            """
        )
        created_at = now()
        connection.executemany(
            """
            INSERT INTO tasks (
              task_id,patent_id,source_order,payload_json,status,created_at,updated_at
            ) VALUES (?,?,?,?,'pending',?,?)
            """,
            [
                (
                    row["task_id"],
                    row["patent_id"],
                    row["source_order"],
                    json.dumps(
                        {
                            field: row.get(field, "")
                            for field in (*REQUEST_FIELDS, "step1_route")
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    created_at,
                    created_at,
                )
                for row in rows
            ],
        )
        connection.commit()
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("任务数据库完整性检查失败")
    finally:
        connection.close()
    for suffix in ("-wal", "-shm"):
        temporary.with_name(temporary.name + suffix).unlink(missing_ok=True)
    os.replace(temporary, path)


def write_requests(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            request = {field: row.get(field, "") for field in REQUEST_FIELDS}
            file.write(
                json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    os.replace(temporary, path)


def read_step1_rows(path: Path, *, encoding: str) -> list[dict[str, str]]:
    with path.open(encoding=encoding, newline="") as file:
        reader = csv.DictReader(file)
        fields = set(reader.fieldnames or ())
        required = {
            "dataset_id",
            "patent_id",
            "title",
            "abstract",
            "claim",
            "ipc",
            "route",
            "selected_for_step2",
        }
        if missing := required - fields:
            raise ValueError(f"{path} 缺少字段：{sorted(missing)}")
        return list(reader)


def _candidate_quality(row: Mapping[str, str]) -> tuple[int, int, int]:
    return (
        1 if row.get("step1_route") == "S" else 0,
        bool(row.get("claim")),
        len(row.get("claim", "")) + len(row.get("abstract", "")),
    )


def stable_score(seed: str, patent_id: str) -> str:
    return hashlib.sha256(f"{seed}|{patent_id}".encode()).hexdigest()


def parse_probability(value: Any, *, field: str) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} 必须是 (0, 1] 内的数值") from error
    if not 0 < probability <= 1:
        raise ValueError(f"{field} 必须是 (0, 1] 内的数值")
    return probability


def format_probability(value: float) -> str:
    return f"{value:.12g}"


def extract_application_year(value: str, *, fallback: str = "") -> str:
    for candidate in (value, fallback):
        if match := re.search(r"(?<!\d)((?:18|19|20|21)\d{2})(?!\d)", candidate):
            return match.group(1)
    return "UNKNOWN"


def connect(path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    if read_only:
        connection = sqlite3.connect(
            f"file:{path}?mode=ro",
            uri=True,
        )
    else:
        connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def generic_error_code(error: Exception) -> str:
    """把异常归为通用类别，不保存可能包含接入细节的原始消息。"""

    if isinstance(error, (TimeoutError, ConnectionError)):
        return "temporary_adapter_error"
    if isinstance(error, (TypeError, ValueError, KeyError)):
        return "invalid_adapter_response"
    return "adapter_error"


def load_adapter(specification: str) -> ModelAdapter:
    """加载使用者自行维护的 module:factory，不读取任何公开仓库凭证。"""

    if ":" not in specification:
        raise ValueError("--adapter 必须使用 module:factory 形式")
    module_name, attribute = specification.split(":", 1)
    factory = getattr(importlib.import_module(module_name), attribute)
    adapter = factory()
    if not callable(getattr(adapter, "classify", None)):
        raise TypeError("适配器必须实现 classify(system_prompt=..., patent=...)")
    return adapter


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"JSON 文件必须是对象：{path}")
    return value


def resolve_from(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def now() -> str:
    return datetime.now(UTC).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="准备固定任务池，不调用模型")
    prepare.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.example.json"),
    )
    prepare.add_argument("--step1-result", type=Path, action="append")
    prepare.add_argument("--output-dir", type=Path)
    prepare.add_argument("--overwrite", action="store_true")

    run = subparsers.add_parser("run", help="使用外部通用适配器逐件识别")
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--adapter", required=True, help="module:factory")
    run.add_argument("--max-attempts", type=int, default=3)
    run.add_argument("--concurrency", type=int, default=1)

    status = subparsers.add_parser("status", help="查看本地任务状态")
    status.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "prepare":
        paths, manifest = prepare_public_step2(
            args.config,
            step1_overrides=args.step1_result,
            output_override=args.output_dir,
            overwrite=args.overwrite,
        )
        output: Mapping[str, Any] = {
            "paths": {key: str(value) for key, value in vars(paths).items()},
            "manifest": manifest,
        }
    else:
        paths = public_step2_paths(args.output_dir)
        if args.command == "status":
            output = read_progress(paths)
        else:
            output = run_public_step2(
                paths,
                load_adapter(args.adapter),
                max_attempts=args.max_attempts,
                concurrency=args.concurrency,
            )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
