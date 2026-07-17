import csv
import json
import sqlite3
from pathlib import Path

from pipeline.step2.__main__ import _format_duration, _print_progress, build_parser
from pipeline.step2.client import ClassificationResponse
from pipeline.step2.prompt import load_prompt_bundle
from pipeline.step2.runner import _exclusive_lock, run_tasks
from pipeline.step2.schema import PatentClassification
from pipeline.step2.tasks import prepare_tasks


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    raw = tmp_path / "patents_2026.csv"
    raw.write_text(
        "申请号,专利名称,摘要文本,主权项内容,IPC分类号,IPC主分类号\n"
        "CN-A,密码方法,采用密钥协商,通过密钥协商加密传输数据,H04L9/00,H04L9/00\n"
        "CN-B,机械装置,普通机械结构,一种齿轮装置,F16H,F16H\n",
        encoding="utf-8",
    )
    step1 = tmp_path / "step1_2026.csv"
    step1.write_text(
        "patent_id,source_row_number,route,selected_for_step2,selection_group,"
        "selection_probability,sample_weight\n"
        "CN-A,2,S,true,S_all,1,1\n"
        "CN-B,3,E,true,E_random,0.02,50\n",
        encoding="utf-8",
    )
    return raw, step1


def test_prepare_binds_patent_id_locally_without_routing_leak(tmp_path: Path) -> None:
    raw, step1 = _write_inputs(tmp_path)
    paths, manifest = prepare_tasks(raw, step1, tmp_path / "step2")

    assert manifest["task_counts"] == {
        "total": 2,
        "by_route": {"E": 1, "S": 1},
        "by_selection_group": {"E_random": 1, "S_all": 1},
    }
    assert manifest["statistics_binding"]["duplicate_task_patent_ids"] == 0
    assert paths.database == (tmp_path / "step2" / "2026" / "tasks.sqlite3").resolve()
    assert paths.manifest.name == "manifest.json"
    assert paths.results.name == "result.csv"
    connection = sqlite3.connect(paths.database)
    row = connection.execute(
        "SELECT patent_id, route, payload_json FROM tasks WHERE patent_id='CN-A'"
    ).fetchone()
    connection.close()
    payload = json.loads(row[2])
    assert row[:2] == ("CN-A", "S")
    assert payload["patent_id"] == "CN-A"
    assert "route" not in payload
    assert "selection_group" not in payload


def test_runner_writes_model_result_under_local_patent_id_and_resumes(tmp_path: Path) -> None:
    raw, step1 = _write_inputs(tmp_path)
    bundle = load_prompt_bundle()
    paths, _ = prepare_tasks(raw, step1, tmp_path / "step2", prompt_bundle=bundle)

    class FakeClient:
        model = "fake-model"
        prompt_bundle = bundle

        def __init__(self) -> None:
            self.calls = 0

        def classify(self, patent):
            self.calls += 1
            label = "DATA_SECURITY" if patent["patent_id"] == "CN-A" else "OTHER"
            basis = ["cryptography"] if label == "DATA_SECURITY" else ["other"]
            classification = PatentClassification(
                label=label,
                confidence=0.9,
                scope_basis=basis,
                processing_activities=(
                    ["transmission"] if label == "DATA_SECURITY" else ["other"]
                ),
                industry_sectors=(
                    ["telecommunications"] if label == "DATA_SECURITY" else ["other"]
                ),
                technical_scope=patent["claim"],
                legal_scope="存在密码技术。" if label == "DATA_SECURITY" else "未建立安全联系。",
                evidence=[{"field": "claim", "quote": patent["claim"]}],
                reason="测试结果",
                review_flag=False,
                review_reason="",
            )
            return ClassificationResponse(
                classification=classification,
                response_id=f"response-{self.calls}",
                requested_model=self.model,
                actual_model=self.model,
                elapsed_seconds=0.1,
                usage={"input_tokens": 100},
                prompt_tokens=100,
                cached_tokens=None,
                cache_write_tokens=None,
                cache_hit_ratio=None,
                cache_mode="ark_responses_stable_prefix",
                prompt_version=bundle.prompt_version,
                prefix_sha256=bundle.prefix_sha256,
                law_sha256=bundle.law_sha256,
                schema_sha256=bundle.schema_sha256,
                raw_text=classification.model_dump_json(),
            )

    client = FakeClient()
    progress = run_tasks(paths, client, retry_delay_seconds=0, concurrency=2)
    assert progress["succeeded"] == 2
    assert progress["concurrency"] == 2
    assert progress["cumulative_request_seconds"] == 0.2
    assert progress["average_request_seconds"] == 0.1
    assert progress["average_completed_task_seconds"] == 0.1
    assert progress["eta_seconds"] == 0
    assert progress["run_elapsed_seconds"] >= 0
    assert progress["run_started_at"]
    assert progress["estimated_finish_at"]
    assert client.calls == 2

    with paths.results.open(encoding="utf-8") as file:
        rows = {row["patent_id"]: row for row in csv.DictReader(file)}
    assert rows["CN-A"]["label"] == "DATA_SECURITY"
    assert rows["CN-B"]["label"] == "OTHER"
    assert json.loads(rows["CN-A"]["processing_activities"]) == ["transmission"]
    assert json.loads(rows["CN-A"]["industry_sectors"]) == ["telecommunications"]
    assert json.loads(rows["CN-B"]["processing_activities"]) == ["other"]
    assert rows["CN-A"]["prefix_sha256"] == bundle.prefix_sha256

    resumed = FakeClient()
    run_tasks(paths, resumed, retry_delay_seconds=0)
    assert resumed.calls == 0


def test_step2_defaults_to_ten_concurrent_requests() -> None:
    args = build_parser().parse_args(["run", "--input", "patents_2026.csv"])
    assert args.concurrency == 10


def test_runner_removes_lock_after_exit(tmp_path: Path) -> None:
    database = tmp_path / "tasks.sqlite3"
    database.touch()
    lock = database.with_name("tasks.sqlite3.run.lock")

    with _exclusive_lock(database):
        assert lock.is_file()

    assert not lock.exists()


def test_progress_duration_format() -> None:
    assert _format_duration(0) == "00:00:00"
    assert _format_duration(3661.4) == "01:01:01"


def test_console_progress_includes_timing_and_eta(capsys) -> None:
    _print_progress(
        {
            "model": "ark-model",
            "completed": 25,
            "total": 100,
            "succeeded": 24,
            "failed": 1,
            "progress_percent": 25.0,
            "concurrency": 10,
            "run_elapsed_seconds": 65,
            "average_request_seconds": 2.5,
            "eta_seconds": 75,
            "usage": {"cached_tokens": 1234},
        }
    )
    output = capsys.readouterr().out
    assert "completed=25/100" in output
    assert "succeeded=24 failed=1" in output
    assert "concurrency=10" in output
    assert "elapsed=00:01:05" in output
    assert "avg=2.50s" in output
    assert "eta=00:01:15" in output
