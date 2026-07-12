import csv
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from patent_data_security.keyword_extraction import extract_keywords_csv
from patent_data_security.step2_prompt import ArkClassificationResponse, PatentClassification
from patent_data_security.step2_runner import (
    prepare_classification_tasks,
    run_classification_tasks,
)


class FakeArkClient:
    model = "fake-ark-model"

    def __init__(self) -> None:
        self.calls = 0

    def classify(self, patent):
        self.calls += 1
        return _successful_response(patent, f"response-{self.calls}")


def _successful_response(
    patent: dict[str, str],
    response_id: str,
    *,
    elapsed_seconds: float = 0.25,
) -> ArkClassificationResponse:
    label = PatentClassification(
        cat=1,
        confidence=0.9,
        subtype="privacy_computing",
        core_invention="通过联邦学习保护联合建模。",
        evidence_chain={
            "protected_object_or_activity": "模型参数",
            "security_goal_or_risk": "参数泄露风险",
            "technical_mechanism": "联邦学习",
            "causal_centrality": "保护是核心效果",
            "missing_or_ambiguous_link": "",
        },
        evidence=[patent["title"] or "测试证据"],
        reason="测试分类",
        review_flag=False,
        review_reason="",
    )
    return ArkClassificationResponse(
        classification=label,
        response_id=response_id,
        requested_model="fake-ark-model",
        actual_model="fake-ark-model",
        elapsed_seconds=elapsed_seconds,
        usage={"input_tokens": 10},
        raw_text=label.model_dump_json(),
    )


def test_step2_prepares_unique_swr_and_e_sample_then_resumes(tmp_path: Path) -> None:
    raw = tmp_path / "patents_2021.csv"
    raw.write_text(
        "申请号,申请年份,专利名称,摘要文本,主权项内容,IPC分类号,IPC主分类号\n"
        "CN-S,2021,强相关,采用联邦学习保护模型,权利要求一,,\n"
        "CN-S,2021,强相关重复,采用联邦学习保护模型,权利要求一,,\n"
        "CN-W,2021,弱相关,采用数据备份恢复业务数据,权利要求二,,\n"
        "CN-R,2021,泛相关,采用防火墙保护网络,权利要求三,,\n"
        "CN-E,2021,未路由,普通机械装置,权利要求四,,\n",
        encoding="utf-8",
    )
    step1 = tmp_path / "step1"
    extract_keywords_csv(raw, step1, progress_every=0)

    paths, summary = prepare_classification_tasks(
        raw,
        step1,
        tmp_path / "step2",
        e_sample_rate=1,
    )
    assert summary["unique_swr"] == {"S": 1, "W": 1, "R": 1}
    assert summary["e_selected"] == 1
    assert summary["total_tasks"] == 4

    client = FakeArkClient()
    progress = run_classification_tasks(paths, client, retry_delay_seconds=0)
    assert progress["completed"] == 4
    assert progress["model"] == "fake-ark-model"
    assert progress["average_request_seconds"] == 0.25
    assert client.calls == 4

    with paths.results.open(encoding="utf-8") as file:
        results = list(csv.DictReader(file))
    assert len(results) == 4
    assert {row["status"] for row in results} == {"succeeded"}

    resumed_client = FakeArkClient()
    resumed = run_classification_tasks(paths, resumed_client, retry_delay_seconds=0)
    assert resumed["completed"] == 4
    assert resumed_client.calls == 0


def test_e_two_percent_sample_records_probability_and_weight(tmp_path: Path) -> None:
    raw = tmp_path / "patents_2020.csv"
    rows = [
        f"CN-E-{index},2020,普通装置{index},机械结构,权利要求{index}"
        for index in range(500)
    ]
    raw.write_text(
        "申请号,申请年份,专利名称,摘要文本,主权项内容\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    step1 = tmp_path / "step1"
    extract_keywords_csv(raw, step1, progress_every=0)
    paths, summary = prepare_classification_tasks(raw, step1, tmp_path / "step2")

    assert summary["e_sample_rate"] == 0.02
    assert 0 < summary["e_selected"] < 30
    connection = sqlite3.connect(paths.database)
    probabilities = connection.execute(
        "SELECT DISTINCT selection_probability, sample_weight FROM tasks"
    ).fetchall()
    connection.close()
    assert probabilities == [(0.02, 50.0)]


def test_same_patent_has_different_task_ids_across_years(tmp_path: Path) -> None:
    task_ids = []
    step1 = tmp_path / "step1"
    step2 = tmp_path / "step2"
    for year in (2020, 2021):
        raw = tmp_path / f"patents_{year}.csv"
        raw.write_text(
            "申请号,申请年份,专利名称,摘要文本,主权项内容\n"
            f"CN-SAME,{year},联邦方法,采用联邦学习,权利要求\n",
            encoding="utf-8",
        )
        extract_keywords_csv(raw, step1, progress_every=0)
        paths, _ = prepare_classification_tasks(raw, step1, step2)
        connection = sqlite3.connect(paths.database)
        task_ids.append(connection.execute("SELECT task_id FROM tasks").fetchone()[0])
        connection.close()

    assert task_ids[0] != task_ids[1]


def test_concurrent_retries_honor_per_task_delay(tmp_path: Path) -> None:
    raw = tmp_path / "patents_2022.csv"
    raw.write_text(
        "申请号,申请年份,专利名称,摘要文本,主权项内容\n"
        "CN-A,2022,联邦学习方法A,采用联邦学习保护模型参数,权利要求一\n"
        "CN-B,2022,联邦学习方法B,采用联邦学习保护模型参数,权利要求二\n",
        encoding="utf-8",
    )
    step1 = tmp_path / "step1"
    extract_keywords_csv(raw, step1, progress_every=0)
    paths, _ = prepare_classification_tasks(
        raw,
        step1,
        tmp_path / "step2",
        e_sample_rate=1,
    )

    class RetryOnceClient:
        model = "fake-ark-model"

        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.call_times: dict[str, list[float]] = {}

        def classify(self, patent):
            with self.lock:
                calls = self.call_times.setdefault(patent["title"], [])
                calls.append(time.monotonic())
                attempt = len(calls)
            if attempt == 1:
                raise RuntimeError("temporary rate limit")
            return _successful_response(patent, f"{patent['title']}-{attempt}")

    client = RetryOnceClient()
    retry_delay = 0.1
    progress = run_classification_tasks(
        paths,
        client,
        concurrency=2,
        max_attempts=2,
        retry_delay_seconds=retry_delay,
    )

    assert progress["succeeded"] == 2
    assert progress["failed"] == 0
    assert all(len(call_times) == 2 for call_times in client.call_times.values())
    assert all(
        call_times[1] - call_times[0] >= retry_delay * 0.8
        for call_times in client.call_times.values()
    )


def test_second_runner_cannot_reset_or_claim_in_flight_tasks(tmp_path: Path) -> None:
    raw = tmp_path / "patents_2023.csv"
    raw.write_text(
        "申请号,申请年份,专利名称,摘要文本,主权项内容\n"
        "CN-A,2023,联邦学习方法,采用联邦学习保护模型参数,权利要求一\n",
        encoding="utf-8",
    )
    step1 = tmp_path / "step1"
    extract_keywords_csv(raw, step1, progress_every=0)
    paths, _ = prepare_classification_tasks(
        raw,
        step1,
        tmp_path / "step2",
        e_sample_rate=1,
    )
    request_started = threading.Event()
    release_request = threading.Event()
    runner_errors: list[Exception] = []

    class BlockingClient:
        model = "fake-ark-model"

        def classify(self, patent):
            request_started.set()
            if not release_request.wait(timeout=3):
                raise TimeoutError("test did not release the request")
            return _successful_response(patent, "blocking-response")

    def run_first() -> None:
        try:
            run_classification_tasks(paths, BlockingClient(), retry_delay_seconds=0)
        except Exception as error:  # noqa: BLE001 - captured for assertion in main thread
            runner_errors.append(error)

    first_runner = threading.Thread(target=run_first)
    first_runner.start()
    assert request_started.wait(timeout=2)
    second_client = FakeArkClient()
    try:
        with pytest.raises(RuntimeError, match="already active"):
            run_classification_tasks(paths, second_client, retry_delay_seconds=0)
    finally:
        release_request.set()
        first_runner.join(timeout=3)

    assert not first_runner.is_alive()
    assert runner_errors == []
    assert second_client.calls == 0
