import csv
import sqlite3
from pathlib import Path

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
        label = PatentClassification(
            cat=1,
            confidence=0.9,
            subtype="privacy_computing",
            evidence=[patent["title"] or "测试证据"],
            reason="测试分类",
            review_flag=False,
            review_reason="",
        )
        return ArkClassificationResponse(
            classification=label,
            response_id=f"response-{self.calls}",
            requested_model=self.model,
            actual_model=self.model,
            elapsed_seconds=0.25,
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
