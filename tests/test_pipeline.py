import csv
import json
from pathlib import Path

from patent_data_security.audit import audit_routes
from patent_data_security.pipeline import SamplingConfig, route_csv, stable_selection


def test_stable_selection_is_deterministic() -> None:
    first = stable_selection("2021|industry|G|both|CN1", 0.5, "seed")
    assert stable_selection("2021|industry|G|both|CN1", 0.5, "seed") is first
    assert stable_selection("anything", 0, "seed") is False
    assert stable_selection("anything", 1, "seed") is True


def test_route_csv_writes_all_routes_and_candidates(tmp_path: Path) -> None:
    source = tmp_path / "patents_2021.csv"
    source.write_text(
        "申请号,申请年份,上市公司行业,专利名称,IPC分类号,IPC主分类号,摘要文本,主权项内容\n"
        "CN1,2021,软件,隐私方法,,,采用联邦学习保护模型,权利要求一\n"
        "CN2,2021,制造,普通装置,G06F16/25,G06F16/25,普通数据处理,权利要求二\n",
        encoding="utf-8",
    )
    sampling = SamplingConfig(both_docs_rate=1, one_doc_rate=1, no_docs_rate=1)

    outputs = route_csv(
        source,
        tmp_path / "out",
        sampling=sampling,
        checkpoint_every=1,
        progress_every=0,
    )

    with outputs.routes.open(encoding="utf-8") as file:
        routes = list(csv.DictReader(file))
    with outputs.candidates.open(encoding="utf-8") as file:
        candidates = [json.loads(line) for line in file]
    summary = json.loads(outputs.summary.read_text(encoding="utf-8"))

    assert [row["route_level"] for row in routes] == ["S", "E"]
    assert routes[1]["is_e_sample"] == "true"
    assert len(candidates) == 2
    assert summary["stats"]["records"] == 2
    assert summary["stats"]["candidate_rows"] == 2
    assert summary["stats"]["candidates"] == 2
    assert summary["stats"]["route_levels"] == {"S": 1, "E": 1}


def test_route_csv_deduplicates_llm_candidates_by_patent_id(tmp_path: Path) -> None:
    source = tmp_path / "patents_2021.csv"
    source.write_text(
        "申请号,申请年份,企业名称,专利名称,摘要文本,主权项内容\n"
        "CN1,2021,甲公司,隐私方法,采用联邦学习保护模型,权利要求一\n"
        "CN1,2021,乙公司,隐私方法,采用联邦学习保护模型,权利要求一\n",
        encoding="utf-8",
    )

    outputs = route_csv(source, tmp_path / "out", progress_every=0)

    with outputs.routes.open(encoding="utf-8") as file:
        routes = list(csv.DictReader(file))
    with outputs.candidates.open(encoding="utf-8") as file:
        candidates = [json.loads(line) for line in file]
    summary = json.loads(outputs.summary.read_text(encoding="utf-8"))

    assert len(routes) == 2
    assert len(candidates) == 1
    assert routes[0]["classification_key"] == routes[1]["classification_key"]
    assert routes[1]["process_status"] == "pending_llm_shared"
    assert summary["stats"]["candidate_rows"] == 2
    assert summary["stats"]["candidates"] == 1

    audit = audit_routes(outputs.routes, outputs.candidates, tmp_path / "audit.json")
    assert audit["all_checks_passed"] is True
