import csv
import json
from pathlib import Path

from pipeline.step1.runner import run_step1


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def test_step1_streams_deduplicates_routes_and_samples(tmp_path: Path) -> None:
    source = tmp_path / "patents_2021.csv"
    source.write_text(
        "申请号,申请年份,专利名称,申请人,申请日,摘要文本,主权项内容,IPC分类号\n"
        "CN1,2021,普通计算,甲公司,2021-01-01,普通处理,,\n"
        "CN1,2021,加密方法,甲公司,2021-01-01,,采用同态加密,G06F21/62\n"
        "CN2,2021,食品安全检测,乙公司,2021-01-02,结果写入数据库,,G06F21/62\n"
        ",2021,密码学协议,丙公司,2021-01-03,,,\n",
        encoding="utf-8",
    )

    outputs = run_step1(
        source,
        tmp_path / "step1",
        workers=1,
        progress_every=0,
        e_sample_rate=1,
    )

    assert outputs.result == (tmp_path / "step1" / "2021" / "result.csv").resolve()
    assert outputs.manifest == (tmp_path / "step1" / "2021" / "manifest.json").resolve()
    rows = _read_csv(outputs.result)
    assert len(rows) == 3
    by_title = {row["title"]: row for row in rows}
    assert by_title["加密方法"]["route"] == "S"
    assert by_title["加密方法"]["association_count"] == "2"
    assert by_title["加密方法"]["source_row_number"] == "3"
    assert by_title["食品安全检测"]["route"] == "E"
    assert by_title["食品安全检测"]["selection_group"] == "E_random"
    assert json.loads(by_title["食品安全检测"]["ipc_audit_hits"])[0]["rule_id"] == (
        "IPC-AUDIT-G06F21"
    )
    synthetic = by_title["密码学协议"]
    assert synthetic["route"] == "S"
    assert synthetic["synthetic_id"] == "True"
    assert synthetic["patent_id"].startswith("synthetic-")

    summary = json.loads(outputs.manifest.read_text(encoding="utf-8"))
    assert summary["stats"] == {
        "input_rows": 4,
        "unique_patents": 3,
        "duplicate_association_rows": 1,
        "route_counts": {"S": 2, "E": 1},
        "selected_for_step2": {"S_all": 2, "E_random": 1},
    }
    assert summary["llm_requests_executed"] == 0
    assert not (tmp_path / "step1" / "2021" / ".tasks.partial.sqlite3").exists()


def test_e_sampling_is_stable_across_runs(tmp_path: Path) -> None:
    source = tmp_path / "patents_2022.csv"
    rows = "\n".join(f"CN{index},普通方法{index},一般机械处理" for index in range(20))
    source.write_text(
        "申请号,专利名称,摘要文本\n" + rows + "\n",
        encoding="utf-8",
    )

    first = run_step1(
        source,
        tmp_path / "first",
        progress_every=0,
        e_sample_rate=0.35,
        e_sample_seed="stable-test",
    )
    second = run_step1(
        source,
        tmp_path / "second",
        progress_every=0,
        e_sample_rate=0.35,
        e_sample_seed="stable-test",
    )

    selected_first = {
        row["patent_id"] for row in _read_csv(first.result) if row["selected_for_step2"] == "true"
    }
    selected_second = {
        row["patent_id"] for row in _read_csv(second.result) if row["selected_for_step2"] == "true"
    }
    assert selected_first == selected_second
    assert 0 < len(selected_first) < 20
