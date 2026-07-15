import csv
import json
from pathlib import Path

from patent_data_security.keyword_extraction import extract_keywords_csv


def test_step1_writes_separate_swre_files_without_llm(tmp_path: Path) -> None:
    source = tmp_path / "patents_2021.csv"
    source.write_text(
        "申请号,申请年份,专利名称,摘要文本,主权项内容\n"
        "CN-S,2021,强相关,采用联邦学习保护模型,\n"
        "CN-W,2021,弱相关,用户数据采用零知识证明。,\n"
        "CN-R,2021,泛相关,用户数据采用加密保护。,\n"
        "CN-E,2021,未路由,普通机械装置,\n",
        encoding="utf-8",
    )

    outputs = extract_keywords_csv(source, tmp_path / "step1", progress_every=0)

    for tier, path in outputs.by_tier().items():
        with path.open(encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        assert len(rows) == 1
        assert rows[0]["keyword_level"] == tier

    with outputs.w.open(encoding="utf-8") as file:
        weak = next(csv.DictReader(file))
    weak_hits = json.loads(weak["keyword_hits"])
    assert weak_hits[0]["context_scope"] == "sentence"
    assert weak_hits[0]["context_hits"][0]["context_id"] == "CTX-DATA-OBJECT"

    summary = json.loads(outputs.summary.read_text(encoding="utf-8"))
    assert summary["stats"]["levels"] == {"S": 1, "W": 1, "R": 1, "E": 1}
    assert summary["llm_requests_executed"] == 0
