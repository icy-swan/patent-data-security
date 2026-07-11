from pathlib import Path

from patent_data_security.records import iter_patent_records, read_patent_records


def test_read_patent_records_normalizes_source_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "patents.csv"
    csv_path.write_text(
        "\ufeff关联股票代码,code,企业名称,专利名称,申请号,IPC分类号,IPC主分类号,摘要文本,主权项内容\n"
        '="002121",2121,上海卡耐新能源有限公司,极耳组件,CN202120102449.4,'
        "H01M50/533; H01M50/528,H01M50/533,摘要内容,权利要求内容\n",
        encoding="utf-8",
    )

    records = read_patent_records(csv_path)

    assert len(records) == 1
    record = records[0]
    assert record.row_number == 2
    assert record.get("stock_code") == "002121"
    assert record.get("company_name") == "上海卡耐新能源有限公司"
    assert record.get("title") == "极耳组件"
    assert record.get("application_number") == "CN202120102449.4"
    assert record.get("ipc") == "H01M50/533; H01M50/528"
    assert record.get("main_ipc") == "H01M50/533"
    assert record.get("abstract") == "摘要内容"
    assert record.get("claim") == "权利要求内容"


def test_iter_patent_records_streams_with_limit(tmp_path: Path) -> None:
    csv_path = tmp_path / "patents.csv"
    csv_path.write_text(
        "关联股票代码,专利名称,摘要文本\n"
        "600000,第一条,摘要一\n"
        "600001,第二条,摘要二\n",
        encoding="utf-8",
    )

    records = list(iter_patent_records(csv_path, limit=1))

    assert len(records) == 1
    assert records[0].get("stock_code") == "600000"
    assert records[0].get("title") == "第一条"


def test_classification_text_uses_available_patent_fields(tmp_path: Path) -> None:
    csv_path = tmp_path / "patents.csv"
    csv_path.write_text(
        "专利名称,IPC分类号,摘要文本,主权项内容\n"
        "访问控制方法,G06F21/62,一种数据访问控制方法,1.一种访问控制方法\n",
        encoding="utf-8",
    )

    record = read_patent_records(csv_path)[0]

    assert record.classification_text == (
        "专利名称：访问控制方法\n"
        "摘要：一种数据访问控制方法\n"
        "主权项：1.一种访问控制方法\n"
        "IPC分类号：G06F21/62"
    )
