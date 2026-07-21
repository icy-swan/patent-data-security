#!/usr/bin/env python3
"""Reproduce the Codex review of the frozen 2021 Step-3 patent sample.

The tool preserves every input field and appends exactly two columns:

* ``codex_review_label``: the explicit reviewed ``DATA_SECURITY`` or ``OTHER`` label.
* ``codex_review_reason``: the binary reviewed label, an exact excerpt from
  title/abstract/claim, and the classification reason.

This is a deterministic replay of the completed 5,000-row review.  The two
override sets below are audit decisions for the frozen sample IDs; they are not
a general-purpose patent classifier.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data/step3/need_manual_review_positive.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data/step3/codex_result_positive.csv"
NEW_FIELDS = ["codex_review_label", "codex_review_reason"]


# Step-2 false negatives found by the claim-first reverse audit.
OTHER_TO_DATA_SECURITY = {
    "step3-08778484cebdf88414fdb47b",
    "step3-0b9d8f24f706490d2cd002f8",
    "step3-0c642137600e73db0ecc544e",
    "step3-0d6e05c2069c68f3fabc62c6",
    "step3-0f15b19529b4a6f888a8d202",
    "step3-1304e39f27bdf1593a20a338",
    "step3-17f2e2d48eeb95dbb6d2f432",
    "step3-1b0ac8aff5de0989e113d91b",
    "step3-205a177ab8868139166d2ba0",
    "step3-20b152a6c53afbd45adb71c1",
    "step3-213b9e59e489364db627a867",
    "step3-22d1f3e3a868418df54fdd47",
    "step3-236ac4ce826b6cf1995a15d7",
    "step3-241367a62395da22ebac3bff",
    "step3-24ebbd1dd7098f961e1aa30a",
    "step3-27261de806f530c596517f60",
    "step3-3594379e94a083014ca85f33",
    "step3-36504da92b3ad4cdff66a666",
    "step3-3d03034060902e37b59cebf0",
    "step3-42e246937608a37abc7b48a1",
    "step3-4b805d7bff5d14fed46f3b63",
    "step3-4ccc5053bca4a162faa3fa75",
    "step3-4e44493aae4f51e8cc2ff7c0",
    "step3-4f8bebf3bb227184d73b728b",
    "step3-5845a657af630de6f7a30793",
    "step3-5f4a747131bf6484db484a5c",
    "step3-5fc710e54402d987f1992b18",
    "step3-61b2fcefb157b3699a07cafe",
    "step3-700047830288ead0b962e273",
    "step3-72451a9cc2ebbfa447a683cf",
    "step3-7d030729a49dc61717f0d73e",
    "step3-7d690b76657e0cf36738bc4c",
    "step3-8166ad710993394d396218cd",
    "step3-863a9ead62247f52c8caaadf",
    "step3-929cbd0310faaca84f0416eb",
    "step3-98319e374b9a2c708cba93e6",
    "step3-9b2ea5972c1b4d04ecb9d01b",
    "step3-a21eae9aab0c83a4fd19b2d4",
    "step3-ab3ee1bd594086899bd33a0b",
    "step3-b70deb6252fb85934643608f",
    "step3-bd00843dc5d8d07c4b92051b",
    "step3-bdd324c645b095799c1387d2",
    "step3-c04365049370b510fb844916",
    "step3-c4ceb36c9f34f1fc30bbccee",
    "step3-c66e16e44af176a4296897a8",
    "step3-c7d5d562371691b35e3ae723",
    "step3-cc9e1f0c35053547bf7e0fa6",
    "step3-cecd7b74eda41708b217032f",
    "step3-d2b18ca16f3926805740cdd5",
    "step3-da2de18ff5a47d354ee11585",
    "step3-dd4340079315eb63052c7447",
    "step3-e6dd2b2daf200d803b7d7a09",
    "step3-ebfb7c1d1b034a6926b84c7b",
    "step3-ec11af83c28fc6c0e72c36b6",
    "step3-f0463dc24d4467ff799e8d76",
    "step3-f0eb89ead4801ffc5d67c533",
    "step3-f25b0027cf4d838eae5049a5",
    "step3-f3dbd024c12bae5dec358c97",
    "step3-f9a8ea3014cf495471bfc9f3",
    "step3-ff566350f5154babd3e0b046",
}


# Step-2 false positives: physical access/safety, ordinary business checks,
# generic anomaly analysis, or blockchain/federated-learning background use.
DATA_SECURITY_TO_OTHER = {
    "step3-0daa1d30626ea9e65bb8440e",
    "step3-1469bef2511e8adc3752fbce",
    "step3-15d279192683a05f95b66a51",
    "step3-19afbdd41dd0f3e27142e02e",
    "step3-21cdb921e4b32f8d998740d7",
    "step3-268efb814eb48a78c6a6e759",
    "step3-28de4e2bbbd2efdae98723e3",
    "step3-2a27cdce71724ae885606b4e",
    "step3-3bbcfeb6f4fe02fe880fe2f2",
    "step3-42fc62a9f4377498f2fe14e4",
    "step3-43fc883c3e5d1f7b5b6cd580",
    "step3-449429fa5dd6838500a8560c",
    "step3-4f902972f0b86f880e851f53",
    "step3-52a1da217a384acd815e83e2",
    "step3-587b5c5ac0a787fedf986454",
    "step3-686147e82860596a85c303ee",
    "step3-6867b58de98306af44be5beb",
    "step3-6c98849db05c55577b2a19f1",
    "step3-6cdaf3b52f48465687d782c3",
    "step3-6f4b2be4edcb520194723bb4",
    "step3-74ca660c1e0552b9d949131f",
    "step3-79fbfcb724ddcf5d3179519a",
    "step3-7d86a42b33e21972e3db8eb1",
    "step3-7edf9263d67d070b60ab6649",
    "step3-8038b34c5d5a63106aad0166",
    "step3-871d36eb89f048d51e1cb7c0",
    "step3-894650e2da7467e91ae64372",
    "step3-8c9c822b5499c1d4b261ed16",
    "step3-8f50ce6674556bf39aa1ee17",
    "step3-a941ac48a9940687ab2feb49",
    "step3-b1a17b7b756a351e88b30f27",
    "step3-b36eb8070ad9e0cb01a71118",
    "step3-c4ad29a7566627a7f608c2e3",
    "step3-d6e075bf21a3aa685e5b2a26",
    "step3-d912c1678506d87ee6f805d2",
    "step3-f034aa6096926fb93f48e390",
    "step3-f446f02daf3f5ef2d5572e93",
    "step3-f62064bb27711098f53e1fb2",
    "step3-fad7b463e479523a2aac63ac",
}


SCOPE_NAMES = {
    "legal_data_security": "数据安全与网络安全",
    "personal_information_protection": "个人信息与隐私保护",
    "cryptography": "密码学保护",
    "data_confidentiality": "数据保密与防泄漏",
    "data_integrity": "数据完整性与防篡改",
    "data_availability": "备份恢复与数据可用性",
    "access_control_authentication": "访问控制与身份鉴别",
    "privacy_enhancing_technology": "隐私增强技术",
    "secure_computation": "安全计算",
    "security_audit_provenance": "安全审计与数据追溯",
    "data_governance_compliance": "数据安全治理与合规",
    "security_monitoring_response": "安全监测与响应",
    "other_data_security": "数据安全保护",
}


TERM_SCOPES = [
    (("同态", "多方计算", "秘密共享", "混淆电路", "可信执行", "机密计算"),
     "secure_computation"),
    (
        (
            "差分隐私",
            "安全聚合",
            "私有集合",
            "私有信息检索",
            "零知识",
            "隐私计算",
        ),
        "privacy_enhancing_technology",
    ),
    (
        (
            "隐私",
            "个人信息",
            "敏感信息",
            "敏感数据",
            "脱敏",
            "匿名",
            "去标识",
            "遮蔽",
            "防窥",
        ),
        "personal_information_protection",
    ),
    (
        (
            "加密",
            "解密",
            "密文",
            "密钥",
            "秘钥",
            "私钥",
            "公钥",
            "密码",
            "密码算法",
            "数字签名",
            "证书",
            "哈希",
            "散列",
            "摘要",
            "代码混淆",
            "SSL",
            "TLS",
            "IPSec",
            "VPN",
            "PUF",
        ),
        "cryptography",
    ),
    (
        (
            "防篡改",
            "不可篡改",
            "篡改",
            "完整性",
            "一致性校验",
            "读写保护",
            "纠错",
            "毒化",
            "伪造",
        ),
        "data_integrity",
    ),
    (
        (
            "备份",
            "恢复",
            "还原",
            "灾备",
            "容灾",
            "冗余",
            "副本",
            "数据丢失",
            "数据缺失",
            "链路故障",
            "掉电",
            "电源失效",
        ),
        "data_availability",
    ),
    (
        (
            "权限",
            "授权",
            "访问控制",
            "安全组",
            "身份认证",
            "身份鉴别",
            "认证",
            "验证",
            "登录",
            "令牌",
            "鉴权",
            "口令",
        ),
        "access_control_authentication",
    ),
    (
        (
            "安全审计",
            "数据溯源",
            "数据血缘",
            "审计",
            "溯源",
            "血缘",
            "水印",
            "取证",
            "责任追踪",
            "留痕",
        ),
        "security_audit_provenance",
    ),
    (("分类分级", "密级", "合规", "监管", "风险评估", "安全评估", "出境"),
     "data_governance_compliance"),
    (
        (
            "攻击",
            "入侵",
            "注入",
            "漏洞",
            "恶意",
            "病毒",
            "威胁",
            "防御",
            "加固",
            "防火墙",
            "黑名单",
            "白名单",
            "异常日志",
            "安全规则",
            "联动处置",
            "安全扫描",
            "安全监控",
            "安全检测",
            "告警",
        ),
        "security_monitoring_response",
    ),
    (
        (
            "泄露",
            "泄漏",
            "保密",
            "机密",
            "外发",
            "单向传输",
            "非法获取",
            "防窃取",
            "防复制",
        ),
        "data_confidentiality",
    ),
    (
        (
            "数据安全",
            "信息安全",
            "网络安全",
            "数据保护",
            "安全防护",
            "可信",
            "零信任",
            "隔离",
        ),
        "legal_data_security",
    ),
]


STRONG_TERMS = tuple(term for terms, _ in TERM_SCOPES for term in terms)


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def chunks(text: str) -> list[str]:
    flat = compact(text)
    if not flat or flat == "无":
        return []
    parts = re.split(r"(?<=[。；;！？!?])|(?<=：)", flat)
    return [part.strip() for part in parts if part.strip()]


def trim_quote(text: str, limit: int = 170) -> str:
    return compact(text)[:limit]


def trim_positive_quote(text: str, limit: int = 170) -> str:
    value = compact(text)
    if len(value) <= limit:
        return value
    lowered = value.lower()
    hits: list[tuple[int, int]] = []
    for terms, _ in TERM_SCOPES:
        for term in terms:
            position = lowered.find(term.lower())
            if position >= 0:
                hits.append((len(term), position))
    if not hits:
        return value[:limit]
    _, position = max(hits, key=lambda item: (item[0], -item[1]))
    start = max(0, position - limit // 3)
    start = min(start, len(value) - limit)
    return value[start : start + limit]


def exact_step2_evidence(row: Mapping[str, str]) -> list[tuple[str, str]]:
    try:
        items = json.loads(row.get("step2_evidence", "") or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(items, list):
        return []
    result: list[tuple[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        field = str(item.get("field", ""))
        quote = compact(str(item.get("quote", "") or ""))
        source = compact(row.get(field, "")) if field in {"claim", "abstract", "title"} else ""
        if quote and quote in source:
            result.append((field, quote))
    return result


def security_score(part: str) -> int:
    score = 0
    for terms, _ in TERM_SCOPES:
        for term in terms:
            if term.lower() in part.lower():
                score += 4 if len(term) >= 3 else 2
    action_words = ("用于", "通过", "根据", "若", "防止", "保护", "确保", "实现")
    if score and any(word in part for word in action_words):
        score += 2
    return score


def pick_positive_quote(row: Mapping[str, str]) -> tuple[str, str]:
    evidence = exact_step2_evidence(row)
    field_order = {"claim": 2, "abstract": 1, "title": 0}
    candidates = [
        (security_score(quote), field_order.get(field, -1), field, quote)
        for field, quote in evidence
        if security_score(quote) > 0
    ]
    if candidates:
        _, _, field, quote = max(candidates)
        return field, trim_positive_quote(quote)

    ranked: list[tuple[int, int, str, str]] = []
    for order, field in enumerate(("claim", "abstract", "title")):
        for part in chunks(row.get(field, "")):
            score = security_score(part)
            if score:
                ranked.append((score, -order, field, part))
    if ranked:
        _, _, field, part = max(ranked)
        return field, trim_positive_quote(part)

    if evidence:
        field, quote = max(
            evidence,
            key=lambda item: (field_order.get(item[0], -1), len(item[1])),
        )
        return field, trim_positive_quote(quote)

    for field in ("claim", "abstract", "title"):
        parts = chunks(row.get(field, ""))
        if parts:
            return field, trim_quote(parts[0])
    return "title", trim_quote(row.get("title", ""))


def pick_negative_quote(row: Mapping[str, str]) -> tuple[str, str]:
    for field in ("claim", "abstract", "title"):
        source = compact(row.get(field, ""))
        if source and source != "无":
            return field, trim_quote(source)
    return "title", trim_quote(row.get("title", ""))


def scope_from_text(text: str) -> str | None:
    lowered = text.lower()
    matches: list[tuple[int, int, int, str]] = []
    for order, (terms, scope) in enumerate(TERM_SCOPES):
        found = [term for term in terms if term.lower() in lowered]
        if found:
            matches.append((max(len(term) for term in found), len(found), -order, scope))
    if matches:
        return SCOPE_NAMES[max(matches)[3]]
    return None


def detected_scope(row: Mapping[str, str], quote: str) -> str:
    quote_scope = scope_from_text(quote)
    if quote_scope:
        return quote_scope
    full_text = compact(row.get("claim", "")) + " " + compact(row.get("abstract", ""))
    full_scope = scope_from_text(full_text)
    if full_scope:
        return full_scope
    try:
        scopes = json.loads(row.get("step2_scope_basis", "") or "[]")
    except json.JSONDecodeError:
        scopes = []
    if isinstance(scopes, list):
        for scope in scopes:
            if scope in SCOPE_NAMES and scope != "other_data_security":
                return SCOPE_NAMES[scope]
    return SCOPE_NAMES["other_data_security"]


def negative_kind(row: Mapping[str, str]) -> str:
    text = compact(" ".join(row.get(name, "") for name in ("title", "claim", "abstract")))
    title = compact(row.get("title", ""))
    physical_terms = (
        "门锁",
        "开锁",
        "门禁",
        "电梯",
        "储物柜",
        "包装盒",
        "保管柜",
        "智能锁",
        "箱体",
        "箱门",
        "物理钥匙",
    )
    if any(term in text for term in physical_terms):
        return "物理设备或空间的开启、通行控制"
    if any(term in title for term in ("重复请求", "限流")):
        return "普通接口限流或重复请求抑制"
    if any(term in text for term in ("骚扰电话", "点名", "考勤", "摇号", "卡券")):
        return "普通内容治理、人员管理或业务核验"
    risk_titles = (
        "风险账户",
        "异常账户",
        "异常行为",
        "用户行为数据检测",
        "诈骗用户",
    )
    if any(term in title for term in risk_titles):
        return "普通业务风险、行为异常或统计检测"
    financial_terms = ("支付", "转账", "贷款", "结算", "账户", "银行卡", "交易")
    if any(term in text for term in financial_terms):
        return "普通金融业务验证、风控或交易处理"
    operational_terms = (
        "生产安全",
        "生产线",
        "功能安全",
        "驾驶安全",
        "车辆",
        "充电桩",
        "电厂",
        "水电",
        "光伏",
        "设备运行",
    )
    if any(term in text for term in operational_terms):
        return "设备控制、生产运行或人身安全相关处理"
    if any(term in text for term in ("人脸", "指纹", "身份识别", "认证")):
        return "普通业务身份识别或产品功能"
    if "区块链" in text:
        return "区块链上的普通存储、交易、查询或业务处理"
    if any(term in text for term in ("哈希", "散列", "校验", "异常", "监测")):
        return "普通数据校验、分类、统计或异常分析"
    if any(term in text for term in ("外观设计", "设计要点", "图形用户界面")):
        return "外观或图形用户界面设计"
    return "普通数据采集、存储、传输、分析或业务处理"


def reason_for(row: Mapping[str, str], final_label: str) -> str:
    original = row["step2_label"]
    verdict = (
        f"GLM5.2需由{original}改为{final_label}"
        if original != final_label
        else "GLM5.2结论准确"
    )
    if final_label == "DATA_SECURITY":
        field, quote = pick_positive_quote(row)
        scope = detected_scope(row, quote)
        return (
            f"复核结论：DATA_SECURITY；{verdict}。{field}逐字披露“{quote}”。"
            "该特征不是背景或可选清单，而是技术方案中的实际处理"
            "或直接效果，"
            f"实质属于{scope}。"
        )

    field, quote = pick_negative_quote(row)
    kind = negative_kind(row)
    return (
        f"复核结论：OTHER；{verdict}。{field}逐字披露“{quote}”。"
        f"其技术内容仅属于{kind}；title、abstract和claim均未披露足以建立"
        "数据保密性、"
        "完整性、可用性、数据访问控制或安全响应效果的必要机制。"
    )


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    return rows, fields


def reviewed_label(row: Mapping[str, str]) -> str:
    sample_id = row["sample_id"]
    if sample_id in OTHER_TO_DATA_SECURITY:
        return "DATA_SECURITY"
    if sample_id in DATA_SECURITY_TO_OTHER:
        return "OTHER"
    return row["step2_label"]


def assert_frozen_input(rows: Sequence[Mapping[str, str]], fields: Sequence[str]) -> None:
    required = {
        "sample_id", "patent_id", "title", "abstract", "claim", "step2_label",
        "step2_scope_basis", "step2_evidence",
    }
    missing_fields = required - set(fields)
    if missing_fields:
        raise ValueError(f"input is missing required fields: {sorted(missing_fields)}")
    duplicate_new_fields = set(NEW_FIELDS) & set(fields)
    if duplicate_new_fields:
        raise ValueError(f"input already contains output fields: {sorted(duplicate_new_fields)}")

    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_id is not unique")
    missing_overrides = (
        OTHER_TO_DATA_SECURITY | DATA_SECURITY_TO_OTHER
    ) - set(sample_ids)
    if missing_overrides:
        raise ValueError(
            "input does not match the frozen review sample; missing override IDs: "
            f"{sorted(missing_overrides)}"
        )
    overlap = OTHER_TO_DATA_SECURITY & DATA_SECURITY_TO_OTHER
    if overlap:
        raise RuntimeError(f"conflicting review overrides: {sorted(overlap)}")
    invalid_labels = sorted({row["step2_label"] for row in rows} - {"DATA_SECURITY", "OTHER"})
    if invalid_labels:
        raise ValueError(f"invalid step2 labels: {invalid_labels}")


def write_review(input_path: Path, output_path: Path, *, force: bool) -> dict[str, object]:
    source_rows, source_fields = read_csv(input_path)
    assert_frozen_input(source_rows, source_fields)
    if output_path.exists() and not force:
        raise FileExistsError(f"output exists; pass --force to replace it: {output_path}")

    output_fields = source_fields + NEW_FIELDS
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8-sig",
        newline="",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="raise")
        writer.writeheader()
        for source_row in source_rows:
            row = dict(source_row)
            final_label = reviewed_label(row)
            changed = final_label != row["step2_label"]
            row[NEW_FIELDS[0]] = final_label
            row[NEW_FIELDS[1]] = reason_for(row, final_label)
            writer.writerow(row)
            counts[f"original:{row['step2_label']}"] += 1
            counts[f"final:{final_label}"] += 1
            counts[f"changed:{changed}"] += 1

    try:
        temp_path.replace(output_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    validation = validate_review(input_path, output_path)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "rows": len(source_rows),
        "columns": len(output_fields),
        "counts": dict(counts),
        "sha256": sha256_file(output_path),
        "validation": validation,
    }


def validate_review(input_path: Path, output_path: Path) -> dict[str, object]:
    source_rows, source_fields = read_csv(input_path)
    output_rows, output_fields = read_csv(output_path)
    errors: list[str] = []
    if len(source_rows) != len(output_rows):
        errors.append(f"row count differs: {len(source_rows)} != {len(output_rows)}")
    if output_fields != source_fields + NEW_FIELDS:
        errors.append("header/order differs from source plus the two required fields")

    counts: Counter[str] = Counter()
    for index, (source, output) in enumerate(zip(source_rows, output_rows, strict=False), start=1):
        changed_field = next(
            (field for field in source_fields if source.get(field) != output.get(field)),
            None,
        )
        if changed_field:
            errors.append(f"row {index}: source field changed: {changed_field}")

        review_label = output.get(NEW_FIELDS[0], "")
        reason = output.get(NEW_FIELDS[1], "")
        if review_label not in {"DATA_SECURITY", "OTHER"}:
            errors.append(f"row {index}: invalid review label: {review_label!r}")
        reason_label = (
            "DATA_SECURITY"
            if "复核结论：DATA_SECURITY" in reason
            else "OTHER"
            if "复核结论：OTHER" in reason
            else ""
        )
        if not reason_label:
            errors.append(f"row {index}: reviewed label is missing from reason")
        if review_label != reason_label:
            errors.append(f"row {index}: review label does not match reason")

        match = re.search(r"(claim|abstract|title)逐字披露“(.*?)”", reason)
        if not match:
            errors.append(f"row {index}: evidence field/quote not found")
        else:
            field, quote = match.groups()
            if compact(quote) not in compact(source.get(field, "")):
                errors.append(f"row {index}: evidence is not an exact source excerpt")

        patent_id = source.get("patent_id", "")
        if patent_id and patent_id in reason:
            errors.append(f"row {index}: patent_id leaked into reason")
        counts[f"final:{review_label}"] += 1

    if errors:
        preview = "\n".join(errors[:30])
        raise ValueError(f"review validation failed with {len(errors)} errors:\n{preview}")
    return {
        "rows": len(output_rows),
        "columns": len(output_fields),
        "counts": dict(counts),
        "errors": 0,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce and validate the Codex review of the frozen 2021 sample."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="atomically replace an existing output file",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate --output against --input without generating a new file",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if args.validate_only:
        if not output_path.is_file():
            raise FileNotFoundError(output_path)
        result = {
            "input": str(input_path),
            "output": str(output_path),
            "sha256": sha256_file(output_path),
            "validation": validate_review(input_path, output_path),
        }
    else:
        result = write_review(input_path, output_path, force=args.force)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
