import csv
import json
import sqlite3
from collections import Counter
from pathlib import Path

import pytest

from patent_data_security.step2_runner import _initialize_database
from patent_data_security.step3_sampling import (
    BLINDED_FIELDS,
    GoldSamplingConfig,
    sample_gold_corpus,
)


def _classification(index: int, cat: int, confidence: float, subtype: str) -> dict:
    review = cat == 2
    return {
        "cat": cat,
        "confidence": confidence,
        "subtype": "potential_data_security" if review else subtype,
        "core_invention": f"测试发明{index}",
        "evidence_chain": {
            "protected_object_or_activity": "测试数据",
            "security_goal_or_risk": "测试风险" if cat != 3 else "",
            "technical_mechanism": "测试机制" if cat != 3 else "",
            "causal_centrality": "测试中心性" if cat != 3 else "不构成数据安全核心",
            "missing_or_ambiguous_link": "关键链路待核验" if review else "",
        },
        "evidence": [f"证据{index}"],
        "reason": f"测试理由{index}",
        "review_flag": review,
        "review_reason": "需要人工复核" if review else "",
    }


def _make_database(
    path: Path,
    *,
    eligible: int = 120,
    e_records: int = 20,
    pending: int = 0,
) -> None:
    connection = sqlite3.connect(path)
    _initialize_database(connection)
    now = "2026-07-13T00:00:00+00:00"
    for index in range(eligible + e_records):
        level = ("S", "W", "R")[index % 3] if index < eligible else "E"
        cat = (1, 2, 3, 3, 3)[index % 5]
        confidence = (0.75, 0.85, 0.90, 0.95, 0.99)[index % 5]
        subtype = "rare_type" if index < 8 else ("other" if cat == 3 else "data_integrity")
        attempts = 2 if index % 17 == 0 else 1
        status = "pending" if index < pending else "succeeded"
        result = _classification(index, cat, confidence, subtype)
        payload = {
            "title": f"专利{index}",
            "abstract": f"摘要{index}",
            "claim": f"主权项{index}",
            "ipc": "G06F21/62",
            "main_ipc": "G06F21/62",
        }
        connection.execute(
            """
            INSERT INTO tasks (
              task_id,dataset_id,patent_id,source_row_number,keyword_level,
              selection_group,selection_probability,sample_weight,payload_json,
              status,attempts,requested_model,actual_model,response_id,result_json,
              raw_response,usage_json,error,elapsed_seconds,created_at,updated_at,completed_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                f"task-{index}",
                "2021",
                f"patent-{index}",
                index + 2,
                level,
                level if level != "E" else "E_sample",
                1.0 if level != "E" else 0.02,
                1.0 if level != "E" else 50.0,
                json.dumps(payload, ensure_ascii=False),
                status,
                attempts,
                "glm-test",
                "glm-test-version",
                f"response-{index}",
                json.dumps(result, ensure_ascii=False),
                json.dumps(result, ensure_ascii=False),
                "{}",
                None,
                1.5,
                now,
                now,
                now if status == "succeeded" else None,
            ),
        )
    connection.commit()
    connection.close()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def test_step3_samples_exact_blinded_swr_gold_set_with_design_weights(
    tmp_path: Path,
) -> None:
    database = tmp_path / "classification_state_2021.sqlite3"
    _make_database(database)
    config = GoldSamplingConfig(
        target_size=40,
        core_size=30,
        seed="test-gold-seed",
        rare_subtype_max_population=8,
    )
    paths, report = sample_gold_corpus([database], tmp_path / "step3", config=config)

    audit = _read_csv(paths.audit)
    blinded = _read_csv(paths.blinded)
    annotator_a = _read_csv(paths.annotator_a)
    annotator_b = _read_csv(paths.annotator_b)
    strata = {row["sampling_stratum"]: row for row in _read_csv(paths.strata)}

    assert len(audit) == 40
    assert len({row["sample_id"] for row in audit}) == 40
    assert {row["keyword_level"] for row in audit} <= {"S", "W", "R"}
    assert Counter(row["sample_stage"] for row in audit) == {
        "representative_core": 30,
        "risk_addition": 10,
    }
    assert all(
        row["risk_group"] != "none"
        for row in audit
        if row["sample_stage"] == "risk_addition"
    )
    assert report["excluded"]["E"] == 20
    assert report["eligible_population"] == 120
    assert report["selected"] == 40

    for row in audit:
        stratum = strata[row["sampling_stratum"]]
        expected_probability = int(stratum["sample_size"]) / int(stratum["population_size"])
        assert float(row["inclusion_probability"]) == pytest.approx(expected_probability)
        assert float(row["evaluation_weight"]) == pytest.approx(1 / expected_probability)

    assert tuple(blinded[0]) == BLINDED_FIELDS
    assert {row["sample_id"] for row in blinded} == {row["sample_id"] for row in audit}
    assert [row["sample_id"] for row in annotator_a] != [
        row["sample_id"] for row in annotator_b
    ]
    forbidden = {
        "keyword_level",
        "glm_cat",
        "raw_confidence",
        "subtype",
        "glm_reason",
        "risk_group",
        "sampling_stratum",
    }
    assert forbidden.isdisjoint(blinded[0])
    assert all(row["annotator_slot"] == "A" for row in annotator_a)
    assert all(row["annotator_slot"] == "B" for row in annotator_b)


def test_step3_rebuild_with_same_seed_is_stable(tmp_path: Path) -> None:
    database = tmp_path / "classification_state_2021.sqlite3"
    _make_database(database, eligible=90, e_records=0)
    config = GoldSamplingConfig(target_size=30, core_size=20, seed="stable-seed")
    paths, _ = sample_gold_corpus([database], tmp_path / "step3", config=config)
    first = [(row["sample_id"], row["task_id"]) for row in _read_csv(paths.audit)]

    with pytest.raises(FileExistsError, match="--rebuild"):
        sample_gold_corpus([database], tmp_path / "step3", config=config)

    rebuilt_paths, _ = sample_gold_corpus(
        [database], tmp_path / "step3", config=config, rebuild=True
    )
    second = [(row["sample_id"], row["task_id"]) for row in _read_csv(rebuilt_paths.audit)]
    assert second == first


def test_step3_rejects_incomplete_swr_tasks(tmp_path: Path) -> None:
    database = tmp_path / "classification_state_2021.sqlite3"
    _make_database(database, eligible=10, e_records=0, pending=1)

    with pytest.raises(ValueError, match="must have succeeded"):
        sample_gold_corpus(
            [database],
            tmp_path / "step3",
            config=GoldSamplingConfig(target_size=5, core_size=4),
        )
