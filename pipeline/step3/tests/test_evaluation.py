from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

from pipeline.step3.evaluation import evaluate_pipeline_results
from pipeline.step3.sampling import (
    FROZEN_RESULT_FIELDS,
    _initialize_task_database,
    step3_paths,
)


def test_evaluation_reports_unweighted_and_design_weighted_metrics(tmp_path: Path) -> None:
    paths = step3_paths(tmp_path / "step3")
    frozen_rows = [
        {
            "sample_id": f"sample-{index}",
            "dataset_id": "2021",
            "application_year": "2021",
            "patent_id": f"CN{index}",
            "title": f"专利{index}",
            "abstract": f"摘要{index}",
            "claim": f"权利要求{index}",
            "ipc": "G06F21/00",
            "main_ipc": "G06F21/00",
        }
        for index in range(4)
    ]
    paths.root.mkdir(parents=True)
    _initialize_task_database(paths.database, frozen_rows)
    paths.manifest.write_text(
        json.dumps(
            {
                "target_size": 4,
                "strata": [
                    {
                        "application_year": "2021",
                        "sampling_group": "positive",
                        "population": 6,
                        "sample": 3,
                    },
                    {
                        "application_year": "2021",
                        "sampling_group": "hard_negative",
                        "population": 1,
                        "sample": 1,
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    step2_inputs = (
        ("CN0", "S", "DATA_SECURITY"),
        ("CN1", "E", "DATA_SECURITY"),
        ("CN2", "S", "OTHER"),
        ("CN3", "S", "DATA_SECURITY"),
    )
    review_labels = ("OTHER", "OTHER", "OTHER", "DATA_SECURITY")
    with paths.results.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                *FROZEN_RESULT_FIELDS,
                "step1_label",
                "step2_label",
                "human_review_label",
            ),
        )
        writer.writeheader()
        for row, review_label, (_, route, step2_label) in zip(
            frozen_rows,
            review_labels,
            step2_inputs,
            strict=True,
        ):
            writer.writerow(
                {
                    **row,
                    "step1_label": "DATA_SECURITY" if route == "S" else "OTHER",
                    "step2_label": step2_label,
                    "human_review_label": review_label,
                }
            )

    step2_database = tmp_path / "step2" / "2021" / "tasks.sqlite3"
    step2_database.parent.mkdir(parents=True)
    connection = sqlite3.connect(step2_database)
    connection.execute(
        "CREATE TABLE tasks ("
        "dataset_id TEXT, patent_id TEXT, route TEXT, status TEXT, result_json TEXT)"
    )
    connection.executemany(
        "INSERT INTO tasks VALUES ('2021', ?, ?, 'succeeded', ?)",
        [
            (patent_id, route, json.dumps({"label": label}))
            for patent_id, route, label in step2_inputs
        ],
    )
    connection.commit()
    connection.close()

    report = evaluate_pipeline_results(paths, [step2_database])

    assert report["sampling_frame"]["eligible_population"] == 7
    assert report["reference"]["label_counts"] == {
        "DATA_SECURITY": 1,
        "OTHER": 3,
    }
    assert report["reference"]["step2_agreement_counts"] == {
        "agreement": 2,
        "disagreement": 2,
    }
    assert report["step1"]["sample_unweighted"]["confusion_matrix"] == {
        "true_positive": 1,
        "true_negative": 1,
        "false_positive": 2,
        "false_negative": 0,
    }
    assert report["step1"]["sample_unweighted"]["accuracy"] == 0.5
    assert report["step1"]["eligible_frame_design_weighted"]["accuracy"] == 0.571429
    assert report["step2"]["sample_unweighted"]["confusion_matrix"] == {
        "true_positive": 1,
        "true_negative": 1,
        "false_positive": 2,
        "false_negative": 0,
    }
    assert report["step2"]["sample_unweighted"]["accuracy"] == 0.5
    assert report["step2"]["eligible_frame_design_weighted"]["accuracy"] == 0.428571
    saved = json.loads(paths.manifest.read_text(encoding="utf-8"))
    assert saved["evaluation"] == report
