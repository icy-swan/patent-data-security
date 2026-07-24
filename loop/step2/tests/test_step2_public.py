from __future__ import annotations

import csv
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Any

from loop.step2.step2_public import (
    RESULT_FIELDS,
    prepare_public_step2,
    run_public_step2,
    validate_decision,
)


class FakeAdapter:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def classify(
        self,
        *,
        system_prompt: str,
        patent: dict[str, str],
    ) -> dict[str, Any]:
        self.calls.append(dict(patent))
        self.assert_prompt_is_minimal(system_prompt)
        return {
            "label": "DATA_SECURITY" if "toy shield" in patent["claim"] else "OTHER",
            "reason": "主权项是否披露 toy shield 决定本条测试标签。",
            "evidence": [
                {
                    "field": "claim",
                    "quote": patent["claim"],
                }
            ],
        }

    @staticmethod
    def assert_prompt_is_minimal(prompt: str) -> None:
        assert '"label"' in prompt
        assert '"reason"' in prompt
        assert '"evidence"' in prompt
        assert "token" in prompt
        assert "置信度" in prompt


class PublicStep2Tests(unittest.TestCase):
    def test_prepare_run_and_export_minimal_paper_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            step1 = root / "step1.csv"
            self.write_step1(step1)
            prompt = root / "prompt.txt"
            prompt.write_text(
                '只返回 {"label":"", "reason":"", "evidence":[]}，'
                "不要返回 token 或置信度。",
                encoding="utf-8",
            )
            config = {
                "step1_results": [],
                "output_dir": "step2",
                "prompt_file": "prompt.txt",
                "pool_size": 2,
                "pool_seed": "toy-pool",
                "encoding": "utf-8-sig",
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            paths, manifest = prepare_public_step2(
                config_path,
                step1_overrides=[step1],
            )

            self.assertEqual(manifest["model_adapter"]["endpoint"], "")
            self.assertEqual(manifest["input_source_addresses"], [])
            self.assertNotIn(str(root), json.dumps(manifest))
            self.assertFalse(paths.result.exists())

            adapter = FakeAdapter()
            progress = run_public_step2(paths, adapter, concurrency=2)

            self.assertEqual(progress["succeeded"], 2)
            self.assertEqual(len(adapter.calls), 2)
            self.assertTrue(
                all(
                    set(patent) == {"title", "abstract", "claim", "ipc"}
                    for patent in adapter.calls
                )
            )
            with paths.result.open(encoding="utf-8-sig", newline="") as file:
                reader = csv.DictReader(file)
                rows = list(reader)
            self.assertEqual(tuple(reader.fieldnames or ()), RESULT_FIELDS)
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["application_year"] for row in rows}, {"2020", "2021"})
            self.assertTrue(
                all(row["combined_step2_inclusion_probability"] for row in rows)
            )
            forbidden = {
                "confidence",
                "usage",
                "tokens",
                "elapsed_seconds",
                "response_id",
                "model",
                "endpoint",
            }
            self.assertFalse(forbidden & set(reader.fieldnames or ()))

    def test_rejects_confidence_and_non_verbatim_evidence(self) -> None:
        patent = {
            "title": "toy title",
            "abstract": "toy abstract",
            "claim": "toy claim",
        }
        with self.assertRaisesRegex(ValueError, "多余"):
            validate_decision(
                {
                    "label": "OTHER",
                    "reason": "这是一个足够长的理由。",
                    "evidence": [{"field": "claim", "quote": "toy claim"}],
                    "confidence": 0.9,
                },
                patent,
            )
        with self.assertRaisesRegex(ValueError, "逐字引文"):
            validate_decision(
                {
                    "label": "OTHER",
                    "reason": "这是一个足够长的理由。",
                    "evidence": [{"field": "claim", "quote": "not in claim"}],
                },
                patent,
            )

    def test_adapter_error_does_not_persist_request_details(self) -> None:
        class FailingAdapter:
            def classify(
                self,
                *,
                system_prompt: str,
                patent: dict[str, str],
            ) -> dict[str, Any]:
                del system_prompt, patent
                raise RuntimeError(
                    "private service location and request identifier must not be stored"
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            step1 = root / "step1.csv"
            self.write_step1(step1)
            prompt = root / "prompt.txt"
            prompt.write_text(
                '只返回 {"label":"", "reason":"", "evidence":[]}',
                encoding="utf-8",
            )
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "step1_results": [],
                        "output_dir": "step2",
                        "prompt_file": "prompt.txt",
                        "pool_size": 1,
                    }
                ),
                encoding="utf-8",
            )
            paths, _ = prepare_public_step2(
                config_path,
                step1_overrides=[step1],
            )

            progress = run_public_step2(
                paths,
                FailingAdapter(),
                max_attempts=1,
            )

            self.assertEqual(progress["failed"], 1)
            connection = sqlite3.connect(paths.database)
            try:
                stored = connection.execute(
                    "SELECT error_code FROM tasks"
                ).fetchone()[0]
            finally:
                connection.close()
            self.assertEqual(stored, "adapter_error")
            self.assertNotIn("private service", paths.database.read_bytes().decode(
                "utf-8",
                errors="ignore",
            ))

    @staticmethod
    def write_step1(path: Path) -> None:
        fields = (
            "dataset_id",
            "patent_id",
            "application_date",
            "title",
            "abstract",
            "claim",
            "ipc",
            "route",
            "selected_for_step2",
            "selection_probability",
        )
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "dataset_id": "toy",
                        "patent_id": "P1",
                        "application_date": "2020-01-01",
                        "title": "toy one",
                        "abstract": "ordinary",
                        "claim": "toy shield",
                        "ipc": "",
                        "route": "S",
                        "selected_for_step2": "true",
                        "selection_probability": "1",
                    },
                    {
                        "dataset_id": "toy",
                        "patent_id": "P2",
                        "application_date": "2021-01-01",
                        "title": "toy two",
                        "abstract": "ordinary",
                        "claim": "ordinary claim",
                        "ipc": "",
                        "route": "E",
                        "selected_for_step2": "true",
                        "selection_probability": "0.5",
                    },
                    {
                        "dataset_id": "toy",
                        "patent_id": "P3",
                        "application_date": "2022-01-01",
                        "title": "toy three",
                        "abstract": "ordinary",
                        "claim": "ordinary claim",
                        "ipc": "",
                        "route": "E",
                        "selected_for_step2": "false",
                        "selection_probability": "0.5",
                    },
                ]
            )


if __name__ == "__main__":
    unittest.main()
