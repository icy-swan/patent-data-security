from __future__ import annotations

import csv
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from loop.step3.step3_public import (
    OUTPUT_FIELDS,
    balanced_capacity_allocation,
    prepare_public_step3,
)


class PublicStep3Tests(unittest.TestCase):
    def test_dual_cohort_sampling_is_exact_disjoint_and_stable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            step2 = root / "step2.csv"
            self.write_population(step2)
            config_path = root / "config.json"
            self.write_config(config_path)

            first_paths, first_manifest = prepare_public_step3(
                config_path,
                step2_override=step2,
                output_override=root / "first",
            )
            second_paths, _ = prepare_public_step3(
                config_path,
                step2_override=step2,
                output_override=root / "second",
            )

            positive = self.read_rows(first_paths.positive_review)
            negative = self.read_rows(first_paths.negative_review)
            combined = self.read_rows(first_paths.combined_sample)
            repeated = self.read_rows(second_paths.combined_sample)

            self.assertEqual(len(positive), 6)
            self.assertEqual(len(negative), 6)
            self.assertEqual(len(combined), 12)
            self.assertFalse(
                {row["patent_id"] for row in positive}
                & {row["patent_id"] for row in negative}
            )
            self.assertEqual(
                Counter(row["sampling_group"] for row in combined),
                {"positive": 6, "hard_negative": 4, "easy_negative": 2},
            )
            self.assertEqual(
                [(row["sample_id"], row["sample_cohort"]) for row in combined],
                [(row["sample_id"], row["sample_cohort"]) for row in repeated],
            )
            self.assertTrue(
                all(not row["human_review_label"] and not row["human_reason"] for row in combined)
            )
            self.assertTrue(
                all(row["combined_inclusion_probability"] for row in combined)
            )
            self.assertEqual(tuple(combined[0]), OUTPUT_FIELDS)
            self.assertEqual(first_manifest["input"]["source_address"], "")
            self.assertNotIn(str(root), json.dumps(first_manifest))
            self.assertFalse(
                first_manifest["review_policy"]["automated_review_included"]
            )
            self.assertEqual(first_manifest["model_requests_executed"], 0)

    def test_balanced_allocation_redistributes_after_capacity_is_full(self) -> None:
        allocation = balanced_capacity_allocation(
            {"2019": 1, "2020": 5, "2021": 5},
            7,
            seed="toy",
        )

        self.assertEqual(allocation["2019"], 1)
        self.assertEqual(sum(allocation.values()), 7)
        self.assertLessEqual(abs(allocation["2020"] - allocation["2021"]), 1)

    def test_rejects_incomplete_step2_population(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            step2 = root / "step2.csv"
            self.write_population(step2)
            config_path = root / "config.json"
            self.write_config(config_path, expected_population=25)

            with self.assertRaisesRegex(ValueError, "完整、冻结"):
                prepare_public_step3(config_path, step2_override=step2)

    @staticmethod
    def write_config(path: Path, *, expected_population: int = 20) -> None:
        path.write_text(
            json.dumps(
                {
                    "step2_result": "",
                    "output_dir": "output",
                    "expected_population_size": expected_population,
                    "encoding": "utf-8-sig",
                    "cohorts": {
                        "positive_priority": {
                            "seed": "toy-positive",
                            "group_targets": {
                                "positive": 4,
                                "hard_negative": 2,
                                "easy_negative": 0,
                            },
                        },
                        "negative_priority": {
                            "seed": "toy-negative",
                            "group_targets": {
                                "positive": 2,
                                "hard_negative": 2,
                                "easy_negative": 2,
                            },
                        },
                    },
                }
            ),
            encoding="utf-8",
        )

    @staticmethod
    def write_population(path: Path) -> None:
        fields = (
            "task_id",
            "dataset_id",
            "patent_id",
            "application_date",
            "application_year",
            "title",
            "abstract",
            "claim",
            "ipc",
            "step1_route",
            "step2_label",
            "step2_reason",
            "step2_evidence",
            "combined_step2_inclusion_probability",
        )
        rows = []
        index = 0
        for year in ("2020", "2021"):
            for group, count in (
                ("positive", 4),
                ("hard_negative", 3),
                ("easy_negative", 3),
            ):
                for _ in range(count):
                    index += 1
                    route = "S" if group == "hard_negative" else "E"
                    label = "DATA_SECURITY" if group == "positive" else "OTHER"
                    rows.append(
                        {
                            "task_id": f"T{index}",
                            "dataset_id": "toy",
                            "patent_id": f"P{index:03d}",
                            "application_date": f"{year}-01-01",
                            "application_year": year,
                            "title": f"title {index}",
                            "abstract": f"abstract {index}",
                            "claim": f"claim {index}",
                            "ipc": "",
                            "step1_route": route,
                            "step2_label": label,
                            "step2_reason": "这是用于公开测试的充分理由。",
                            "step2_evidence": json.dumps(
                                [{"field": "claim", "quote": f"claim {index}"}]
                            ),
                            "combined_step2_inclusion_probability": "0.5",
                        }
                    )
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def read_rows(path: Path) -> list[dict[str, str]]:
        with path.open(encoding="utf-8-sig", newline="") as file:
            return list(csv.DictReader(file))


if __name__ == "__main__":
    unittest.main()
