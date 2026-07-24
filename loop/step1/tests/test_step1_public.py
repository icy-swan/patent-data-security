from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from loop.step1.step1_public import run_public_step1


class PublicStep1Tests(unittest.TestCase):
    def test_routes_deduplicates_and_keeps_source_address_blank(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rules = {
                "schema_version": "test",
                "matching": {
                    "fields": ["claim", "abstract", "title"],
                    "context_window_chars": 20,
                    "ipc_changes_route": False,
                },
                "concepts": [
                    {
                        "concept_id": "TOY-STANDALONE",
                        "category": "technical",
                        "canonical_term": "toy standalone",
                        "variants": ["alpha shield"],
                        "match_policy": {"mode": "standalone"},
                    },
                    {
                        "concept_id": "TOY-CONTEXT",
                        "category": "descriptive",
                        "canonical_term": "toy contextual",
                        "variants": ["protect"],
                        "match_policy": {
                            "mode": "cooccurrence",
                            "required_any": ["TOY-DATA"],
                        },
                    },
                ],
                "context_lexicons": [
                    {
                        "context_id": "TOY-DATA",
                        "kind": "object",
                        "variants": ["dataset"],
                    }
                ],
                "diagnostic_patterns": [
                    {
                        "pattern_id": "TOY-DIAGNOSTIC",
                        "variants": ["physical safety"],
                    }
                ],
                "ipc_audit_rules": [
                    {"rule_id": "TOY-IPC", "symbol": "G00X"}
                ],
            }
            (root / "rules.json").write_text(
                json.dumps(rules),
                encoding="utf-8",
            )
            (root / "input.csv").write_text(
                "patent_id,title,abstract,claim,ipc,applicant,application_date\n"
                "P1,ordinary,ordinary text,,,A,2020-01-01\n"
                "P1,better,contains alpha shield,,,A,2020-01-01\n"
                "P2,contextual,,protect the dataset,,B,2020-01-02\n"
                "P3,physical safety,ordinary text,,G00X,B,2020-01-03\n",
                encoding="utf-8",
            )
            config = {
                "input_csv": "",
                "output_dir": "output",
                "rules_file": "rules.json",
                "e_sample_rate": 1.0,
                "e_sample_seed": "toy-seed",
                "columns": {
                    field: field
                    for field in (
                        "patent_id",
                        "title",
                        "abstract",
                        "claim",
                        "ipc",
                        "applicant",
                        "application_date",
                    )
                },
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            outputs = run_public_step1(
                config_path,
                input_override=root / "input.csv",
            )

            with outputs.result.open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))
            by_id = {row["patent_id"]: row for row in rows}
            self.assertEqual(len(rows), 3)
            self.assertEqual(by_id["P1"]["route"], "S")
            self.assertEqual(by_id["P1"]["association_count"], "2")
            self.assertEqual(by_id["P2"]["route"], "S")
            self.assertEqual(by_id["P3"]["route"], "E")
            self.assertEqual(by_id["P3"]["selection_group"], "E_random")
            self.assertEqual(
                json.loads(by_id["P3"]["diagnostic_hits"])[0]["pattern_id"],
                "TOY-DIAGNOSTIC",
            )
            self.assertTrue(json.loads(by_id["P3"]["ipc_audit_hits"])[0]["audit_only"])

            manifest = json.loads(outputs.manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest["input_source"], "")
            self.assertNotIn(str(root), json.dumps(manifest))
            self.assertFalse(manifest["disclosure"]["production_taxonomy_included"])
            self.assertEqual(manifest["llm_requests_executed"], 0)

    def test_empty_public_rules_route_every_record_to_e(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rules_path = root / "rules.json"
            rules_path.write_text(
                json.dumps(
                    {
                        "schema_version": "test-empty",
                        "matching": {
                            "fields": ["claim", "abstract", "title"],
                            "context_window_chars": 48,
                            "ipc_changes_route": False,
                        },
                        "concepts": [],
                        "context_lexicons": [],
                        "diagnostic_patterns": [],
                        "ipc_audit_rules": [],
                    }
                ),
                encoding="utf-8",
            )
            input_path = root / "input.csv"
            input_path.write_text(
                "patent_id,title,abstract,claim,ipc,applicant,application_date\n"
                "P1,example,example,example,,A,2020-01-01\n",
                encoding="utf-8",
            )
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "input_csv": "",
                        "output_dir": "output",
                        "rules_file": "rules.json",
                        "e_sample_rate": 1.0,
                        "columns": {
                            field: field
                            for field in (
                                "patent_id",
                                "title",
                                "abstract",
                                "claim",
                                "ipc",
                                "applicant",
                                "application_date",
                            )
                        },
                    }
                ),
                encoding="utf-8",
            )

            outputs = run_public_step1(config_path, input_override=input_path)

            with outputs.result.open(encoding="utf-8-sig", newline="") as file:
                row = next(csv.DictReader(file))
            self.assertEqual(row["route"], "E")
            self.assertEqual(row["step1_label"], "OTHER")


if __name__ == "__main__":
    unittest.main()
