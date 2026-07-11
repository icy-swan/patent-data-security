"""Post-run consistency and evidence-distribution audit."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


def audit_routes(
    routes_path: str | Path,
    candidates_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    routes = Path(routes_path)
    candidates = Path(candidates_path)
    route_levels: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    groups: Counter[str] = Counter()
    keywords: Counter[str] = Counter()
    ipc_rules: Counter[str] = Counter()
    diagnostics: Counter[str] = Counter()
    classification_keys: set[str] = set()
    route_rows = 0
    e_samples = 0

    with routes.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            route_rows += 1
            route_levels[row["route_level"]] += 1
            statuses[row["process_status"]] += 1
            e_samples += int(row["is_e_sample"] == "true")
            if row["classification_key"]:
                classification_keys.add(row["classification_key"])
            for hit in json.loads(row["keyword_hits"]):
                groups[hit["group_id"]] += 1
                keywords[hit["keyword"]] += 1
            for hit in json.loads(row["ipc_hits"]):
                ipc_rules[hit["rule_id"]] += 1
            for hit in json.loads(row["diagnostic_hits"]):
                diagnostics[hit["keyword"]] += 1

    candidate_rows = 0
    candidate_ids: set[str] = set()
    candidate_custom_ids: set[str] = set()
    with candidates.open(encoding="utf-8") as file:
        for line in file:
            candidate = json.loads(line)
            candidate_rows += 1
            candidate_ids.add(candidate["patent_id"])
            candidate_custom_ids.add(candidate["custom_id"])

    expected_candidate_route_rows = route_rows - route_levels["E"] + e_samples
    checks = {
        "route_level_sum_matches_rows": sum(route_levels.values()) == route_rows,
        "candidate_route_statuses_match": (
            statuses["pending_llm"] + statuses["pending_llm_shared"]
            == expected_candidate_route_rows
        ),
        "candidate_jsonl_is_unique_by_patent": candidate_rows == len(candidate_ids),
        "candidate_custom_ids_are_unique": candidate_rows == len(candidate_custom_ids),
        "route_keys_match_candidate_custom_ids": classification_keys == candidate_custom_ids,
    }
    result = {
        "schema_version": "1.0.0",
        "routes_path": str(routes.resolve()),
        "candidates_path": str(candidates.resolve()),
        "route_rows": route_rows,
        "candidate_route_rows": expected_candidate_route_rows,
        "unique_candidates": candidate_rows,
        "e_samples": e_samples,
        "route_levels": dict(route_levels),
        "process_statuses": dict(statuses),
        "top_keyword_groups": groups.most_common(30),
        "top_keywords": keywords.most_common(50),
        "top_ipc_rules": ipc_rules.most_common(30),
        "top_diagnostics": diagnostics.most_common(30),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, target)
    return result
