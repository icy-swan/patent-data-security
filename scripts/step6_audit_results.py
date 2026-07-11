#!/usr/bin/env python3
"""Step 6 executable: audit route/candidate consistency into a dedicated artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from patent_data_security.audit import audit_routes

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--routes",
        type=Path,
        default=PROJECT_ROOT / "data" / "step2" / "patent_routes_2021.csv",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=PROJECT_ROOT / "data" / "step2" / "patent_llm_candidates_2021.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "step6" / "route_audit_2021.json",
    )
    args = parser.parse_args()
    result = audit_routes(args.routes, args.candidates, args.output)
    print(json.dumps({"output": str(args.output), "checks": result["checks"]}, indent=2))
    return 0 if result["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
