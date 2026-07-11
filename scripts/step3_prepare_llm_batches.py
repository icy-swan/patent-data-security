#!/usr/bin/env python3
"""Step 3 executable: create LLM Batch request files locally without API calls."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from patent_data_security.llm import prepare_batch_files

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=PROJECT_ROOT / "data" / "step2" / "patent_llm_candidates_2021.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "step3")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL"))
    parser.add_argument("--max-requests", type=int, default=20_000)
    parser.add_argument("--max-bytes", type=int, default=180_000_000)
    args = parser.parse_args()
    if not args.model:
        raise SystemExit("--model or LLM_MODEL is required")
    prepared = prepare_batch_files(
        args.candidates,
        args.output_dir,
        model=args.model,
        max_requests=args.max_requests,
        max_bytes=args.max_bytes,
    )
    print(
        json.dumps(
            {
                "requests": prepared.requests,
                "files": [str(path) for path in prepared.files],
                "manifest": str(prepared.manifest),
                "llm_requests_executed": 0,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
