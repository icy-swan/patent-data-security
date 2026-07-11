#!/usr/bin/env python3
"""Step 1 executable: extract keyword/context evidence into S/W/R/E files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from patent_data_security.keyword_extraction import extract_keywords_csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "上市公司专利明细_2021年申请.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "step1",
    )
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--worker-chunksize", type=int, default=100)
    parser.add_argument("--progress-every", type=int, default=100_000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    outputs = extract_keywords_csv(
        args.input,
        args.output_dir,
        encoding=args.encoding,
        workers=args.workers,
        worker_chunksize=args.worker_chunksize,
        progress_every=args.progress_every,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                **{tier: str(path) for tier, path in outputs.by_tier().items()},
                "summary": str(outputs.summary),
                "llm_requests_executed": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
