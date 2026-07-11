#!/usr/bin/env python3
"""Step 1 executable: extract keyword/context evidence into S/W/R/E files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from patent_data_security.datasets import dataset_id, discover_files
from patent_data_security.keyword_extraction import extract_keywords_csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        help="Input CSV; repeat for multiple files. Defaults to scanning --input-dir.",
    )
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--pattern", default="*.csv")
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
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip datasets whose complete Step 1 outputs already exist.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    inputs = discover_files(args.input, args.input_dir, args.pattern)
    results = []
    for source in inputs:
        dataset = dataset_id(source)
        expected = [
            args.output_dir / f"keyword_{tier}_{dataset}.csv" for tier in "SWRE"
        ] + [args.output_dir / f"keyword_summary_{dataset}.json"]
        if args.skip_existing and not args.overwrite and all(path.is_file() for path in expected):
            results.append({"dataset_id": dataset, "input": str(source), "status": "skipped"})
            continue
        outputs = extract_keywords_csv(
            source,
            args.output_dir,
            encoding=args.encoding,
            workers=args.workers,
            worker_chunksize=args.worker_chunksize,
            progress_every=args.progress_every,
            limit=args.limit,
            overwrite=args.overwrite,
        )
        results.append(
            {
                "dataset_id": dataset,
                "input": str(source),
                "status": "completed",
                "outputs": {
                    **{tier: str(path) for tier, path in outputs.by_tier().items()},
                    "summary": str(outputs.summary),
                },
            }
        )
    print(
        json.dumps(
            {"datasets": results, "llm_requests_executed": 0},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
