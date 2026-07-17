"""CLI for Step 1 S/E keyword and context routing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.common.datasets import dataset_id, discover_files
from pipeline.step1.runner import run_step1
from pipeline.step1.taxonomy import DEFAULT_RESOURCE_DIR, load_keyword_bundle

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append")
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "step1")
    parser.add_argument("--resources", type=Path, default=DEFAULT_RESOURCE_DIR)
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--worker-chunksize", type=int, default=100)
    parser.add_argument("--sqlite-batch-size", type=int, default=5_000)
    parser.add_argument("--progress-every", type=int, default=100_000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--e-sample-rate", type=float, default=0.02)
    parser.add_argument("--e-sample-seed", default="step1-e-random-v2")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    inputs = discover_files(args.input, args.input_dir, args.pattern)
    bundle = load_keyword_bundle(args.resources)
    datasets = []
    for source in inputs:
        token = dataset_id(source)
        expected = (
            args.output_dir / token / "result.csv",
            args.output_dir / token / "manifest.json",
        )
        if args.skip_existing and not args.overwrite and all(path.is_file() for path in expected):
            datasets.append({"dataset_id": token, "input": str(source), "status": "skipped"})
            continue
        outputs = run_step1(
            source,
            args.output_dir,
            bundle=bundle,
            encoding=args.encoding,
            workers=args.workers,
            worker_chunksize=args.worker_chunksize,
            sqlite_batch_size=args.sqlite_batch_size,
            progress_every=args.progress_every,
            limit=args.limit,
            e_sample_rate=args.e_sample_rate,
            e_sample_seed=args.e_sample_seed,
            overwrite=args.overwrite,
        )
        datasets.append(
            {
                "dataset_id": token,
                "input": str(source),
                "status": "completed",
                "result": str(outputs.result),
                "manifest": str(outputs.manifest),
            }
        )
    print(json.dumps({"datasets": datasets, "llm_requests_executed": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
