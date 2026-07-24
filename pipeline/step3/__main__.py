"""Prepare and simulate independent Step 3 annotations."""

from __future__ import annotations

import argparse
import json
import os
import signal
from pathlib import Path
from typing import Any

from pipeline.step3.client import DEFAULT_MODEL, CodexAnnotationClient
from pipeline.step3.evaluation import evaluate_pipeline_results
from pipeline.step3.runner import read_progress, run_simulation
from pipeline.step3.sampling import (
    NEGATIVE_SAMPLE_SEED,
    SamplingConfig,
    discover_step2_databases,
    finalize_human_results,
    merge_review_results,
    prepare_negative_sample,
    prepare_sample,
    step3_paths,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STEP2 = PROJECT_ROOT / "data" / "step2"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "step3"
STOP_REQUESTED = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Freeze a new 5,000-record sample")
    sources = prepare.add_mutually_exclusive_group()
    sources.add_argument("--step2-dir", type=Path, default=DEFAULT_STEP2)
    sources.add_argument("--database", type=Path, action="append")
    prepare.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    prepare.add_argument("--seed", default=SamplingConfig().seed)
    prepare.add_argument("--rebuild", action="store_true")

    prepare_negative = subparsers.add_parser(
        "prepare-negative",
        help=(
            "Append 2,000 predicted positives, 1,000 hard negatives "
            "and 2,000 easy negatives"
        ),
    )
    negative_sources = prepare_negative.add_mutually_exclusive_group()
    negative_sources.add_argument("--step2-dir", type=Path, default=DEFAULT_STEP2)
    negative_sources.add_argument("--database", type=Path, action="append")
    prepare_negative.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    prepare_negative.add_argument("--seed", default=NEGATIVE_SAMPLE_SEED)

    merge = subparsers.add_parser(
        "merge",
        help="Validate result_positive.csv and result_negative.csv, then create result.csv",
    )
    merge.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)

    simulate = subparsers.add_parser("simulate", help="Run provisional local Codex annotations")
    simulate.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    simulate.add_argument("--model", default=os.getenv("CODEX_MODEL", DEFAULT_MODEL))
    simulate.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high", "xhigh", "max"),
        default="low",
    )
    simulate.add_argument("--batch-size", type=int, default=20)
    simulate.add_argument("--max-attempts", type=int, default=3)
    simulate.add_argument("--retry-delay-seconds", type=float, default=2)
    simulate.add_argument("--timeout-seconds", type=float, default=1_800)

    status = subparsers.add_parser("status", help="Show simulation progress")
    status.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)

    finalize = subparsers.add_parser(
        "finalize", help="Validate result.csv and create clean 8:1:1 human splits"
    )
    finalize.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    finalize.add_argument("--step2-dir", type=Path, default=DEFAULT_STEP2)
    finalize.add_argument("--database", type=Path, action="append")
    finalize.add_argument("--split-seed", default="step3-human-split-v2.6.0")

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate Step 1 and Step 2 against result.csv",
    )
    evaluate.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    evaluate.add_argument("--step2-dir", type=Path, default=DEFAULT_STEP2)
    evaluate.add_argument("--database", type=Path, action="append")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "prepare":
        databases = args.database or discover_step2_databases(args.step2_dir)
        paths, manifest = prepare_sample(
            databases,
            args.output_dir,
            config=SamplingConfig(seed=args.seed),
            rebuild=args.rebuild,
        )
        print(
            json.dumps(
                {"paths": _paths_json(paths), "manifest": manifest}, ensure_ascii=False, indent=2
            )
        )
        return 0
    if args.command == "prepare-negative":
        databases = args.database or discover_step2_databases(args.step2_dir)
        paths, manifest = prepare_negative_sample(
            databases,
            args.output_dir,
            seed=args.seed,
        )
        print(
            json.dumps(
                {"paths": _paths_json(paths), "manifest": manifest},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    paths = step3_paths(args.output_dir)
    if args.command == "merge":
        print(json.dumps(merge_review_results(paths), ensure_ascii=False, indent=2))
        return 0
    if args.command == "status":
        print(json.dumps(read_progress(paths), ensure_ascii=False, indent=2))
        return 0
    if args.command == "finalize":
        if not paths.results.is_file():
            raise SystemExit(f"Missing human annotation file: {paths.results}")
        split_report = finalize_human_results(paths, split_seed=args.split_seed)
        evaluation = evaluate_pipeline_results(
            paths,
            args.database or discover_step2_databases(args.step2_dir),
        )
        print(
            json.dumps(
                {"split_report": split_report, "evaluation": evaluation},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "evaluate":
        evaluation = evaluate_pipeline_results(
            paths,
            args.database or discover_step2_databases(args.step2_dir),
        )
        print(json.dumps(evaluation, ensure_ascii=False, indent=2))
        return 0
    if not paths.database.is_file():
        raise SystemExit(f"Run prepare first; missing {paths.database}")

    global STOP_REQUESTED
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    client = CodexAnnotationClient(
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        workspace=PROJECT_ROOT,
        timeout_seconds=args.timeout_seconds,
    )
    progress = run_simulation(
        paths,
        client,
        batch_size=args.batch_size,
        max_attempts=args.max_attempts,
        retry_delay_seconds=args.retry_delay_seconds,
        stop_requested=lambda: STOP_REQUESTED,
        progress_callback=lambda value: print(
            f"step3 {value['completed']}/{value['total']} "
            f"ok={value['succeeded']} failed={value['failed']} eta={value['eta_seconds']}s",
            flush=True,
        ),
    )
    print(json.dumps(progress, ensure_ascii=False, indent=2))
    return 0


def _request_stop(_signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def _paths_json(paths: Any) -> dict[str, str]:
    return {key: str(value) for key, value in vars(paths).items()}


if __name__ == "__main__":
    raise SystemExit(main())
