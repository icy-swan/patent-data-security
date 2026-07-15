"""Prepare, run and inspect resumable Step 2 classifications."""

from __future__ import annotations

import argparse
import json
import os
import signal
from pathlib import Path
from typing import Any

from pipeline.common.datasets import dataset_id
from pipeline.step2.client import ARK_BASE_URL, OpenAICompatibleClient
from pipeline.step2.runner import read_progress, run_tasks
from pipeline.step2.tasks import prepare_tasks, task_paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STOP_REQUESTED = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--step1-results", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "step2")
    prepare.add_argument("--encoding", default="utf-8-sig")
    prepare.add_argument("--rebuild", action="store_true")

    run = subparsers.add_parser("run")
    run.add_argument("--input", type=Path, required=True, help="Used to resolve dataset ID")
    run.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "step2")
    run.add_argument("--model", default=os.getenv("ARK_MODEL") or os.getenv("OPENAI_MODEL"))
    run.add_argument("--base-url", default=os.getenv("ARK_BASE_URL", ARK_BASE_URL))
    run.add_argument("--timeout-seconds", type=float, default=180)
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument("--max-attempts", type=int, default=3)
    run.add_argument("--retry-delay-seconds", type=float, default=2)
    run.add_argument("--prompt-cache-key")

    status = subparsers.add_parser("status")
    status.add_argument("--input", type=Path, required=True)
    status.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "step2")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "prepare":
        paths, manifest = prepare_tasks(
            args.input,
            args.step1_results,
            args.output_dir,
            encoding=args.encoding,
            rebuild=args.rebuild,
        )
        print(
            json.dumps(
                {"paths": _paths_json(paths), "manifest": manifest},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    paths = task_paths(args.output_dir, dataset_id(args.input))
    if args.command == "status":
        print(json.dumps(read_progress(paths), ensure_ascii=False, indent=2))
        return 0
    if not args.model:
        raise SystemExit("--model, ARK_MODEL or OPENAI_MODEL is required")

    global STOP_REQUESTED
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    client = OpenAICompatibleClient(
        model=args.model,
        base_url=args.base_url,
        timeout_seconds=args.timeout_seconds,
        prompt_cache_key=args.prompt_cache_key,
    )
    progress = run_tasks(
        paths,
        client,
        max_attempts=args.max_attempts,
        retry_delay_seconds=args.retry_delay_seconds,
        concurrency=args.concurrency,
        stop_requested=lambda: STOP_REQUESTED,
        progress_callback=_print_progress,
    )
    print(json.dumps(progress, ensure_ascii=False, indent=2))
    return 0


def _request_stop(_signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def _print_progress(progress: dict[str, Any]) -> None:
    print(
        f"completed={progress['completed']}/{progress['total']} "
        f"({progress['progress_percent']:.2f}%) "
        f"cached_tokens={progress['usage']['cached_tokens']}",
        flush=True,
    )


def _paths_json(paths: Any) -> dict[str, str]:
    return {
        "database": str(paths.database),
        "manifest": str(paths.manifest),
        "results": str(paths.results),
        "progress": str(paths.progress),
    }


if __name__ == "__main__":
    raise SystemExit(main())
