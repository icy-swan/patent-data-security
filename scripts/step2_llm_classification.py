#!/usr/bin/env python3
"""Prepare, run, monitor, and stop resumable one-by-one Ark classification."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from patent_data_security.datasets import discover_files
from patent_data_security.step2_prompt import ARK_BASE_URL, VolcengineArkClient
from patent_data_security.step2_runner import (
    prepare_classification_tasks,
    run_classification_tasks,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STOP_REQUESTED = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run", "start"):
        command = subparsers.add_parser(name)
        _add_dataset_arguments(command)
        command.add_argument("--e-sample-rate", type=float, default=0.02)
        command.add_argument("--e-sample-seed", default="step2-e-sample-v1")
        command.add_argument("--rebuild", action="store_true")
        if name in {"run", "start"}:
            command.add_argument("--model", default=os.getenv("ARK_MODEL"))
            command.add_argument("--base-url", default=os.getenv("ARK_BASE_URL", ARK_BASE_URL))
            command.add_argument("--timeout-seconds", type=float, default=180)
            command.add_argument("--max-attempts", type=int, default=3)
            command.add_argument("--retry-delay-seconds", type=float, default=2)

    status = subparsers.add_parser("status")
    status.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "step2")

    stop = subparsers.add_parser("stop")
    stop.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "step2")
    return parser


def _add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        help="Raw CSV; repeat for multiple years. Defaults to scanning --input-dir.",
    )
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--step1-dir", type=Path, default=PROJECT_ROOT / "data" / "step1")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "step2")
    parser.add_argument("--encoding", default="utf-8-sig")


def main() -> int:
    load_dotenv()
    args = build_parser().parse_args()
    if args.command == "status":
        return _status(args.output_dir)
    if args.command == "stop":
        return _stop(args.output_dir)
    if args.command == "start":
        return _start_background(args)
    return _prepare_or_run(args)


def _prepare_or_run(args: argparse.Namespace) -> int:
    global STOP_REQUESTED
    inputs = discover_files(args.input, args.input_dir, args.pattern)
    summaries = []
    if args.command == "run":
        if not args.model:
            raise SystemExit("--model or ARK_MODEL is required")
        if not os.getenv("ARK_API_KEY"):
            raise SystemExit("ARK_API_KEY is required before running model requests")
        signal.signal(signal.SIGTERM, _request_stop)
        signal.signal(signal.SIGINT, _request_stop)
        client = VolcengineArkClient(
            model=args.model,
            base_url=args.base_url,
            timeout_seconds=args.timeout_seconds,
        )
    else:
        client = None

    for source in inputs:
        if STOP_REQUESTED:
            break
        try:
            paths, prepared = prepare_classification_tasks(
                source,
                args.step1_dir,
                args.output_dir,
                e_sample_rate=args.e_sample_rate,
                e_sample_seed=args.e_sample_seed,
                encoding=args.encoding,
                rebuild=args.rebuild,
                stop_requested=lambda: STOP_REQUESTED,
            )
        except InterruptedError:
            break
        item: dict[str, Any] = {"prepared": prepared, "paths": _paths_json(paths)}
        if client is not None:
            item["progress"] = run_classification_tasks(
                paths,
                client,
                max_attempts=args.max_attempts,
                retry_delay_seconds=args.retry_delay_seconds,
                stop_requested=lambda: STOP_REQUESTED,
                progress_callback=_print_progress,
            )
        summaries.append(item)
    print(json.dumps({"datasets": summaries}, ensure_ascii=False, indent=2))
    _remove_own_pid_file(args.output_dir)
    return 0


def _start_background(args: argparse.Namespace) -> int:
    if not args.model:
        raise SystemExit("--model or ARK_MODEL is required")
    if not os.getenv("ARK_API_KEY"):
        raise SystemExit("ARK_API_KEY is required before starting background requests")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pid_path = args.output_dir / "step2_runner.pid"
    if pid_path.is_file() and _pid_is_running(int(pid_path.read_text().strip())):
        raise SystemExit(f"Step 2 runner is already active with PID {pid_path.read_text().strip()}")
    log_path = args.output_dir / "step2_runner.log"
    command = [sys.executable, str(Path(__file__).resolve()), "run"]
    command.extend(_forward_run_arguments(args))
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(  # noqa: S603 - fixed interpreter and local script
            command,
            cwd=PROJECT_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    print(json.dumps({"pid": process.pid, "log": str(log_path), "status": "started"}))
    return 0


def _forward_run_arguments(args: argparse.Namespace) -> list[str]:
    values = [
        "--input-dir",
        str(args.input_dir),
        "--pattern",
        args.pattern,
        "--step1-dir",
        str(args.step1_dir),
        "--output-dir",
        str(args.output_dir),
        "--encoding",
        args.encoding,
        "--e-sample-rate",
        str(args.e_sample_rate),
        "--e-sample-seed",
        args.e_sample_seed,
        "--model",
        args.model,
        "--base-url",
        args.base_url,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--max-attempts",
        str(args.max_attempts),
        "--retry-delay-seconds",
        str(args.retry_delay_seconds),
    ]
    for path in args.input or []:
        values.extend(("--input", str(path)))
    if args.rebuild:
        values.append("--rebuild")
    return values


def _status(output_dir: Path) -> int:
    progress_files = sorted(output_dir.glob("classification_progress_*.json"))
    progress = [json.loads(path.read_text(encoding="utf-8")) for path in progress_files]
    pid_path = output_dir / "step2_runner.pid"
    pid = int(pid_path.read_text().strip()) if pid_path.is_file() else None
    print(
        json.dumps(
            {
                "runner_pid": pid,
                "runner_active": bool(pid and _pid_is_running(pid)),
                "datasets": progress,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _stop(output_dir: Path) -> int:
    pid_path = output_dir / "step2_runner.pid"
    if not pid_path.is_file():
        print(json.dumps({"status": "not_running"}))
        return 0
    pid = int(pid_path.read_text().strip())
    if _pid_is_running(pid):
        os.kill(pid, signal.SIGTERM)
        print(json.dumps({"status": "stop_requested", "pid": pid}))
    else:
        pid_path.unlink(missing_ok=True)
        print(json.dumps({"status": "stale_pid_removed", "pid": pid}))
    return 0


def _request_stop(_signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def _print_progress(progress: dict[str, Any]) -> None:
    eta = progress["eta_seconds"]
    eta_text = "unknown" if eta is None else f"{eta / 3600:.2f}h"
    print(
        f"model={progress['model']} completed={progress['completed']}/{progress['total']} "
        f"({progress['progress_percent']:.2f}%) avg={progress['average_request_seconds']:.2f}s "
        f"eta={eta_text}",
        flush=True,
    )


def _paths_json(paths: Any) -> dict[str, str]:
    return {
        "database": str(paths.database),
        "results": str(paths.results),
        "progress": str(paths.progress),
    }


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _remove_own_pid_file(output_dir: Path) -> None:
    pid_path = output_dir / "step2_runner.pid"
    if pid_path.is_file() and pid_path.read_text().strip() == str(os.getpid()):
        pid_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
