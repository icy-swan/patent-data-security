"""Prepare, run and inspect resumable Step 2 classifications."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from pipeline.common.datasets import dataset_id
from pipeline.step2.client import ARK_BASE_URL, VolcengineArkClient
from pipeline.step2.runner import read_progress, run_tasks
from pipeline.step2.tasks import prepare_tasks, task_paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STEP1 = PROJECT_ROOT / "data" / "step1"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "step2"
STOP_REQUESTED = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument(
        "--step1-results",
        type=Path,
        help="Defaults to data/step1/{dataset}/result.csv",
    )
    prepare.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    prepare.add_argument("--encoding", default="utf-8-sig")
    prepare.add_argument("--rebuild", action="store_true")

    run = subparsers.add_parser("run")
    run.add_argument("--input", type=Path, required=True, help="Used to resolve dataset ID")
    run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    run.add_argument("--model", default=os.getenv("ARK_MODEL"))
    run.add_argument("--base-url", default=os.getenv("ARK_BASE_URL", ARK_BASE_URL))
    run.add_argument("--timeout-seconds", type=float, default=180)
    run.add_argument("--concurrency", type=int, default=10)
    run.add_argument("--max-attempts", type=int, default=3)
    run.add_argument("--retry-delay-seconds", type=float, default=2)
    run.add_argument("--pid-file", type=Path, help=argparse.SUPPRESS)

    start = subparsers.add_parser("start")
    start.add_argument("--input", type=Path, required=True, help="Used to resolve dataset ID")
    start.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    start.add_argument("--env-file", type=Path, default=PROJECT_ROOT / "v1" / ".env")
    start.add_argument("--model")
    start.add_argument("--base-url")
    start.add_argument("--timeout-seconds", type=float, default=180)
    start.add_argument("--concurrency", type=int, default=10)
    start.add_argument("--max-attempts", type=int, default=3)
    start.add_argument("--retry-delay-seconds", type=float, default=2)

    stop = subparsers.add_parser("stop")
    stop.add_argument("--input", type=Path, required=True, help="Used to resolve dataset ID")
    stop.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)

    status = subparsers.add_parser("status")
    status.add_argument("--input", type=Path, required=True)
    status.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "prepare":
        token = dataset_id(args.input)
        step1_results = args.step1_results or DEFAULT_STEP1 / token / "result.csv"
        if not step1_results.is_file():
            raise SystemExit(f"Missing Step 1 result: {step1_results}")
        paths, manifest = prepare_tasks(
            args.input,
            step1_results,
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
    if args.command == "start":
        return _start_background(args, paths)
    if args.command == "stop":
        return _stop_background(paths)
    if args.command == "status":
        pid_path, log_path = _runner_paths(paths)
        pid = _read_pid(pid_path)
        print(
            json.dumps(
                {
                    "runner_pid": pid,
                    "runner_active": bool(pid and _pid_is_running(pid)),
                    "log": str(log_path),
                    "progress": read_progress(paths),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if not args.model:
        raise SystemExit("--model or ARK_MODEL is required")

    global STOP_REQUESTED
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    client = VolcengineArkClient(
        model=args.model,
        base_url=args.base_url,
        timeout_seconds=args.timeout_seconds,
    )
    progress: dict[str, Any] | None = None
    try:
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
    finally:
        _remove_own_pid_file(args.pid_file)
    if progress is not None:
        _cleanup_completed_runtime(paths, progress)
    return 0


def _start_background(args: argparse.Namespace, paths: Any) -> int:
    pid_path, log_path = _runner_paths(paths)
    existing_pid = _read_pid(pid_path)
    if existing_pid and _pid_is_running(existing_pid):
        raise SystemExit(f"Step 2 is already running with PID {existing_pid}")
    if not paths.database.is_file():
        raise SystemExit(f"Prepared Step 2 database does not exist: {paths.database}")

    environment = os.environ.copy()
    environment.update(_read_env_file(args.env_file))
    model = args.model or environment.get("ARK_MODEL")
    base_url = args.base_url or environment.get("ARK_BASE_URL") or ARK_BASE_URL
    if not environment.get("ARK_API_KEY"):
        raise SystemExit(f"ARK_API_KEY is missing from environment and {args.env_file}")
    if not model:
        raise SystemExit(f"ARK_MODEL is missing from arguments, environment and {args.env_file}")
    environment["ARK_MODEL"] = model
    environment["ARK_BASE_URL"] = base_url
    environment["PYTHONPATH"] = str(PROJECT_ROOT)

    command = [
        sys.executable,
        "-m",
        "pipeline.step2",
        "run",
        "--input",
        str(args.input.resolve()),
        "--output-dir",
        str(args.output_dir.resolve()),
        "--model",
        model,
        "--base-url",
        base_url,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--concurrency",
        str(args.concurrency),
        "--max-attempts",
        str(args.max_attempts),
        "--retry-delay-seconds",
        str(args.retry_delay_seconds),
        "--pid-file",
        str(pid_path),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(  # noqa: S603 - fixed local module invocation
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "runner_pid": process.pid,
                "model": model,
                "base_url": base_url,
                "concurrency": args.concurrency,
                "log": str(log_path),
                "pid_file": str(pid_path),
                "progress": str(paths.progress),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _stop_background(paths: Any) -> int:
    pid_path, _ = _runner_paths(paths)
    pid = _read_pid(pid_path)
    if not pid or not _pid_is_running(pid):
        print("Step 2 is not running.")
        pid_path.unlink(missing_ok=True)
        return 0
    os.kill(pid, signal.SIGTERM)
    print(f"Sent SIGTERM to Step 2 PID {pid}; in-flight requests will finish before exit.")
    return 0


def _request_stop(_signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def _print_progress(progress: dict[str, Any]) -> None:
    eta = progress["eta_seconds"]
    print(
        f"model={progress['model']} "
        f"completed={progress['completed']}/{progress['total']} "
        f"succeeded={progress['succeeded']} failed={progress['failed']} "
        f"({progress['progress_percent']:.2f}%) "
        f"concurrency={progress['concurrency']} "
        f"elapsed={_format_duration(progress['run_elapsed_seconds'])} "
        f"avg={progress['average_request_seconds']:.2f}s "
        f"eta={_format_duration(eta) if eta is not None else 'unknown'} "
        f"cached_tokens={progress['usage']['cached_tokens']}",
        flush=True,
    )


def _format_duration(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _runner_paths(paths: Any) -> tuple[Path, Path]:
    root = paths.database.parent
    return root / "runner.pid", root / "runner.log"


def _read_pid(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _remove_own_pid_file(path: Path | None) -> None:
    if path is None or _read_pid(path) != os.getpid():
        return
    path.unlink(missing_ok=True)


def _cleanup_completed_runtime(paths: Any, progress: dict[str, Any]) -> None:
    """Keep only the four documented Step 2 artifacts after a complete run."""

    if progress.get("succeeded") != progress.get("total") or progress.get("failed"):
        return
    pid_path, log_path = _runner_paths(paths)
    for path in (
        pid_path,
        log_path,
        paths.database.with_name(paths.database.name + "-wal"),
        paths.database.with_name(paths.database.name + "-shm"),
        paths.database.with_name(paths.database.name + ".run.lock"),
    ):
        path.unlink(missing_ok=True)


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def _paths_json(paths: Any) -> dict[str, str]:
    return {
        "database": str(paths.database),
        "manifest": str(paths.manifest),
        "result": str(paths.results),
        "progress": str(paths.progress),
    }


if __name__ == "__main__":
    raise SystemExit(main())
