"""Prepare and simulate independent Step 3 annotations."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from pipeline.step3.client import DEFAULT_MODEL, CodexAnnotationClient
from pipeline.step3.evaluation import evaluate_pipeline_results
from pipeline.step3.k3_review import (
    AGENT_PLAN_BASE_URL,
    DEFAULT_K3_MODEL,
    AgentPlanKimiReviewClient,
    k3_review_paths,
    prepare_k3_review,
    read_k3_progress,
    reset_failed_k3_reviews,
    run_k3_reviews,
)
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

    prepare_k3 = subparsers.add_parser(
        "prepare-k3",
        help="Freeze both review queues for the optional Kimi K3 Agent Plan review",
    )
    prepare_k3.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    prepare_k3.add_argument("--rebuild", action="store_true")

    for command, help_text in (
        ("run-k3", "Run resumable one-patent Kimi K3 review in the foreground"),
        ("start-k3", "Start Kimi K3 review in the background"),
    ):
        k3_run = subparsers.add_parser(command, help=help_text)
        k3_run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
        k3_run.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
        k3_run.add_argument("--model", default=DEFAULT_K3_MODEL)
        k3_run.add_argument("--base-url")
        k3_run.add_argument("--timeout-seconds", type=float, default=300)
        k3_run.add_argument("--concurrency", type=int, default=5)
        k3_run.add_argument("--max-attempts", type=int, default=3)
        k3_run.add_argument("--retry-delay-seconds", type=float, default=2)
        if command == "run-k3":
            k3_run.add_argument("--pid-file", type=Path, help=argparse.SUPPRESS)

    status_k3 = subparsers.add_parser("status-k3", help="Show Kimi K3 review progress")
    status_k3.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)

    stop_k3 = subparsers.add_parser("stop-k3", help="Stop the background Kimi K3 review")
    stop_k3.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)

    retry_k3 = subparsers.add_parser(
        "retry-k3",
        help="Reset terminal failed Kimi K3 rows without changing successful rows",
    )
    retry_k3.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
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
    if args.command == "prepare-k3":
        paths, manifest = prepare_k3_review(
            args.output_dir,
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
    if args.command == "start-k3":
        return _start_k3_background(args)
    if args.command == "status-k3":
        paths = k3_review_paths(args.output_dir)
        pid = _read_pid(paths.pid)
        print(
            json.dumps(
                {
                    "runner_pid": pid,
                    "runner_active": bool(pid and _pid_is_running(pid)),
                    "log": str(paths.log),
                    "progress": read_k3_progress(paths),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "stop-k3":
        return _stop_k3_background(k3_review_paths(args.output_dir))
    if args.command == "retry-k3":
        paths = k3_review_paths(args.output_dir)
        print(
            json.dumps(
                {"reset_failed": reset_failed_k3_reviews(paths)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "run-k3":
        return _run_k3(args)

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


def _run_k3(args: argparse.Namespace) -> int:
    paths = k3_review_paths(args.output_dir)
    if not paths.database.is_file():
        raise SystemExit(f"Run prepare-k3 first; missing {paths.database}")
    environment = os.environ.copy()
    environment.update(_read_env_file(args.env_file))
    base_url = args.base_url or environment.get("ARK_BASE_URL") or AGENT_PLAN_BASE_URL
    api_key = environment.get("ARK_API_KEY")
    api_key_kind = environment.get("ARK_API_KEY_KIND")
    if not api_key:
        raise SystemExit(f"ARK_API_KEY is missing from environment and {args.env_file}")

    global STOP_REQUESTED
    STOP_REQUESTED = False
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    client = AgentPlanKimiReviewClient(
        model=args.model,
        api_key=api_key,
        api_key_kind=api_key_kind,
        base_url=base_url,
        timeout_seconds=args.timeout_seconds,
    )
    progress: dict[str, Any] | None = None
    try:
        progress = run_k3_reviews(
            paths,
            client,
            max_attempts=args.max_attempts,
            retry_delay_seconds=args.retry_delay_seconds,
            concurrency=args.concurrency,
            stop_requested=lambda: STOP_REQUESTED,
            progress_callback=_print_k3_progress,
        )
        print(json.dumps(progress, ensure_ascii=False, indent=2))
    finally:
        _remove_own_pid_file(args.pid_file)
    return 0


def _start_k3_background(args: argparse.Namespace) -> int:
    paths = k3_review_paths(args.output_dir)
    if not paths.database.is_file():
        raise SystemExit(f"Run prepare-k3 first; missing {paths.database}")
    existing_pid = _read_pid(paths.pid)
    if existing_pid and _pid_is_running(existing_pid):
        raise SystemExit(f"K3 review is already running with PID {existing_pid}")

    environment = os.environ.copy()
    environment.update(_read_env_file(args.env_file))
    base_url = args.base_url or environment.get("ARK_BASE_URL") or AGENT_PLAN_BASE_URL
    if not environment.get("ARK_API_KEY"):
        raise SystemExit(f"ARK_API_KEY is missing from environment and {args.env_file}")
    if environment.get("ARK_API_KEY_KIND") != "agent-plan":
        raise SystemExit("K3 review requires ARK_API_KEY_KIND=agent-plan")
    if base_url.rstrip("/") != AGENT_PLAN_BASE_URL:
        raise SystemExit(f"K3 review requires Agent Plan base URL {AGENT_PLAN_BASE_URL}")
    environment["ARK_BASE_URL"] = base_url
    environment["PYTHONPATH"] = str(PROJECT_ROOT)

    command = [
        sys.executable,
        "-m",
        "pipeline.step3",
        "run-k3",
        "--output-dir",
        str(args.output_dir.resolve()),
        "--env-file",
        str(args.env_file.resolve()),
        "--model",
        args.model,
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
        str(paths.pid),
    ]
    with paths.log.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(  # noqa: S603 - fixed local module invocation
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    paths.pid.write_text(f"{process.pid}\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "runner_pid": process.pid,
                "model": args.model,
                "base_url": base_url,
                "request_granularity": "one_patent_per_request",
                "concurrency": args.concurrency,
                "log": str(paths.log),
                "progress": str(paths.progress),
                "result": str(paths.result),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _stop_k3_background(paths: Any) -> int:
    pid = _read_pid(paths.pid)
    if not pid or not _pid_is_running(pid):
        print("K3 review is not running.")
        paths.pid.unlink(missing_ok=True)
        return 0
    os.kill(pid, signal.SIGTERM)
    print(f"Sent SIGTERM to K3 review PID {pid}; in-flight requests will finish before exit.")
    return 0


def _print_k3_progress(progress: dict[str, Any]) -> None:
    print(
        f"k3 {progress['succeeded']}/{progress['total']} "
        f"failed={progress['failed']} running={progress['running']} "
        f"eta={progress['eta_seconds']}s",
        flush=True,
    )


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
    if path is not None and _read_pid(path) == os.getpid():
        path.unlink(missing_ok=True)


def _paths_json(paths: Any) -> dict[str, str]:
    return {key: str(value) for key, value in vars(paths).items()}


if __name__ == "__main__":
    raise SystemExit(main())
