"""Command-line interface for annual patent processing."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from patent_data_security.pipeline import route_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="patent-data-security")
    subparsers = parser.add_subparsers(dest="command", required=True)

    route = subparsers.add_parser("route", help="Run DOCS/IPC routing and E sampling")
    route.add_argument("--input", required=True, type=Path)
    route.add_argument("--output-dir", required=True, type=Path)
    route.add_argument("--encoding", default="utf-8-sig")
    route.add_argument("--limit", type=int)
    route.add_argument("--resume", action="store_true")
    route.add_argument("--overwrite", action="store_true")
    route.add_argument("--checkpoint-every", type=int, default=50_000)
    route.add_argument("--progress-every", type=int, default=100_000)
    route.add_argument("--workers", type=int, default=1)
    route.add_argument("--worker-chunksize", type=int, default=100)

    prepare = subparsers.add_parser("prepare-batches", help="Build upload-ready Batch files")
    prepare.add_argument("--candidates", required=True, type=Path)
    prepare.add_argument("--output-dir", required=True, type=Path)
    prepare.add_argument("--model", default=os.getenv("LLM_MODEL"))
    prepare.add_argument("--max-requests", type=int, default=20_000)
    prepare.add_argument("--max-bytes", type=int, default=180_000_000)

    submit = subparsers.add_parser("submit-batch", help="Submit one prepared Batch file")
    submit.add_argument("--file", required=True, type=Path)

    merge = subparsers.add_parser("merge-batches", help="Validate and merge Batch outputs")
    merge.add_argument("--outputs", required=True, nargs="+", type=Path)
    merge.add_argument("--destination", required=True, type=Path)
    merge.add_argument("--model", required=True)

    audit = subparsers.add_parser("audit", help="Audit route and candidate consistency")
    audit.add_argument("--routes", required=True, type=Path)
    audit.add_argument("--candidates", required=True, type=Path)
    audit.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    if args.command == "route":
        outputs = route_csv(
            args.input,
            args.output_dir,
            encoding=args.encoding,
            limit=args.limit,
            resume=args.resume,
            overwrite=args.overwrite,
            checkpoint_every=args.checkpoint_every,
            progress_every=args.progress_every,
            workers=args.workers,
            worker_chunksize=args.worker_chunksize,
        )
        print(json.dumps({name: str(path) for name, path in outputs.__dict__.items()}, indent=2))
        return 0
    if args.command == "prepare-batches":
        from patent_data_security.llm import prepare_batch_files

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
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "submit-batch":
        from patent_data_security.llm import openai_client_from_env, submit_batch_file

        client = openai_client_from_env()
        print(submit_batch_file(args.file, client=client))
        return 0
    if args.command == "merge-batches":
        from patent_data_security.llm import merge_batch_outputs

        counts = merge_batch_outputs(args.outputs, args.destination, model_name=args.model)
        print(json.dumps(counts, indent=2))
        return 0
    if args.command == "audit":
        from patent_data_security.audit import audit_routes

        result = audit_routes(args.routes, args.candidates, args.output)
        print(json.dumps(result["checks"], indent=2))
        return 0 if result["all_checks_passed"] else 1
    return 2
