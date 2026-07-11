#!/usr/bin/env python3
"""Step 2 executable: combine DOCS/IPC routing and build local model candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from patent_data_security.pipeline import route_csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "上市公司专利明细_2021年申请.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "step2")
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--worker-chunksize", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=50_000)
    parser.add_argument("--progress-every", type=int, default=100_000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    outputs = route_csv(
        args.input,
        args.output_dir,
        encoding=args.encoding,
        workers=args.workers,
        worker_chunksize=args.worker_chunksize,
        checkpoint_every=args.checkpoint_every,
        progress_every=args.progress_every,
        limit=args.limit,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    print(json.dumps({key: str(value) for key, value in outputs.__dict__.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
