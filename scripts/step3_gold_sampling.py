#!/usr/bin/env python3
"""Extract a reproducible, blinded Human Gold annotation sample from Step 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from patent_data_security.step3_sampling import (
    GoldSamplingConfig,
    discover_step2_databases,
    sample_gold_corpus,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        action="append",
        help="Step 2 SQLite database; repeat for multiple years. Defaults to --step2-dir.",
    )
    parser.add_argument(
        "--step2-dir", type=Path, default=PROJECT_ROOT / "data" / "step2"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "data" / "step3"
    )
    parser.add_argument("--target-size", type=int, default=2_000)
    parser.add_argument("--core-size", type=int, default=1_500)
    parser.add_argument("--seed", default="step3-gold-v1")
    parser.add_argument("--rare-subtype-max-population", type=int, default=100)
    parser.add_argument("--high-confidence-threshold", type=float, default=0.90)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Replace an existing frozen Step 3 sample with the requested configuration.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    databases = args.database or discover_step2_databases(args.step2_dir)
    paths, report = sample_gold_corpus(
        databases,
        args.output_dir,
        config=GoldSamplingConfig(
            target_size=args.target_size,
            core_size=args.core_size,
            seed=args.seed,
            rare_subtype_max_population=args.rare_subtype_max_population,
            high_confidence_threshold=args.high_confidence_threshold,
        ),
        rebuild=args.rebuild,
    )
    print(
        json.dumps(
            {
                "status": "sampled",
                "selected": report["selected"],
                "eligible_population": report["eligible_population"],
                "seed": report["seed"],
                "paths": {key: str(value) for key, value in vars(paths).items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
