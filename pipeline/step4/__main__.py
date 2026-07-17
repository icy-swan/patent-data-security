"""Prepare Step 4 datasets and train the paper-style RoBERTa classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.step4.data import DATASET_VERSION, paths_as_json, prepare_datasets, step4_paths
from pipeline.step4.train import DEFAULT_MODEL, TEXT_FIELDS, train_roberta

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STEP3 = PROJECT_ROOT / "data" / "step3" / "positive-priority-v2.2.0"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "step4" / DATASET_VERSION


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="Validate Step 3 and export classifier plus MaaS JSONL"
    )
    prepare.add_argument("--step3-dir", type=Path, default=DEFAULT_STEP3)
    prepare.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    prepare.add_argument("--rebuild", action="store_true")

    train = subparsers.add_parser("train-roberta", help="Fine-tune and evaluate RoBERTa")
    train.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    train.add_argument("--model", default=DEFAULT_MODEL)
    train.add_argument(
        "--text-fields",
        nargs="+",
        choices=TEXT_FIELDS,
        default=["abstract"],
        help="Paper-compatible default is abstract only",
    )
    train.add_argument("--max-length", type=int, default=512)
    train.add_argument("--epochs", type=float, default=4)
    train.add_argument("--learning-rate", type=float, default=2e-5)
    train.add_argument("--train-batch-size", type=int, default=16)
    train.add_argument("--eval-batch-size", type=int, default=32)
    train.add_argument("--gradient-accumulation-steps", type=int, default=1)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--warmup-ratio", type=float, default=0.1)
    train.add_argument("--seed", type=int, default=42)
    precision = train.add_mutually_exclusive_group()
    precision.add_argument("--fp16", action="store_true")
    precision.add_argument("--bf16", action="store_true")
    train.add_argument("--gradient-checkpointing", action="store_true")
    train.add_argument("--resume-from-checkpoint", type=Path)
    train.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "prepare":
        paths, manifest = prepare_datasets(
            args.step3_dir,
            args.output_dir,
            rebuild=args.rebuild,
        )
        print(
            json.dumps(
                {"paths": paths_as_json(paths), "manifest": manifest},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    paths = step4_paths(args.output_dir)
    report = train_roberta(
        paths,
        model_name=args.model,
        text_fields=args.text_fields,
        max_length=args.max_length,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        resume_from_checkpoint=args.resume_from_checkpoint,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
