#!/usr/bin/env python3
"""Step 5 executable: validate downloaded Batch outputs and merge classifications."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from patent_data_security.llm import merge_batch_outputs

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", required=True, nargs="+", type=Path)
    parser.add_argument(
        "--destination",
        type=Path,
        default=PROJECT_ROOT / "data" / "step5" / "patent_classifications_2021.csv",
    )
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    counts = merge_batch_outputs(args.outputs, args.destination, model_name=args.model)
    print(json.dumps({"destination": str(args.destination), "counts": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
