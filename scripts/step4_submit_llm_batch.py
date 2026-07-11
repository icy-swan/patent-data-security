#!/usr/bin/env python3
"""Step 4 executable: explicitly submit one prepared file to the external LLM API."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from patent_data_security.llm import openai_client_from_env, submit_batch_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "step4")
    args = parser.parse_args()
    batch_id = submit_batch_file(args.file, client=openai_client_from_env())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipt = args.output_dir / f"submission_{batch_id}.json"
    receipt.write_text(
        json.dumps(
            {
                "batch_id": batch_id,
                "submitted_file": str(args.file.resolve()),
                "submitted_at_utc": datetime.now(UTC).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"batch_id": batch_id, "receipt": str(receipt)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
