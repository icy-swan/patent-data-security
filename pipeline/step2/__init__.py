"""Step 2 independent LLM classification for the S/E task pool.

Keep package initialization lightweight: importing ``pipeline.step2.schema`` from
Step 3 must not also import the Step 1 keyword matcher and its ``ahocorasick``
runtime. The two historical package-level exports remain available lazily.
"""

from __future__ import annotations

from typing import Any

__all__ = ["PROMPT_VERSION", "load_prompt_bundle"]


def __getattr__(name: str) -> Any:
    if name == "PROMPT_VERSION":
        from pipeline.step2.prompt import PROMPT_VERSION

        return PROMPT_VERSION
    if name == "load_prompt_bundle":
        from pipeline.step2.prompt import load_prompt_bundle

        return load_prompt_bundle
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
