"""Step 1 keyword/context routing with lightweight package imports."""

from __future__ import annotations

from typing import Any

__all__ = [
    "KeywordBundle",
    "KeywordMatcher",
    "MatchResult",
    "Step1Outputs",
    "load_keyword_bundle",
    "run_step1",
]


def __getattr__(name: str) -> Any:
    if name in {"KeywordMatcher", "MatchResult"}:
        from pipeline.step1.matcher import KeywordMatcher, MatchResult

        return {"KeywordMatcher": KeywordMatcher, "MatchResult": MatchResult}[name]
    if name in {"Step1Outputs", "run_step1"}:
        from pipeline.step1.runner import Step1Outputs, run_step1

        return {"Step1Outputs": Step1Outputs, "run_step1": run_step1}[name]
    if name in {"KeywordBundle", "load_keyword_bundle"}:
        from pipeline.step1.taxonomy import KeywordBundle, load_keyword_bundle

        return {"KeywordBundle": KeywordBundle, "load_keyword_bundle": load_keyword_bundle}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
